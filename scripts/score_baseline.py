"""Phase 2: score every passage once and write data/baseline_scores.parquet.

Scores go to data/cache/<endpoint>.jsonl as they arrive, so an interrupted run
resumes where it stopped. At SlopTotal's CPU speed, analyze over all ~3.2k
passages takes ~3 h; quick-score ~30 min.

    .venv\\Scripts\\python scripts\\build_data.py          # first
    .venv\\Scripts\\python scripts\\score_baseline.py
    .venv\\Scripts\\python scripts\\score_baseline.py --limit 20     # smoke test
    .venv\\Scripts\\python scripts\\score_baseline.py --summary-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deai.data import BASELINE, DATA_DIR, load_passages  # noqa: E402
from deai.harness import HarnessError, SlopTotalClient, TextRejected  # noqa: E402
from deai.metrics import auroc, tpr_at_fpr  # noqa: E402

CACHE_DIR = DATA_DIR / "cache"
ENGINES_JSON = DATA_DIR / "engines.json"
SLOPTOTAL_DB = Path(r"D:\AI\sloptotal\data")
PREFIX = {"quick": "q_", "analyze": "a_"}
LIKELY_AI = 55.0


def read_cache(endpoint: str) -> dict[str, dict]:
    path = CACHE_DIR / f"{endpoint}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def db_size_mb() -> float:
    return sum(p.stat().st_size for p in SLOPTOTAL_DB.glob("sloptotal.db*")) / 2**20


async def score_endpoint(st: SlopTotalClient, df: pd.DataFrame, endpoint: str) -> None:
    done = read_cache(endpoint)
    todo = df[~df.id.isin(done)]
    print(f"[{endpoint}] {len(done)} cached, {len(todo)} to score", flush=True)
    if todo.empty:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_ok = n_fail = 0

    async def one(pid: str, text: str):
        try:
            return pid, await st.score(text, endpoint)  # type: ignore[arg-type]
        except (HarnessError, TextRejected) as e:
            return pid, e

    with (CACHE_DIR / f"{endpoint}.jsonl").open("a", encoding="utf-8") as f:
        tasks = [asyncio.create_task(one(r.id, r.text)) for r in todo.itertuples()]
        for fut in asyncio.as_completed(tasks):
            pid, res = await fut
            if isinstance(res, Exception):
                n_fail += 1
                print(f"  FAILED {pid}: {res}", flush=True)
                continue
            f.write(json.dumps({"id": pid, "overall": res.overall, "verdict": res.verdict,
                                "engines": res.engines, "latency_s": round(res.latency_s, 3),
                                "report_id": res.report_id}) + "\n")
            f.flush()
            n_ok += 1
            if n_ok % 50 == 0 or n_ok == len(todo):
                el = time.perf_counter() - t0
                rate = n_ok / el
                eta = (len(todo) - n_ok - n_fail) / rate if rate else 0
                print(f"  {n_ok}/{len(todo)}  {rate:.2f}/s  ETA {eta / 60:.0f} min", flush=True)
    print(f"[{endpoint}] done: {n_ok} ok, {n_fail} failed, {time.perf_counter() - t0:.0f}s", flush=True)


def build_table(df: pd.DataFrame, endpoints: list[str]) -> pd.DataFrame:
    out = df.copy()
    for ep in endpoints:
        cache = read_cache(ep)
        p = PREFIX[ep]
        rows = []
        for pid, r in cache.items():
            row = {"id": pid, f"{p}overall": r["overall"], f"{p}verdict": r["verdict"]}
            row.update({f"{p}{k}": v for k, v in r["engines"].items()})
            if ep == "analyze":
                row[f"{p}report_id"] = r.get("report_id")
            rows.append(row)
        if rows:
            out = out.merge(pd.DataFrame(rows), on="id", how="left")
    return out


def summarize(t: pd.DataFrame, endpoints: list[str], families: dict[str, str]) -> None:
    for ep in endpoints:
        col = f"{PREFIX[ep]}overall"
        if col not in t or t[col].isna().all():
            continue
        s = t[t[col].notna() & (t.split != "excluded")]
        print(f"\n=== {ep}: {len(s)}/{(t.split != 'excluded').sum()} scored ===")
        g = s.groupby(["split", "label"])[col]
        print(pd.DataFrame({"n": g.size(), "mean": g.mean().round(1),
                            f">= {LIKELY_AI:.0f}": g.apply(lambda x: (x >= LIKELY_AI).mean()).round(3)}))

        ai, hum = s[s.label == "ai"][col].tolist(), s[s.label == "human"][col].tolist()
        print(f"\n  AUROC {auroc(ai, hum):.3f}   TPR@1%FPR {tpr_at_fpr(ai, hum, 0.01):.3f}   "
              f"TPR@5%FPR {tpr_at_fpr(ai, hum, 0.05):.3f}")
        src = s[s.is_source]
        share = (src[col] >= LIKELY_AI).mean()
        print(f"  Rewrite sources >= {LIKELY_AI:.0f}: {share:.1%} "
              f"({'PASS' if share > 0.5 else 'FAIL'}: Phase 2 wants most of the AI set at >= 55)")

        by = lambda key: src.groupby(key)[col].agg(  # noqa: E731
            n="size", mean="mean", likely_ai=lambda x: (x >= LIKELY_AI).mean()).round(3)
        hum_by = s[s.label == "human"].groupby("source")[col].agg(
            human_mean="mean", human_fpr=lambda x: (x >= LIKELY_AI).mean()).round(3)
        print("\n  Rewrite sources by generator:")
        print(by("model").sort_values("n", ascending=False).to_string().replace("\n", "\n  "))
        print("\n  By source (AI rewrite sources | human):")
        print(by("source").join(hum_by).to_string().replace("\n", "\n  "))

    if "a_overall" in t:
        s = t[t.a_overall.notna() & (t.split != "excluded")]
        engines = [c[2:] for c in t.columns if c.startswith("a_") and c[2:] in families]
        print("\n=== analyze per-engine separation (all non-excluded rows) ===")
        rows = []
        for k in engines:
            ai = s.loc[s.label == "ai", f"a_{k}"].dropna().tolist()
            hum = s.loc[s.label == "human", f"a_{k}"].dropna().tolist()
            rows.append({"engine": k, "family": families[k], "human": sum(hum) / len(hum),
                         "ai": sum(ai) / len(ai), "auroc": auroc(ai, hum),
                         "tpr@1%": tpr_at_fpr(ai, hum, 0.01)})
        e = pd.DataFrame(rows).sort_values(["family", "auroc"], ascending=[True, False]).round(3)
        print(e.to_string(index=False))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--endpoints", default="quick,analyze")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N passages (smoke test)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--quick-concurrency", type=int, default=1, help="quick-score didn't scale past 1 in Phase 1")
    ap.add_argument("--analyze-concurrency", type=int, default=2)
    args = ap.parse_args()
    endpoints = [e for e in args.endpoints.split(",") if e]

    df = load_passages()
    if args.limit:
        df = df.sample(n=args.limit, random_state=0)

    if not args.summary_only:
        async with SlopTotalClient(args.base_url, concurrency={"quick": args.quick_concurrency,
                                                               "analyze": args.analyze_concurrency}) as st:
            if not await st.health():
                print(f"SlopTotal is not answering at {args.base_url}. Start it with scripts\\start.bat.")
                return 1
            engines = await st.engines()
            ENGINES_JSON.write_text(json.dumps(
                {k: {"name": e.name, "engine_type": e.engine_type} for k, e in engines.items()}, indent=1),
                encoding="utf-8")
            db0 = db_size_mb()
            for ep in endpoints:
                await score_endpoint(st, df, ep)
            print(f"SlopTotal DB: {db0:.1f} MB -> {db_size_mb():.1f} MB")

    families = {k: v["engine_type"] for k, v in json.loads(ENGINES_JSON.read_text("utf-8")).items()}
    table = build_table(load_passages(), endpoints)
    if not args.limit:
        table.to_parquet(BASELINE, index=False)
        print(f"Wrote {BASELINE} ({len(table)} rows)")
    summarize(table if not args.limit else table[table.id.isin(df.id)], endpoints, families)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
