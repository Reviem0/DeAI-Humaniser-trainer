"""Phase 1 check: score known-human and known-AI passages and measure throughput.

Passages come from SlopTotal's SlopBench corpus (human text from before
November 2022, paired AI text from 14 models). SlopTotal's calibration was
fitted on SlopBench, so the separation printed here is in-distribution and
optimistic; it's a smoke test, not an evaluation.

/api/analyze caches by text hash, so every passage is trimmed by a random number
of leading/trailing sentences each run, and each concurrency level gets its own
passages. Reports older than the request are still counted as cache hits.

    .venv\\Scripts\\python scripts\\check_harness.py
    .venv\\Scripts\\python scripts\\check_harness.py --n 10 --quick-levels 1,4 --analyze-levels 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deai.harness import Score, SlopTotalClient  # noqa: E402
from deai.metrics import auroc  # noqa: E402

DEFAULT_SLOPBENCH = Path(r"D:\AI\sloptotal\tests\eval\slopbench\corpus")
LIKELY_AI = 55.0  # the "Likely AI" line used in PLAN.md


def load_pool(corpus: Path, label: str, min_words: int, max_words: int) -> list[dict]:
    rows = []
    for path in sorted(corpus.glob(f"{label}-*.jsonl")):
        if "mixed" in path.name:  # grey-zone sets, label is ambiguous
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if min_words <= len(r["text"].split()) <= max_words:
                    rows.append({"label": label, "source": r["source"], "text": r["text"],
                                 "model": r.get("model", "")})
    return rows


def stratified_sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows spread round-robin across sources."""
    by_source: dict[str, list[dict]] = {}
    for r in pool:
        by_source.setdefault(r["source"], []).append(r)
    for rows in by_source.values():
        rng.shuffle(rows)
    sources = sorted(by_source)
    rng.shuffle(sources)
    out, i = [], 0
    while len(out) < n and any(by_source.values()):
        rows = by_source[sources[i % len(sources)]]
        if rows:
            out.append(rows.pop())
        i += 1
    return out


def trim(text: str, rng: random.Random, min_words: int) -> str:
    """Drop 1-2 leading and 0-2 trailing sentences so the text hash is new."""
    parts = re.split(r"(?<=[.!?])(\s+)", text.strip())
    sents, seps = parts[0::2], parts[1::2] + [""]
    n = len(sents)
    for _ in range(10):
        a, c = rng.randint(1, 2), rng.randint(0, 2)
        if n - a - c < 3:
            continue
        out = "".join(s + sep for s, sep in zip(sents[a:n - c], seps[a:n - c])).strip()
        if len(out.split()) >= min_words:
            return out
    return text.strip()


def gpu_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def is_cache_hit(s: Score, started_utc: datetime) -> bool:
    if not s.created_at:
        return False
    created = datetime.fromisoformat(s.created_at).replace(tzinfo=timezone.utc)
    return (started_utc - created).total_seconds() > 2


async def run_level(base_url: str, items: list[dict], endpoint: str, conc: int) -> dict:
    async with SlopTotalClient(base_url, concurrency={endpoint: conc}) as st:
        await st.engines()
        started_utc = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        results = await st.score_many([it["text"] for it in items], endpoint)  # type: ignore[arg-type]
        wall = time.perf_counter() - t0

    ok = [(it, r) for it, r in zip(items, results) if isinstance(r, Score)]
    errors = [f"{type(r).__name__}: {r}" for r in results if not isinstance(r, Score)]
    lat = sorted(r.latency_s for _, r in ok)
    hits = sum(is_cache_hit(r, started_utc) for _, r in ok)
    return {
        "endpoint": endpoint,
        "concurrency": conc,
        "n": len(items),
        "ok": len(ok),
        "errors": errors,
        "wall_s": wall,
        "texts_per_s": len(ok) / wall if wall else float("nan"),
        "words_per_s": sum(len(it["text"].split()) for it, _ in ok) / wall if wall else float("nan"),
        "latency_p50_s": statistics.median(lat) if lat else float("nan"),
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else float("nan"),
        "retries": sum(r.attempts - 1 for _, r in ok),
        "cache_hits": hits,
        "rows": [
            {"label": it["label"], "source": it["source"], "model": it["model"],
             "words": len(it["text"].split()), "overall": r.overall, "latency_s": r.latency_s,
             "engines": r.engines}
            for it, r in ok
        ],
    }


def print_scores(level: dict) -> None:
    rows = level["rows"]
    print(f"\n  {'label':<6} {'source':<26} {'words':>5} {'score':>6} {'lat s':>6}")
    for r in sorted(rows, key=lambda r: (r["label"], r["source"])):
        print(f"  {r['label']:<6} {r['source']:<26} {r['words']:>5} {r['overall']:>6.1f} {r['latency_s']:>6.2f}")
    hum = [r["overall"] for r in rows if r["label"] == "human"]
    ai = [r["overall"] for r in rows if r["label"] == "ai"]
    if hum and ai:
        print(f"\n  mean human {statistics.mean(hum):5.1f}   mean AI {statistics.mean(ai):5.1f}   "
              f"AUROC {auroc(ai, hum):.3f}")
        print(f"  AI >= {LIKELY_AI:.0f}: {sum(s >= LIKELY_AI for s in ai)}/{len(ai)}   "
              f"human >= {LIKELY_AI:.0f} (false positives): {sum(s >= LIKELY_AI for s in hum)}/{len(hum)}")


def print_engines(level: dict, families: dict[str, str]) -> None:
    rows = level["rows"]
    keys = sorted({k for r in rows for k in r["engines"]}, key=lambda k: (families.get(k, ""), k))
    print(f"\n  {'engine':<26} {'family':<12} {'human':>6} {'AI':>6} {'AUROC':>6}")
    for k in keys:
        hum = [r["engines"][k] for r in rows if r["label"] == "human" and k in r["engines"]]
        ai = [r["engines"][k] for r in rows if r["label"] == "ai" and k in r["engines"]]
        if hum and ai:
            print(f"  {k:<26} {families.get(k, '?'):<12} {statistics.mean(hum):6.3f} "
                  f"{statistics.mean(ai):6.3f} {auroc(ai, hum):6.3f}")


def print_timing(levels: list[dict]) -> None:
    print(f"\n  {'endpoint':<8} {'conc':>4} {'ok/n':>7} {'wall s':>7} {'texts/s':>8} {'words/s':>8} "
          f"{'p50 s':>6} {'p95 s':>6} {'retry':>5} {'cached':>6}")
    for lv in levels:
        print(f"  {lv['endpoint']:<8} {lv['concurrency']:>4} {lv['ok']:>3}/{lv['n']:<3} {lv['wall_s']:>7.1f} "
              f"{lv['texts_per_s']:>8.2f} {lv['words_per_s']:>8.0f} {lv['latency_p50_s']:>6.2f} "
              f"{lv['latency_p95_s']:>6.2f} {lv['retries']:>5} {lv['cache_hits']:>6}")
        for e in lv["errors"][:3]:
            print(f"      error: {e}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--slopbench", type=Path, default=DEFAULT_SLOPBENCH)
    ap.add_argument("--n", type=int, default=20, help="passages per label per level")
    ap.add_argument("--quick-levels", default="1,4,8", help="client concurrency levels for quick-score")
    ap.add_argument("--analyze-levels", default="1,2", help="client concurrency levels for analyze")
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None, help="default: new passages every run")
    ap.add_argument("--out", type=Path, default=None, help="JSON results (default: runs/check_harness-<time>.json)")
    args = ap.parse_args()

    quick_levels = [int(x) for x in args.quick_levels.split(",") if x]
    analyze_levels = [int(x) for x in args.analyze_levels.split(",") if x]
    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    pools = {lab: load_pool(args.slopbench, lab, args.min_words, args.max_words) for lab in ("human", "ai")}
    print(f"SlopBench pool: {len(pools['human'])} human, {len(pools['ai'])} AI "
          f"({args.min_words}-{args.max_words} words); seed {seed}")

    # One disjoint set of passages per concurrency level, shared by both endpoints
    # (quick-score has no cache, so it can't warm analyze's).
    n_sets = max(len(quick_levels), len(analyze_levels))
    sets = []
    for _ in range(n_sets):
        items = []
        for lab in ("human", "ai"):
            picked = stratified_sample(pools[lab], args.n, rng)
            chosen = {id(r) for r in picked}
            pools[lab] = [r for r in pools[lab] if id(r) not in chosen]
            items += [{**r, "text": trim(r["text"], rng, min_words=120)} for r in picked]
        rng.shuffle(items)
        sets.append(items)

    async with SlopTotalClient(args.base_url) as st:
        if not await st.health():
            print(f"SlopTotal is not answering at {args.base_url}. Start it with scripts\\start.bat.")
            return 1
        engines = await st.engines()
        families = {k: e.engine_type for k, e in engines.items()}
    print(f"Server up, {len(engines)} engines. GPU memory before: {gpu_mem()}")

    levels = []
    for endpoint, lvls in (("quick", quick_levels), ("analyze", analyze_levels)):
        for i, conc in enumerate(lvls):
            print(f"\n[{endpoint}] concurrency {conc}: {len(sets[i])} texts ...", flush=True)
            lv = await run_level(args.base_url, sets[i], endpoint, conc)
            levels.append(lv)
            print(f"  {lv['ok']}/{lv['n']} ok in {lv['wall_s']:.1f}s -> {lv['texts_per_s']:.2f} texts/s")
    gpu_after = gpu_mem()

    first = {ep: next(lv for lv in levels if lv["endpoint"] == ep) for ep in ("quick", "analyze")}
    for ep in ("quick", "analyze"):
        print(f"\n=== Scores: {ep}, concurrency {first[ep]['concurrency']} ===")
        print_scores(first[ep])
    print("\n=== Per-engine means (analyze) ===")
    print_engines(first["analyze"], families)
    print("\n=== Throughput ===")
    print_timing(levels)
    print(f"\n  GPU memory after: {gpu_after}")

    best = {ep: max((lv for lv in levels if lv["endpoint"] == ep), key=lambda lv: lv["texts_per_s"])
            for ep in ("quick", "analyze")}
    print("\n=== Training budget at best throughput ===")
    for ep, lv in best.items():
        per_hour = lv["texts_per_s"] * 3600
        print(f"  {ep:<8} {lv['texts_per_s']:.2f} texts/s (concurrency {lv['concurrency']}) = "
              f"{per_hour:,.0f}/hour; one 4A round on 3k sources x 8 samples = "
              f"{24_000 / per_hour:.1f} h")

    out = args.out or ROOT / "runs" / f"check_harness-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": seed, "base_url": args.base_url, "levels": levels,
                               "gpu_after": gpu_after}, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0 if all(not lv["errors"] for lv in levels) else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
