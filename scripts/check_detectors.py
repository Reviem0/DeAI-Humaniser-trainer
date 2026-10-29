"""Phase 1 check: score known-human and known-AI passages and measure throughput.

Passages come from SlopBench (human text from before November 2022, paired AI
text from 14 models). SlopTotal's calibration, which deai.detectors reuses, was
fitted on SlopBench, so the separation printed here is in-distribution and
optimistic; it's a smoke test, not an evaluation.

Detectors run in-process. Model loading is timed separately (one warm-up text
per endpoint); throughput is measured with ``score_many`` in chunks of --chunk.
quick runs only the engines its overall score needs (as the reward does);
analyze runs all 23 so the per-engine table is complete.

    .venv\\Scripts\\python scripts\\check_detectors.py
    .venv\\Scripts\\python scripts\\check_detectors.py --n 10 --endpoints quick
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deai.data import SLOPBENCH  # noqa: E402
from deai.detectors import Detectors, Score  # noqa: E402
from deai.metrics import auroc  # noqa: E402

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


def gpu_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def torch_peak_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None


def run_endpoint(det: Detectors, items: list[dict], endpoint: str, chunk: int) -> dict:
    engines = det.endpoint_engines(endpoint) if endpoint == "analyze" else ()
    t0 = time.perf_counter()
    det.score_many([items[0]["text"]], endpoint, engines=engines)  # loads the models
    load_s = time.perf_counter() - t0

    texts = [it["text"] for it in items]
    results: list = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), chunk):
        results += det.score_many(texts[i:i + chunk], endpoint, engines=engines)
    wall = time.perf_counter() - t0

    ok = [(it, r) for it, r in zip(items, results) if isinstance(r, Score)]
    errors = [f"{type(r).__name__}: {r}" for r in results if not isinstance(r, Score)]
    return {
        "endpoint": endpoint,
        "n": len(items),
        "ok": len(ok),
        "errors": errors,
        "load_s": load_s,
        "wall_s": wall,
        "texts_per_s": len(ok) / wall if wall else float("nan"),
        "words_per_s": sum(len(it["text"].split()) for it, _ in ok) / wall if wall else float("nan"),
        "engines_run": sorted({k for _, r in ok for k in r.engines}),
        "rows": [
            {"label": it["label"], "source": it["source"], "model": it["model"],
             "words": len(it["text"].split()), "overall": r.overall, "engines": r.engines}
            for it, r in ok
        ],
    }


def print_scores(res: dict) -> None:
    rows = res["rows"]
    print(f"\n  {'label':<6} {'source':<26} {'words':>5} {'score':>6}")
    for r in sorted(rows, key=lambda r: (r["label"], r["source"])):
        print(f"  {r['label']:<6} {r['source']:<26} {r['words']:>5} {r['overall']:>6.1f}")
    hum = [r["overall"] for r in rows if r["label"] == "human"]
    ai = [r["overall"] for r in rows if r["label"] == "ai"]
    if hum and ai:
        print(f"\n  mean human {statistics.mean(hum):5.1f}   mean AI {statistics.mean(ai):5.1f}   "
              f"AUROC {auroc(ai, hum):.3f}")
        print(f"  AI >= {LIKELY_AI:.0f}: {sum(s >= LIKELY_AI for s in ai)}/{len(ai)}   "
              f"human >= {LIKELY_AI:.0f} (false positives): {sum(s >= LIKELY_AI for s in hum)}/{len(hum)}")


def print_engines(res: dict, families: dict[str, str]) -> None:
    rows = res["rows"]
    keys = sorted({k for r in rows for k in r["engines"]}, key=lambda k: (families.get(k, ""), k))
    print(f"\n  {'engine':<26} {'family':<12} {'human':>6} {'AI':>6} {'AUROC':>6}")
    for k in keys:
        hum = [r["engines"][k] for r in rows if r["label"] == "human" and k in r["engines"]]
        ai = [r["engines"][k] for r in rows if r["label"] == "ai" and k in r["engines"]]
        if hum and ai:
            print(f"  {k:<26} {families.get(k, '?'):<12} {statistics.mean(hum):6.3f} "
                  f"{statistics.mean(ai):6.3f} {auroc(ai, hum):6.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slopbench", type=Path, default=SLOPBENCH)
    ap.add_argument("--n", type=int, default=20, help="passages per label")
    ap.add_argument("--endpoints", default="quick,analyze")
    ap.add_argument("--chunk", type=int, default=32, help="texts per score_many call")
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None, help="default: new passages every run")
    ap.add_argument("--out", type=Path, default=None, help="JSON results (default: runs/check_detectors-<time>.json)")
    args = ap.parse_args()

    endpoints = [e for e in args.endpoints.split(",") if e]
    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    pools = {lab: load_pool(args.slopbench, lab, args.min_words, args.max_words) for lab in ("human", "ai")}
    print(f"SlopBench pool ({args.slopbench}): {len(pools['human'])} human, {len(pools['ai'])} AI "
          f"({args.min_words}-{args.max_words} words); seed {seed}")
    if not pools["human"] or not pools["ai"]:
        print("No passages found. Run scripts\\fetch_slopbench.py first.")
        return 1
    items = [r for lab in ("human", "ai") for r in stratified_sample(pools[lab], args.n, rng)]
    rng.shuffle(items)

    det = Detectors()
    families = {k: e.engine_type for k, e in det.engines().items()}
    device = os.environ.get("DEAI_DETECTOR_DEVICE", "default")
    print(f"{len(families)} engines, device {device}. GPU memory before: {gpu_mem()}")

    results = {}
    for ep in endpoints:
        print(f"\n[{ep}] {len(items)} texts ...", flush=True)
        res = results[ep] = run_endpoint(det, items, ep, args.chunk)
        print(f"  load {res['load_s']:.1f}s; {res['ok']}/{res['n']} ok in {res['wall_s']:.1f}s "
              f"-> {res['texts_per_s']:.2f} texts/s; GPU memory {gpu_mem()}", flush=True)
    peak = torch_peak_mb()

    for ep, res in results.items():
        print(f"\n=== Scores: {ep} ===")
        print_scores(res)
    if "analyze" in results:
        print("\n=== Per-engine means (analyze) ===")
        print_engines(results["analyze"], families)

    print("\n=== Throughput ===")
    print(f"\n  {'endpoint':<8} {'engines':>7} {'ok/n':>7} {'load s':>7} {'wall s':>7} {'texts/s':>8} {'words/s':>8}")
    for ep, r in results.items():
        print(f"  {ep:<8} {len(r['engines_run']):>7} {r['ok']:>3}/{r['n']:<3} {r['load_s']:>7.1f} "
              f"{r['wall_s']:>7.1f} {r['texts_per_s']:>8.2f} {r['words_per_s']:>8.0f}")
        for e in r["errors"][:3]:
            print(f"      error: {e}")
    gpu_after = gpu_mem()
    print(f"\n  GPU memory after: {gpu_after}" + (f"; torch peak {peak:,.0f} MB" if peak is not None else ""))

    print("\n=== Training budget ===")
    for ep, r in results.items():
        per_hour = r["texts_per_s"] * 3600
        if not per_hour:
            continue
        print(f"  {ep:<8} {r['texts_per_s']:.2f} texts/s = {per_hour:,.0f}/hour; "
              f"one 4A round on 3k sources x 8 samples = {24_000 / per_hour:.1f} h")

    out = args.out or ROOT / "runs" / f"check_detectors-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": seed, "device": device, "results": results,
                               "gpu_after": gpu_after, "torch_peak_mb": peak}, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")
    det.unload()
    return 0 if all(not r["errors"] for r in results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
