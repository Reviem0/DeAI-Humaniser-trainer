"""Phase 3: calibrate the fluency and LM-gate thresholds on human passages and sanity-check the gates.

Scores CoLA fluency for train+val human passages (test humans are left alone)
and for the AI rewrite sources, and suggests ``min_fluency`` as a low quantile
of the human scores. Likewise suggests ``max_perplexity`` and ``max_burstiness``
(the ``use_lm_gates`` upper bounds) as a high quantile of the human values,
and the ``use_structure_gate`` bounds (semicolons, clause joiners, longest
sentence) the same way.
Also reports how often real text fails the unicode gate on its own, and the
gate results for word-shuffled sources (should be ~0% pass).

    .venv\\Scripts\\python scripts\\calibrate_reward.py
    .venv\\Scripts\\python scripts\\calibrate_reward.py --limit 100     # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deai.data import load_passages  # noqa: E402
from deai.reward import LM_MODEL, Reward, RewardConfig, structure, unicode_problems  # noqa: E402

RUNS = ROOT / "runs"
# Grade 7-10 student essays with @CAPS1-style anonymisation tokens: a fluency bar
# set on them would let broken rewrites through.
EXCLUDED_SOURCES = ("essay-hewlett",)


def shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def quantiles(x: list[float]) -> dict[str, float]:
    qs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
    return {f"q{int(q * 100):02d}": round(float(np.quantile(x, q)), 4) for q in qs}


def upper_quantiles(x: list[float]) -> dict[str, float]:
    qs = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0]
    return {f"q{int(q * 100):02d}": round(float(np.quantile(x, q)), 4) for q in qs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantile", type=float, default=0.02,
                    help="min_fluency = this quantile of human fluency (share of humans that fail)")
    ap.add_argument("--lm-quantile", type=float, default=0.98,
                    help="max_perplexity / max_burstiness = this quantile of the human values")
    ap.add_argument("--structure-quantile", type=float, default=0.99,
                    help="max_semicolons / max_joiners / max_sentence_words = this quantile of the human values")
    ap.add_argument("--limit", type=int, default=None, help="sample N passages per group")
    ap.add_argument("--include", default="", help="comma-separated EXCLUDED_SOURCES to keep")
    args = ap.parse_args()

    df = load_passages()
    excluded = [s for s in EXCLUDED_SOURCES if s not in args.include.split(",")]
    human = df[(df.label == "human") & df.split.isin(["train", "val"]) & ~df.source.isin(excluded)]
    src = df[df.is_source & df.split.isin(["train", "val"])]
    if args.limit:
        human = human.sample(n=min(args.limit, len(human)), random_state=0)
        src = src.sample(n=min(args.limit, len(src)), random_state=0)
    reward = Reward()
    report: dict = {"n_human": len(human), "n_sources": len(src), "quantile": args.quantile,
                    "excluded_sources": excluded}

    for name, part in (("human", human), ("sources", src)):
        # Without the source exemption: how much real text uses such characters at all.
        bad = [(r.id, p) for r in part.itertuples() if (p := unicode_problems(r.text))]
        report[f"unicode_strict_{name}"] = len(bad) / len(part)
        report[f"unicode_strict_{name}_examples"] = bad[:10]
        print(f"{name} text with characters the unicode gate only allows from the source: "
              f"{len(bad)}/{len(part)}")
        for pid, p in bad[:5]:
            print(f"   {pid}: {'; '.join(p)}")

    t0 = time.perf_counter()
    flu_h = reward.fluency(human.text.tolist())
    flu_s = reward.fluency(src.text.tolist())
    print(f"\nfluency scored {len(flu_h) + len(flu_s)} passages in {time.perf_counter() - t0:.0f}s")
    report["fluency_human"] = quantiles(flu_h)
    report["fluency_sources"] = quantiles(flu_s)
    threshold = round(float(np.quantile(flu_h, args.quantile)), 3)
    report["suggested_min_fluency"] = threshold
    print(f"human fluency   {report['fluency_human']}")
    print(f"sources fluency {report['fluency_sources']}")
    print(f"human fluency by source:")
    by = human.assign(f=flu_h).groupby("source").f.agg(["size", "mean", "min"]).round(3)
    print(by.to_string())

    print(f"\nsuggested min_fluency = {threshold} (fails {args.quantile:.0%} of human passages)")
    print(f"AI sources below it: {np.mean(np.array(flu_s) < threshold):.1%}")

    t0 = time.perf_counter()
    lm_h = reward.lm_stats(human.text.tolist())
    lm_s = reward.lm_stats(src.text.tolist())
    print(f"\nLM stats ({LM_MODEL}) for {len(lm_h) + len(lm_s)} passages in {time.perf_counter() - t0:.0f}s")
    lm_cfg = {}
    for key, cfg_key in (("perplexity", "max_perplexity"), ("burstiness", "max_burstiness")):
        h = [getattr(s, key) for s in lm_h if getattr(s, key) is not None]
        s = [getattr(x, key) for x in lm_s if getattr(x, key) is not None]
        report[f"{key}_human"], report[f"{key}_sources"] = upper_quantiles(h), upper_quantiles(s)
        lm_cfg[cfg_key] = round(float(np.quantile(h, args.lm_quantile)), 3)
        report[f"suggested_{cfg_key}"] = lm_cfg[cfg_key]
        print(f"human {key:<10} {report[f'{key}_human']}  ({len(lm_h) - len(h)} unmeasured)")
        print(f"sources {key:<8} {report[f'{key}_sources']}")
        print(f"suggested {cfg_key} = {lm_cfg[cfg_key]} (fails {1 - args.lm_quantile:.0%} of human passages); "
              f"AI sources above it: {np.mean(np.array(s) > lm_cfg[cfg_key]):.1%}")
    by = human.assign(p=[s.perplexity for s in lm_h], b=[s.burstiness for s in lm_h])
    print(by.groupby("source")[["p", "b"]].agg(["median", "max"]).round(2).to_string())

    st_h = [structure(t) for t in human.text]
    st_s = [structure(t) for t in src.text]
    print()
    for key, cfg_key in (("semicolons", "max_semicolons"), ("joiners", "max_joiners"),
                         ("longest_sentence", "max_sentence_words")):
        h, s = [getattr(x, key) for x in st_h], [getattr(x, key) for x in st_s]
        report[f"{key}_human"], report[f"{key}_sources"] = upper_quantiles(h), upper_quantiles(s)
        v = float(np.quantile(h, args.structure_quantile))
        lm_cfg[cfg_key] = round(v) if key == "longest_sentence" else round(v, 1)
        report[f"suggested_{cfg_key}"] = lm_cfg[cfg_key]
        print(f"human {key:<16} {report[f'{key}_human']}")
        print(f"suggested {cfg_key} = {lm_cfg[cfg_key]}; AI sources above it: "
              f"{np.mean(np.array(s) > lm_cfg[cfg_key]):.1%} (they keep their own value)")

    cfg = RewardConfig(min_fluency=threshold, use_lm_gates=True, use_structure_gate=True, **lm_cfg)
    reward.config = cfg

    sample = src.sample(n=min(200, len(src)), random_state=1).text.tolist()
    rng = random.Random(0)
    for label, rewrites in (("identity", sample),
                            ("shuffled", [shuffle_words(t, rng) for t in sample])):
        res = reward.check_gates(sample, rewrites)
        failed = [r.failed_gate or "passed" for r in res]
        counts = {k: failed.count(k) for k in sorted(set(failed))}
        sims = [r.similarity for r in res if r.similarity is not None]
        report[f"{label}_gates"] = counts
        report[f"{label}_similarity"] = quantiles(sims) if sims else {}
        print(f"{label:>8} rewrites: {counts}; similarity median "
              f"{np.median(sims) if sims else float('nan'):.3f}")

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"calibrate_reward-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
