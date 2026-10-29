"""Phase 4 plumbing shared by the DPO and GRPO trainers: sources, scoring, validation.

Scored samples are written as JSONL, one row per rewrite, so every stage can
resume after an interruption and every number can be traced back to text.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from deai.data import load_passages
from deai.detectors import MIN_TEXT_LENGTH, Detectors
from deai.reward import Gate, HFQualityModels, Reward, RewardConfig, RewardResult, make_detector

LIKELY_AI = 55.0


def load_sources(split: str, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    df = load_passages()
    df = df[df.is_source & (df.split == split)][["id", "text", "model", "source", "words"]]
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    return df.sort_values("id").reset_index(drop=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def result_row(r: RewardResult) -> dict:
    return {
        "reward": r.reward,
        "passed": r.passed,
        "failed_gate": r.failed_gate,
        "similarity": r.similarity,
        "detector": r.detector_score,
        "gates": {g.name: {"passed": g.passed, "value": g.value, "detail": g.detail} for g in r.gates},
        "error": r.error,
    }


class Scorer:
    """Reward models and detectors stay loaded across calls until ``release()``."""

    def __init__(self, reward_source: str, config: RewardConfig = RewardConfig(),
                 detectors: Detectors | None = None):
        self.reward_source = reward_source
        self.detectors = detectors or Detectors()
        self.detector = make_detector(self.detectors, reward_source)
        self.models = HFQualityModels()
        self.config = config

    def score(self, sources: Sequence[str], rewrites: Sequence[str],
              finished: Sequence[bool] | None = None, use_nli: bool = False) -> list[RewardResult]:
        return asyncio.run(self._score(sources, rewrites, finished, use_nli))

    def detect(self, texts: Sequence[str]) -> list[float | None]:
        """Detector score for any text (no gates); None where it's too short or an engine failed."""
        ok = [i for i, t in enumerate(texts) if len(t.strip()) >= MIN_TEXT_LENGTH]
        scores = asyncio.run(self.detector([texts[i].strip() for i in ok])) if ok else []
        out: list[float | None] = [None] * len(texts)
        for i, s in zip(ok, scores):
            out[i] = None if isinstance(s, BaseException) else float(s)
        return out

    def release(self) -> None:
        """Free the detectors' and quality models' GPU memory; they reload on the next score."""
        self.detectors.unload()
        self.models.release()

    async def _score(self, sources, rewrites, finished, use_nli):
        reward = Reward(self.detector, self.models, dataclasses.replace(self.config, use_nli=use_nli))
        finished = finished if finished is not None else [True] * len(rewrites)
        idx = [i for i, f in enumerate(finished) if f]
        scored = await reward.score([sources[i] for i in idx], [rewrites[i] for i in idx]) if idx else []
        # A rewrite cut off at max_new_tokens gets 0 like any other gate failure.
        out = [RewardResult(0.0, [Gate("finished", False, detail="hit max_new_tokens")]) for _ in rewrites]
        for i, r in zip(idx, scored):
            out[i] = r
        return out


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def evaluate(model, tok, val: pd.DataFrame, scorer: Scorer, out_dir: Path,
             source_scores: dict[str, float | None], gen_batch: int = 16) -> dict:
    """Greedy rewrite of every val source, gates with NLI, detector on every rewrite.

    The detector runs on rewrites that fail a gate too, so evasion and quality
    can be read separately; ``evasion_and_quality`` is the share that does both.
    """
    from deai.policy import generate

    t0 = time.perf_counter()
    samples = [s[0] for s in generate(model, tok, val.text.tolist(), n=1, temperature=0.0,
                                      max_seqs_per_batch=gen_batch)]
    gen_s = time.perf_counter() - t0
    # The policy (~8 GB) and the detectors don't both fit beside the quality models, and an
    # out-of-memory engine would score None. Callers drop the model after evaluate().
    model.to("cpu")
    from deai.policy import release_gpu
    release_gpu()
    return score_eval(val, samples, scorer, out_dir, source_scores, gen_s)


def score_eval(val: pd.DataFrame, samples: Sequence, scorer: Scorer, out_dir: Path,
               source_scores: dict[str, float | None], generation_s: float) -> dict:
    """``evaluate`` after generation: one greedy ``Sample`` per val source, in val's order.
    Writes eval.jsonl, metrics.json and spotcheck.md to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rewrites = [s.text for s in samples]
    results = scorer.score(val.text.tolist(), rewrites, [s.finished for s in samples], use_nli=True)
    det = scorer.detect(rewrites)

    rows = []
    for r, s, res, d in zip(val.itertuples(), samples, results, det):
        rows.append({"id": r.id, "model": r.model, "source_text": r.text, "rewrite": s.text,
                     "finished": s.finished, "words": len(s.text.split()), "source_words": r.words,
                     "source_detector": source_scores.get(r.id), "rewrite_detector": d,
                     **result_row(res)})
    write_jsonl(out_dir / "eval.jsonl", rows)
    metrics = summarize_eval(rows)
    metrics["generation_s"] = round(generation_s, 1)
    if metrics["detector_failed"]:
        print(f"WARNING: {metrics['detector_failed']} detector failures in eval (see eval.jsonl 'error')",
              flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    write_spotcheck(rows, out_dir / "spotcheck.md")
    return metrics


def summarize_eval(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    gate_names = sorted({g for gs in df.gates for g in gs})
    reached = {g: [gs[g]["passed"] for gs in df.gates if g in gs] for g in gate_names}
    det = df.rewrite_detector.dropna()
    det_pass = df.loc[df.passed, "rewrite_detector"].dropna()
    src = df.source_detector.dropna()
    return {
        "n": len(df),
        "pass_all_gates": round(float(df.passed.mean()), 3),
        # Gates run in order, so a later gate is only judged on rewrites that reached it.
        "gate_pass_rate_when_reached": {g: round(float(np.mean(v)), 3) for g, v in reached.items() if v},
        "failed_gate_counts": df.failed_gate.fillna("passed").value_counts().to_dict(),
        "finished": round(float(df.finished.mean()), 3),
        "mean_reward": round(float(df.reward.fillna(0).mean()), 4),
        # None scores (engine failed) count as 0 / not evading above; this says how many there were.
        "detector_failed": int(df["error"].notna().sum()) if "error" in df else 0,
        "source_detector_mean": round(float(src.mean()), 2) if len(src) else None,
        "source_likely_ai": round(float((src >= LIKELY_AI).mean()), 3) if len(src) else None,
        "rewrite_detector_mean": round(float(det.mean()), 2) if len(det) else None,
        "rewrite_detector_mean_passing": round(float(det_pass.mean()), 2) if len(det_pass) else None,
        "rewrite_below_likely_ai": round(float((det < LIKELY_AI).mean()), 3) if len(det) else None,
        "evasion_and_quality": round(float((df.passed & (df.rewrite_detector < LIKELY_AI)).mean()), 3),
        "similarity_mean": round(float(df.similarity.dropna().mean()), 4) if df.similarity.notna().any() else None,
        "length_ratio_median": round(float((df.words / df.source_words).median()), 3),
    }


def write_spotcheck(rows: list[dict], path: Path, n: int = 30, seed: int = 0) -> None:
    pick = random.Random(seed).sample(rows, min(n, len(rows)))
    lines = [f"# Spot check ({len(pick)} of {len(rows)} val rewrites)\n",
             "Read for: meaning drift, invented or dropped facts, odd style, prompt leakage.\n"]
    for r in pick:
        gates = ", ".join(f"{k} {'ok' if v['passed'] else 'FAIL'}"
                          + (f" {v['value']:.3f}" if isinstance(v["value"], float) else "")
                          for k, v in r["gates"].items())
        lines += [f"## {r['id']} ({r['model']})",
                  f"detector {r['source_detector']} -> {r['rewrite_detector']}; reward {r['reward']}; {gates}\n",
                  "**Source**\n", r["source_text"], "\n**Rewrite**\n", r["rewrite"], "\n---\n"]
    path.write_text("\n".join(lines), encoding="utf-8")


def source_scores(scorer: Scorer, sources: pd.DataFrame, cache: Path) -> dict[str, float | None]:
    """The no-rewrite baseline: detector score of each source, cached per run."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    scores = dict(zip(sources.id, scorer.detect(sources.text.tolist())))
    cache.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    return scores
