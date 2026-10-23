"""TensorBoard logging for Phase 4 runs.

Every run writes under ``runs/<run>/tb/``:

- ``tb/rounds``: one point per round (x = round), so runs can be overlaid.
  ``val/*`` are the validation metrics (``metrics.json``), ``val_gates/*`` the
  per-gate pass rates, ``best_of_n/*`` the training-sample stats
  (``round_stats.json``), plus a text panel of sample val rewrites.
- ``tb/scoring``: live progress while a round's samples are scored, per chunk
  (x = rewrites scored so far in that round; GRPO logs everything as round 0).
- ``tb/train_round_k`` (DPO) or ``tb/train`` (GRPO): the TRL trainer's own
  logs (loss, reward margins, KL, ...).

Rounds already logged are recorded in ``tb/logged.json``, so a resumed run or a
backfill never writes a round's point twice. (Live scoring points aren't
deduplicated; a resumed round just continues its curve.)

    .venv\\Scripts\\python -m deai.tblog            # backfill every run in runs/
    .venv\\Scripts\\tensorboard --logdir runs
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"

# metrics.json keys plotted per round; higher-is-better unless noted.
VAL_SCALARS = (
    "pass_all_gates", "mean_reward", "evasion_and_quality",
    "rewrite_detector_mean",          # lower = more human-like
    "rewrite_detector_mean_passing",  # same, only rewrites that pass every gate
    "rewrite_below_likely_ai", "source_detector_mean",
    "similarity_mean", "length_ratio_median", "finished",
)
ROUND_SCALARS = ("pairs", "mean_reward", "mean_best_of_n")


def _writer(path: Path):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(str(path))


class RunLog:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "tb"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._marker = self.dir / "logged.json"
        self._done = set(json.loads(self._marker.read_text("utf-8"))) if self._marker.exists() else set()
        self._writers: dict[str, object] = {}

    def _w(self, name: str):
        if name not in self._writers:
            self._writers[name] = _writer(self.dir / name)
        return self._writers[name]

    def _mark(self, key: str) -> bool:
        """True the first time ``key`` is seen."""
        if key in self._done:
            return False
        self._done.add(key)
        self._marker.write_text(json.dumps(sorted(self._done)), encoding="utf-8")
        return True

    # -- per round ------------------------------------------------------

    def log_round(self, run_dir: Path, k: int, eval_dir: Path | None = None) -> None:
        """Log whatever round k has produced so far (metrics, round stats, samples)."""
        rd = eval_dir or Path(run_dir) / f"round_{k}"
        w = self._w("rounds")
        metrics = rd / "metrics.json"
        if metrics.exists() and self._mark(f"val/{k}"):
            m = json.loads(metrics.read_text("utf-8"))
            for key in VAL_SCALARS:
                if isinstance(m.get(key), (int, float)):
                    w.add_scalar(f"val/{key}", m[key], k)
            for gate, rate in m.get("gate_pass_rate_when_reached", {}).items():
                w.add_scalar(f"val_gates/{gate}", rate, k)
            n = m.get("n") or 1
            for gate, count in m.get("failed_gate_counts", {}).items():
                w.add_scalar(f"val_failed_at/{gate}", count / n, k)
            eval_rows = rd / "eval.jsonl"
            if eval_rows.exists():
                w.add_text("val/samples", _samples_markdown(eval_rows), k)
        stats = rd / "round_stats.json"
        if stats.exists() and self._mark(f"best_of_n/{k}"):
            s = json.loads(stats.read_text("utf-8"))
            for key in ROUND_SCALARS:
                if isinstance(s.get(key), (int, float)):
                    w.add_scalar(f"best_of_n/{key}", s[key], k)
            if s.get("sources"):
                w.add_scalar("best_of_n/pair_yield", s["pairs"] / s["sources"], k)
            total = sum(s.get("samples_by_failed_gate", {}).values()) or 1
            for gate, count in s.get("samples_by_failed_gate", {}).items():
                w.add_scalar(f"best_of_n_failed_at/{gate}", count / total, k)
        w.flush()

    # -- live scoring ---------------------------------------------------

    def log_scoring(self, k: int, step: int, samples: list[dict]) -> None:
        """One point per scored chunk of a round (``step`` = rewrites scored so far in this round)."""
        if not samples:
            return
        w = self._w("scoring")
        rewards = [s["reward"] for s in samples if s["reward"] is not None]
        passed = [s for s in samples if s["passed"]]
        det = [s["detector"] for s in passed if s["detector"] is not None]
        w.add_scalar(f"round_{k}/mean_reward", sum(rewards) / len(rewards) if rewards else 0.0, step)
        w.add_scalar(f"round_{k}/pass_all_gates", len(passed) / len(samples), step)
        if det:
            w.add_scalar(f"round_{k}/detector_mean_passing", sum(det) / len(det), step)
        w.flush()

    def trainer_dir(self, name: str) -> str:
        return str(self.dir / name)

    def close(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers.clear()


def _samples_markdown(eval_rows: Path, n: int = 5, seed: int = 0) -> str:
    rows = [json.loads(line) for line in eval_rows.open(encoding="utf-8") if line.strip()]
    pick = random.Random(seed).sample(rows, min(n, len(rows)))  # same ids every round
    parts = []
    for r in pick:
        parts.append(f"**{r['id']}**: detector {r['source_detector']} → {r['rewrite_detector']}, "
                     f"reward {r['reward']}, failed {r['failed_gate'] or '—'}\n\n"
                     f"> {r['rewrite'][:1200]}")
    return "\n\n---\n\n".join(parts)


def backfill(runs: Path = RUNS) -> None:
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "config.json").exists()):
        log = RunLog(run_dir)
        rounds = sorted(int(p.name.split("_")[1]) for p in run_dir.glob("round_*"))
        for k in rounds:
            log.log_round(run_dir, k)
        if (run_dir / "eval").exists():  # GRPO
            log.log_round(run_dir, 1, run_dir / "eval")
        log.close()
        print(f"{run_dir.name}: rounds {rounds or '-'}")


if __name__ == "__main__":
    backfill(Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS)
