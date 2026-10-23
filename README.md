# DeAI

A research project on how well AI-text detectors hold up against a rewriter that is trained to evade them.

We train a small rewriter model to turn AI-written text into text that [SlopTotal](#sloptotal-harness)'s detectors score as human. The rewrite has to keep the meaning and stay fluent. We then use the rewriter to study two questions:

1. **Transfer:** when the rewriter is optimized against one family of detectors (neural classifiers, statistical/perplexity-based, or heuristic), does it also evade the other families?
2. **Defense:** how much does adversarial retraining (in the style of RADAR) make the detectors more robust?

The study is the contribution. Trained humanizers already exist (DIPPER, RADAR, Nicks et al. 2024, AuthorMist), so this is not meant to be another one.

> **Status:** Phase 1 done: the venv, the harness client ([deai/harness.py](deai/harness.py)) and the throughput check ([scripts/check_harness.py](scripts/check_harness.py)) are in, and the results are in PLAN.md. Phases 2–6 aren't written yet. The full plan, with "done when" criteria for each phase, is in [PLAN.md](PLAN.md).

## Approach

| Phase | What happens |
|---|---|
| 1. Harness | Async client for SlopTotal's `quick-score` and `analyze` endpoints, with throughput measurement |
| 2. Data | 3–5k AI passages from several generators and genres, about 1k pre-2022 human passages, split by generator and topic; baseline scores cached |
| 3. Reward | Detector score × semantic similarity, given only when every quality gate passes |
| 4. Training | Qwen-class 1.5–3B instruct model with LoRA. Best-of-N + DPO first, GRPO if needed. One run per reward source (quick, neural, statistical, full) |
| 5. Evaluation | Evasion rate, TPR at 1% FPR, the train-family × eval-family transfer matrix, quality metrics, comparison with baselines |
| 6. Defense | Retrain SlopTotal classifiers on the rewrites and measure how much robustness each round adds |

### Reward gates

A rewrite gets **zero reward** if it fails any of these checks, so the model can't trade quality for a lower detector score:

- **Unicode:** NFKC-stable, with no zero-width characters or homoglyphs
- **Length:** 0.8–1.2× the source word count, and at least 80 words
- **Meaning:** embedding cosine similarity ≥ 0.85 (all-mpnet-base-v2), plus a two-way NLI entailment check on validation
- **Fluency:** grammatical-acceptability classifier above a threshold calibrated on human text. Perplexity is deliberately not used, because low perplexity is what statistical detectors flag.

## SlopTotal harness

The detectors come from a local SlopTotal install, which runs as a FastAPI server on `http://localhost:8000` (start it with `scripts\start.bat` in the SlopTotal folder).

| Endpoint | Used for |
|---|---|
| `POST /api/quick-score` | Cheap training reward (6 engines; mostly Fakespot) |
| `POST /api/analyze` | Evaluation (all 23 engines with per-engine scores) |
| `POST /api/paragraph-score` | Failure analysis |
| `GET /api/queue/ticket/{id}` | Polling when a request comes back `202 queued` |

Inputs must be at least 50 characters. Engines with zero weight in the overall score (GLTR, log-rank, perplexity, DiveEye, …) still report their own scores, so they serve as held-out detectors for measuring transfer.

## Setup (planned)

Requirements:

- A CUDA GPU. Development uses one RTX 5070 Ti (16 GB) shared with SlopTotal: about 3–5 GB for the detectors and about 11 GB for training.
- Python **3.12**, in a venv separate from SlopTotal's 3.14 one.
- PyTorch built for CUDA 12.8 or newer (needed for Blackwell GPUs).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers trl peft datasets sentence-transformers httpx pandas
```

If GPU memory runs short, start SlopTotal with `SLOPTOTAL_DEVICE=cpu`, or alternate between generating and scoring in each training round.

## Repository layout (planned)

```
DeAI-project/
  PLAN.md
  pyproject.toml
  deai/
    harness.py      # SlopTotal client (quick/analyze/ticket polling)
    reward.py       # gates + reward
    data.py         # generation, splits, caching
    train_dpo.py    # best-of-N + DPO loop
    train_grpo.py   # GRPO (Phase 4B)
    evaluate.py     # evaluation tables + transfer matrix
  scripts/
    check_harness.py
  data/             # gitignored
  runs/             # gitignored checkpoints + logs
  tests/
    test_reward.py  # shortcut examples must score 0
```

## Responsible use

This project is for red-teaming detectors and for research. Trained rewriter weights will not be released publicly unless the defense results (hardened detectors and adversarial data) are released with them. Where it is useful, results and adversarial data will be contributed back to SlopTotal.
