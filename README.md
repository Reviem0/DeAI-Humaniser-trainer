# DeAI

A research project on how well AI-text detectors hold up against a rewriter that is trained to evade them.

## What this project is

We train a small language model to rewrite AI-written text so that the detectors in [SlopTotal](#sloptotal-the-detector-server) score it as human. The rewrite has to keep the meaning and stay fluent. We then use the trained rewriter to study two questions:

1. **Transfer:** when the rewriter is optimized against one family of detectors (neural classifiers, statistical/perplexity-based, or heuristic), does it also evade the other families?
2. **Defense:** how much does adversarial retraining (in the style of RADAR) make the detectors more robust?

The study is the contribution. Trained humanizers already exist (DIPPER, RADAR, Nicks et al. 2024, AuthorMist), so this is not meant to be another one.

> **Status:** These are in: the harness client, the data and baseline scores, the reward, and both trainers. The trainers have been smoke-tested, and the zero-shot baseline has been evaluated. The full training runs haven't been done yet. The evaluation and defense code isn't written yet.

## How it works

```
 AI passage ──► rewriter (Qwen + LoRA) ──► N rewrites
                                              │
                                              ▼
                        quality gates: unicode → length → meaning → fluency → NLI
                                              │ (fail any = reward 0)
                                              ▼
                        SlopTotal detector score (0–100) via HTTP
                                              │
                                              ▼
                        reward = (1 − detector/100) × similarity
                                              │
                                              ▼
                        best vs. worst rewrite → DPO pair → train next LoRA round
```

| Component | What it does | Code |
|---|---|---|
| Harness | Async client for SlopTotal's `quick-score` and `analyze` endpoints, with queue polling, retries and a concurrency limit. A script measures throughput. | [deai/harness.py](deai/harness.py), [scripts/check_harness.py](scripts/check_harness.py) |
| Data | Passages from SlopTotal's SlopBench corpus: AI passages from several generators and genres, each paired with a human passage written from the same seed. Split into train/val/test by topic group, with one generator held out for test only. Every passage is scored once to give the "before" table. | [deai/data.py](deai/data.py), [scripts/build_data.py](scripts/build_data.py), [scripts/score_baseline.py](scripts/score_baseline.py) |
| Reward | Quality gates, then detector score × semantic similarity. | [deai/reward.py](deai/reward.py), [scripts/calibrate_reward.py](scripts/calibrate_reward.py) |
| Best-of-N + DPO | Each round samples N = 8 rewrites per training source, scores them, pairs the best with the worst, and trains a fresh LoRA adapter with one epoch of DPO. Then it evaluates on val. | [deai/train_dpo.py](deai/train_dpo.py), [deai/policy.py](deai/policy.py), [deai/rollout.py](deai/rollout.py) |
| GRPO | TRL `GRPOTrainer` with the same reward. It can start from a DPO-trained policy. | [deai/train_grpo.py](deai/train_grpo.py) |
| Evaluation *(not written yet)* | Evasion rate, TPR at 1% FPR, the train-family × eval-family transfer matrix, quality metrics, baselines | — |
| Defense *(not written yet)* | Retrain SlopTotal classifiers on the rewrites and measure how much robustness each round adds | — |

### The rewriter

The default base model is `Qwen/Qwen3-4B-Instruct-2507` (`policy.BASE_MODEL`); you can pick another with `--base`. After round *k*, the policy is the base model plus LoRA adapters 1..*k*, merged in order. DPO's reference model is that policy with the newest adapter switched off, so it's always the previous round's policy. Round 0 is the untouched base model, which is only evaluated.

### Reward gates

A rewrite gets **zero reward** if it fails any of these checks, so the model can't trade quality for a lower detector score. The gates run cheapest first, and the detector is only called for rewrites that pass all of them:

- **Unicode:** NFKC-stable, with no zero-width characters or homoglyphs (characters the source itself uses are exempt)
- **Length:** 0.8–1.2× the source word count, and at least 80 words
- **Meaning:** embedding cosine similarity ≥ 0.85 (all-mpnet-base-v2, chunked so long passages aren't truncated)
- **Fluency:** mean per-sentence CoLA acceptability ≥ 0.70, the 2nd percentile of human passages. Perplexity is deliberately not used, because low perplexity is what statistical detectors flag.
- **NLI:** two-way, per-sentence entailment: ≥ 90% of sentences entailed in each direction. This catches dropped or flipped facts. It runs during training too (`--no-train-nli` turns it off).

If the detector server fails, the reward is `None` rather than 0, so the trainer skips that sample instead of learning from it.

### Reward sources

Each training run optimizes against one detector family (`--reward`):

| `--reward` | Detector score used |
|---|---|
| `quick` | `quick-score` overall (mostly Fakespot) |
| `neural-quick` | mean of the 4 neural engines that quick-score already runs (a cheap stand-in for `neural`) |
| `neural` | mean of the neural / classifier / embedding engines from `analyze` |
| `stat` | mean of the statistical engines (Binoculars, Fast-DetectGPT, …) |
| `full` | `analyze` overall score (expensive) |

Comparing these runs against all engine families is the transfer study.

## SlopTotal: the detector server

The detectors come from a local SlopTotal install, which runs as a FastAPI server on `http://localhost:8000`. Start it with `scripts\start.bat` (or `scripts/start.sh`) in the SlopTotal folder. This project expects SlopTotal at `D:\AI\sloptotal`, and reads the SlopBench corpus from `D:\AI\sloptotal\tests\eval\slopbench\corpus`.

| Endpoint | Used for |
|---|---|
| `POST /api/quick-score` | Cheap training reward (6 engines; mostly Fakespot), ~2 texts/s on CPU |
| `POST /api/analyze` | Evaluation (all 23 engines with per-engine scores), ~0.3 texts/s on CPU |
| `POST /api/paragraph-score` | Failure analysis |
| `GET /api/queue/ticket/{id}` | Polling when a request comes back `202 queued` |

Inputs must be at least 50 characters. Engines with zero weight in the overall score (GLTR, log-rank, perplexity, DiveEye, …) still report their own scores, so they serve as held-out detectors for measuring transfer.

## How to run

### 1. Requirements

- A CUDA GPU. Development uses one RTX 5070 Ti (16 GB) shared with SlopTotal: about 3–5 GB for the detectors and about 11 GB for training.
- Python **3.12**, in a venv separate from SlopTotal's 3.14 one.
- PyTorch built for CUDA 12.8 or newer (needed for Blackwell GPUs).
- A working SlopTotal install (see above).

### 2. Install

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

If GPU memory runs short, start SlopTotal with `SLOPTOTAL_DEVICE=cpu`.

### 3. Check the harness

With SlopTotal running, score known-human and known-AI passages and measure throughput:

```powershell
.venv\Scripts\python scripts\check_harness.py
```

### 4. Build the data and baseline scores

```powershell
.venv\Scripts\python scripts\build_data.py        # -> data/passages.parquet
.venv\Scripts\python scripts\score_baseline.py    # -> data/baseline_scores.parquet
```

`score_baseline.py` caches scores as they arrive, so an interrupted run resumes where it stopped. On a CPU-only SlopTotal, `analyze` takes about 3 hours over all passages and `quick-score` about 30 minutes. Use `--limit 20` for a smoke test, or `--endpoints quick` to skip `analyze`.

### 5. (Optional) Re-calibrate the reward

```powershell
.venv\Scripts\python scripts\calibrate_reward.py
```

This sets the fluency threshold from human passages and sanity-checks every gate.

### 6. Train

The easiest way to start best-of-N + DPO training is the launcher script. It starts SlopTotal if it isn't running, starts TensorBoard and opens it in the browser, and then runs `deai.train_dpo`:

```powershell
train.bat                                                    # Windows
./train.sh                                                   # Git Bash / Linux / macOS
```

The default is run `r-quick-qwen3-4b`: the quick-score reward, 3 rounds. Any extra arguments are passed through to the trainer and override the defaults:

```powershell
train.bat --rounds 1 --train-limit 8 --eval-limit 8 --n 4 --run smoke    # end-to-end smoke test
train.bat --reward neural-quick --run r-neural-quick-qwen3-4b
train.bat --base Qwen/Qwen2.5-1.5B-Instruct --gen-batch 64 --run r-quick
```

You can also set the defaults with the environment variables `DEAI_RUN`, `DEAI_REWARD`, `DEAI_ROUNDS`, `TB_PORT` and `SLOPTOTAL_DIR`.

To run the trainers directly:

```powershell
.venv\Scripts\python -m deai.train_dpo --run r-quick --reward quick --rounds 3
.venv\Scripts\python -m deai.train_grpo --run r-quick --reward quick --init-from runs\dpo-r-quick
```

Every stage skips work whose output already exists, so **rerunning the same command resumes an interrupted run**. Run either module with `--help` to see all options (learning rate, β, LoRA rank, N, temperature, `--loss-type sigmoid,sft` for RPO-style training, and others).

### 7. Monitor and inspect

```powershell
.venv\Scripts\tensorboard --logdir runs    # http://localhost:6006
```

Each round's validation metrics and per-gate pass rates, live scoring progress and the trainer's loss are logged under `runs/<run>/tb/`. Runs overlay on the same charts. Runs from before TensorBoard was added can be logged with `python -m deai.tblog`; running it again doesn't log anything twice.

Each DPO round writes its outputs to `runs/dpo-<run>/round_k/`:

| File | Contents |
|---|---|
| `metrics.json` | Validation summary: gate pass rates, mean detector score, share below 55, and `evasion_and_quality` (passes every gate *and* scores < 55) |
| `eval.jsonl` | Every greedy val rewrite, with its gate results and detector score |
| `spotcheck.md` | 30 sample rewrites next to their sources, to read by hand |
| `samples.jsonl`, `scored.jsonl`, `pairs.jsonl`, `round_stats.json` | Best-of-N samples, their rewards, the DPO pairs, and summary stats |
| `adapter/` | The LoRA adapter trained this round |

Read `spotcheck.md` after every round. Metrics alone won't show a shortcut that nobody anticipated.

### 8. Tests

```powershell
.venv\Scripts\python -m pytest                  # everything
.venv\Scripts\python -m pytest -m "not models"  # skip tests that load the reward models
```

## Repository layout

```
DeAI-project/
  pyproject.toml
  train.bat, train.sh # DPO training launcher (SlopTotal + TensorBoard + training)
  deai/
    harness.py        # SlopTotal client (quick/analyze/ticket polling)
    data.py           # SlopBench passages, splits, baseline table
    metrics.py        # small detector metrics (AUROC, TPR at FPR)
    reward.py         # gates + reward, reward sources
    policy.py         # rewriter prompt, model + adapter loading, sampling
    rollout.py        # reward scoring, validation, spot checks
    train_dpo.py      # best-of-N + DPO loop
    train_grpo.py     # GRPO
    tblog.py          # TensorBoard logging and backfill
  scripts/
    check_harness.py    # detector scores and throughput
    build_data.py       # data/passages.parquet
    score_baseline.py   # data/baseline_scores.parquet
    calibrate_reward.py # fluency threshold + gate sanity checks
  tests/              # data, harness, reward (shortcut examples must score 0), training
  data/               # gitignored
  runs/               # gitignored: checkpoints, logs, TensorBoard
```

## Responsible use

This project is for red-teaming detectors and for research. Trained rewriter weights will not be released publicly unless the defense results (hardened detectors and adversarial data) are released with them. Where it is useful, results and adversarial data will be contributed back to SlopTotal.
