# DeAI

A research project on how well AI-text detectors hold up against a rewriter that is trained to evade them.

## What this project is

We train a small language model to rewrite AI-written text so that the detectors ported from [SlopTotal](#the-detectors) score it as human. The rewrite has to keep the meaning and stay fluent. We then use the trained rewriter to study two questions:

1. **Transfer:** when the rewriter is optimized against one family of detectors (neural classifiers, statistical/perplexity-based, or heuristic), does it also evade the other families?
2. **Defense:** how much does adversarial retraining (in the style of RADAR) make the detectors more robust?

The study is the contribution. Trained humanizers already exist (DIPPER, RADAR, Nicks et al. 2024, AuthorMist), so this is not meant to be another one.

> **Status:** These are in: the in-process detectors, the data and baseline scores, the reward, and both trainers. The trainers have been smoke-tested, and the zero-shot baseline has been evaluated. The full training runs haven't been done yet. The evaluation and defense code isn't written yet.

## How it works

```
 AI passage ──► rewriter (Qwen + LoRA) ──► N rewrites
                                              │
                                              ▼
                        quality gates: unicode → length → meaning → fluency → NLI
                                              │ (fail any = reward 0)
                                              ▼
                        detector score (0–100), in-process
                                              │
                                              ▼
                        reward = (1 − detector/100) × similarity
                                              │
                                              ▼
                        best vs. worst rewrite → DPO pair → train next LoRA round
```

| Component | What it does | Code |
|---|---|---|
| Detectors | In-process ports of SlopTotal's 23 engines and its `quick` and `analyze` overall scores. A script checks separation and measures throughput. | [deai/detectors/](deai/detectors/__init__.py), [scripts/check_detectors.py](scripts/check_detectors.py) |
| Data | Passages from the SlopBench corpus (shipped with SlopTotal): AI passages from several generators and genres, each paired with a human passage written from the same seed. Split into train/val/test by topic group, with one generator held out for test only. Every passage is scored once to give the "before" table. | [deai/data.py](deai/data.py), [scripts/build_data.py](scripts/build_data.py), [scripts/score_baseline.py](scripts/score_baseline.py) |
| Reward | Quality gates, then detector score × semantic similarity. | [deai/reward.py](deai/reward.py), [scripts/calibrate_reward.py](scripts/calibrate_reward.py) |
| Best-of-N + DPO | Each round samples N = 8 rewrites per training source, scores them, pairs the best with the worst, and trains a fresh LoRA adapter with one epoch of DPO. Then it evaluates on val. | [deai/train_dpo.py](deai/train_dpo.py), [deai/policy.py](deai/policy.py), [deai/rollout.py](deai/rollout.py) |
| GRPO | TRL `GRPOTrainer` with the same reward. It can start from a DPO-trained policy. | [deai/train_grpo.py](deai/train_grpo.py) |
| Evaluation *(not written yet)* | Evasion rate, TPR at 1% FPR, the train-family × eval-family transfer matrix, quality metrics, baselines | — |
| Defense *(not written yet)* | Retrain the classifiers on the rewrites and measure how much robustness each round adds | — |

### The rewriter

The default base model is `Qwen/Qwen3-4B-Instruct-2507` (`policy.BASE_MODEL`); you can pick another with `--base`. After round *k*, the policy is the base model plus LoRA adapters 1..*k*, merged in order. DPO's reference model is that policy with the newest adapter switched off, so it's always the previous round's policy. Round 0 is the untouched base model, which is only evaluated.

### Reward gates

A rewrite gets **zero reward** if it fails any of these checks, so the model can't trade quality for a lower detector score. The gates run cheapest first, and the detector is only called for rewrites that pass all of them:

- **Unicode:** NFKC-stable, with no zero-width characters or homoglyphs (characters the source itself uses are exempt)
- **Length:** 0.8–1.2× the source word count, and at least 80 words
- **Meaning:** embedding cosine similarity ≥ 0.85 (all-mpnet-base-v2, chunked so long passages aren't truncated)
- **Fluency:** mean per-sentence CoLA acceptability ≥ 0.70, the 2nd percentile of human passages. Perplexity is deliberately not used, because low perplexity is what statistical detectors flag.
- **NLI:** two-way, per-sentence entailment: ≥ 90% of sentences entailed in each direction. This catches dropped or flipped facts. It runs during training too (`--no-train-nli` turns it off).

If a detector engine fails, the reward is `None` rather than 0, so the trainer skips that sample instead of learning from it.

### Reward sources

Each training run optimizes against one detector family (`--reward`):

| `--reward` | Detector score used |
|---|---|
| `quick` | `quick` overall (mostly Fakespot) |
| `neural-quick` | mean of the 4 neural engines that `quick` already runs (a cheap stand-in for `neural`) |
| `neural` | mean of the neural / classifier / embedding engines from `analyze` |
| `stat` | mean of the statistical engines (Binoculars, Fast-DetectGPT, …) |
| `full` | `analyze` overall score (expensive) |

Comparing these runs against all engine families is the transfer study.

## The detectors

The 23 detector engines and their calibration come from [SlopTotal](https://github.com/pablocaeg/sloptotal) (MIT). [deai/detectors/](deai/detectors/__init__.py) ports them to run in-process, so no server is needed: `base` (engine interface, device), `neural` (9 classifiers), `statistical` (GPT-2-based: Binoculars, Fast-DetectGPT, perplexity, GLTR, …), `heuristic`, `language` and `calibration`. Tests check them against SlopTotal's own code on shared texts: overall scores match exactly, engine scores within 0.001.

| Endpoint | Used for |
|---|---|
| `quick` | Cheap training reward (6 engines, 4 with non-zero weight; mostly Fakespot) |
| `analyze` | Evaluation (all 23 engines with per-engine scores) |

```python
from deai.detectors import Detectors
det = Detectors()
det.score(text, "quick").overall                                   # 0–100
det.score_many(texts, "analyze", engines=det.endpoint_engines("analyze"))
```

Only the engines a request needs are loaded and run. Models run in fp32: the 9 classifiers take about 6.7 GB of GPU memory and gpt2-medium/distilgpt2 about 1.7 GB if all 23 engines load; the `quick` reward needs only 4. Set `DEAI_DETECTOR_DEVICE=cpu` to keep them off the GPU. Inputs must be at least 50 characters. Engines with zero weight in the overall score (GLTR, log-rank, perplexity, DiveEye, …) still report their own scores, so they serve as held-out detectors for measuring transfer.

One deliberate difference from SlopTotal: a failing engine raises (`DetectorError`), so the reward becomes `None` and the sample is skipped, where SlopTotal would score it 0.5 (quick) or 0.0 (analyze) and carry on.

The SlopBench corpus is fetched into `data/slopbench` at a pinned SlopTotal commit by `scripts/fetch_slopbench.py` (override the location with `DEAI_SLOPBENCH`).

## How to run

### 1. Requirements

- A CUDA GPU with 16 GB (development: one RTX 5070 Ti) in the default `--vram medium`; see [GPU memory](#gpu-memory).
- Python **3.12**.
- PyTorch built for CUDA 12.8 or newer (needed for Blackwell GPUs).

### 2. Install

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

### GPU memory

The policy and the scorers don't fit a 16 GB card together. Worst case (600-word sources, Qwen3-4B, `--max-length 2048`): DPO training peaks at 12.3 GB and sampling 16 sequences at 11.6 GB; the detectors take 8.4 GB if all 23 load (11.3 GB while scoring) and the gates' quality models 1.5 GB. `--vram` (or `DEAI_VRAM`) chooses how they share the GPU:

| Mode | What it does | Needs |
|---|---|---|
| `high` | Every model stays loaded. Fastest. | ~24 GB |
| `medium` (default) | The detectors and quality models are freed before every policy stage (sample, train, eval generation) and reload to score. | 16 GB |
| `low` | `medium`, plus activation offloading during DPO and at most 8 sequences per `generate()`. | ~12 GB |

GRPO keeps the policy and the scorers loaded together, so in `medium` and `low` it runs the detectors on the CPU. No mode changes a score: the detectors are fp32 on any device and agree across CPU and GPU to 0.001. `import deai` also selects PyTorch's `cudaMallocAsync` allocator (unless `PYTORCH_CUDA_ALLOC_CONF` is set); with the default allocator, DPO training fragments to 18 GB reserved on Windows and spills into shared system memory. Each stage's peak is printed to `train.log`. To keep the detectors off the GPU entirely, set `DEAI_DETECTOR_DEVICE=cpu`.

### 3. Fetch SlopBench and check the detectors

```powershell
.venv\Scripts\python scripts\fetch_slopbench.py     # -> data/slopbench
.venv\Scripts\python scripts\check_detectors.py     # separation + texts/s for quick and analyze
```

### 4. Build the data and baseline scores

```powershell
.venv\Scripts\python scripts\build_data.py        # -> data/passages.parquet
.venv\Scripts\python scripts\score_baseline.py    # -> data/baseline_scores.parquet
```

`score_baseline.py` records every engine of each endpoint and caches scores in chunks, so an interrupted run resumes where it stopped. Use `--limit 20 --out-dir <tmp>` for a smoke test, or `--endpoints quick` to skip `analyze`.

### 5. (Optional) Re-calibrate the reward

```powershell
.venv\Scripts\python scripts\calibrate_reward.py
```

This sets the fluency threshold from human passages and sanity-checks every gate.

### 6. Train

The easiest way to start best-of-N + DPO training is the launcher script. It starts TensorBoard and opens it in the browser, and then runs `deai.train_dpo`:

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

#### Settings in `.env`

To avoid retyping arguments, put the training settings in a `.env` file. The launchers read it at startup:

```powershell
copy .env.example .env    # then edit .env
```

[.env.example](.env.example) lists every setting with its default. Each line is `KEY=value`, with no spaces around `=`, no quotes and no comment after the value. A setting that is commented out or left empty uses the trainer's default.

| Variable | Trainer flag | Default |
|---|---|---|
| `DEAI_RUN` | `--run` | `r-quick-qwen3-4b` |
| `DEAI_REWARD` | `--reward` | `quick` |
| `DEAI_ROUNDS` | `--rounds` | `3` |
| `DEAI_BASE` | `--base` | `Qwen/Qwen3-4B-Instruct-2507` |
| `DEAI_N` | `--n` | `8` |
| `DEAI_TEMPERATURE` | `--temperature` | `0.9` |
| `DEAI_GEN_BATCH` | `--gen-batch` | `16` |
| `DEAI_TRAIN_LIMIT`, `DEAI_EVAL_LIMIT` | `--train-limit`, `--eval-limit` | all sources |
| `DEAI_LR` | `--lr` | `2e-5` |
| `DEAI_BETA` | `--beta` | `0.1` |
| `DEAI_LORA_R` | `--lora-r` | `16` |
| `DEAI_LOSS_TYPE` | `--loss-type` | `sigmoid` |
| `DEAI_EFFECTIVE_BATCH` | `--effective-batch` | `16` |
| `DEAI_VRAM` | `--vram` | `medium` |
| `DEAI_EXTRA_ARGS` | any other flags, space-separated | — |
| `DEAI_DETECTOR_DEVICE` | — (`cuda` or `cpu` for the detectors) | `cuda` when available |
| `DEAI_SLOPBENCH` | — (SlopBench corpus folder) | `data\slopbench` |
| `TB_PORT` | — (TensorBoard port) | `6006` |
| `HF_HOME` | — (Hugging Face model cache) | Hugging Face's default |

Precedence, highest first: arguments passed to `train.bat`/`train.sh`, then variables already set in your shell, then `.env`. `.env` is gitignored, so each machine can have its own.

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

Everything a run prints, including warnings and tracebacks, is also appended to `runs/<run>/train.log`, with the time on each line and a header (command, git commit) each time the run starts or resumes.

Each DPO round writes its outputs to `runs/dpo-<run>/round_k/`:

| File | Contents |
|---|---|
| `metrics.json` | Validation summary: gate pass rates, mean detector score, share below 55, and `evasion_and_quality` (passes every gate *and* scores < 55) |
| `eval.jsonl` | Every greedy val rewrite, with its gate results and detector score |
| `spotcheck.md` | 30 sample rewrites next to their sources, to read by hand |
| `samples.jsonl`, `scored.jsonl`, `pairs.jsonl`, `round_stats.json` | Best-of-N samples, their rewards, the DPO pairs, and summary stats |
| `adapter/` | The LoRA adapter trained this round: the round's checkpoint. Only the LoRA weights are saved, never a merged model. The policy after round *k* is the base model plus `round_1/adapter` … `round_k/adapter`. |

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
  train.bat, train.sh # DPO training launcher (TensorBoard + training)
  .env.example        # launcher settings; copy to .env
  deai/
    detectors/        # SlopTotal's 23 engines + calibration, in-process
      base.py         #   engine interface, device
      neural.py       #   9 classifiers
      statistical.py  #   GPT-2-based engines
      heuristic.py    #   linguistic, formulaic, structural, ...
      language.py     #   language detection
      calibration.py  #   quick/analyze overall scores and verdicts
    data.py           # SlopBench passages, splits, baseline table
    metrics.py        # small detector metrics (AUROC, TPR at FPR)
    reward.py         # gates + reward, reward sources
    policy.py         # rewriter prompt, model + adapter loading, sampling
    rollout.py        # reward scoring, validation, spot checks
    train_dpo.py      # best-of-N + DPO loop
    train_grpo.py     # GRPO
    tblog.py          # TensorBoard logging and backfill
    consolelog.py     # runs/<run>/train.log: console output of every run
  scripts/
    fetch_slopbench.py  # data/slopbench (pinned SlopTotal commit)
    check_detectors.py  # detector separation and throughput
    build_data.py       # data/passages.parquet
    score_baseline.py   # data/baseline_scores.parquet
    calibrate_reward.py # fluency threshold + gate sanity checks
  tests/              # data, detectors (parity with SlopTotal), reward (shortcut examples must score 0), training
  data/               # gitignored
  runs/               # gitignored: checkpoints, logs, TensorBoard
```

## Responsible use

This project is for red-teaming detectors and for research. Trained rewriter weights will not be released publicly unless the defense results (hardened detectors and adversarial data) are released with them. Where it is useful, results and adversarial data will be contributed back to SlopTotal.
