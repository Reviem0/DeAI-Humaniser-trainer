# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project that trains a small rewriter (Qwen3-4B + LoRA) to make AI text score as human on in-process ports of the **SlopTotal** detectors, then studies (1) transfer across detector families and (2) adversarial-retraining defenses. The study is the contribution, not the humanizer. `PLAN.md` is the phase-by-phase design doc (Phases 1–4 done; Phase 5 evaluation and Phase 6 defense not written yet) and records past findings that explain current design choices; read the relevant phase before changing that area.

## Environment

- Windows, Python 3.12 venv at `.venv`. Always invoke via `.venv\Scripts\python`.
- PyTorch CUDA 12.8 build on one 16 GB GPU shared by the policy and the scorers, which only fit one at a time (worst case: DPO training 12.3 GB, sampling 11.6 GB; detectors 8.4 GB if all 23 load, quality models 1.5 GB). `--vram` / `DEAI_VRAM` (`deai/vram.py`) picks `high` (all resident, ~24 GB), `medium` (default: `Scorer.release()` before every policy stage, fits 16 GB) or `low` (+ activation offloading, generate ≤ 8). `import deai` selects the cudaMallocAsync allocator if torch isn't loaded yet; without it training fragments to 18 GB reserved. `DEAI_DETECTOR_DEVICE=cpu` keeps detectors off the GPU (GRPO does this in medium/low).
- No server: the engines and calibration are ported from SlopTotal (MIT) into `deai/detectors/`. The SlopBench corpus lives in `data/slopbench` (`scripts\fetch_slopbench.py`, pinned commit; `deai.data.SLOPBENCH`, override with `DEAI_SLOPBENCH`).
- `data/` and `runs/` are gitignored. `.env` holds per-machine launcher settings; edit `.env.example` instead (a project hook blocks edits to `.env`).

## Commands

```powershell
.venv\Scripts\python -m pytest -m "not models"            # fast suite (~15s), no reward models loaded
.venv\Scripts\python -m pytest                            # also the tests that load real HF reward models
.venv\Scripts\python -m pytest tests/test_reward.py::test_name -q   # single test

.venv\Scripts\python scripts\fetch_slopbench.py           # -> data/slopbench (pinned SlopTotal commit)
.venv\Scripts\python scripts\check_detectors.py           # detector sanity + texts/s (quick, analyze)
.venv\Scripts\python scripts\build_data.py                # -> data/passages.parquet
.venv\Scripts\python scripts\score_baseline.py --limit 20 # -> data/baseline_scores.parquet (resumable)
.venv\Scripts\python scripts\calibrate_reward.py          # fluency threshold + gate sanity checks

train.bat --rounds 1 --train-limit 8 --eval-limit 8 --n 4 --run smoke   # end-to-end smoke test
.venv\Scripts\python -m deai.train_dpo --run <name> --reward quick --rounds 3
.venv\Scripts\python -m deai.train_grpo --run <name> --reward quick --init-from runs\dpo-<name>
.venv\Scripts\tensorboard --logdir runs
```

`train.bat`/`train.sh` start TensorBoard if needed, then run `deai.train_dpo`. Settings precedence: CLI args > shell env > `.env` (`DEAI_*` variables map to trainer flags; see `.env.example`). No linter or formatter is configured.

Test notes: `tests/test_data.py` skips when the SlopBench corpus is absent; `models`-marked tests skip if HF weights can't load. `tests/test_detectors*.py` check parity with SlopTotal on fixed texts (overall exact, engines within 0.001); reward tests inject stub detectors/`QualityModels`, so they need no GPU.

## Architecture

Pipeline: AI passage → rewriter samples N rewrites → quality gates → detector score (in-process) → reward → DPO pairs → next LoRA round.

- **`detectors/`** — SlopTotal's 23 engines (`neural`, `statistical`, `heuristic`, `language`; interface and device in `base`) and its calibration (`calibration`). `Detectors().score(text, "quick"|"analyze", engines=(), overall=True)` / `score_many` return `Score` (overall 0–100, engines 0–1); only engines the request needs are loaded and run, and `unload()` frees them. Texts under 50 chars raise `TextRejected`; a failing engine raises `DetectorError` (a deliberate difference from SlopTotal's 0.5/0.0 fallback). Don't change engine or calibration code without rerunning the parity tests.
- **`reward.py`** — `Reward.score()` runs gates cheapest-first (unicode → length → meaning → fluency → NLI) and only calls the detector for rewrites that passed all of them. Failing a gate gives reward **0**; a detector failure gives reward **`None`**, which trainers must skip rather than learn from. Reward = `(1 − detector/100) × similarity`. `REWARD_SOURCES` maps `--reward` names (`quick`, `neural-quick`, `neural`, `stat`, `full`) to an endpoint plus optional engine family; family scores are the mean of that family's engines. Perplexity is deliberately not a gate (low perplexity is what statistical detectors flag). Models are lazy and injectable through the `QualityModels` protocol.
- **`data.py`** — SlopBench AI/human pairs, filtered to 100–600 words. Splits are hashed by topic group so a group and a pair never cross splits; one generator (`HELD_OUT_GENERATORS`) is test-only. Changing split logic invalidates every existing run and baseline.
- **`policy.py`** — prompt (`INSTRUCTION`, non-thinking chat template), model loading, sampling. The policy after round *k* is the base model with `round_1..k/adapter` merged in order; only LoRA adapters are saved, never merged weights.
- **`rollout.py`** — `Scorer` (detectors + reward; `release()` frees the detectors' and quality models' GPU memory), validation (`evaluate`, `summarize_eval`, `spotcheck.md`), and the `LIKELY_AI = 55` threshold behind `evasion_and_quality`.
- **`train_dpo.py`** — per round, the stages `sample → score → pairs → train → eval` each write files under `runs/dpo-<run>/round_k/` and are **skipped when their output exists**, so rerunning the same command resumes. DPO trains a fresh adapter on top of the previous rounds; the reference model is that same stack with the newest adapter off (i.e. the previous round's policy). Pairs are best vs. worst rewrite per source, dropped when the gap is below `--min-margin` or nothing passed. `config.json` pins `reward`, `base`, `n`, `train_limit`, `train_nli`, `instruction` and `detectors` (runs scored by the old SlopTotal server lack it, so they can't be resumed); changing any of them for an existing run name is refused, so use a new `--run`.
- **`train_grpo.py`** — TRL `GRPOTrainer` with the same gated reward (`GateReward`), optionally initialised from a DPO run.
- **`tblog.py`** — TensorBoard logging, idempotent; `python -m deai.tblog` backfills older runs.
- **`vram.py`** — the `--vram` modes and the measurements behind them; the trainers log each stage's peak memory to `train.log`.
- **`consolelog.py`** — `start(run_dir)` tees all console output (with a time per line, one header per launch) into `runs/<run>/train.log`; both trainers call it.

When judging a training change, read `round_k/spotcheck.md`, not just `metrics.json`: the gates exist because metrics alone miss shortcuts, and `tests/test_reward.py` asserts known shortcut rewrites score 0.

## Responsible use

Trained rewriter weights are not to be released publicly unless the defense results (hardened detectors, adversarial data) are released with them.
