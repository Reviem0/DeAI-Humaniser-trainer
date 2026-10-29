#!/usr/bin/env bash
# DeAI - Phase 4A training (best-of-N + DPO) as a Slurm batch job on one Iridis X A100.
#
# One-time, on the login node first:  scripts/hpc_setup.sh   (venv, data, models, hpc_env.sh)
#
# Usage:  sbatch slurm_train_a100.sh [extra train_dpo arguments]
#   sbatch slurm_train_a100.sh
#   sbatch slurm_train_a100.sh --reward neural-quick --run r-neural-quick-qwen3-4b
#   DEAI_RUN=smoke sbatch slurm_train_a100.sh --rounds 1 --train-limit 8 --eval-limit 8 --n 4
#
# Later arguments win, so anything passed here overrides the defaults below.
# Resubmitting the same command resumes the run (finished stages are skipped).
#
# A100 settings, versus the 16 GB defaults:
#   --vram high      policy, detectors and quality models all stay resident (~24 GB),
#                    no reloading of the scorers around every policy stage
#   --gen-batch      64 sequences per generate() on 80 GB, 32 on 40 GB (16 on 16 GB)
#   --batch-size     4 pairs per DPO step on 80 GB, 2 on 40 GB (1 on 16 GB); the
#                    effective batch stays 16, so the optimisation matches local runs
# The card's memory is read at start-up, so the same script works on both A100 sizes.
# These are for the default Qwen3-4B. Qwen/Qwen3.5-9B (DEAI_BASE, ~18 GB of weights) fits
# an 80 GB card as is; on 40 GB use DEAI_VRAM=medium DEAI_BATCH_SIZE=1. Download it first:
# scripts/hpc_setup.sh --base Qwen/Qwen3.5-9B. Several GPUs: raise --gres, all are used.
# None of these are pinned in config.json, so they can change when resuming a run.

#SBATCH --job-name=deai-dpo
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ ! -f hpc_env.sh ]]; then
    echo "Error: hpc_env.sh not found. Run scripts/hpc_setup.sh on the login node first." >&2
    exit 1
fi
source hpc_env.sh   # cd to the repo, activate .venv, HF cache offline

DEAI_RUN="${DEAI_RUN:-r-quick-qwen3-4b}"
DEAI_REWARD="${DEAI_REWARD:-quick}"
DEAI_ROUNDS="${DEAI_ROUNDS:-3}"
DEAI_VRAM="${DEAI_VRAM:-high}"

GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
if (( GPU_MB >= 70000 )); then
    DEAI_GEN_BATCH="${DEAI_GEN_BATCH:-64}" DEAI_BATCH_SIZE="${DEAI_BATCH_SIZE:-4}"
else
    DEAI_GEN_BATCH="${DEAI_GEN_BATCH:-32}" DEAI_BATCH_SIZE="${DEAI_BATCH_SIZE:-2}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Training run \"$DEAI_RUN\" (reward $DEAI_REWARD, $DEAI_ROUNDS rounds, vram $DEAI_VRAM," \
     "gen-batch $DEAI_GEN_BATCH, batch-size $DEAI_BATCH_SIZE). Resubmit to resume."
echo "TensorBoard: run it on the login node, e.g. tensorboard --logdir runs --port 6006"
echo

# Optional settings become flags only when set, so the trainer's defaults apply otherwise.
args=()
opt() { [[ -n "${!1:-}" ]] && args+=("$2" "${!1}"); return 0; }
opt DEAI_BASE --base
opt DEAI_N --n
opt DEAI_TEMPERATURE --temperature
opt DEAI_TRAIN_LIMIT --train-limit
opt DEAI_EVAL_LIMIT --eval-limit
opt DEAI_LR --lr
opt DEAI_BETA --beta
opt DEAI_LORA_R --lora-r
opt DEAI_LOSS_TYPE --loss-type
opt DEAI_EFFECTIVE_BATCH --effective-batch
read -ra extra <<< "${DEAI_EXTRA_ARGS:-}"
args+=("${extra[@]}")

srun python -m deai.train_dpo --run "$DEAI_RUN" --reward "$DEAI_REWARD" --rounds "$DEAI_ROUNDS" \
    --vram "$DEAI_VRAM" --gen-batch "$DEAI_GEN_BATCH" --batch-size "$DEAI_BATCH_SIZE" \
    "${args[@]}" "$@"
