#!/usr/bin/env bash
# DeAI - Phase 4A training (best-of-N + DPO) of Qwen/Qwen3.5-9B on 2 Iridis X A100s.
#
# Sampling, scoring and eval generation run one process per GPU (deai/multigpu.py);
# DPO training runs on the first GPU. Estimated ~5-6 h per run (round 0 + 3 rounds)
# on 2x 80 GB; time one round with --train-limit 64 to pin that down.
#
# One-time, on the login node first (venv, data, models incl. the 9B base, the
# linear-attention kernels, hpc_env.sh):
#   scripts/hpc_setup.sh --base Qwen/Qwen3.5-9B
#
# Usage:  sbatch slurm_train_a100x2_qwen35_9b.sh [extra train_dpo arguments]
#   sbatch slurm_train_a100x2_qwen35_9b.sh                          # quick reward, run r-quick-qwen3.5-9b
#   DEAI_REWARD=neural sbatch slurm_train_a100x2_qwen35_9b.sh       # run r-neural-qwen3.5-9b
#   DEAI_RUN=smoke-9b sbatch slurm_train_a100x2_qwen35_9b.sh --rounds 1 --train-limit 8 --eval-limit 8 --n 4
#
# Later arguments win, so anything passed here overrides the defaults below.
# Resubmitting the same command resumes the run (finished stages are skipped). Submit one job per reward source.
#
# Settings for Qwen3.5-9B (~18 GB of bf16 weights, ~32 KB of KV cache per token):
#   80 GB cards: --vram high (scorers stay loaded), --gen-batch 128, --batch-size 2
#   40 GB cards: --vram medium, --gen-batch 32, --batch-size 1 (DPO peaks ~25 GB)
# The effective batch stays 16 either way. These are estimates: check the per-stage
# peaks in runs/dpo-<run>/train.log after a smoke run. None of them are pinned in
# config.json, so they can change when resuming; the base is pinned.

#SBATCH --job-name=deai-dpo-9b
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ ! -f hpc_env.sh ]]; then
    echo "Error: hpc_env.sh not found. Run scripts/hpc_setup.sh --base Qwen/Qwen3.5-9B on the login node first." >&2
    exit 1
fi
source hpc_env.sh   # cd to the repo, activate .venv, HF cache offline

DEAI_BASE="${DEAI_BASE:-Qwen/Qwen3.5-9B}"
DEAI_REWARD="${DEAI_REWARD:-quick}"
DEAI_RUN="${DEAI_RUN:-r-$DEAI_REWARD-qwen3.5-9b}"
DEAI_ROUNDS="${DEAI_ROUNDS:-3}"

GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -n1)
if (( GPU_MB >= 70000 )); then
    DEAI_VRAM="${DEAI_VRAM:-high}" DEAI_GEN_BATCH="${DEAI_GEN_BATCH:-128}" DEAI_BATCH_SIZE="${DEAI_BATCH_SIZE:-2}"
else
    DEAI_VRAM="${DEAI_VRAM:-medium}" DEAI_GEN_BATCH="${DEAI_GEN_BATCH:-32}" DEAI_BATCH_SIZE="${DEAI_BATCH_SIZE:-1}"
fi

# Fail in seconds, not after queueing: the base must be in the offline cache.
python - "$DEAI_BASE" <<'EOF' || exit 1
import sys
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained(sys.argv[1])
except OSError as e:
    sys.exit(f"Error: {sys.argv[1]} is not in the HF cache ({e}). "
             f"Run scripts/hpc_setup.sh --base {sys.argv[1]} on the login node.")
EOF
python -c "import fla, causal_conv1d" 2>/dev/null \
    || echo "Warning: flash-linear-attention / causal-conv1d missing: Qwen3.5 falls back to slow PyTorch kernels." >&2

NGPU=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / NGPU > 0 ? ${SLURM_CPUS_PER_TASK:-16} / NGPU : 1 ))
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname): $NGPU GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Training run \"$DEAI_RUN\" (base $DEAI_BASE, reward $DEAI_REWARD, $DEAI_ROUNDS rounds, vram $DEAI_VRAM," \
     "gen-batch $DEAI_GEN_BATCH, batch-size $DEAI_BATCH_SIZE). Resubmit to resume."
echo "TensorBoard: run it on the login node, e.g. tensorboard --logdir runs --port 6006"
echo

# Optional settings become flags only when set, so the trainer's defaults apply otherwise.
args=()
opt() { [[ -n "${!1:-}" ]] && args+=("$2" "${!1}"); return 0; }
opt DEAI_N --n
opt DEAI_TEMPERATURE --temperature
opt DEAI_TRAIN_LIMIT --train-limit
opt DEAI_EVAL_LIMIT --eval-limit
opt DEAI_LR --lr
opt DEAI_BETA --beta
opt DEAI_LORA_R --lora-r
opt DEAI_LOSS_TYPE --loss-type
opt DEAI_EFFECTIVE_BATCH --effective-batch
opt DEAI_GPUS --gpus
read -ra extra <<< "${DEAI_EXTRA_ARGS:-}"
args+=("${extra[@]}")

srun python -m deai.train_dpo --run "$DEAI_RUN" --base "$DEAI_BASE" --reward "$DEAI_REWARD" \
    --rounds "$DEAI_ROUNDS" --vram "$DEAI_VRAM" --gen-batch "$DEAI_GEN_BATCH" --batch-size "$DEAI_BATCH_SIZE" \
    "${args[@]}" "$@"
