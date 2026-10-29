#!/usr/bin/env bash
# DeAI - one Phase 4A training run (best-of-N + DPO) spread over several Iridis X L4s.
#
# Sampling, scoring and eval generation run one process per GPU (deai/multigpu.py);
# DPO training runs on the first GPU. Results match a one-GPU run, so this only
# changes the wall time. For several runs at once use slurm_train_l4_array.sh instead.
#
# One-time, on the login node first:  scripts/hpc_setup.sh   (venv, data, models, hpc_env.sh)
#
# Usage:  sbatch slurm_train_l4_multi.sh [extra train_dpo arguments]
#   sbatch slurm_train_l4_multi.sh
#   sbatch slurm_train_l4_multi.sh --reward neural-quick --run r-neural-quick-qwen3-4b
#   sbatch --gres=gpu:4 slurm_train_l4_multi.sh          # fewer GPUs: every visible one is used
#
# Later arguments win, so anything passed here overrides the defaults below.
# Resubmitting the same command resumes the run (finished stages are skipped), which
# is also why the job is requeueable.
#
# L4 settings (24 GB per GPU, each worker has its own), versus the 16 GB defaults:
#   --vram medium    high needs ~24 GB, the whole card, so the scorers still swap out
#   --gen-batch 32   16 sequences used 11.6 GB with the policy alone (16 on 16 GB)
#   --batch-size 2   effective batch stays 16, so the optimisation matches local runs (1)
# None of these are pinned in config.json, so they can change when resuming a run.
# Host memory: each worker loads its own policy (~8 GB while loading), hence --mem.

#SBATCH --job-name=deai-dpo-l4x7
#SBATCH -p l4
#SBATCH --gres=gpu:7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
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
DEAI_VRAM="${DEAI_VRAM:-medium}"
DEAI_GEN_BATCH="${DEAI_GEN_BATCH:-32}"
DEAI_BATCH_SIZE="${DEAI_BATCH_SIZE:-2}"

# Each worker process gets a share of the CPUs, not all of them.
NGPU=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU > 0 ? ${SLURM_CPUS_PER_TASK:-8} / NGPU : 1 ))
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname): $NGPU GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
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
opt DEAI_GPUS --gpus
read -ra extra <<< "${DEAI_EXTRA_ARGS:-}"
args+=("${extra[@]}")

srun python -m deai.train_dpo --run "$DEAI_RUN" --reward "$DEAI_REWARD" --rounds "$DEAI_ROUNDS" \
    --vram "$DEAI_VRAM" --gen-batch "$DEAI_GEN_BATCH" --batch-size "$DEAI_BATCH_SIZE" \
    "${args[@]}" "$@"
