#!/usr/bin/env bash
# DeAI - Phase 4A training (best-of-N + DPO) as a Slurm job array on Iridis X L4s.
#
# Several runs at once: each array task trains one reward source on its own L4.
# Together they are the per-family rewriters the transfer study compares. To spread
# one run over several L4s instead, use slurm_train_l4_multi.sh.
#
# One-time, on the login node first:  scripts/hpc_setup.sh   (venv, data, models, hpc_env.sh)
#
# Usage:  sbatch slurm_train_l4_array.sh [extra train_dpo arguments, applied to every task]
#   sbatch slurm_train_l4_array.sh                    # all 5 rewards, runs r-<reward>-qwen3-4b
#   sbatch --array=0,2 slurm_train_l4_array.sh        # only quick and neural
#   DEAI_RUN_SUFFIX=lr1e-5 sbatch slurm_train_l4_array.sh --lr 1e-5   # runs r-<reward>-lr1e-5
#
# Resubmitting the same command resumes every run (finished stages are skipped),
# which is also why the job is requeueable.
#
# L4 settings (24 GB), versus the 16 GB defaults:
#   --vram medium    high needs ~24 GB, the whole card, so the scorers still swap out
#   --gen-batch 32   16 sequences used 11.6 GB with the policy alone (16 on 16 GB)
#   --batch-size 2   effective batch stays 16, so the optimisation matches local runs (1)
# None of these are pinned in config.json, so they can change when resuming a run.
# An L4 has about a third of an A100's bf16 throughput and memory bandwidth, so allow
# roughly 2-3x the A100 wall time.

#SBATCH --job-name=deai-dpo-l4
#SBATCH -p l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ ! -f hpc_env.sh ]]; then
    echo "Error: hpc_env.sh not found. Run scripts/hpc_setup.sh on the login node first." >&2
    exit 1
fi
source hpc_env.sh   # cd to the repo, activate .venv, HF cache offline

# One reward source per array index (deai.reward.REWARD_SOURCES).
REWARDS=(quick neural-quick neural stat full)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
DEAI_REWARD="${REWARDS[$TASK]:?no reward for array index $TASK (0-$((${#REWARDS[@]} - 1)))}"
DEAI_RUN="r-$DEAI_REWARD-${DEAI_RUN_SUFFIX:-qwen3-4b}"
DEAI_ROUNDS="${DEAI_ROUNDS:-3}"
DEAI_VRAM="${DEAI_VRAM:-medium}"
DEAI_GEN_BATCH="${DEAI_GEN_BATCH:-32}"
DEAI_BATCH_SIZE="${DEAI_BATCH_SIZE:-2}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID (array task $TASK) on $(hostname):" \
     "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
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
