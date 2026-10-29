#!/usr/bin/env bash
# DeAI - start Phase 4A training (best-of-N + DPO) and open TensorBoard.
#
# Usage:  ./train.sh [extra train_dpo arguments]
#   ./train.sh                                  Qwen3-4B, quick-score reward, 3 rounds
#   ./train.sh --reward neural-quick --run r-neural-quick-qwen3-4b
#   ./train.sh --rounds 1 --train-limit 8 --eval-limit 8 --n 4 --run smoke   (smoke test)
#   ./train.sh --base Qwen/Qwen2.5-1.5B-Instruct --gen-batch 64 --run r-quick
#
# Later arguments win, so anything passed here overrides the defaults below.
# Rerunning the same command resumes an interrupted run.
# Works in Git Bash on Windows and on Linux/macOS.
#
# Settings: copy .env.example to .env and edit it (DEAI_RUN, DEAI_REWARD, DEAI_ROUNDS,
# DEAI_BASE, DEAI_LR, ..., TB_PORT, HF_HOME). Variables already set in
# the environment win over .env.

set -euo pipefail
cd "$(dirname "$0")"

# --- .env: KEY=value lines, # comments; doesn't override the environment -------
if [[ -f .env ]]; then
    while IFS='=' read -r key val || [[ -n "$key" ]]; do
        key="${key%$'\r'}" val="${val%$'\r'}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue   # blank lines, comments
        if [[ -z "${!key+x}" ]]; then export "$key=$val"; fi
    done < .env
fi

DEAI_RUN="${DEAI_RUN:-r-quick-qwen3-4b}"
DEAI_REWARD="${DEAI_REWARD:-quick}"
DEAI_ROUNDS="${DEAI_ROUNDS:-3}"
TB_PORT="${TB_PORT:-6006}"

if [[ -x .venv/Scripts/python.exe ]]; then
    PY=.venv/Scripts/python.exe   # Windows venv
elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    echo "Error: no venv in .venv. Create it first, see README.md \"Setup\"." >&2
    exit 1
fi

up() { curl -sf "$1" >/dev/null 2>&1; }

wait_for() {  # url seconds
    local i
    for ((i = 0; i < $2; i++)); do
        up "$1" && return 0
        sleep 1
    done
    return 1
}

open_url() {
    if command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$1" >/dev/null 2>&1
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1
    elif command -v open >/dev/null 2>&1; then open "$1"
    else echo "Open $1 in a browser."
    fi
}

mkdir -p runs

# --- TensorBoard: start once, reuse if already running --------------------------
TB_URL="http://localhost:$TB_PORT/"
if ! up "$TB_URL"; then
    echo "Starting TensorBoard on port $TB_PORT (log: runs/tensorboard.log)..."
    nohup "$PY" -m tensorboard.main --logdir runs --port "$TB_PORT" --reload_interval 15 \
        >runs/tensorboard.log 2>&1 &
    wait_for "$TB_URL" 60 || echo "Warning: TensorBoard did not start; training continues without it."
fi
up "$TB_URL" && open_url "$TB_URL"

# --- Training -------------------------------------------------------------------
# Optional settings become flags only when set, so the trainer's defaults apply otherwise.
args=()
opt() { [[ -n "${!1:-}" ]] && args+=("$2" "${!1}"); return 0; }
opt DEAI_BASE --base
opt DEAI_N --n
opt DEAI_TEMPERATURE --temperature
opt DEAI_GEN_BATCH --gen-batch
opt DEAI_TRAIN_LIMIT --train-limit
opt DEAI_EVAL_LIMIT --eval-limit
opt DEAI_LR --lr
opt DEAI_BETA --beta
opt DEAI_LORA_R --lora-r
opt DEAI_LOSS_TYPE --loss-type
opt DEAI_EFFECTIVE_BATCH --effective-batch
opt DEAI_VRAM --vram
read -ra extra <<< "${DEAI_EXTRA_ARGS:-}"
args+=("${extra[@]}")

export PYTHONIOENCODING=utf-8
echo
echo "Training run \"$DEAI_RUN\" (reward $DEAI_REWARD, $DEAI_ROUNDS rounds). Ctrl+C stops; rerun to resume."
if (( ${#args[@]} )); then echo "Settings: ${args[*]}"; fi
echo
exec "$PY" -m deai.train_dpo --run "$DEAI_RUN" --reward "$DEAI_REWARD" --rounds "$DEAI_ROUNDS" "${args[@]}" "$@"
