#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/glm-4-9b-chat-abliterated-Q4_K_M.gguf}"
GPU_LAYERS="${NGL:-auto}"
FIT_MARGIN="${FIT_TARGET:-512}"
CTX_SIZE="${CTX:-8192}"
PARALLEL_SLOTS="${PARALLEL:-1}"
THREADS="${THREADS:-6}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

if ! command -v llama-server >/dev/null 2>&1; then
    echo "Error: llama-server is not installed." >&2
    exit 1
fi

if pgrep -x "llama-server" >/dev/null 2>&1; then
    echo "Waiting for existing llama-server process to release GPU memory and port..." >&2
    for _ in {1..5}; do
        pgrep -x "llama-server" >/dev/null 2>&1 || break
        sleep 1
    done
    if pgrep -x "llama-server" >/dev/null 2>&1; then
        echo "Error: Another llama-server instance is still running (PID: $(pgrep -x llama-server | tr '\n' ' '))." >&2
        echo "Stop it first with: pkill -x llama-server" >&2
        exit 1
    fi
fi

exec llama-server \
    -m "${MODEL_PATH}" \
    -ngl "${GPU_LAYERS}" \
    -fit on \
    -fitt "${FIT_MARGIN}" \
    -c "${CTX_SIZE}" \
    -np "${PARALLEL_SLOTS}" \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    -t "${THREADS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
