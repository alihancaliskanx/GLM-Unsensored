#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/glm-4-9b-chat-abliterated-Q4_K_M.gguf}"
RULES_PATH="${RULES_PATH:-${SCRIPT_DIR}/rules.md}"
GPU_LAYERS="${NGL:-auto}"
FIT_MARGIN="${FIT_TARGET:-512}"
CTX_SIZE="${CTX:-8192}"
THREADS="${THREADS:-6}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

if ! command -v llama-cli >/dev/null 2>&1; then
    echo "Error: llama-cli is not installed." >&2
    exit 1
fi

if pgrep -x "llama-server" >/dev/null 2>&1; then
    echo "Waiting for existing llama-server process to release GPU memory..." >&2
    for _ in {1..5}; do
        pgrep -x "llama-server" >/dev/null 2>&1 || break
        sleep 1
    done
    if pgrep -x "llama-server" >/dev/null 2>&1; then
        echo "Error: Another llama-server instance is still running (PID: $(pgrep -x llama-server | tr '\n' ' ')) and holding VRAM." >&2
        echo "Stop it first with: pkill -x llama-server" >&2
        exit 1
    fi
fi

EXTRA_ARGS=()
if [[ -f "${RULES_PATH}" ]]; then
    EXTRA_ARGS+=(-sysf "${RULES_PATH}")
fi

exec llama-cli \
    -m "${MODEL_PATH}" \
    -ngl "${GPU_LAYERS}" \
    -fit on \
    -fitt "${FIT_MARGIN}" \
    -c "${CTX_SIZE}" \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    -t "${THREADS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
