#!/bin/bash
# Quality A/B of one GGUF: llmprobe (capability card + reasoning eval, greedy,
# thinking off) against this engine and against a llama.cpp llama-server on
# the same file. Reports land in ~/.llmprobe (llmprobe --library), the raw
# output next to the parity logs.
#
#   scripts/llmprobe_ab.sh /path/model.gguf [llmprobe args...]
set -euo pipefail
MODEL="$1"; shift
BIN="${MLX_SERVE_BIN:-$(dirname "$0")/../../mlx-serve/zig-out/bin/mlx-serve}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/projects/agents/llama.cpp/build/bin/llama-server}"
LLMPROBE="${LLMPROBE:-node $HOME/projects/agents/llmprobe/bin/dist/llmprobe.mjs}"
OUT="${OUT:-$HOME/claude-tmp/gguf-prod/llmprobe}"
TAG="$(basename "$MODEL" .gguf)"
# llmprobe flags come after the model path: default is the full run plus the reasoning eval.
[ $# -eq 0 ] && set -- --eval
mkdir -p "$OUT"

wait_health() {
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || { echo "server died"; return 1; }
    sleep 1
  done
  return 1
}

run() { # name port cmd...
  local name="$1" port="$2"; shift 2
  "$@" > "$OUT/$TAG.$name.server.log" 2>&1 &
  local pid=$!
  wait_health "$port" "$pid"
  $LLMPROBE "http://127.0.0.1:$port" -m "$TAG" --no-bench --reasoning off --eval-max-tokens "${EVAL_MAX_TOKENS:-2048}" ${EXTRA[@]+"${EXTRA[@]}"} > "$OUT/$TAG.$name.txt" 2>&1 || true
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
}

EXTRA=("$@")
# ENGINES="mlx" for a native MLX checkpoint (llama.cpp can't serve it).
for engine in ${ENGINES:-mlx llamacpp}; do
  case "$engine" in
    mlx) run mlx 18201 "$BIN" --model "$MODEL" --serve --port 18201 --mlx-gguf ;;
    llamacpp) run llamacpp 18202 "$LLAMA_SERVER" -m "$MODEL" --port 18202 -ngl 99 --jinja -c 16384 ;;
  esac
done
echo "done: $OUT/$TAG.{mlx,llamacpp}.txt"
