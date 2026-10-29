#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Half-hour field test for vllm-ampere-extended: serve a model with the fork
# wheel, run the fixed benchmark matrix (recipes/bench/serve_matrix.py), and
# print a report ready to paste into a GitHub issue.
#
# Usage: recipes/bench/field_test.sh <model-path> [port] [extra vllm serve args]

set -euo pipefail

MODEL=${1:?usage: field_test.sh <model-path> [port] [extra vllm serve args]}
PORT=${2:-8321}
shift 2 2>/dev/null || true

BASE_URL="http://localhost:$PORT"
SERVER_LOG=$(mktemp /tmp/field_test_serve.XXXXXX.log)
RESULTS=$(mktemp /tmp/field_test_results.XXXXXX.json)
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

WHEEL=$(python -c 'import vllm; print(vllm.__version__)')
GPUS=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort -u | tr '\n' ';')

echo "== vllm-ampere-extended field test =="
echo "model:  $MODEL"
echo "wheel:  vllm $WHEEL"
echo "GPUs:   $GPUS"
echo "port:   $PORT  (server log: $SERVER_LOG)"

vllm serve "$MODEL" --served-model-name field-test --port "$PORT" "$@" \
  >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "waiting for /health (first model load can take many minutes)..."
for _ in $(seq 1 120); do
  if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup — last 30 log lines:"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  sleep 10
done
if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
  echo "server not healthy after 20 min — last 30 log lines:"
  tail -30 "$SERVER_LOG"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
python "$SCRIPT_DIR/serve_matrix.py" \
  --base-url "$BASE_URL/v1" --model field-test --out "$RESULTS" >/dev/null

echo
echo "== paste from here into an issue =="
echo "https://github.com/practical-inference/vllm-ampere-extended/issues"
echo
echo "**Field test report**"
echo
echo "- wheel: vllm $WHEEL"
echo "- GPUs: $GPUS"
echo "- model: \`$MODEL\`"
echo "- serve args: \`vllm serve $MODEL --served-model-name field-test --port $PORT $*\`"
echo
echo '```json'
cat "$RESULTS"
echo '```'
echo
echo "(server log: $SERVER_LOG — attach it if the numbers look off)"
