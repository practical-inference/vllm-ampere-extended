#!/bin/bash
# capture_stock_failure.sh — one dev-slot window, both stacks.
#
# Bounces the DEV unit to a STOCK upstream vLLM serving the same model with
# fork-only features stripped (fp8 KV emu, KV offload connector, fork MTP
# schedule), captures the boot/serve outcome — exact failure traceback OR a
# working server, which then gets the same serve_matrix.py benchmark — and
# restores the fork unit. Never touches prod units.
#
# Usage: capture_stock_failure.sh <stock-venv> [port]
# Env: MODEL_DIR, GPU, PORT (defaults mirror serve-leoncca-tp1.sh exp2 unit)
set -uo pipefail

STOCK_VENV=${1:?usage: capture_stock_failure.sh <stock-venv> [port]}
PORT=${2:-8017}
UNIT=vllm-leoncca-exp2
MODEL_DIR=${MODEL_DIR:-/data/models/cyankiwi}
GPU=${GPU:-GPU-cb99124e-dd81-01d3-0101-82a664c7cbd5}
GPU_IDX=${GPU_IDX:-1}
OUT_DIR=${OUT_DIR:-/tmp/opencode/stock-capture}
# Cold weight load of this checkpoint is ~10 min (39 shards); budget the
# health poll accordingly (2026-10-26: a 300s poll expired mid-load and
# misreported a healthy boot as a failure).
BOOT_POLL_SECS=${BOOT_POLL_SECS:-900}
mkdir -p "$OUT_DIR"

log() { echo "$(date '+%F %T') [stock-capture] $*"; }

log "fleet state BEFORE"
/data/models/bin/fleet_state.sh | head -20

log "stopping $UNIT (dev slot)"
systemctl --user stop "$UNIT"

# Minimal honest stock command: model + standard serving flags only. Fork-only
# features (fp8 KV emu, KV offload connector, MTP schedule, mamba prefix
# checkpoint) are stripped — a flag-parse failure would be a false "stock
# can't run it". PLE CPU offload IS an upstream feature this model needs to
# fit 96GB (verified upstream 0.29, serve-leoncca-tp1.sh header).
log "launching stock vLLM ($STOCK_VENV) on port $PORT"
export VLLM_PLE_CPU_OFFLOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
numactl --cpunodebind=0 --membind=0 \
  env CUDA_VISIBLE_DEVICES="$GPU_IDX" \
  "$STOCK_VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9 \
  --disable-uvicorn-access-log > "$OUT_DIR/stock-serve.log" 2>&1 &
SERVE_PID=$!

BOOTED=0
for i in $(seq 1 $((BOOT_POLL_SECS / 5))); do
  sleep 5
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then BOOTED=1; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then break; fi
done

if [ "$BOOTED" = 1 ]; then
  log "stock vLLM BOOTS — running the same benchmark matrix"
  "$STOCK_VENV/bin/python" "$(dirname "$0")/serve_matrix.py" \
    --base-url "http://localhost:$PORT/v1" --model qwen3.8-flash-next \
    --out "$OUT_DIR/bench-stock.json" > "$OUT_DIR/bench-stock.log" 2>&1 \
    && log "stock benchmark done" || log "stock benchmark FAILED (see bench-stock.log)"
else
  log "stock vLLM DID NOT BOOT — capturing failure"
  cp "$OUT_DIR/stock-serve.log" "$OUT_DIR/stock-failure.log"
  grep -m1 -B2 -A15 -E "Error|Traceback|NotImplemented|assert" "$OUT_DIR/stock-serve.log" \
    > "$OUT_DIR/stock-failure-excerpt.txt" || true
fi

log "stopping stock serve (own PID $SERVE_PID)"
kill "$SERVE_PID" 2>/dev/null
for i in $(seq 1 12); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 5; done
kill -9 "$SERVE_PID" 2>/dev/null || true
# EngineCore children survive a parent kill (2026-10-26: orphaned EngineCore
# held 81GB, the fork unit OOM'd restarting into it). Kill descendants by
# exact PID after verifying they belong to this stock serve.
for pid in $(pgrep -f "$STOCK_VENV" 2>/dev/null); do
  [ "$pid" = "$$" ] && continue
  grep -qs "$STOCK_VENV" "/proc/$pid/environ" "/proc/$pid/cmdline" 2>/dev/null || continue
  log "killing surviving stock child $pid"
  kill -9 "$pid" 2>/dev/null || true
done

# Wait for GPU memory to actually drain before restarting the unit —
# restarting into a draining GPU OOMs the weight load (same incident).
for i in $(seq 1 36); do
  USED=$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${USED:-999999}" -lt 1000 ] && break
  sleep 5
done
log "GPU${GPU_IDX} drained to ${USED:-?} MiB"

log "restoring $UNIT (fork wheel)"
systemctl --user start "$UNIT"
for i in $(seq 1 90); do
  sleep 5
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
done
log "fleet state AFTER"
/data/models/bin/fleet_state.sh | head -20
log "done — outcome in $OUT_DIR"
