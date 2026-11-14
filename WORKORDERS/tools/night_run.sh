#!/bin/bash
# night_run.sh <ID> [port] — wait health, run storm repro, dump log, analyze.
set -u
ID="$1"; PORT="${2:-8080}"
export XDG_RUNTIME_DIR=/run/user/1001
BASE="http://127.0.0.1:${PORT}"
DIR=/tmp/opencode
PY=/home/columbo/source/vllm-nightly/.venv/bin/python
REPRO=/tmp/opencode/oom-snap-lane/WORKORDERS/tools/glm_storm_repro.py
ANALYZER=/tmp/opencode/oom-snap-lane/WORKORDERS/tools/glm_storm_analyzer.py
OUT="$DIR/night-${ID}-repro.jsonl"
ERR="$DIR/night-${ID}-repro.err"
SLOG="$DIR/night-${ID}-service.log"
BOOT_T=$(date +"%m-%d %T")
echo "[$ID] boot marker $BOOT_T; waiting health $BASE ..."
for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$BASE/health" || true)
  [ "$code" = "200" ] && break
  sleep 5
done
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$BASE/health" || true)
if [ "$code" != "200" ]; then echo "[$ID] FATAL: health never came up (last=$code)"; exit 1; fi
echo "[$ID] healthy at $(date +%T); running repro"
START_TS=$(date -Iseconds)
"$PY" "$REPRO" --base-url "$BASE" > "$OUT" 2> "$ERR"
rc=$?
echo "[$ID] repro rc=$rc done $(date +%T)"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$BASE/health" || true)
echo "[$ID] post-repro health=$code"
tail -c 120000000 /data/models/vllm_glm53_flash_service.log > "$SLOG"
echo "=== CRASH SIGNATURES ==="
grep -c -E "Cannot get 2 free blocks|illegal memory access|Traceback" "$SLOG" || true
echo "=== ANALYZER (since boot) ==="
"$PY" "$ANALYZER" "$SLOG" --since "$BOOT_T"
echo "=== CANARY/GAP LINES ==="
grep -E '"phase": "canary' "$OUT" | tail -5
grep -E 'gaps' "$OUT"
grep -E 'storm' "$OUT" | tail -12
echo "[$ID] RUN-DONE $(date +%T)"
