#!/bin/bash
# night_soak.sh <ID> — 2h stability soak: 60s health+kv samples, batch storm
# probe at ~t+100min, final summary. Assumes boot+repro already done.
set -u
ID="$1"; PORT="${2:-8080}"
export XDG_RUNTIME_DIR=/run/user/1001
BASE="http://127.0.0.1:${PORT}"
DIR=/tmp/opencode
SAMPLES="$DIR/night-${ID}-soak.csv"
echo "ts,health,kv" > "$SAMPLES"
START=$(date +%s)
for i in $(seq 1 120); do
  h=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || true)
  k=$(curl -s --max-time 5 "$BASE/metrics" 2>/dev/null | grep -a kv_cache_usage_perc | grep -av '^#' | awk '{print $NF}' | tail -1)
  echo "$(date -Is),$h,${k:-?}" >> "$SAMPLES"
  if [ $(( $(date +%s) - START )) -ge 6000 ] && [ -z "${RAN:-}" ]; then
    RAN=1
    /home/columbo/source/vllm-nightly/.venv/bin/python "$DIR/night_e1b_ramp.py" "$BASE" 88 > "$DIR/night-${ID}-soak-ramp.jsonl" 2>&1
    echo "ramp-rc=$?" >> "$DIR/night-${ID}-soak-ramp.jsonl"
  fi
  sleep 55
done
echo "=== SOAK SUMMARY ==="
total=$(($(wc -l < "$SAMPLES") - 1))
dead=$(awk -F, 'NR>1 && $2!="200"' "$SAMPLES" | wc -l)
first_ts=$(sed -n '2p' "$SAMPLES" | cut -d, -f1)
last_ts=$(tail -1 "$SAMPLES" | cut -d, -f1)
echo "samples=$total dead_samples=$dead"
awk -F, 'NR>1{ts=$1; print}' "$SAMPLES" | tail -1 | head -c 80
echo
echo "soak $first_ts -> $last_ts"
grep -c "^" "$DIR/night-${ID}-soak-ramp.jsonl" 2>/dev/null && grep -E "FINAL|rc=" "$DIR/night-${ID}-soak-ramp.jsonl" | tail -2
echo "SOAK-DONE $(date -Is)"
