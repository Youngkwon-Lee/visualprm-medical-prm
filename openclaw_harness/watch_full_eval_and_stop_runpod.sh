#!/usr/bin/env bash
set -euo pipefail

RESULT_FILE="${RESULT_FILE:-/Users/youngkwon/projects/visualprm_openclaw_harness/results_native_openclaw/full_pathvqa-vision-direct_gateway_0_1147_20260513.jsonl}"
TARGET_LINES="${TARGET_LINES:-1147}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-fp8nzcdkersqy5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUNPODCTL_BIN="${RUNPODCTL_BIN:-$(command -v runpodctl)}"
PID_FILE="${PID_FILE:-/Users/youngkwon/projects/visualprm_openclaw_harness/.codex_logs/stop_runpod_when_full_done.pid}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

current_lines() {
  if [ -f "$RESULT_FILE" ]; then
    wc -l < "$RESULT_FILE" | tr -d ' '
  else
    echo 0
  fi
}

cleanup() {
  rm -f "$PID_FILE"
}

trap cleanup EXIT
mkdir -p "$(dirname "$PID_FILE")"
echo "$$" > "$PID_FILE"

echo "[$(timestamp)] watcher started"
echo "[$(timestamp)] result_file=$RESULT_FILE target_lines=$TARGET_LINES pod_id=$RUNPOD_POD_ID poll_seconds=$POLL_SECONDS"

if [ ! -x "$RUNPODCTL_BIN" ]; then
  echo "[$(timestamp)] runpodctl not found: $RUNPODCTL_BIN"
  exit 1
fi

while true; do
  lines="$(current_lines)"
  echo "[$(timestamp)] progress lines=$lines/$TARGET_LINES"

  if [ "$lines" -ge "$TARGET_LINES" ]; then
    echo "[$(timestamp)] target reached, stopping RunPod pod $RUNPOD_POD_ID"
    until "$RUNPODCTL_BIN" pod stop "$RUNPOD_POD_ID"; do
      echo "[$(timestamp)] runpod stop failed, retrying in $POLL_SECONDS seconds"
      sleep "$POLL_SECONDS"
    done
    echo "[$(timestamp)] RunPod stop command succeeded"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
