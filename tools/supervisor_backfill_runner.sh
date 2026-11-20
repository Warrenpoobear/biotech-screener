#!/bin/bash
# supervisor_backfill_runner.sh — runs supervisor + sentinel for missed days, then today.
#
# Detects missing supervisor artifacts since last successful run and backfills them
# before running the current day's triage.
#
# Called by cron at 20:30 ET (supervisor) and 20:40 ET (sentinel).
# Idempotent: safe to run multiple times.

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
SUPERVISOR_DIR="$REPO/artifacts/ops_supervisor"
SUPERVISOR_SCRIPT="$REPO/agents/ops_supervisor/supervisor.py"
SENTINEL_SCRIPT="$REPO/tools/agent_supervisor_sentinel.py"

# Ensure output dir exists
mkdir -p "$SUPERVISOR_DIR"

# Helper: run supervisor for a specific date
run_supervisor_for_date() {
    local date="$1"
    local json_path="$SUPERVISOR_DIR/${date}_supervisor.json"

    # Skip if artifact already exists (idempotent)
    if [ -f "$json_path" ]; then
        echo "[backfill] $date supervisor already exists, skipping"
        return 0
    fi

    echo "[backfill] Running supervisor for $date..."
    /usr/bin/python3 "$SUPERVISOR_SCRIPT" --as-of "$date" \
        >> "$REPO/logs/ops_supervisor.log" 2>&1

    # Check success
    if [ -f "$json_path" ]; then
        echo "[backfill] $date supervisor OK"
        return 0
    else
        echo "[backfill] $date supervisor FAILED (artifact missing)" >&2
        return 1
    fi
}

# Helper: run sentinel for a specific date
run_sentinel_for_date() {
    local date="$1"
    local json_path="$SUPERVISOR_DIR/${date}_sentinel.json"

    # Skip if artifact already exists (idempotent)
    if [ -f "$json_path" ]; then
        echo "[backfill] $date sentinel already exists, skipping"
        return 0
    fi

    echo "[backfill] Running sentinel for $date..."
    /usr/bin/python3 "$SENTINEL_SCRIPT" --as-of "$date" \
        >> "$REPO/logs/ops_supervisor.log" 2>&1

    # Check success
    if [ -f "$json_path" ]; then
        echo "[backfill] $date sentinel OK"
        return 0
    else
        echo "[backfill] $date sentinel FAILED (artifact missing)" >&2
        return 1
    fi
}

# Helper: detect last successful supervisor run date
get_last_supervisor_date() {
    # Find the most recent supervisor JSON that exists
    local latest=$(ls -1 "$SUPERVISOR_DIR"/*_supervisor.json 2>/dev/null | tail -1)
    if [ -n "$latest" ]; then
        basename "$latest" | sed 's/_supervisor.json//'
    else
        echo ""
    fi
}

# Helper: check if date is weekday (1=Mon, 5=Fri; 6-7=weekend)
is_weekday() {
    local date="$1"
    local dow=$(date -d "$date" +%u 2>/dev/null || echo "0")
    [ "$dow" -ge 1 ] && [ "$dow" -le 5 ]
}

# Main backfill logic
echo "[backfill] Starting supervisor/sentinel backfill check..."

LAST_DATE=$(get_last_supervisor_date)
echo "[backfill] Last supervisor artifact: $LAST_DATE"

# If no artifacts at all, start from 05-09 (first missed weekday)
if [ -z "$LAST_DATE" ]; then
    START_DATE="2026-05-09"
    echo "[backfill] No supervisor artifacts found; assuming gap from 05-08 shutdown"
else
    # Calculate next date after LAST_DATE
    START_DATE=$(date -d "$LAST_DATE + 1 day" +%Y-%m-%d)
fi

TODAY=$(date +%Y-%m-%d)
echo "[backfill] Backfill range: $START_DATE → $TODAY"

# Generate list of missing weekday dates
MISSING_DATES=()
CURRENT_DATE="$START_DATE"
# Strictly-before comparison (ISO dates sort lexicographically). The loop used
# to be `!= "$TODAY"`, which never terminates when START_DATE is already past
# TODAY — i.e. whenever today's supervisor artifact exists before this runs
# (START_DATE = LAST_DATE + 1 day = tomorrow). That spun at ~27% CPU forking
# `date` from 2026-11-10 19:19 until it was killed 2026-11-11 ~11:40.
while [[ "$CURRENT_DATE" < "$TODAY" ]]; do
    if is_weekday "$CURRENT_DATE"; then
        MISSING_DATES+=("$CURRENT_DATE")
    fi
    CURRENT_DATE=$(date -d "$CURRENT_DATE + 1 day" +%Y-%m-%d)
done

if [ ${#MISSING_DATES[@]} -eq 0 ]; then
    echo "[backfill] No missing weekdays detected"
else
    echo "[backfill] Found ${#MISSING_DATES[@]} missing weekday(s): ${MISSING_DATES[*]}"

    # Backfill supervisor + sentinel for each missing date
    for date in "${MISSING_DATES[@]}"; do
        run_supervisor_for_date "$date" || echo "[backfill] WARNING: supervisor for $date may have failed"
        run_sentinel_for_date "$date" || echo "[backfill] WARNING: sentinel for $date may have failed"
    done

    echo "[backfill] Backfill complete"
fi

# Always run supervisor + sentinel for TODAY (normal flow)
echo "[backfill] Running supervisor/sentinel for today ($TODAY)..."
run_supervisor_for_date "$TODAY" || true
run_sentinel_for_date "$TODAY" || true

echo "[backfill] Done"
