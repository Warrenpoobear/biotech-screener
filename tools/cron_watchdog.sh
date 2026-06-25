#!/usr/bin/env bash
# cron_watchdog.sh — detect and recover missed cron jobs on WSL2
#
# WSL2 sleep/wake corrupts cron's internal timer. This watchdog:
#   1. Checks if today's production ran (by looking for cron.log entry)
#   2. If not, runs data_refresh + daily_production manually
#   3. Logs to logs/watchdog.log
#
# Run this from Windows Task Scheduler or a @reboot cron entry.
# Usage: bash tools/cron_watchdog.sh

set -uo pipefail

REPO="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG="$REPO/logs/watchdog.log"
CRON_LOG="$REPO/logs/cron.log"

cd "$REPO" || exit 1

TODAY=$(TZ=America/Detroit date +%Y-%m-%d)
DOW=$(TZ=America/Detroit date +%u)  # 1=Mon, 7=Sun

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] watchdog: $*" >> "$LOG"; }

# Skip weekends
if [ "$DOW" -ge 6 ]; then
    log "Weekend ($DOW) — skipping"
    exit 0
fi

# Check if today's production produced its snapshot. The "Starting" line in
# cron.log is written by the wrapper before any work happens, so it can't
# distinguish a successful run from one killed mid-pipeline (e.g. WSL2
# reaping the parent during a subprocess call — observed 2026-04-27 and
# 2026-04-28). data/snapshots/$TODAY/rankings.csv exists once the pipeline
# reaches promotion; this is the same gate the wrapper uses for its own
# rank-change monitor (see cron_daily_production.sh).
if [ -f "$REPO/data/snapshots/$TODAY/rankings.csv" ]; then
    PROD_RAN=true
else
    PROD_RAN=false
fi

if [ "$PROD_RAN" = false ]; then
    # If daily_production is currently mid-flight (lock held by an active PID),
    # skip the recovery — it's already running. The wrapper has its own lock,
    # so re-invoking would just emit "SKIP", but we'd waste 1.5 min on
    # data_refresh first. Avoiding that by short-circuiting here.
    DAILY_LOCK="$REPO/logs/.daily_production.lock"
    if [ -f "$DAILY_LOCK" ]; then
        DAILY_PID=$(cat "$DAILY_LOCK" 2>/dev/null || echo "")
        if [ -n "$DAILY_PID" ] && kill -0 "$DAILY_PID" 2>/dev/null; then
            log "Production active (PID $DAILY_PID) — skipping recovery to avoid wasteful data_refresh"
            PROD_RAN=skip
        fi
    fi
fi

if [ "$PROD_RAN" = false ]; then
    log "MISSED: No production run for $TODAY — triggering manual recovery"

    $PYTHON "$REPO/tools/notify_cron_missed.py" \
        --date "$TODAY" \
        --reason "production_snapshot_missing" \
        --recovery-triggered >> "$LOG" 2>&1 \
        || log "notify_cron_missed failed (exit $?)"

    # Try to restart cron (may fail without sudo)
    service cron restart >> "$LOG" 2>&1 || log "cron restart failed (no sudo) — running jobs directly"

    # Run data refresh first
    log "Running data_refresh..."
    bash "$REPO/tools/cron_data_refresh.sh" all >> "$REPO/logs/data_refresh.log" 2>&1
    log "data_refresh done (exit $?)"

    # Run daily production
    log "Running daily_production..."
    bash "$REPO/tools/cron_daily_production.sh" >> "$CRON_LOG" 2>&1
    log "daily_production done (exit $?)"
elif [ "$PROD_RAN" = "skip" ]; then
    : # already logged above
else
    log "Production already ran for $TODAY — skipping production recovery"
fi

# Post-snapshot task supervisor (Spec 069 phase 1).
# When the snapshot promoted but daily_production was reaped before reaching
# Step 5n (AACT) / 5l.5 (Herald), those tasks never produced their artifacts.
# Supervisor re-runs them idempotently. Gate fires when:
#   - snapshot's rankings.csv exists (PROD_RAN=true OR we just re-ran above)
#   - complete marker missing for $TODAY
# Each task has its own done predicate, so a no-op pass is cheap.
SUPERVISOR_MARKER="$REPO/artifacts/post_snapshot_done/$TODAY.complete"
if [ -f "$REPO/data/snapshots/$TODAY/rankings.csv" ] && [ ! -f "$SUPERVISOR_MARKER" ]; then
    log "Snapshot present, post-snapshot tasks incomplete — running supervisor"
    bash "$REPO/tools/run_post_snapshot_supervisor.sh" "$TODAY" >> "$LOG" 2>&1
    log "post_snapshot_supervisor done (exit $?)"
fi

# Phase-2 agent recovery runs UNCONDITIONALLY.
# WSL sleep between 17:30 and 19:00 ET can miss evening builder slots even
# when morning production succeeded. Detection uses artifact presence for
# $YESTERDAY (not agents_direct JSONL — Class F LLM path retired in #399).
#
# Idempotency: artifacts/phase2_recovery_done/$YESTERDAY.complete
YESTERDAY=$(TZ=America/Detroit date -d "yesterday" +%Y-%m-%d)
PHASE2_MARKER_DIR="$REPO/artifacts/phase2_recovery_done"
PHASE2_MARKER="$PHASE2_MARKER_DIR/$YESTERDAY.complete"

mkdir -p "$PHASE2_MARKER_DIR"

phase2_artifact_present() {
    [ -f "$REPO/$1" ]
}

recover_phase2_tool() {
    local name=$1 artifact=$2 cmd=$3
    if phase2_artifact_present "$artifact"; then
        return 0
    fi
    log "Recovering phase-2 tool: $name (missing $artifact)"
    # shellcheck disable=SC2086
    eval "$cmd" >> "$REPO/logs/phase2_recovery.log" 2>&1 || log "Phase-2 $name recovery failed (exit $?)"
}

if [ -f "$PHASE2_MARKER" ]; then
    log "Phase-2 recovery already completed for $YESTERDAY — skipping"
else
    missed=""
    phase2_artifact_present "artifacts/price_action_watch/${YESTERDAY}_watch.json" || missed="$missed price_action_watch"
    phase2_artifact_present "artifacts/options_watch/${YESTERDAY}_watch.json" || missed="$missed options_watch"
    phase2_artifact_present "agents/postmortem/memory/${YESTERDAY}.md" || missed="$missed postmortem"
    phase2_artifact_present "agents/review_queue_steward/memory/${YESTERDAY}.md" || missed="$missed review_queue_steward"

    if [ -n "$missed" ]; then
        log "MISSED phase-2 artifacts for $YESTERDAY:$missed — triggering deterministic recovery"
        recover_phase2_tool price_action_watch \
            "artifacts/price_action_watch/${YESTERDAY}_watch.json" \
            "$PYTHON $REPO/tools/build_price_action_watch.py --as-of-date $YESTERDAY"
        recover_phase2_tool options_watch \
            "artifacts/options_watch/${YESTERDAY}_watch.json" \
            "$PYTHON $REPO/tools/build_options_watch.py --as-of-date $YESTERDAY"
        recover_phase2_tool postmortem \
            "agents/postmortem/memory/${YESTERDAY}.md" \
            "$PYTHON $REPO/agents/postmortem/scripts/run_postmortem.py --as-of-date $YESTERDAY"
        recover_phase2_tool review_queue_steward \
            "agents/review_queue_steward/memory/${YESTERDAY}.md" \
            "$PYTHON $REPO/tools/run_review_queue_steward.py --as-of-date $YESTERDAY"
        log "Phase-2 deterministic recovery complete"
        {
            echo "phase2_recovery_complete: $YESTERDAY"
            echo "recovered_tools:$missed"
            echo "recovery_timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"
        } > "$PHASE2_MARKER"
    else
        log "All phase-2 artifacts present for $YESTERDAY — no recovery needed"
        {
            echo "phase2_recovery_complete: $YESTERDAY"
            echo "recovered_tools: none"
            echo "recovery_timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"
        } > "$PHASE2_MARKER"
    fi
fi

# Monitoring layer recovery — heartbeat receipt + ops_supervisor for $TODAY when
# production snapshot exists but evening observability was missed (WSL sleep).
if phase2_artifact_present "data/snapshots/${TODAY}/rankings.csv"; then
    if ! phase2_artifact_present "artifacts/heartbeat/${TODAY}_receipt.md"; then
        log "MISSED heartbeat receipt for $TODAY — recovering"
        $PYTHON "$REPO/tools/agent_heartbeat_checks.py" --date "$TODAY" \
            >> "$REPO/logs/heartbeat_checks.log" 2>&1 \
            || log "Heartbeat recovery failed (exit $?)"
    fi
    if phase2_artifact_present "artifacts/heartbeat/${TODAY}_receipt.md" \
        && ! phase2_artifact_present "artifacts/ops_supervisor/${TODAY}_supervisor.json"; then
        log "MISSED ops_supervisor for $TODAY — recovering"
        $PYTHON "$REPO/agents/ops_supervisor/supervisor.py" --as-of "$TODAY" \
            >> "$REPO/logs/ops_supervisor.log" 2>&1 \
            || log "ops_supervisor recovery failed (exit $?)"
    fi
fi

# Pre-market feed checks run on every invocation, regardless of production state.
# Each check has its own per-run marker so a missed morning slot can recover later in the day.

# Bellringer morning reminder (06:30 ET slot)
BELLRINGER_LOG="$REPO/logs/bellringer.log"
if ! grep -qE "^\[$TODAY " "$BELLRINGER_LOG" 2>/dev/null; then
    log "MISSED bellringer for $TODAY — recovering"
    bash "$REPO/tools/cron_bellringer.sh" all >> "$BELLRINGER_LOG" 2>&1 || log "Bellringer recovery failed (exit $?)"
fi

# Conference abstracts refresh (06:00 ET slot). Grep anchored to today's cache-file
# line so it doesn't false-match older dates mentioned in summary tables.
CONF_LOG="$REPO/logs/conference_refresh.log"
if ! grep -q "abstracts_$TODAY\.json" "$CONF_LOG" 2>/dev/null; then
    log "MISSED conference abstracts for $TODAY — recovering"
    source "$REPO/.env" 2>/dev/null || true
    $PYTHON "$REPO/tools/fetch_conference_abstracts_grok.py" --all >> "$CONF_LOG" 2>&1 || log "Conference refresh failed (exit $?)"
fi

# Trapops monitor recovery — check artifact, not log mtime.
# The production pipeline writes trapops_daily_summary.json into the promoted
# snapshot dir. If that file is missing, run trapops standalone to generate it.
# Using artifact presence avoids false-negative from log touched by earlier run.
TRAPOPS_LOG="$REPO/logs/trapops_cron.log"
TRAPOPS_ARTIFACT="$REPO/data/snapshots/$TODAY/trapops_daily_summary.json"
if [ ! -f "$TRAPOPS_ARTIFACT" ]; then
    log "MISSED trapops artifact for $TODAY — recovering"
    (cd "$REPO" && $PYTHON "$REPO/tools/trapops_monitor.py" --snapshot-date "$TODAY" >> "$TRAPOPS_LOG" 2>&1) \
        && log "Trapops recovery OK — artifact written" \
        || log "Trapops recovery FAILED (exit $?) — artifact may still be missing"
else
    log "Trapops artifact present for $TODAY — skipping recovery"
fi

# Evening cron recovery — prefer cron_evening_catchup.sh (deterministic, #399).
# Legacy per-job list kept as fallback when catchup log shows no activity today.
EVENING_CATCHUP_LOG="$REPO/logs/evening_catchup.log"
EVENING_MARKER_DIR="$REPO/artifacts/evening_recovery_done"
EVENING_MARKER="$EVENING_MARKER_DIR/$TODAY.complete"
mkdir -p "$EVENING_MARKER_DIR"

if [ "$PROD_RAN" = true ] && [ ! -f "$EVENING_MARKER" ]; then
    if [ -f "$EVENING_CATCHUP_LOG" ] && grep -q "$TODAY" "$EVENING_CATCHUP_LOG" 2>/dev/null; then
        log "Evening catchup already ran for $TODAY — skipping watchdog evening block"
    else
        log "Evening catchup missing for $TODAY — running cron_evening_catchup.sh"
        bash "$REPO/tools/cron_evening_catchup.sh" >> "$EVENING_CATCHUP_LOG" 2>&1 \
            || log "Evening catchup recovery failed (exit $?)"
    fi
    {
        echo "evening_recovery_complete: $TODAY"
        echo "recovery_timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    } > "$EVENING_MARKER"
    log "Evening cron recovery complete for $TODAY"
fi

log "Recovery complete for $TODAY"
