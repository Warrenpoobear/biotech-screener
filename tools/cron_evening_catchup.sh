#!/usr/bin/env bash
# cron_evening_catchup.sh — catch-up for evening observability agents
#
# WSL2 sleep/wake can leave the 17:00–18:55 ET observability window
# uncovered. This script checks each evening agent/tool against its
# today-log (or today-artifact) and runs any that are missing AND whose
# scheduled time has already passed.
#
# Idempotent: safe to re-run. Skips weekends.
#
# Triggers:
#   - 22:00 ET weekdays (late-evening safety net while WSL still awake)
#   - @reboot (catches reboot-based wakes)

set -uo pipefail

REPO="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG="$REPO/logs/evening_catchup.log"
AGENTS_LOG="$REPO/logs/agents.log"
AGENTS_DIRECT_DIR="$REPO/logs/agents_direct"

cd "$REPO" || exit 1

TODAY=$(TZ=America/Detroit date +%Y-%m-%d)
TODAY_COMPACT=$(TZ=America/Detroit date +%Y%m%d)
NOW_HM=$(TZ=America/Detroit date +%H%M)
DOW=$(TZ=America/Detroit date +%u)

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] evening_catchup: $*" >> "$LOG"; }

if [ "$DOW" -ge 6 ]; then
    log "Weekend (DOW=$DOW) — skipping"
    exit 0
fi

log "=== Starting evening catch-up for $TODAY (now=$NOW_HM ET) ==="

# Load .env for ANTHROPIC_API_KEY etc.
set -a
# shellcheck disable=SC1091
source "$REPO/.env" 2>/dev/null || true
set +a

# True if an agent already has a logs/agents_direct/<agent>_<YYYYMMDD>*.json today
agent_ran_today() {
    local agent=$1
    ls "$AGENTS_DIRECT_DIR/${agent}_${TODAY_COMPACT}"*.json >/dev/null 2>&1
}

# True if a tool's log file already contains today's date
tool_log_has_today() {
    local logfile=$1
    [ -f "$logfile" ] && grep -q "$TODAY" "$logfile" 2>/dev/null
}

# True if a file exists
file_exists() {
    [ -f "$1" ]
}

# Run an agent via tools/run_agent_direct.py
run_agent() {
    local agent=$1 sched=$2
    if [ "$NOW_HM" -lt "$sched" ]; then
        log "defer $agent — scheduled $sched not yet past"
        return
    fi
    if agent_ran_today "$agent"; then
        log "skip  $agent — already ran today"
        return
    fi
    log "RUN   $agent (scheduled $sched)"
    $PYTHON "$REPO/tools/run_agent_direct.py" --agent "$agent" --message "HEARTBEAT" \
        >> "$AGENTS_LOG" 2>&1 && log "done  $agent" || log "FAIL  $agent (exit $?)"
}

# Run a direct-tool agent, guarded by a check_fn returning 0 if already done
run_tool() {
    local name=$1 sched=$2 logfile=$3 check_fn=$4 cmd=$5
    if [ "$NOW_HM" -lt "$sched" ]; then
        log "defer $name — scheduled $sched not yet past"
        return
    fi
    if $check_fn; then
        log "skip  $name — already ran today"
        return
    fi
    log "RUN   $name (scheduled $sched)"
    # shellcheck disable=SC2086
    eval "$cmd" >> "$logfile" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        log "done  $name"
    elif [ "$rc" -eq 1 ]; then
        # Exit 1 is the idiomatic "alerts found but ran successfully" signal
        # for heartbeat_checks / data_auditor / production_qa_check. Real
        # crashes exit >1.
        log "alert $name (exit 1 — see $logfile)"
    else
        log "FAIL  $name (exit $rc)"
    fi
}

# ---- Agents via run_agent_direct.py ----
run_agent ops                     1700
run_agent sentinel                1715
run_agent crt_resolution_watcher  1800
run_agent policy_shadow_watch     1805
run_agent catalyst_delta          1820
run_agent price_action_watch      1830
run_agent postmortem              1835

# ---- Direct-tool agents ----

# heartbeat_checks (17:30) — log: logs/heartbeat_checks.log
check_heartbeat() { tool_log_has_today "$REPO/logs/heartbeat_checks.log"; }
run_tool heartbeat_checks 1730 "$REPO/logs/heartbeat_checks.log" check_heartbeat \
    "$PYTHON $REPO/tools/agent_heartbeat_checks.py"

# data_auditor (18:00) — log: logs/data_auditor.log
check_data_auditor() { tool_log_has_today "$REPO/logs/data_auditor.log"; }
run_tool data_auditor 1800 "$REPO/logs/data_auditor.log" check_data_auditor \
    "$PYTHON $REPO/agents/data_auditor/run_audit.py --daily-only"

# event_feedback (18:02) — log: logs/event_feedback.log
check_event_feedback() { tool_log_has_today "$REPO/logs/event_feedback.log"; }
run_tool event_feedback 1802 "$REPO/logs/event_feedback.log" check_event_feedback \
    "$PYTHON $REPO/tools/build_event_feedback.py --as-of-date $TODAY"

# production_qa_check (18:55) — artifact: artifacts/production_qa/<TODAY>_report.json
check_production_qa() { file_exists "$REPO/artifacts/production_qa/${TODAY}_report.json"; }
run_tool production_qa_check 1855 "$REPO/logs/production_qa.log" check_production_qa \
    "$PYTHON $REPO/tools/production_qa_check.py --as-of-date $TODAY"

# ---- Friday-only direct-tool agents (calibration cadence) ----
# Crontab schedules: build_calibration_evidence.py at Fri 19:00,
# build_event_feedback_metrics.py at Fri 19:30. WSL2 sleep-cliff at ~17:30
# means these are sleep-roulette without an explicit catchup gate.
if [ "$DOW" -eq 5 ]; then

    # calibration_evidence (Fri 19:00) — log: logs/calibration_evidence.log
    check_calibration_evidence() { tool_log_has_today "$REPO/logs/calibration_evidence.log"; }
    run_tool calibration_evidence 1900 "$REPO/logs/calibration_evidence.log" check_calibration_evidence \
        "$PYTHON $REPO/tools/build_calibration_evidence.py"

    # event_feedback_metrics (Fri 19:30) — log: logs/event_feedback_metrics.log
    check_event_feedback_metrics() { tool_log_has_today "$REPO/logs/event_feedback_metrics.log"; }
    run_tool event_feedback_metrics 1930 "$REPO/logs/event_feedback_metrics.log" check_event_feedback_metrics \
        "$PYTHON $REPO/tools/build_event_feedback_metrics.py --as-of-date $TODAY"

fi

log "=== Evening catch-up complete ==="
