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

cd "$REPO" || exit 1

TODAY=$(TZ=America/Detroit date +%Y-%m-%d)
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

# True if a tool's log file already contains today's date
tool_log_has_today() {
    local logfile=$1
    [ -f "$logfile" ] && grep -q "$TODAY" "$logfile" 2>/dev/null
}

# True if a file exists
file_exists() {
    [ -f "$1" ]
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

# ---- Direct-tool agents (deterministic — Class F LLM HEARTBEAT retired) ----

# ops_digest (17:00) — artifact: artifacts/ops_digest/<TODAY>_digest.json
check_ops_digest() { file_exists "$REPO/artifacts/ops_digest/${TODAY}_digest.json"; }
run_tool ops_digest 1700 "$REPO/logs/ops_digest.log" check_ops_digest \
    "$PYTHON $REPO/tools/build_ops_digest.py --as-of-date $TODAY"

# ruleset_sentinel (17:15) — sidecar: data/snapshots/<TODAY>/ruleset_health.json
check_ruleset_sentinel() { file_exists "$REPO/data/snapshots/${TODAY}/ruleset_health.json"; }
run_tool ruleset_sentinel 1715 "$REPO/logs/ruleset_health.log" check_ruleset_sentinel \
    "$PYTHON $REPO/tools/ruleset_health_monitor.py --as-of-date $TODAY"

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

# hermes_knowledge_layer (18:00) — artifact: artifacts/ops/knowledge_layer/latest_state.json
check_hermes_knowledge() {
    local f="$REPO/artifacts/ops/knowledge_layer/latest_state.json"
    [ -f "$f" ] && grep -q "\"as_of_date\": \"$TODAY\"" "$f" 2>/dev/null
}
run_tool hermes_knowledge 1800 "$REPO/logs/hermes_knowledge.log" check_hermes_knowledge \
    "$PYTHON $REPO/tools/build_hermes_knowledge_layer.py"

# policy_shadow (18:05) — artifact: artifacts/policy_shadow/tier_weighted/<TODAY>_comparison.json
# Mirrors the daily cron 5 18 * * 1-5 build_policy_shadow_compare.py --as-of-date $(date +%Y-%m-%d).
# Replaces the previous `run_agent policy_shadow_watch 1805` LLM HEARTBEAT call.
check_policy_shadow() { file_exists "$REPO/artifacts/policy_shadow/tier_weighted/${TODAY}_comparison.json"; }
run_tool policy_shadow 1805 "$REPO/logs/agents_direct_cron.log" check_policy_shadow \
    "$PYTHON $REPO/tools/build_policy_shadow_compare.py --as-of-date $TODAY"

# hermes_contradiction_detector (18:05) — log: logs/hermes_contradiction.log
# Runs after knowledge_layer; reads latest_state.json from that build.
check_hermes_contradiction() { tool_log_has_today "$REPO/logs/hermes_contradiction.log"; }
run_tool hermes_contradiction 1805 "$REPO/logs/hermes_contradiction.log" check_hermes_contradiction \
    "$PYTHON $REPO/agents/hermes-contradiction-detector/run_job.py --from-build"

# crt_resolution (18:00) — artifact: output/catalyst_ev/crt_options_join.json
check_crt_resolution() { file_exists "$REPO/output/catalyst_ev/crt_options_join.json"; }
run_tool crt_resolution 1800 "$REPO/logs/crt_resolution.log" check_crt_resolution \
    "$PYTHON $REPO/tools/catalyst_resolution_tracker.py --as-of-date $TODAY && $PYTHON $REPO/scripts/research/build_crt_options_join.py"

# catalyst_delta (18:20) — artifact: artifacts/catalyst_delta/<TODAY>_delta.json
check_catalyst_delta() { file_exists "$REPO/artifacts/catalyst_delta/${TODAY}_delta.json"; }
run_tool catalyst_delta 1820 "$REPO/logs/agents_direct_cron.log" check_catalyst_delta \
    "$PYTHON $REPO/tools/build_catalyst_delta.py --as-of-date $TODAY"

# price_action_watch (18:30) — artifact: artifacts/price_action_watch/<TODAY>_watch.json
check_price_action_watch() { file_exists "$REPO/artifacts/price_action_watch/${TODAY}_watch.json"; }
run_tool price_action_watch 1830 "$REPO/logs/price_action_watch.log" check_price_action_watch \
    "$PYTHON $REPO/tools/build_price_action_watch.py --as-of-date $TODAY"

# postmortem (18:35) — memory: agents/postmortem/memory/<TODAY>.md
check_postmortem() {
    file_exists "$REPO/agents/postmortem/memory/${TODAY}.md" || \
        tool_log_has_today "$REPO/logs/postmortem.log"
}
run_tool postmortem 1835 "$REPO/logs/postmortem.log" check_postmortem \
    "$PYTHON $REPO/agents/postmortem/scripts/run_postmortem.py"

# production_qa_check (18:55) — artifact: artifacts/production_qa/<TODAY>_report.json
check_production_qa() { file_exists "$REPO/artifacts/production_qa/${TODAY}_report.json"; }
run_tool production_qa_check 1855 "$REPO/logs/production_qa.log" check_production_qa \
    "$PYTHON $REPO/tools/production_qa_check.py --as-of-date $TODAY"

# herald_health (14:35) — artifact: artifacts/herald/health_check_<TODAY>.json
check_herald_health() { file_exists "$REPO/artifacts/herald/health_check_${TODAY}.json"; }
run_tool herald_health 1435 "$REPO/logs/herald_health.log" check_herald_health \
    "$PYTHON $REPO/tools/herald_health_check.py --as-of-date $TODAY"

# ops_supervisor (19:00) — artifact: artifacts/ops_supervisor/<TODAY>_supervisor.json
check_ops_supervisor() { file_exists "$REPO/artifacts/ops_supervisor/${TODAY}_supervisor.json"; }
run_tool ops_supervisor 1900 "$REPO/logs/ops_supervisor.log" check_ops_supervisor \
    "$PYTHON $REPO/agents/ops_supervisor/supervisor.py --as-of $TODAY"

# supervisor_sentinel (19:15) — artifact: artifacts/ops_supervisor/<TODAY>_sentinel.json
check_supervisor_sentinel() { file_exists "$REPO/artifacts/ops_supervisor/${TODAY}_sentinel.json"; }
run_tool supervisor_sentinel 1915 "$REPO/logs/ops_supervisor.log" check_supervisor_sentinel \
    "$PYTHON $REPO/tools/agent_supervisor_sentinel.py --as-of $TODAY"

log "=== Evening catch-up complete ==="
