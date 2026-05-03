#!/usr/bin/env bash
# cron_weekend_calibration_catchup.sh
#
# Saturday/Sunday safety net for the Friday-19:00 calibration_evidence
# and Friday-19:30 event_feedback_metrics builds. The weekday catchup
# (cron_evening_catchup.sh) exits early on weekends; if Friday's WSL2
# host slept through both 19:00 cron and 22:00 catchup, the weekly
# evidence stays silent until next Friday — exactly the failure mode
# that produced the 14d-stale fleet finding.
#
# This script does ONE job: if last Friday's calibration logs are
# missing today's-Friday date, run the builds with --as-of-date set
# to that Friday so the artifacts land at the correct timestamp.
#
# Idempotent: re-running on Sunday after a successful Saturday run
# is a no-op (logs grep finds Friday's date and skips).
#
# Recommended cron:
#   0 9 * * 6 /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_weekend_calibration_catchup.sh
#   0 9 * * 0 /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_weekend_calibration_catchup.sh

set -uo pipefail

REPO="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG="$REPO/logs/weekend_calibration_catchup.log"
CAL_LOG="$REPO/logs/calibration_evidence.log"
EVF_LOG="$REPO/logs/event_feedback_metrics.log"

cd "$REPO" || exit 1

DOW=$(TZ=America/Detroit date +%u)
TODAY=$(TZ=America/Detroit date +%Y-%m-%d)

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] weekend_calibration_catchup: $*" >> "$LOG"; }

# Only fire on Sat (6) or Sun (7)
if [ "$DOW" -lt 6 ]; then
    log "Not a weekend (DOW=$DOW) — exiting"
    exit 0
fi

# Compute last Friday's date
if [ "$DOW" -eq 6 ]; then
    FRI_DAYS_AGO=1
else
    FRI_DAYS_AGO=2
fi
FRI=$(TZ=America/Detroit date -d "$FRI_DAYS_AGO day ago" +%Y-%m-%d)

log "=== Starting weekend calibration catch-up — today=$TODAY (DOW=$DOW), checking Friday $FRI ==="

# Load .env
set -a
# shellcheck disable=SC1091
source "$REPO/.env" 2>/dev/null || true
set +a

# Idempotency: check for Friday-dated ARTIFACTS, not log content. The build
# tools tag calibration_evidence by run-date (so a Saturday catchup writes
# <SAT>_evidence.md), but event_feedback_metrics tags by --as-of-date. We
# detect "did Friday's evidence already get produced" by:
#   - calibration_evidence: any *_evidence.md in artifacts/calibration_evidence/
#     with mtime >= Friday 00:00 ET (the production cron run on Friday OR a
#     prior catchup that targeted this Friday)
#   - event_feedback_metrics: existence of metrics_${FRI}.md
FRI_EPOCH=$(TZ=America/Detroit date -d "$FRI" +%s)

cal_artifact_present() {
    local d="$REPO/artifacts/calibration_evidence"
    [ -d "$d" ] || return 1
    # Any *_evidence.md with mtime >= Friday 00:00 means evidence covering
    # this Friday's window has already landed.
    find "$d" -maxdepth 1 -name '*_evidence.md' -newermt "$FRI 00:00" \
        2>/dev/null | grep -q .
}

evf_artifact_present() {
    [ -f "$REPO/artifacts/event_feedback_metrics/metrics_${FRI}.md" ] \
        || [ -f "$REPO/artifacts/event_feedback/metrics_${FRI}.md" ]
}

# postmortem agent — runs from cron `35 18 * * 1-5` via run_agent_direct.py.
# Outputs go to artifacts/postmortem/<DATE>/<TICKER>.{json,md} and to
# logs/agents_direct/postmortem_YYYYMMDD_*.json (per-invocation log).
# Idempotency: a Friday-dated artifact subdir with files inside means
# postmortem already ran for that Friday. Postmortem may legitimately
# produce no artifacts when there are no resolved catalysts (T+3 not
# yet available), so we ALSO accept a Friday/Saturday invocation log.
pm_artifact_present() {
    local d="$REPO/artifacts/postmortem/${FRI}"
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
        return 0
    fi
    local fri_compact="${FRI//-/}"
    local sat_compact
    sat_compact=$(TZ=America/Detroit date -d "${FRI} +1 day" +%Y%m%d 2>/dev/null || echo "")
    ls "$REPO/logs/agents_direct/postmortem_${fri_compact}_"*.json \
        2>/dev/null | grep -q . && return 0
    [ -n "$sat_compact" ] && \
        ls "$REPO/logs/agents_direct/postmortem_${sat_compact}_"*.json \
            2>/dev/null | grep -q . && return 0
    return 1
}

# bellringer results email (Fri 18:30) — sits in WSL2 sleep window.
# Produces artifacts/earnings_sync/results_<DATE>.json when there ARE
# earnings reporters; produces ONLY a log line when there aren't. So a
# false-negative idempotency on no-earnings Fridays is acceptable —
# the Saturday rerun is a single yfinance call with zero email side-effect.
bell_artifact_present() {
    [ -f "$REPO/artifacts/earnings_sync/results_${FRI}.json" ]
}

# news_digest evening (Fri 18:00) — sits in WSL2 sleep window.
# Produces artifacts/news_digest/biotech_news_digest_<DATE>_<HHMM>.{json,txt,html}
# A Saturday catchup writes a Saturday-stamped file, so we check for ANY
# Friday-dated digest file. False-positive risk: morning/midday digests
# satisfy the check even if evening missed — acceptable because morn+midday
# are pre-cliff and almost always succeed; if WSL slept ALL of Friday,
# none of the three digests exist and catchup correctly fires.
news_digest_evening_present() {
    ls "$REPO/artifacts/news_digest/biotech_news_digest_${FRI}"*.json \
        2>/dev/null | grep -q .
}

cal_done=$(cal_artifact_present && echo y || echo n)
evf_done=$(evf_artifact_present && echo y || echo n)
pm_done=$(pm_artifact_present && echo y || echo n)
bell_done=$(bell_artifact_present && echo y || echo n)
news_eve_done=$(news_digest_evening_present && echo y || echo n)

if [ "$cal_done" = "y" ] && [ "$evf_done" = "y" ] && [ "$pm_done" = "y" ] \
   && [ "$bell_done" = "y" ] && [ "$news_eve_done" = "y" ]; then
    log "All Friday builds present (cal=$cal_done evf=$evf_done pm=$pm_done bell=$bell_done news=$news_eve_done) — nothing to do"
    exit 0
fi

log "Friday backfill needed: cal=$cal_done evf=$evf_done pm=$pm_done bell=$bell_done news_eve=$news_eve_done"

if [ "$cal_done" = "n" ]; then
    log "RUN  build_calibration_evidence.py"
    rc=0
    $PYTHON "$REPO/tools/build_calibration_evidence.py" >> "$CAL_LOG" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        log "done build_calibration_evidence.py"
    else
        log "FAIL build_calibration_evidence.py (exit $rc)"
    fi
fi

if [ "$evf_done" = "n" ]; then
    log "RUN  build_event_feedback_metrics.py --as-of-date $FRI"
    rc=0
    $PYTHON "$REPO/tools/build_event_feedback_metrics.py" --as-of-date "$FRI" >> "$EVF_LOG" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        log "done build_event_feedback_metrics.py"
    else
        log "FAIL build_event_feedback_metrics.py (exit $rc)"
    fi
fi

if [ "$pm_done" = "n" ]; then
    log "RUN  postmortem agent (run_agent_direct.py HEARTBEAT)"
    rc=0
    $PYTHON "$REPO/tools/run_agent_direct.py" \
        --agent postmortem --message "HEARTBEAT" \
        >> "$REPO/logs/agents.log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        log "done postmortem"
    else
        log "FAIL postmortem (exit $rc)"
    fi
fi

if [ "$bell_done" = "n" ]; then
    log "RUN  cron_bellringer.sh results"
    rc=0
    "$REPO/tools/cron_bellringer.sh" results \
        >> "$REPO/logs/bellringer.log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        log "done bellringer-results"
    else
        log "FAIL bellringer-results (exit $rc)"
    fi
fi

if [ "$news_eve_done" = "n" ]; then
    log "RUN  build_news_digest.py --window evening"
    rc=0
    $PYTHON "$REPO/scripts/build_news_digest.py" --window evening \
        >> "$REPO/logs/news_digest.log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        log "done news_digest-evening"
    else
        log "FAIL news_digest-evening (exit $rc)"
    fi
fi

log "=== Weekend calibration catch-up complete ==="
