#!/usr/bin/env bash
# cron_daily_production.sh — Automated daily production screen runner.
#
# Designed for cron on WSL2. Runs the full daily production pipeline
# for today's date, logging output to a dated log file.
#
# Usage:
#   ./tools/cron_daily_production.sh              # run for today
#   ./tools/cron_daily_production.sh 2026-03-20   # run for specific date
#   ./tools/cron_daily_production.sh --catch-up   # backfill missed weekdays
#
# Cron example (weekdays at 4:30 PM ET, after market close):
#   30 16 * * 1-5 /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_daily_production.sh >> /mnt/c/Projects/biotech_screener/biotech-screener/logs/cron.log 2>&1
# On WSL2 reboot, catch up any missed days:
#   @reboot sleep 60 && /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_daily_production.sh --catch-up >> /mnt/c/Projects/biotech_screener/biotech-screener/logs/cron.log 2>&1

set -euo pipefail

REPO_ROOT="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG_DIR="${REPO_ROOT}/logs"
LOCK_FILE="${REPO_ROOT}/logs/.daily_production.lock"
SNAPSHOT_DIR="${REPO_ROOT}/data/snapshots"
MAX_CATCHUP_DAYS=5

# Class P: cron may invoke this script by absolute path with cwd != repo.
cd "${REPO_ROOT}"

# --- Catch-up mode: find and run missed weekdays ---
if [ "${1:-}" = "--catch-up" ]; then
    echo "[$(date -Iseconds)] Catch-up: scanning last ${MAX_CATCHUP_DAYS} weekdays for missed runs"
    MISSED=0
    for i in $(seq 1 ${MAX_CATCHUP_DAYS}); do
        CHECK_DATE=$(date -d "-${i} days" +%Y-%m-%d 2>/dev/null || continue)
        CHECK_DOW=$(date -d "${CHECK_DATE}" +%u 2>/dev/null || continue)
        # Skip weekends
        [ "${CHECK_DOW}" -gt 5 ] && continue
        # Contract-aware skip. PASS (rc=0) or optional-only-missing (rc=2)
        # means the snapshot is good enough; FAIL (rc=1) means at least one
        # required artifact is missing or empty. We only do an expensive
        # full re-run when rankings.csv itself is missing — diagnostic-only
        # gaps are left for cron_diagnostics_backstop.sh to repair.
        if [ -d "${SNAPSHOT_DIR}/${CHECK_DATE}" ]; then
            CONTRACT_RC=0
            ${PYTHON} tools/check_output_contract.py \
                --as-of "${CHECK_DATE}" \
                --snapshot-root "${SNAPSHOT_DIR}" \
                >/dev/null 2>&1 || CONTRACT_RC=$?
            if [ "${CONTRACT_RC}" -eq 0 ] || [ "${CONTRACT_RC}" -eq 2 ]; then
                continue
            fi
            if [ -f "${SNAPSHOT_DIR}/${CHECK_DATE}/rankings.csv" ]; then
                echo "[$(date -Iseconds)] Catch-up: ${CHECK_DATE} has rankings.csv but missing required diagnostics — leaving for diagnostics backstop"
                continue
            fi
            echo "[$(date -Iseconds)] Catch-up: ${CHECK_DATE} snapshot dir exists but rankings.csv missing — full re-run"
        fi
        echo "[$(date -Iseconds)] Catch-up: missed ${CHECK_DATE}, running backfill"
        "$0" "${CHECK_DATE}" || true
        MISSED=$((MISSED + 1))
    done
    if [ ${MISSED} -eq 0 ]; then
        echo "[$(date -Iseconds)] Catch-up: no missed runs found"
    else
        echo "[$(date -Iseconds)] Catch-up: backfilled ${MISSED} missed day(s)"
    fi
    exit 0
fi

# Date: use argument or today
AS_OF_DATE="${1:-$(date +%Y-%m-%d)}"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Prevent concurrent runs
if [ -f "${LOCK_FILE}" ]; then
    LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null || echo "")
    if [ -n "${LOCK_PID}" ] && kill -0 "${LOCK_PID}" 2>/dev/null; then
        echo "[$(date -Iseconds)] SKIP: another daily production run is active (PID ${LOCK_PID})"
        exit 0
    else
        echo "[$(date -Iseconds)] WARN: stale lock file removed (PID ${LOCK_PID} not running)"
        rm -f "${LOCK_FILE}"
    fi
fi

# Write lock
echo $$ > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

# Load environment
if [ -f "${REPO_ROOT}/.env" ]; then
    # Use python-dotenv style parsing to avoid bash expansion of $ in values.
    # `source .env` breaks on values like passwords containing $, backticks, etc.
    while IFS= read -r line || [ -n "${line}" ]; do
        # Skip comments and blank lines
        [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
        # Strip surrounding quotes from value
        key="${line%%=*}"
        val="${line#*=}"
        val="${val#\"}" ; val="${val%\"}"
        val="${val#\'}" ; val="${val%\'}"
        export "${key}=${val}"
    done < "${REPO_ROOT}/.env"
fi

LOG_FILE="${LOG_DIR}/daily_production_${AS_OF_DATE}.log"

echo "[$(date -Iseconds)] Starting daily production for ${AS_OF_DATE}" | tee -a "${LOG_FILE}"

# Check if it's a weekday (skip weekends)
DOW=$(date -d "${AS_OF_DATE}" +%u 2>/dev/null || echo "0")
if [ "${DOW}" -gt 5 ]; then
    echo "[$(date -Iseconds)] SKIP: ${AS_OF_DATE} is a weekend (day ${DOW})" | tee -a "${LOG_FILE}"
    exit 0
fi

# --- Drift guard (issue #484): warn if this checkout is behind origin/main ---
# The cron checkout does not auto-pull, so merged fixes do not take effect
# until it is fast-forwarded. Surface drift loudly instead of silently running
# stale code. Non-blocking: this only logs — it never aborts the run.
DRIFT_WARN_THRESHOLD=5
if command -v git >/dev/null 2>&1 && git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    timeout 30 git -C "${REPO_ROOT}" fetch --quiet origin main 2>/dev/null \
        || echo "[$(date -Iseconds)] WARN: drift-guard fetch failed (offline?) — comparing against last-known origin/main" | tee -a "${LOG_FILE}"
    LOCAL_HEAD=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "?")
    BEHIND=$(git -C "${REPO_ROOT}" rev-list --count HEAD..origin/main 2>/dev/null || echo "0")
    # Spec 115 Phase 1: bind HEAD *here*, at run start. The capture step below
    # used to read HEAD at its own invocation time, so any mid-run HEAD move
    # (a pull, a commit from an interactive session) guaranteed a mismatch
    # against the snapshot's stamp and silently cost a mandate-eligible window.
    # This is what happened on 2026-07-27 — see docs/incidents/FV_GAP_2026_07_27.md.
    RUN_START_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "")"

    # Spec 115 Phase 2 (visibility half): surface mandate evidence that is
    # sitting uncommitted. On 2026-07-23 a working-tree revert destroyed an
    # already-written capture and nobody noticed for five days, because nothing
    # reported the gap. This does not prevent Mode B — it makes it visible
    # within one run instead of a week. Non-blocking: warn only, never abort.
    CAPTURES_REL="artifacts/forward_validation/captures.jsonl"
    CAPTURES_ABS="${REPO_ROOT}/${CAPTURES_REL}"
    if [ -f "${CAPTURES_ABS}" ]; then
        COMMITTED_LINES=$(git -C "${REPO_ROOT}" show "HEAD:${CAPTURES_REL}" 2>/dev/null | grep -c . || echo 0)
        WORKING_LINES=$(grep -c . "${CAPTURES_ABS}" 2>/dev/null || echo 0)
        if [ "${WORKING_LINES}" -gt "${COMMITTED_LINES}" ]; then
            UNCOMMITTED_DATES=$(tail -n +"$((COMMITTED_LINES + 1))" "${CAPTURES_ABS}" 2>/dev/null \
                | ${PYTHON} -c 'import sys,json;print(", ".join(json.loads(l)["date"] for l in sys.stdin if l.strip()))' 2>/dev/null || echo "")
            echo "[$(date -Iseconds)] WARN: UNCOMMITTED_MANDATE_EVIDENCE — $((WORKING_LINES - COMMITTED_LINES)) capture(s) in ${CAPTURES_REL} are not committed: ${UNCOMMITTED_DATES:-<unparsed>}. A working-tree revert would destroy them silently (this is how 2026-07-23 was lost). Commit them." | tee -a "${LOG_FILE}"
        fi
    fi
    if [ "${BEHIND}" -gt "${DRIFT_WARN_THRESHOLD}" ]; then
        echo "[$(date -Iseconds)] WARN: checkout drift — HEAD ${LOCAL_HEAD} is ${BEHIND} commits behind origin/main. Cron is running STALE code; merged fixes are not live until this checkout is fast-forwarded (issue #484)." | tee -a "${LOG_FILE}"
    elif [ "${BEHIND}" -gt 0 ]; then
        echo "[$(date -Iseconds)] INFO: checkout is ${BEHIND} commit(s) behind origin/main (HEAD ${LOCAL_HEAD})." | tee -a "${LOG_FILE}"
    fi
fi

# Run the production pipeline (timeout: 75 minutes).
# Budget breakdown observed: snapshot ~25 min + Herald ≤10 min + AACT ≤30 min
# (Mondays only) + tail steps ≤5 min = ~70 min worst case. The previous 45-min
# budget was killing the python child mid-AACT before the wrapper could reach
# its own PASS/FAIL summary block.
PIPELINE_TIMEOUT=6000
# NOTE: the pipeline invocation MUST be guarded (|| EXIT_CODE=$?). Under
# `set -euo pipefail` a bare non-zero exit aborts the wrapper before it can
# reach its post-pipeline tail (diagnostics + forward-validation capture).
# run_daily_production.py exits 2 on WARN-status runs, which is routine, so an
# unguarded call silently skips the tail on every warning day. Capturing the
# code here keeps the tail running while still recording the true exit status.
if command -v timeout >/dev/null 2>&1; then
    EXIT_CODE=0
    timeout --signal=TERM --kill-after=60 ${PIPELINE_TIMEOUT} \
        ${PYTHON} tools/run_daily_production.py \
        --as-of-date "${AS_OF_DATE}" \
        >> "${LOG_FILE}" 2>&1 || EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 124 ]; then
        echo "[$(date -Iseconds)] TIMEOUT: pipeline exceeded ${PIPELINE_TIMEOUT}s — killed" | tee -a "${LOG_FILE}"
    fi
else
    EXIT_CODE=0
    ${PYTHON} tools/run_daily_production.py \
        --as-of-date "${AS_OF_DATE}" \
        >> "${LOG_FILE}" 2>&1 || EXIT_CODE=$?
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date -Iseconds)] PASS: daily production completed successfully" | tee -a "${LOG_FILE}"
elif [ ${EXIT_CODE} -eq 2 ]; then
    echo "[$(date -Iseconds)] WARN: daily production completed with warnings (exit 2)" | tee -a "${LOG_FILE}"
else
    echo "[$(date -Iseconds)] FAIL: daily production failed (exit ${EXIT_CODE})" | tee -a "${LOG_FILE}"
fi

# --- Failure notification ---
# Send alert on non-zero exit so pipeline failures don't go unnoticed.
# Configure PIPELINE_ALERT_WEBHOOK in .env (e.g. Slack/Discord incoming webhook URL).
if [ ${EXIT_CODE} -ne 0 ] && [ -n "${PIPELINE_ALERT_WEBHOOK:-}" ]; then
    _STATUS="FAIL"
    [ ${EXIT_CODE} -eq 2 ] && _STATUS="WARN"
    curl -sf -X POST "${PIPELINE_ALERT_WEBHOOK}" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"[Wake Robin] ${_STATUS}: daily production ${AS_OF_DATE} exited ${EXIT_CODE}. Log: ${LOG_FILE}\"}" \
        >> "${LOG_FILE}" 2>&1 || true
fi

# --- Rank-change monitor (read-only diagnostic) ---
# Compares today's rankings.csv to the most recent prior snapshot and writes
# rank_change_alerts.{csv,md,json} into the snapshot dir. Read-only — does NOT
# change scoring, selectors, ranking, eligibility, or portfolio construction.
# Gated on the snapshot's rankings.csv existing rather than the pipeline's
# overall exit status: post-snapshot tasks (Herald, AACT, etc.) sometimes
# hang and never return EXIT_CODE, but the snapshot itself is already
# complete by then — this gate fires whenever the diagnostic has data.
if [ -f "${SNAPSHOT_DIR}/${AS_OF_DATE}/rankings.csv" ]; then
    ${PYTHON} tools/build_rank_change_monitor.py \
        --as-of-date "${AS_OF_DATE}" \
        --print-alerts \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: rank-change monitor exited non-zero" | tee -a "${LOG_FILE}"

    # Webhook on CRITICAL rank-change alerts (system-level or per-ticker).
    # Mirrors the pipeline-failure webhook pattern. WARN stays log-only.
    ALERT_JSON="${SNAPSHOT_DIR}/${AS_OF_DATE}/rank_change_alerts.json"
    if [ -f "${ALERT_JSON}" ] && [ -n "${PIPELINE_ALERT_WEBHOOK:-}" ]; then
        PAYLOAD=$(${PYTHON} - "${ALERT_JSON}" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("summary", {})
if s.get("n_critical", 0) == 0:
    sys.exit(1)
lines = [
    f"[Wake Robin] CRITICAL rank-change alerts on {d.get('as_of_date')}: "
    f"{s['n_critical']} critical, {s.get('n_warn', 0)} warn, "
    f"cohort_churn={s.get('cohort_churn_pct', 0)}%"
]
for sa in d.get("system_alerts", []):
    if sa.get("severity") == "CRITICAL":
        extras = {k: v for k, v in sa.items() if k not in ("kind", "severity")}
        lines.append(f"  SYSTEM: {sa['kind']} {extras}")
for a in d.get("alerts", []):
    if a.get("severity") == "CRITICAL":
        rd = a.get("rank_delta")
        rd_str = f"{rd:+d}" if isinstance(rd, int) else "∅"
        flags = ",".join(a.get("flags", [])[:3])
        lines.append(f"  {a['ticker']}: rankΔ={rd_str} {a['likely_reason']} ({flags})")
print(json.dumps({"text": "\n".join(lines)}))
PY
        ) || PAYLOAD=""
        if [ -n "${PAYLOAD}" ]; then
            curl -sf -X POST "${PIPELINE_ALERT_WEBHOOK}" \
                -H "Content-Type: application/json" \
                -d "${PAYLOAD}" \
                >> "${LOG_FILE}" 2>&1 || true
            echo "[$(date -Iseconds)] Posted CRITICAL rank-change alert to webhook" | tee -a "${LOG_FILE}"
        fi
    fi
fi

# --- Diagnostic reports (read-only) ---
# Per-snapshot artifacts emitted whenever rankings.csv exists. Each tool runs
# independently — a failure in one does not block the others. None modify
# scoring, selectors, ranking, eligibility, or portfolio construction. The
# tools always exit 0; FAIL signals are carried in the JSON's
# `overall_severity` field for any downstream consumer.
if [ -f "${SNAPSHOT_DIR}/${AS_OF_DATE}/rankings.csv" ]; then
    ${PYTHON} tools/build_snapshot_integrity_report.py \
        --as-of-date "${AS_OF_DATE}" \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: snapshot_integrity_report exited non-zero" | tee -a "${LOG_FILE}"

    ${PYTHON} tools/build_feature_coverage_report.py \
        --as-of-date "${AS_OF_DATE}" \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: feature_coverage_report exited non-zero" | tee -a "${LOG_FILE}"

    ${PYTHON} tools/build_distribution_drift_report.py \
        --as-of-date "${AS_OF_DATE}" \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: distribution_drift_report exited non-zero" | tee -a "${LOG_FILE}"

    ${PYTHON} tools/build_sentinel_ticker_report.py \
        --as-of-date "${AS_OF_DATE}" \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: sentinel_ticker_report exited non-zero" | tee -a "${LOG_FILE}"
fi

# --- Forward validation: capture today's Top-30 EW basket + fill completed returns ---
# Immutable daily capture appended to artifacts/forward_validation/captures.jsonl.
# Fill script updates returns for any pending capture whose forward window is now observable.
# Both are read-only with respect to the model; no scoring or selection logic changes.
#
# Capture ONLY when production completed (exit 0 = clean, exit 2 = completed with
# governance warnings). On a hard failure (exit 1), timeout (124), SIGKILL (137),
# or any other code, the snapshot may be partial or stale — capturing then would
# feed a stale-output false positive into the forward mandate. The capture script
# applies its own freshness/provenance gate as defense-in-depth, and is passed the
# wrapper's HEAD commit so it can confirm the snapshot came from this invocation.
# Spec 115 Phase 1: RUN_START_COMMIT is bound at run start (drift guard above),
# NOT here. Re-reading HEAD at this point is the bug that cost the 2026-07-27
# window. If HEAD moved during the run, the run may have executed a mix of two
# commits, so the capture is still refused — but with an unambiguous, greppable
# diagnosis instead of a generic provenance refusal that reads like a pipeline
# fault. Early-binding alone would MASK a genuinely mixed-code run.
CURRENT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "")"
INVOCATION_COMMIT="${RUN_START_COMMIT:-${CURRENT_COMMIT}}"
HEAD_MOVED=0
if [ -n "${RUN_START_COMMIT}" ] && [ -n "${CURRENT_COMMIT}" ] && [ "${RUN_START_COMMIT}" != "${CURRENT_COMMIT}" ]; then
    HEAD_MOVED=1
    echo "[$(date -Iseconds)] ERROR: RUN_HEAD_MOVED — HEAD changed during this run: start=${RUN_START_COMMIT} now=${CURRENT_COMMIT}." | tee -a "${LOG_FILE}"
    echo "[$(date -Iseconds)] ERROR: RUN_HEAD_MOVED — refusing the mandate capture; this run may have executed a mix of two commits. Do not --force (that yields REPLAY, which is never mandate-eligible). Do not run git state changes while production is in flight." | tee -a "${LOG_FILE}"
fi
if { [ "${EXIT_CODE}" -eq 0 ] || [ "${EXIT_CODE}" -eq 2 ]; } && [ -f "${SNAPSHOT_DIR}/${AS_OF_DATE}/rankings.csv" ]; then
    # Only the capture is gated on HEAD stability — it is the one step whose
    # provenance depends on this run. The fill and summary steps below operate on
    # PRIOR captures, so skipping them would delay evidence maturation for no
    # reason.
    if [ "${HEAD_MOVED}" -eq 0 ]; then
        ${PYTHON} tools/run_forward_validation.py \
            --as-of-date "${AS_OF_DATE}" \
            --expect-commit "${INVOCATION_COMMIT}" \
            --capture-mode LIVE \
            2>&1 | tee -a "${LOG_FILE}" || \
            echo "[$(date -Iseconds)] WARN: run_forward_validation exited non-zero" | tee -a "${LOG_FILE}"
    else
        echo "[$(date -Iseconds)] SKIP: mandate capture for ${AS_OF_DATE} — RUN_HEAD_MOVED (see ERROR above). Window is lost and is NOT recoverable." | tee -a "${LOG_FILE}"
    fi

    ${PYTHON} tools/fill_forward_returns.py \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: fill_forward_returns exited non-zero" | tee -a "${LOG_FILE}"

    ${PYTHON} tools/weekly_validation_summary.py \
        2>&1 | tee -a "${LOG_FILE}" || \
        echo "[$(date -Iseconds)] WARN: weekly_validation_summary exited non-zero" | tee -a "${LOG_FILE}"
else
    echo "[$(date -Iseconds)] SKIP: forward-validation capture — pipeline exit=${EXIT_CODE} or rankings.csv absent (capture only on exit 0/2 with a fresh snapshot)" | tee -a "${LOG_FILE}"
fi

# --- Housekeeping: prune pre-staging, old logs, and old caches ---
# Pre-staging (__pre_*) dirs older than 7 days are removed (temporary staging).
# Snapshots older than 18 months are compressed to tar.gz archives.
# PIT cache anchors older than 12 months are removed (data/caches/price_pit/).
# Artifact watch/digest dirs older than 6 months are removed.
# Logs older than 60 days are removed.
PRE_STAGING_DAYS=7
LOG_RETENTION_DAYS=60
SNAPSHOT_ARCHIVE_DAYS=548  # ~18 months
CACHE_PRUNE_DAYS=365       # 12 months
ARTIFACT_PRUNE_DAYS=180    # 6 months

prune_count=0
for dir in "${SNAPSHOT_DIR}"/*__pre_*; do
    [ -d "${dir}" ] || continue
    dirname=$(basename "${dir}")
    snap_date="${dirname:0:10}"
    snap_epoch=$(date -d "${snap_date}" +%s 2>/dev/null || continue)
    now_epoch=$(date +%s)
    age_days=$(( (now_epoch - snap_epoch) / 86400 ))
    if [ ${age_days} -gt ${PRE_STAGING_DAYS} ]; then
        rm -rf "${dir}"
        prune_count=$((prune_count + 1))
    fi
done

# Prune old logs
for logfile in "${LOG_DIR}"/daily_production_*.log; do
    [ -f "${logfile}" ] || continue
    log_epoch=$(stat -c %Y "${logfile}" 2>/dev/null || continue)
    now_epoch=$(date +%s)
    age_days=$(( (now_epoch - log_epoch) / 86400 ))
    if [ ${age_days} -gt ${LOG_RETENTION_DAYS} ]; then
        rm -f "${logfile}"
        prune_count=$((prune_count + 1))
    fi
done

# Compress old snapshots to archives (>18 months, not already archived)
ARCHIVE_DIR="${REPO_ROOT}/data/archives"
mkdir -p "${ARCHIVE_DIR}"
for dir in "${SNAPSHOT_DIR}"/20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
    [ -d "${dir}" ] || continue
    snap_date=$(basename "${dir}")
    snap_epoch=$(date -d "${snap_date}" +%s 2>/dev/null || continue)
    now_epoch=$(date +%s)
    age_days=$(( (now_epoch - snap_epoch) / 86400 ))
    if [ ${age_days} -gt ${SNAPSHOT_ARCHIVE_DAYS} ]; then
        archive_path="${ARCHIVE_DIR}/${snap_date}.tar.gz"
        if [ ! -f "${archive_path}" ]; then
            tar -czf "${archive_path}" -C "${SNAPSHOT_DIR}" "${snap_date}" 2>/dev/null && \
                rm -rf "${dir}" && prune_count=$((prune_count + 1))
        else
            # Archive exists, safe to remove snapshot
            rm -rf "${dir}" && prune_count=$((prune_count + 1))
        fi
    fi
done

# Prune old PIT price cache anchors (>12 months)
PRICE_CACHE="${REPO_ROOT}/data/caches/price_pit/PIT"
if [ -d "${PRICE_CACHE}" ]; then
    for dir in "${PRICE_CACHE}"/20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
        [ -d "${dir}" ] || continue
        cache_date=$(basename "${dir}")
        cache_epoch=$(date -d "${cache_date}" +%s 2>/dev/null || continue)
        now_epoch=$(date +%s)
        age_days=$(( (now_epoch - cache_epoch) / 86400 ))
        if [ ${age_days} -gt ${CACHE_PRUNE_DAYS} ]; then
            rm -rf "${dir}" && prune_count=$((prune_count + 1))
        fi
    done
fi

# Prune old artifact watch/digest dirs (>6 months)
for artifact_type in price_action_watch ops_digest shadow_monitor; do
    artifact_dir="${REPO_ROOT}/artifacts/${artifact_type}"
    [ -d "${artifact_dir}" ] || continue
    find "${artifact_dir}" -name "20[0-9][0-9]-*" -mtime +${ARTIFACT_PRUNE_DAYS} -delete 2>/dev/null
    _pruned=$?
    [ ${_pruned} -eq 0 ] && prune_count=$((prune_count + 1))
done

if [ ${prune_count} -gt 0 ]; then
    echo "[$(date -Iseconds)] Housekeeping: pruned ${prune_count} old item(s) (pre-staging, logs, snapshots, caches, artifacts)" | tee -a "${LOG_FILE}"
fi

# NOTE: OpenClaw agents (ops/sentinel/qa) run on their own cron schedule
# staggered after production: 5:00 / 5:15 / 5:30 PM ET.
# They are NOT triggered from this script to avoid inspecting half-built packets.

echo "[$(date -Iseconds)] Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
exit ${EXIT_CODE}
