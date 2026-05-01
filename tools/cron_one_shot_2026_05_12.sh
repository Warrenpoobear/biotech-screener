#!/usr/bin/env bash
# cron_one_shot_2026_05_12.sh — Verify the build_event_analyst.py cron entry
# (added 2026-05-01: `10 19 * * 1-5`) has been firing daily and writing
# artifacts/event_analyst/{date}_summary.json.
#
# Fires once on 2026-05-12 09:00 ET, after one full week of natural fires.
# Self-skips on any other date and on re-invocations (marker file).
#
# Cron entry:
#   0 9 12 5 * /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_one_shot_2026_05_12.sh >> /mnt/c/Projects/biotech_screener/biotech-screener/logs/one_shot.log 2>&1

set -euo pipefail

REPO_ROOT="/mnt/c/Projects/biotech_screener/biotech-screener"
TARGET_DATE="2026-05-12"
MARKER="${REPO_ROOT}/logs/.one_shot_${TARGET_DATE}.done"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/audit"
ARTIFACT="${ARTIFACT_DIR}/event_analyst_builder_verify_${TARGET_DATE}.md"
LOG_PREFIX="[$(date -Iseconds)]"

if [ -f "$MARKER" ]; then
    echo "${LOG_PREFIX} SKIP: already fired (marker $MARKER exists)"
    exit 0
fi

TODAY=$(date +%Y-%m-%d)
if [ "$TODAY" != "$TARGET_DATE" ]; then
    echo "${LOG_PREFIX} SKIP: today=${TODAY} != target=${TARGET_DATE}"
    exit 0
fi

echo "${LOG_PREFIX} Firing event_analyst builder verification for ${TARGET_DATE}"

cd "$REPO_ROOT"
mkdir -p "$ARTIFACT_DIR" logs

EXPECTED_DATES=(2026-05-04 2026-05-05 2026-05-06 2026-05-07 2026-05-08 2026-05-11)
PRESENT=()
MISSING=()
SIZE_FLAGS=()

for d in "${EXPECTED_DATES[@]}"; do
    f="artifacts/event_analyst/${d}_summary.json"
    if [ -f "$f" ]; then
        PRESENT+=("$d")
        # Flag if file is suspiciously small (< 500 bytes suggests empty/error)
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -lt 500 ]; then
            SIZE_FLAGS+=("${d} (${sz} bytes)")
        fi
    else
        MISSING+=("$d")
    fi
done

PRESENT_STR="${PRESENT[*]:-(none)}"
MISSING_STR="${MISSING[*]:-(none)}"
SIZE_FLAGS_STR="${SIZE_FLAGS[*]:-(none)}"

if [ ${#MISSING[@]} -eq 0 ] && [ ${#SIZE_FLAGS[@]} -eq 0 ]; then
    VERDICT="PASS — all 6/6 weekday artifacts present and non-trivially sized"
elif [ ${#MISSING[@]} -eq 0 ]; then
    VERDICT="WARN — all 6/6 present but ${#SIZE_FLAGS[@]} suspiciously small (possible error/empty)"
else
    VERDICT="FAIL — ${#MISSING[@]}/${#EXPECTED_DATES[@]} missing"
fi

LOG_FILE="logs/event_analyst_builder.log"
if [ -f "$LOG_FILE" ]; then
    LOG_TAIL=$(tail -50 "$LOG_FILE")
    LOG_BYTES=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
else
    LOG_TAIL="(log file does not exist — cron entry may not have fired at all)"
    LOG_BYTES=0
fi

ERROR_HITS=$(grep -ciE "error|exception|traceback|failed" "$LOG_FILE" 2>/dev/null || echo 0)

{
    echo "# Event Analyst Builder Verification — ${TARGET_DATE}"
    echo
    echo "Cron entry under verification: \`10 19 * * 1-5\` (added 2026-05-01)"
    echo "Reason added: build_event_analyst.py was unscheduled — only LLM HEARTBEAT was wired, so artifacts/event_analyst/ was frozen at 2026-04-03."
    echo
    echo "## Verdict: ${VERDICT}"
    echo
    echo "## Expected weekday artifacts (2026-05-04 → 2026-05-11, skipping weekend)"
    echo
    echo "- Present (${#PRESENT[@]}): ${PRESENT_STR}"
    echo "- Missing (${#MISSING[@]}): ${MISSING_STR}"
    echo "- Size-flagged (<500 bytes): ${SIZE_FLAGS_STR}"
    echo
    echo "## Builder log"
    echo
    echo "- Path: ${LOG_FILE}"
    echo "- Size: ${LOG_BYTES} bytes"
    echo "- Error/exception/traceback line count: ${ERROR_HITS}"
    echo
    echo "### Last 50 lines"
    echo
    echo '```'
    echo "${LOG_TAIL}"
    echo '```'
    echo
    echo "## Failure-mode hint"
    echo
    if [ ${#MISSING[@]} -eq ${#EXPECTED_DATES[@]} ]; then
        echo "All 6 dates missing → cron entry likely not firing at all. Check \`crontab -l | grep build_event_analyst\` and WSL uptime during 19:10 ET windows on the missing dates."
    elif [ ${#MISSING[@]} -gt 0 ]; then
        echo "Partial coverage — likely WSL was down on the missing dates. Cross-check \`last reboot\` for those evenings."
    elif [ ${#SIZE_FLAGS[@]} -gt 0 ]; then
        echo "Files exist but small — script ran but produced near-empty output. Check log for postmortem source data issues."
    elif [ "$ERROR_HITS" -gt 0 ]; then
        echo "Artifacts present but log contains error tokens — review log tail for context."
    else
        echo "Clean — issue can be marked closed in MEMORY.md."
    fi
} > "$ARTIFACT"

echo "${LOG_PREFIX} Wrote ${ARTIFACT}"
echo "${LOG_PREFIX} Verdict report ↓↓↓"
cat "$ARTIFACT"
echo "${LOG_PREFIX} Verdict report ↑↑↑"

touch "$MARKER"
echo "${LOG_PREFIX} Done. Marker: $MARKER"
