#!/usr/bin/env bash
# Weekly self-improvement loop review — operator host (WSL).
# Advisory only. Does not modify skills or production behavior unless gates are set.
#
# Suggested crontab (Friday 19:30 ET):
#   30 19 * * 5 cd /mnt/c/Projects/biotech_screener/biotech-screener && bash tools/cron_weekly_skills_review.sh >> logs/weekly_skills_review.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
DATE="${1:-$(date +%Y-%m-%d)}"

log() { echo "[$(date -Iseconds)] $*"; }

log "Weekly skills review for $DATE"
log "Rule 12 checklist: docs/governance/RULE_12_PROMOTION_CHECKLIST.md"

$PYTHON tools/fleet_completion_audit.py --write --json >> logs/weekly_skills_review.log 2>&1 || log "fleet_completion_audit reported FAIL (see log)"
$PYTHON tools/fleet_ops_status.py --write --no-telemetry || true
$PYTHON tools/fleet_crontab_verify.py --write || true
GATES_MSG=$($PYTHON -c "from tools.skills_loop_review import selfimprove_gates_status; print(selfimprove_gates_status()['message'])")
log "$GATES_MSG"

$PYTHON tools/weekly_skills_digest.py --date "$DATE"
$PYTHON tools/audit_learnings.py

# Draft skill patches only when operator has cleared stalled-loop gates
if [[ "${SELFIMPROVE_GATES_MET:-0}" == "1" ]]; then
  log "SELFIMPROVE_GATES_MET=1 — running pattern_to_skillpatch (drafts only)"
  $PYTHON tools/pattern_to_skillpatch.py --min-recurrence 3 --out artifacts/skill_patch_drafts || true
else
  log "SELFIMPROVE_GATES_MET not set — skipping pattern_to_skillpatch"
fi

log "Done"
