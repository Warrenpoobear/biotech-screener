#!/usr/bin/env bash
# run_forward_evidence_package.sh — freeze-lift forward evidence (governance Tier 0)
#
# Assembles Path C close status, forward-eval IC, coinvest_score_z IC, final_score IC,
# and coinvest shadow summary. Does NOT lift the architecture freeze.
#
# Usage:
#   bash tools/run_forward_evidence_package.sh --dry-run
#   FREEZE_LIFT_ACK=1 bash tools/run_forward_evidence_package.sh --write
#   FREEZE_LIFT_ACK=1 bash tools/run_forward_evidence_package.sh --write 2026-06-24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
MODE="dry-run"
DATE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --write) MODE="write"; shift ;;
        --dry-run) MODE="dry-run"; shift ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            DATE="$1"
            shift
            ;;
    esac
done

DATE="${DATE:-$(TZ=America/Detroit date +%Y-%m-%d)}"

log() { echo "[$(date -Iseconds)] forward_evidence: $*"; }

log "Freeze-lift forward evidence package for $DATE"
log "Memo: docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md"
log "Does NOT lift freeze — produces evidence for operator decision"
echo ""

log "Step 1 — Path C retrospective window close"
$PYTHON tools/path_c_window_close_decision.py --window-end 2026-06-03 --output-json || true
echo ""

log "Step 2 — Refresh forward-eval IC ledger (best effort)"
$PYTHON tools/monitor_forward_eval_ic.py --window-end "$DATE" 2>&1 | tail -5 || true
echo ""

if [ "$MODE" = "write" ]; then
    if [ "${FREEZE_LIFT_ACK:-}" != "1" ]; then
        log "ABORT — set FREEZE_LIFT_ACK=1 to write artifacts"
        log "  export FREEZE_LIFT_ACK=1"
        exit 1
    fi
    log "Step 3 — Build and write evidence package"
    $PYTHON tools/forward_evidence_package.py --as-of-date "$DATE" --write
else
    log "Step 3 — Dry-run evidence package (no writes)"
    $PYTHON tools/forward_evidence_package.py --as-of-date "$DATE" --dry-run --json | head -40
    log "Use --write with FREEZE_LIFT_ACK=1 to persist artifacts"
fi
