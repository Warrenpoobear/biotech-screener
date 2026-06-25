#!/usr/bin/env bash
# run_fleet_operator_checklist.sh — post-pull / post-crontab host verification
#
# Read-only checks + artifact writes. Safe to re-run.
#
# Usage:
#   bash tools/run_fleet_operator_checklist.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

log() { echo "[$(date -Iseconds)] checklist: $*"; }

log "Fleet operator checklist — repo=$REPO_ROOT"
log "Rule 12: docs/governance/RULE_12_PROMOTION_CHECKLIST.md"
log "Crontab reference (paste manually): bash tools/install_agent_fleet_crontab.sh"

log "Herald health"
$PYTHON tools/herald_health_check.py --stdout || true

log "Completion audit (before fleet_ops so status.json embeds registry_coverage)"
if ! $PYTHON tools/fleet_completion_audit.py --write; then
    log "FAIL — completion audit reported wiring gaps (see output above)"
    exit 1
fi

log "Fleet ops status write"
$PYTHON tools/fleet_ops_status.py --write --no-telemetry || true

log "Self-improve gates"
$PYTHON -c "from tools.skills_loop_review import selfimprove_gates_status; print(selfimprove_gates_status()['message'])"

log "Done — artifacts under artifacts/fleet_ops/"
