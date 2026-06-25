#!/usr/bin/env bash
# run_fleet_host_onboarding.sh — one-shot WSL host onboarding after git pull
#
# Prints crontab install reference, runs the full operator checklist
# (audit → fleet_ops → crontab verify), and surfaces Rule 12 stalled-loop gates.
#
# Usage:
#   bash tools/run_fleet_host_onboarding.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -Iseconds)] onboarding: $*"; }

log "Agent fleet host onboarding — migration phases 2–14 code-complete"
log "Architecture index: docs/AGENT_FLEET_ARCHITECTURE_INDEX.md"
log "Rule 12 gates: docs/governance/RULE_12_PROMOTION_CHECKLIST.md"
echo ""
echo "Step 1 — Install crontab (manual paste):"
echo "  bash tools/install_agent_fleet_crontab.sh"
echo "  crontab -e   # paste printed block on WSL host"
echo ""
echo "Step 2 — Run verification checklist:"
bash "$REPO_ROOT/tools/run_fleet_operator_checklist.sh"
echo ""
log "Step 3 — Close host blockers before SELFIMPROVE_GATES_MET=1:"
log "  F-2026-005 Herald: confirm classified JSONL on host; watchdog runs --recover on FAIL"
log "  F-2026-006 CI: restore GitHub Actions budget; green tests on main"
log "  Update .learnings/memory.md stalled-loop table when confirmed"
log "Done — artifacts under artifacts/fleet_ops/"
