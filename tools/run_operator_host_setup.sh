#!/usr/bin/env bash
# run_operator_host_setup.sh — unified WSL operator setup (fleet + optional research)
#
# Step 1 always runs fleet onboarding (crontab hint + checklist + gate reminders).
# Step 2 runs the research host battery when PIT snapshots and price history exist.
#
# Usage:
#   bash tools/run_operator_host_setup.sh
#   bash tools/run_operator_host_setup.sh --skip-research
#   bash tools/run_operator_host_setup.sh --research-only
#   bash tools/run_operator_host_setup.sh 2026-06-24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SKIP_RESEARCH=0
RESEARCH_ONLY=0
DATE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-research) SKIP_RESEARCH=1; shift ;;
        --research-only) RESEARCH_ONLY=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            if [ -z "$DATE" ]; then
                DATE="$1"
            else
                echo "Unknown argument: $1" >&2
                exit 2
            fi
            shift
            ;;
    esac
done

DATE="${DATE:-$(TZ=America/Detroit date +%Y-%m-%d)}"

log() { echo "[$(date -Iseconds)] operator_setup: $*"; }

research_prereqs_met() {
    [ -e "$REPO_ROOT/data/snapshots_pit_v2" ] \
        && [ -f "$REPO_ROOT/production_data/price_history.csv" ] \
        && [ -f "$REPO_ROOT/data/snapshots/$DATE/rankings.csv" ]
}

run_research_battery() {
    log "Research prerequisites present — running research host battery for $DATE"
    bash "$REPO_ROOT/tools/run_research_host_battery.sh" "$DATE"
}

if [ "$RESEARCH_ONLY" -eq 1 ]; then
    log "Research-only mode for $DATE"
    if research_prereqs_met; then
        run_research_battery
    else
        log "Abort — research prerequisites missing"
        log "  need: data/snapshots_pit_v2/, production_data/price_history.csv,"
        log "        data/snapshots/$DATE/rankings.csv"
        log "See docs/research/CHECKLIST_V2_FINAL_SCORE_BLOCKER_2026_06_24.md"
        exit 1
    fi
    exit 0
fi

log "Unified operator host setup — fleet + optional research"
log "Fleet index: docs/AGENT_FLEET_ARCHITECTURE_INDEX.md"
log "Research blocker: docs/research/CHECKLIST_V2_FINAL_SCORE_BLOCKER_2026_06_24.md"
echo ""

bash "$REPO_ROOT/tools/run_fleet_host_onboarding.sh"

if [ "$SKIP_RESEARCH" -eq 1 ]; then
    log "Skipping research battery (--skip-research)"
    exit 0
fi

echo ""
if research_prereqs_met; then
    run_research_battery
else
    log "Research battery skipped — prerequisites not on host"
    log "  missing one or more of:"
    log "    data/snapshots_pit_v2/"
    log "    production_data/price_history.csv"
    log "    data/snapshots/$DATE/rankings.csv"
    log "When ready: bash tools/run_research_host_battery.sh $DATE"
fi
