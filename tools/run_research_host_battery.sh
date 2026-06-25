#!/usr/bin/env bash
# run_research_host_battery.sh — WSL host research battery (non-fleet)
#
# Runs Checklist v2, Spec 100 final_score IC, Spec 105 live QA, and optional
# Sci-Cart normalization sample review. Requires production artifacts on host.
#
# Usage:
#   bash tools/run_research_host_battery.sh
#   bash tools/run_research_host_battery.sh 2026-06-24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
DATE="${1:-$(TZ=America/Detroit date +%Y-%m-%d)}"
START_DATE="${RESEARCH_START_DATE:-2024-01-01}"
END_DATE="${RESEARCH_END_DATE:-$DATE}"

log() { echo "[$(date -Iseconds)] research_battery: $*"; }

missing=0
require_path() {
    if [ ! -e "$1" ]; then
        log "MISSING prerequisite: $1"
        missing=1
    fi
}

log "Research host battery for $DATE (Spec 100/105 + Checklist v2)"
log "Governance: research-only — does not modify production scoring"

require_path "$REPO_ROOT/data/snapshots_pit_v2"
require_path "$REPO_ROOT/production_data/price_history.csv"
require_path "$REPO_ROOT/data/snapshots/$DATE/rankings.csv"

if [ "$missing" -ne 0 ]; then
    log "Abort — see docs/research/CHECKLIST_V2_FINAL_SCORE_BLOCKER_2026_06_24.md"
    exit 1
fi

log "Step 1 — build research panel (if stale)"
if [ ! -f "$REPO_ROOT/output/signals/research_panel.csv" ]; then
    $PYTHON scripts/research/build_signal_research_panel.py --no-parquet
else
    log "research_panel.csv exists — skipping build (delete to force rebuild)"
fi

log "Step 2 — Checklist v2 battery"
$PYTHON scripts/research/checklist_v2_rerun.py

log "Step 3 — Spec 100 final_score IC"
$PYTHON tools/measure_final_score_ic_spec100.py \
    --start-date "$START_DATE" --end-date "$END_DATE" \
    --snapshot-dir data/snapshots_pit_v2

log "Step 4 — rank IC backtest (final_score default)"
$PYTHON run_rank_ic_backtest.py --signal final_score --universe eligible || true

log "Step 5 — Spec 105 expectation coverage (live QA)"
$PYTHON tools/verify_expectation_coverage_spec105.py --as-of-date "$DATE" --write
$PYTHON tools/production_qa_check.py --as-of-date "$DATE" || true

log "Step 6 — Sci-Cart normalization sample review (optional)"
if [ -f "$REPO_ROOT/production_data/trial_records.json" ]; then
    $PYTHON tools/sciart_normalization_sample_review.py --write || true
else
    log "skip sciart sample review — production_data/trial_records.json missing"
fi

log "Done — outputs under output/checklist_v2_rerun/, output/dem_ranker_*, artifacts/spec105/"
