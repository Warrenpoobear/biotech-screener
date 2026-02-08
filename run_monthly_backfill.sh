#!/usr/bin/env bash
# =============================================================================
# Monthly Archive Backfill
# =============================================================================
# Runs the scoring pipeline for 22 monthly dates (2024-01 through 2025-10),
# creating archives for each.  All dates clear the 60-trading-day forward
# return fence (latest usable as-of: ~2025-11-13).
#
# PIT caveat: fundamentals/trials/13F use current-state production_data
# with --pit-mode degrade.  Prices/timing are true as-of.  This is
# "PIT-lite" — sufficient for directional A/B signal comparison and
# regime/stage attribution, but not strict PIT claims.
#
# Usage:
#   bash run_monthly_backfill.sh                # Run all 22 dates
#   bash run_monthly_backfill.sh --dry-run      # Validate only, no scoring
#   bash run_monthly_backfill.sh --start 2024-06 --end 2025-03  # Subset
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="production_data"
SNAPSHOT_DIR="data/snapshots"
ARCHIVE_DIR="data/archives"
OUTPUT_DIR="backfill_results"
LOG_DIR="backfill_logs"

DRY_RUN=false
START_FILTER=""
END_FILTER=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=true; shift ;;
        --start)     START_FILTER="$2"; shift 2 ;;
        --end)       END_FILTER="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# End-of-month dates: 2024-01 through 2025-10
# Using last business day of each month (approx)
ALL_DATES=(
    "2024-01-31"
    "2024-02-29"
    "2024-03-29"
    "2024-04-30"
    "2024-05-31"
    "2024-06-28"
    "2024-07-31"
    "2024-08-30"
    "2024-09-30"
    "2024-10-31"
    "2024-11-29"
    "2024-12-31"
    "2025-01-31"
    "2025-02-28"
    "2025-03-31"
    "2025-04-30"
    "2025-05-30"
    "2025-06-30"
    "2025-07-31"
    "2025-08-29"
    "2025-09-30"
    "2025-10-31"
)

# Apply date filters if specified
DATES=()
for d in "${ALL_DATES[@]}"; do
    ym="${d:0:7}"  # YYYY-MM
    if [[ -n "$START_FILTER" && "$ym" < "$START_FILTER" ]]; then continue; fi
    if [[ -n "$END_FILTER" && "$ym" > "$END_FILTER" ]]; then continue; fi
    DATES+=("$d")
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$ARCHIVE_DIR"

echo "============================================================"
echo "MONTHLY ARCHIVE BACKFILL"
echo "============================================================"
echo "  Data dir:    $DATA_DIR"
echo "  Archive dir: $ARCHIVE_DIR"
echo "  Dates:       ${#DATES[@]}"
echo "  Dry run:     $DRY_RUN"
if [[ -n "$START_FILTER" || -n "$END_FILTER" ]]; then
    echo "  Filter:      ${START_FILTER:-*} to ${END_FILTER:-*}"
fi
echo "  Range:       ${DATES[0]} to ${DATES[-1]}"
echo "============================================================"
echo ""

# Skip dates that already have archives
TODO_DATES=()
SKIPPED=0
for d in "${DATES[@]}"; do
    if [[ -f "$ARCHIVE_DIR/${d}.tar.gz" ]]; then
        echo "  SKIP $d  (archive exists)"
        SKIPPED=$((SKIPPED + 1))
    else
        TODO_DATES+=("$d")
    fi
done
echo ""
echo "  To run:  ${#TODO_DATES[@]}  (skipped $SKIPPED existing)"
echo ""

if [[ ${#TODO_DATES[@]} -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi

SUCCESS=0
FAIL=0
FAIL_DATES=()

for d in "${TODO_DATES[@]}"; do
    echo "============================================================"
    echo "[$d] Starting..."
    echo "============================================================"

    OUTPUT_FILE="$OUTPUT_DIR/results_${d}.json"
    LOG_FILE="$LOG_DIR/backfill_${d}.log"

    if $DRY_RUN; then
        echo "  DRY RUN: python3 run_screen.py --as-of-date $d --data-dir $DATA_DIR --pit-mode degrade --output $OUTPUT_FILE"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    # Step 1: Run the scoring pipeline
    if python3 run_screen.py \
        --as-of-date "$d" \
        --data-dir "$DATA_DIR" \
        --pit-mode degrade \
        --output "$OUTPUT_FILE" \
        --snapshot-dir "$SNAPSHOT_DIR" \
        --log-level WARNING \
        2>&1 | tee "$LOG_FILE"; then

        echo "  Screen OK"

        # Step 2: Create archive
        if python3 archive_snapshot.py \
            --date "$d" \
            --snapshot-dir "$SNAPSHOT_DIR" \
            --data-dir "$DATA_DIR" \
            --archive-dir "$ARCHIVE_DIR" \
            2>&1 | tee -a "$LOG_FILE"; then

            echo "  Archive OK: $ARCHIVE_DIR/${d}.tar.gz"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  FAIL: archive creation failed for $d"
            FAIL=$((FAIL + 1))
            FAIL_DATES+=("$d")
        fi
    else
        echo "  FAIL: scoring failed for $d"
        FAIL=$((FAIL + 1))
        FAIL_DATES+=("$d")
    fi

    echo ""
done

# Summary
echo "============================================================"
echo "BACKFILL COMPLETE"
echo "============================================================"
echo "  Success:  $SUCCESS"
echo "  Failed:   $FAIL"
echo "  Skipped:  $SKIPPED"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed dates: ${FAIL_DATES[*]}"
fi
echo ""
echo "Next steps:"
echo "  1. Verify archives:  python3 archive_snapshot.py --list"
echo "  2. Run backtest:     python3 run_rank_ic_backtest.py"
echo "  3. A/B comparison:   python3 run_rank_ic_backtest.py --compare-signals score_rank_pct,score_rank_pct_attn"
echo ""
