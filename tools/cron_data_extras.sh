#!/usr/bin/env bash
# cron_data_extras.sh — Scheduled refresh for data sources that previously
# required manual runs (Form 4, short interest, PIT financials, Purple Book).
#
# Companion to cron_data_refresh.sh. Runs BEFORE that script so its outputs
# are available to the 14:00 ctgov/herald/iv/universe pipeline and the 16:30
# production screen.
#
# Stages:
#   form4        Incremental SEC Form 4 fetch (insider transactions)
#   short        Yahoo Finance short interest snapshot
#   pit_fin      Incremental SEC EDGAR XBRL facts refresh (per-ticker PIT)
#   fin_records  Aggregate financial_records.json refresh (base layer for PIT override)
#   burn         Quarterly cash burn history (Module 2 burn-acceleration)
#   purple       FDA Purple Book download + ingest (biologics competition)
#   all          Run all stages (default)
#
# Recommended cron schedule (proposed — NOT yet installed):
#   # Daily incrementals — 13:30 ET, before cron_data_refresh.sh at 14:00
#   30 13 * * 1-5 /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_data_extras.sh form4 short pit_fin >> /mnt/c/Projects/biotech_screener/biotech-screener/logs/data_extras.log 2>&1
#   # Weekly — Monday 13:00 ET (Purple Book updates monthly; weekly is plenty)
#   0  13 * * 1   /mnt/c/Projects/biotech_screener/biotech-screener/tools/cron_data_extras.sh purple >> /mnt/c/Projects/biotech_screener/biotech-screener/logs/data_extras.log 2>&1

set -euo pipefail

REPO_ROOT="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG_DIR="${REPO_ROOT}/logs"

cd "$REPO_ROOT"
source .env 2>/dev/null || true

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] data-extras: $*"
}

stage_form4() {
    # 2026-05-01 operational repair (Spec 065 stable-snapshot gate):
    # Switched from fetch_form4_bulk.py to fetch_form4_insider.py (incremental).
    # Bulk re-downloaded all historical filings per ticker (~72s/ticker × 326 = ~6.5h),
    # consistently timing out at 1800s after only ~25 tickers and skipping the
    # state-save + panel-rebuild steps. Incremental fetches only new accessions
    # per ticker, scaling with new-filing volume rather than total history.
    # See FORM4_OPERATIONAL_REPAIR_2026_05_01.md.
    log "Form 4 incremental fetch (timeout 2400s)..."
    local rc=0
    timeout 2400 $PYTHON tools/fetch_form4_insider.py 2>&1 | tail -10 || rc=$?
    if [ $rc -eq 124 ]; then
        log "Form 4 fetch TIMED OUT after 2400s — continuing with partial data"
    elif [ $rc -ne 0 ]; then
        log "Form 4 fetch failed (exit $rc) — continuing"
    else
        log "Form 4 fetch done"
    fi

    # Always rebuild panel from current raw files, even if the main fetch
    # timed out or hit partial failure. Pass B enrichment (rankings-assembly)
    # reads from the panel; a stale panel hides the partial-fetch state.
    log "Form 4 panel rebuild (timeout 300s)..."
    local prc=0
    timeout 300 $PYTHON tools/fetch_form4_insider.py --panel-only 2>&1 | tail -3 || prc=$?
    if [ $prc -ne 0 ]; then
        log "Form 4 panel rebuild failed (exit $prc) — panel may be stale"
    else
        log "Form 4 panel rebuild done"
    fi
}

stage_short() {
    log "Short interest snapshot (timeout 900s)..."
    local rc=0
    timeout 900 $PYTHON collect_short_interest.py 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "Short interest TIMED OUT after 900s"
    elif [ $rc -ne 0 ]; then
        log "Short interest failed (exit $rc) — continuing"
    else
        log "Short interest done"
    fi
}

stage_pit_fin() {
    log "PIT financials incremental refresh (timeout 2400s)..."
    local rc=0
    timeout 2400 $PYTHON tools/build_pit_financials.py --workers 3 2>&1 | tail -10 || rc=$?
    if [ $rc -eq 124 ]; then
        log "PIT financials TIMED OUT after 2400s — continuing with partial data"
    elif [ $rc -ne 0 ]; then
        log "PIT financials failed (exit $rc) — continuing"
    else
        log "PIT financials done"
    fi
}

stage_fin_records() {
    log "Aggregate financial_records.json refresh (timeout 1800s)..."
    local rc=0
    timeout 1800 $PYTHON collect_financial_data.py --universe production_data/universe.json 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "fin_records TIMED OUT after 1800s"
    elif [ $rc -ne 0 ]; then
        log "fin_records failed (exit $rc) — continuing"
    else
        log "fin_records done"
    fi
}

stage_burn() {
    log "Quarterly burn history (timeout 1200s)..."
    local rc=0
    timeout 1200 $PYTHON fetch_quarterly_burn.py 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "burn TIMED OUT after 1200s"
    elif [ $rc -ne 0 ]; then
        log "burn failed (exit $rc) — continuing"
    else
        log "burn done"
    fi
}

stage_purple() {
    log "Purple Book download + ingest (timeout 600s)..."
    local rc=0
    timeout 600 $PYTHON scripts/ingest_purple_book.py --download 2>&1 | tail -10 || rc=$?
    if [ $rc -eq 124 ]; then
        log "Purple Book TIMED OUT after 600s"
    elif [ $rc -ne 0 ]; then
        log "Purple Book failed (exit $rc) — continuing"
    else
        log "Purple Book done"
    fi
}

if [ $# -eq 0 ]; then
    set -- all
fi

for mode in "$@"; do
    case "$mode" in
        form4)       stage_form4 ;;
        short)       stage_short ;;
        pit_fin)     stage_pit_fin ;;
        fin_records) stage_fin_records ;;
        burn)        stage_burn ;;
        purple)      stage_purple ;;
        all)
            stage_form4
            stage_short
            stage_pit_fin
            stage_fin_records
            stage_burn
            stage_purple
            ;;
        *)
            echo "Usage: $0 {form4|short|pit_fin|fin_records|burn|purple|all} [...]"
            exit 1
            ;;
    esac
done

log "done ($*)"
