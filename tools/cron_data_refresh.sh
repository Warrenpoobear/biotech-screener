#!/usr/bin/env bash
# cron_data_refresh.sh — Daily data pipeline refresh.
#
# Runs data collection/refresh jobs that feed the production pipeline.
# Designed to run BEFORE the main production screen (16:30 ET).
#
# Stages:
#   ctgov            Warm CTgov trial cache (trial_records.json)
#   sec_8k           Warm SEC 8-K catalyst cache (cache/sec/8k_catalysts/)
#   fda_adcom        Warm FDA AdCom calendar cache (cache/fda/)
#   fda_regulatory   Warm FDA regulatory notices cache (cache/fda/)
#   euctr            Warm EUCTR trial cache (~20 min; NOT in production pipeline)
#   ctis             Warm CTIS trial cache (~17 min; NOT in production pipeline)
#   isrctn           Warm ISRCTN trial cache (~17 min; NOT in production pipeline)
#   merged_trials    Rebuild merged trial cache from ctgov+euctr+ctis+isrctn
#   pdufa_extracted  Build extracted PDUFA sidecar (Phase 1, review-only)
#   herald           Fetch + dedupe + classify company press releases
#   firecrawl        Research-only: search biotech news via Firecrawl (research-only, no alpha)
#   iv               Rebuild historical IV features from surface data
#   universe         Run universe maintenance health check
#   status           Write logs/data_refresh_status_{date}.json summary
#   all              Run all stages (default)
#
# NOTE: euctr/ctis/isrctn/merged_trials are intentionally kept here and NOT in
# cron_daily_production.sh — they take 15-25 min each and would cause the
# production pipeline (step 1.5) to time out. run_daily_production.py defaults
# to essential sources only: sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory.
#
# Cron schedule:
#   0 14 * * 1-5  (2:00 PM ET weekdays — 2.5 hours before production)
#   0 8  * * 1-5  (8:00 AM ET weekdays — second run for firecrawl research discovery)

set -euo pipefail

REPO_ROOT="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG_DIR="${REPO_ROOT}/logs"

cd "$REPO_ROOT"
set -a
source .env 2>/dev/null || true
set +a

# Set PYTHONPATH for imports from repo root (common, tools, etc.)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TODAY=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] data-refresh: $*"
}

stage_ctgov() {
    log "CTgov warm cache..."
    $PYTHON warm_caches.py --sources ctgov --as-of-date "$TODAY" 2>&1 | tail -5
    log "CTgov done"
}

stage_sec_8k() {
    log "SEC 8-K warm cache (timeout 1800s)..."
    local rc=0
    timeout 1800 $PYTHON warm_caches.py --sources sec_8k --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "SEC 8-K warm TIMED OUT after 1800s — partial cache may be present"
    elif [ $rc -ne 0 ]; then
        log "SEC 8-K warm failed (exit $rc) — continuing"
    else
        log "SEC 8-K warm done"
    fi
}

stage_fda_adcom() {
    log "FDA AdCom warm cache (timeout 300s)..."
    local rc=0
    timeout 300 $PYTHON warm_caches.py --sources fda_adcom --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "FDA AdCom warm TIMED OUT after 300s"
    elif [ $rc -ne 0 ]; then
        log "FDA AdCom warm failed (exit $rc) — continuing"
    else
        log "FDA AdCom warm done"
    fi
}

stage_fda_regulatory() {
    log "FDA regulatory notices warm cache (timeout 300s)..."
    local rc=0
    timeout 300 $PYTHON warm_caches.py --sources fda_regulatory --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "FDA regulatory warm TIMED OUT after 300s"
    elif [ $rc -ne 0 ]; then
        log "FDA regulatory warm failed (exit $rc) — continuing"
    else
        log "FDA regulatory warm done"
    fi
}

# --- Slow international trial registries ---
# EUCTR/CTIS/ISRCTN each take 15-25 min. They live here (not in
# cron_daily_production.sh / run_daily_production.py step 1.5) so they do NOT
# block the production pipeline. merged_trials depends on all three being fresh.
stage_euctr() {
    log "EUCTR warm cache (sources: euctr, timeout 1500s)..."
    local rc=0
    timeout 1500 $PYTHON warm_caches.py --sources euctr --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "EUCTR warm TIMED OUT after 1500s — partial cache may be present"
    elif [ $rc -ne 0 ]; then
        log "EUCTR warm failed (exit $rc) — continuing"
    else
        log "EUCTR warm done"
    fi
}

stage_ctis() {
    # 1800s headroom above the collector's internal 900s enrichment budget so a
    # cold-cache run still reaches the atomic cache write (see ctis_collector).
    log "CTIS warm cache (sources: ctis, timeout 1800s)..."
    local rc=0
    timeout 1800 $PYTHON warm_caches.py --sources ctis --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "CTIS warm TIMED OUT after 1800s — partial cache may be present"
    elif [ $rc -ne 0 ]; then
        log "CTIS warm failed (exit $rc) — continuing"
    else
        log "CTIS warm done"
    fi
}

stage_isrctn() {
    log "ISRCTN warm cache (sources: isrctn, timeout 1200s)..."
    local rc=0
    timeout 1200 $PYTHON warm_caches.py --sources isrctn --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "ISRCTN warm TIMED OUT after 1200s — partial cache may be present"
    elif [ $rc -ne 0 ]; then
        log "ISRCTN warm failed (exit $rc) — continuing"
    else
        log "ISRCTN warm done"
    fi
}

stage_merged_trials() {
    log "Merged-trials rebuild (sources: merged_trials, timeout 300s)..."
    local rc=0
    timeout 300 $PYTHON warm_caches.py --sources merged_trials --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "merged_trials rebuild TIMED OUT after 300s"
    elif [ $rc -ne 0 ]; then
        log "merged_trials rebuild failed (exit $rc) — continuing"
    else
        log "merged_trials rebuild done"
    fi
}

stage_pdufa_extracted() {
    log "Building extracted PDUFA sidecar (timeout 120s, Phase 1 review-only)..."
    local rc=0
    timeout 120 $PYTHON -m tools.build_pdufa_dates_extracted --as-of-date "$TODAY" 2>&1 | tail -10 || rc=$?
    if [ $rc -eq 124 ]; then
        log "PDUFA extracted sidecar TIMED OUT after 120s"
    elif [ $rc -ne 0 ]; then
        log "PDUFA extracted sidecar failed (exit $rc) — continuing"
    else
        log "PDUFA extracted sidecar done"
    fi
}

stage_herald() {
    log "Herald press release fetch (timeout 1500s)..."
    local rc=0
    timeout 1500 $PYTHON tools/fetch_company_press_releases.py --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "Herald fetch TIMED OUT after 1500s — continuing with partial data"
    elif [ $rc -ne 0 ]; then
        log "Herald fetch failed (exit $rc) — continuing"
    else
        log "Herald fetch done"
    fi

    RELEASES_FILE="data/press_releases/releases_${TODAY}.jsonl"
    DEDUPED_FILE="data/press_releases/deduped/deduped_${TODAY}.jsonl"
    if [ -f "$RELEASES_FILE" ]; then
        log "Herald dedupe (timeout 120s)..."
        local rc_dedupe=0
        timeout 120 $PYTHON tools/dedupe_press_releases.py --input "$RELEASES_FILE" 2>&1 | tail -5 || rc_dedupe=$?
        if [ $rc_dedupe -eq 124 ]; then
            log "Herald dedupe TIMED OUT after 120s"
        elif [ $rc_dedupe -ne 0 ]; then
            log "Herald dedupe failed (exit $rc_dedupe)"
        else
            log "Herald dedupe done"
        fi
    else
        log "No new releases file for $TODAY — skip dedupe/classify"
        return 0
    fi

    if [ -f "$DEDUPED_FILE" ]; then
        log "Herald classify (timeout 300s)..."
        local rc2=0
        timeout 300 $PYTHON tools/classify_press_releases.py --input "$DEDUPED_FILE" 2>&1 | tail -5 || rc2=$?
        if [ $rc2 -eq 124 ]; then
            log "Herald classify TIMED OUT after 300s"
        else
            log "Herald classify done"
        fi
    else
        log "No deduped file for $TODAY — classify skipped"
    fi

    log "Herald health check..."
    local rc_health=0
    $PYTHON tools/herald_health_check.py --as-of-date "$TODAY" 2>&1 | tail -5 || rc_health=$?
    if [ $rc_health -eq 0 ]; then
        log "Herald health: OK"
    elif [ $rc_health -eq 1 ]; then
        log "Herald health: WARN (exit 1)"
    else
        log "Herald health: FAIL (exit $rc_health)"
    fi
}

stage_firecrawl() {
    log "Firecrawl research discovery (timeout 180s)..."

    # Check if API key is available
    if [ -z "${FIRECRAWL_API_KEY:-}" ]; then
        log "Firecrawl research skipped: FIRECRAWL_API_KEY not set in environment"
        return 0
    fi

    local rc=0
    timeout 180 $PYTHON tools/firecrawl_research_ingest.py \
        --query "GLP-1 obesity drug clinical trial (site:biopharmadive.com OR site:fiercebiotech.com OR site:endpoints.news OR site:statnews.com OR site:biospace.com OR site:xconomy.com)" \
        --limit 20 \
        --timeout 30 \
        --out "artifacts/research/firecrawl/$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "Firecrawl research TIMED OUT after 180s"
    elif [ $rc -ne 0 ]; then
        log "Firecrawl research failed (exit $rc) — continuing"
    else
        log "Firecrawl research done → artifacts/research/firecrawl/$TODAY/"
    fi
}

stage_iv() {
    log "IV features rebuild..."
    $PYTHON scripts/research/build_historical_iv_features.py 2>&1 | tail -5
    log "IV features done"
}

stage_universe() {
    log "Universe maintenance..."
    $PYTHON tools/build_universe_maintenance.py --as-of-date "$TODAY" 2>&1 | tail -5
    log "Universe done"
}

stage_status() {
    log "Writing status report..."
    $PYTHON - "$REPO_ROOT" "$TODAY" <<'PYEOF'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(sys.argv[1])
today = sys.argv[2]


def count_events(p):
    if p is None or not p.exists():
        return -1
    try:
        with open(p) as fh:
            data = json.load(fh)
        return len(data) if isinstance(data, list) else len(data.keys())
    except Exception:
        return -2


# SEC 8-K cache filename includes PATTERN_VERSION; glob and pick newest.
sec_matches = sorted(
    repo_root.glob(f"cache/sec/8k_catalysts/8k_catalysts_{today}_*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
sec_latest = sec_matches[0] if sec_matches else None
adcom_path = repo_root / f"cache/fda/adcom_calendar_{today}.json"
reg_path = repo_root / f"cache/fda/fda_regulatory_{today}.json"
press_path = repo_root / f"data/press_releases/releases_{today}.jsonl"
ctgov_path = repo_root / f"cache/ctgov/trial_records_{today}.json"
pdufa_extracted_path = repo_root / "production_data" / "pdufa_dates_extracted.json"
pdufa_dated_path = repo_root / "artifacts" / "regulatory" / f"pdufa_dates_extracted_{today}.json"
pdufa_diff_md = repo_root / "artifacts" / "regulatory" / f"pdufa_extracted_vs_canonical_{today}.md"

status = {
    "as_of_date": today,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "sec_8k": {
        "exists": sec_latest is not None,
        "path": str(sec_latest) if sec_latest else None,
        "event_count": count_events(sec_latest),
        "pattern_version": (
            sec_latest.stem.rsplit("_", 1)[-1] if sec_latest else None
        ),
    },
    "fda_adcom": {
        "exists": adcom_path.exists(),
        "path": str(adcom_path),
        "event_count": count_events(adcom_path),
    },
    "fda_regulatory": {
        "exists": reg_path.exists(),
        "path": str(reg_path),
        "event_count": count_events(reg_path),
    },
    "ctgov": {
        "exists": ctgov_path.exists(),
        "path": str(ctgov_path),
        "event_count": count_events(ctgov_path),
    },
    "press_releases": {
        "exists": press_path.exists(),
        "path": str(press_path),
        "line_count": (
            sum(1 for _ in open(press_path)) if press_path.exists() else -1
        ),
    },
    "pdufa_extracted": {
        "latest_exists": pdufa_extracted_path.exists(),
        "latest_path": str(pdufa_extracted_path),
        "latest_record_count": count_events(pdufa_extracted_path),
        "dated_exists": pdufa_dated_path.exists(),
        "dated_path": str(pdufa_dated_path),
        "diff_md_exists": pdufa_diff_md.exists(),
        "diff_md_path": str(pdufa_diff_md),
    },
}

# Diff bucket counts (best-effort: parse the dated extracted snapshot for
# event_status breakdown; full bucket counts live in the diff MD).
if pdufa_dated_path.exists():
    try:
        with open(pdufa_dated_path) as fh:
            dated_records = json.load(fh)
        status["pdufa_extracted"]["status_breakdown"] = {
            s: sum(1 for r in dated_records if r.get("event_status") == s)
            for s in ("upcoming", "extended", "resubmission_accepted")
        }
    except Exception:
        status["pdufa_extracted"]["status_breakdown"] = {}

# Overall pass/fail: every required cache exists with > 0 events.
required = ["sec_8k", "fda_adcom", "fda_regulatory"]
status["overall_pass"] = all(
    status[k]["exists"] and status[k].get("event_count", 0) > 0 for k in required
)

out = repo_root / "logs" / f"data_refresh_status_{today}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(status, indent=2))
print(json.dumps(status, indent=2))
PYEOF
    log "Status report → logs/data_refresh_status_${TODAY}.json"
}

MODE="${1:-all}"

case "$MODE" in
    ctgov)
        stage_ctgov
        ;;
    sec_8k)
        stage_sec_8k
        ;;
    fda_adcom)
        stage_fda_adcom
        ;;
    fda_regulatory)
        stage_fda_regulatory
        ;;
    pdufa_extracted)
        stage_pdufa_extracted
        ;;
    euctr)
        stage_euctr
        ;;
    ctis)
        stage_ctis
        ;;
    isrctn)
        stage_isrctn
        ;;
    merged_trials)
        stage_merged_trials
        ;;
    herald)
        stage_herald
        ;;
    firecrawl)
        stage_firecrawl
        ;;
    iv)
        stage_iv
        ;;
    universe)
        stage_universe
        ;;
    status)
        stage_status
        ;;
    all)
        stage_ctgov
        stage_sec_8k
        stage_fda_adcom
        stage_fda_regulatory
        # Slow international registries run here (not in production pipeline) to
        # avoid blocking step 1.5 of run_daily_production.py. Each has its own
        # timeout; failures are non-fatal and logged. merged_trials runs last as
        # it depends on euctr/ctis/isrctn caches being present.
        stage_euctr
        stage_ctis
        stage_isrctn
        stage_merged_trials
        stage_pdufa_extracted
        stage_herald
        stage_firecrawl
        stage_iv
        stage_universe
        stage_status
        ;;
    *)
        echo "Usage: $0 {ctgov|sec_8k|fda_adcom|fda_regulatory|euctr|ctis|isrctn|merged_trials|pdufa_extracted|herald|firecrawl|iv|universe|status|all}"
        exit 1
        ;;
esac

log "done ($MODE)"
