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
#   pdufa_extracted  Build extracted PDUFA sidecar (Phase 1, review-only)
#   herald           Fetch + classify company press releases
#   iv               Rebuild historical IV features from surface data
#   universe         Run universe maintenance health check
#   status           Write logs/data_refresh_status_{date}.json summary
#   all              Run all stages (default)
#
# Cron schedule:
#   0 14 * * 1-5  (2:00 PM ET weekdays — 2.5 hours before production)

set -euo pipefail

REPO_ROOT="/mnt/c/Projects/biotech_screener/biotech-screener"
PYTHON="/usr/bin/python3"
LOG_DIR="${REPO_ROOT}/logs"

cd "$REPO_ROOT"
source .env 2>/dev/null || true

TODAY=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] data-refresh: $*"
}

stage_ctgov() {
    log "CTgov warm cache (timeout 600s)..."
    local rc=0
    timeout 600 $PYTHON warm_caches.py --sources ctgov --as-of-date "$TODAY" 2>&1 | tail -5 || rc=$?
    if [ $rc -eq 124 ]; then
        log "CTgov warm TIMED OUT after 600s — continuing with partial data"
    elif [ $rc -ne 0 ]; then
        log "CTgov warm failed (exit $rc) — continuing"
    else
        log "CTgov done"
    fi
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

    # Classify new releases (timeout 300s)
    RELEASES_FILE="data/press_releases/releases_${TODAY}.jsonl"
    if [ -f "$RELEASES_FILE" ]; then
        log "Herald classify (timeout 300s)..."
        local rc2=0
        timeout 300 $PYTHON tools/classify_press_releases.py --input "$RELEASES_FILE" 2>&1 | tail -5 || rc2=$?
        if [ $rc2 -eq 124 ]; then
            log "Herald classify TIMED OUT after 300s"
        else
            log "Herald classify done"
        fi
    else
        log "No new releases file for $TODAY"
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

# SEC 13F PIT cache freshness — feeds B6 selector via institutional_summary.
# No deterministic daily file path; we look for a today-dated subdir under
# data/caches/sec_13f/PIT/ and compute its age. Stale = no subdir within
# the last 3 calendar days. Advisory only — does not gate overall_pass.
sec_13f_pit_dir = repo_root / "data" / "caches" / "sec_13f" / "PIT"
if sec_13f_pit_dir.exists():
    subdirs = sorted(
        (p for p in sec_13f_pit_dir.iterdir() if p.is_dir() and p.name >= "2020"),
        key=lambda p: p.name,
        reverse=True,
    )
    latest = subdirs[0].name if subdirs else None
    latest_age_d = None
    if latest:
        try:
            from datetime import date as _date
            ly, lm, ld = (int(x) for x in latest.split("-"))
            latest_age_d = (_date.fromisoformat(today) - _date(ly, lm, ld)).days
        except Exception:
            latest_age_d = -1
    status["sec_13f_pit"] = {
        "exists": latest is not None,
        "latest_subdir": latest,
        "latest_age_days": latest_age_d,
        "stale": (latest_age_d is None) or (latest_age_d > 3),
    }
else:
    status["sec_13f_pit"] = {
        "exists": False,
        "latest_subdir": None,
        "latest_age_days": None,
        "stale": True,
    }
status["sec_13f_pit_advisory"] = status["sec_13f_pit"]["stale"]

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
    herald)
        stage_herald
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
        stage_pdufa_extracted
        stage_herald
        stage_iv
        stage_universe
        stage_status
        ;;
    *)
        echo "Usage: $0 {ctgov|sec_8k|fda_adcom|fda_regulatory|pdufa_extracted|herald|iv|universe|status|all}"
        exit 1
        ;;
esac

log "done ($MODE)"
