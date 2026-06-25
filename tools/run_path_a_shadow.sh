#!/usr/bin/env bash
# run_path_a_shadow.sh — Path A timing gates shadow run (Spec 106 A0)
#
# Uses portfolio_policy_path_a_shadow.json (gates enabled). Does NOT change production policy.
#
# Writes:
#   artifacts/live_shadow_path_a/positions/{date}.json
#   artifacts/portfolio_construction/{date}_path_a_manifest.json
#
# Usage:
#   bash tools/run_path_a_shadow.sh 2026-06-24
#   bash tools/run_path_a_shadow.sh 2026-06-24 --write   # alias (manifest auto-written)
#
# Daily (post-production on WSL):
#   bash tools/run_path_a_shadow.sh $(date +%Y-%m-%d)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATE="${1:-}"
shift || true
if [[ -z "$DATE" ]]; then
  echo "Usage: bash tools/run_path_a_shadow.sh YYYY-MM-DD [--write]" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
POLICY="${REPO_ROOT}/production_data/portfolio_policy_path_a_shadow.json"
WRITE=0
for arg in "$@"; do
  if [[ "$arg" == "--write" ]]; then
    WRITE=1
  fi
done

log() { echo "[$(date -Iseconds)] path_a_shadow: $*"; }

log "Path A shadow portfolio for $DATE (policy: portfolio_policy_path_a_shadow.json)"
$PYTHON tools/live_shadow_portfolio.py \
  --as-of-date "$DATE" \
  --policy "$POLICY" \
  --out-dir "${REPO_ROOT}/artifacts/live_shadow_path_a"

MANIFEST="${REPO_ROOT}/artifacts/portfolio_construction/${DATE}_path_a_manifest.json"
if [[ -f "$MANIFEST" ]]; then
  log "Manifest: $MANIFEST"
elif [[ "$WRITE" -eq 1 ]]; then
  log "WARN: expected manifest missing at $MANIFEST"
  exit 1
fi
