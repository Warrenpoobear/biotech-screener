#!/usr/bin/env bash
# run_path_a_shadow.sh — Path A timing gates shadow run (Spec 106 A0)
#
# Uses portfolio_policy_path_a_shadow.json (gates enabled). Does NOT change production policy.
#
# Usage:
#   bash tools/run_path_a_shadow.sh 2026-06-24
#   bash tools/run_path_a_shadow.sh 2026-06-24 --write

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

echo "[$(date -Iseconds)] Path A shadow portfolio for $DATE"
$PYTHON tools/live_shadow_portfolio.py \
  --as-of-date "$DATE" \
  --policy "$POLICY" \
  --out-dir "${REPO_ROOT}/artifacts/live_shadow_path_a"

if [[ "$WRITE" -eq 1 ]]; then
  $PYTHON tools/path_a_timing_gates.py \
    --as-of-date "$DATE" \
    --policy "$POLICY" \
    --write
fi
