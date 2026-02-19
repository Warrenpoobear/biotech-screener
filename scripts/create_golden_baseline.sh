#!/bin/bash
# create_golden_baseline.sh -- Bootstrap a golden replay baseline
#
# Usage: ./scripts/create_golden_baseline.sh [AS_OF_DATE]
#
# Steps:
#   1. Run pipeline from production_data to produce a replay bundle
#   2. Run pipeline FROM that bundle to produce deterministic baseline rankings
#   3. Copy baseline rankings to golden/baseline_<DATE>/
#   4. Upload bundle to GitHub Release (tag: golden-baseline)
#
# Prerequisites:
#   - Pipeline must be runnable locally (deps installed, production_data/ populated)
#   - gh CLI must be authenticated (gh auth status)
#
# After running:
#   - git add golden/baseline_<DATE>/rankings.csv && git commit && git push
#   - The workflow will pick up the bundle from the release and baseline from the repo

set -euo pipefail

# Use python3 if python isn't available (common on Ubuntu/WSL)
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

DATE="${1:-$(date +%Y-%m-%d)}"
BUNDLE_PATH="/tmp/replay_bundle_${DATE}.tgz"
SNAP_DIR="/tmp/golden_snap"
BASELINE_SNAP="/tmp/golden_baseline"
GOLDEN_DIR="golden/baseline_${DATE}"

echo "=== Golden baseline bootstrap for ${DATE} ==="
echo ""

# ── Step 1: Generate replay bundle from production data ──────────
echo "[1/4] Running pipeline to generate replay bundle..."
mkdir -p "$SNAP_DIR"
$PYTHON run_screen.py \
  --as-of-date "$DATE" \
  --data-dir production_data \
  --output "${SNAP_DIR}/${DATE}/screen_output.json" \
  --decision-mode phase2 \
  --snapshot-dir "$SNAP_DIR" \
  --replay-bundle-out "$BUNDLE_PATH"

if [ ! -f "$BUNDLE_PATH" ]; then
  echo "ERROR: Bundle not created at ${BUNDLE_PATH}"
  exit 1
fi
echo "Bundle created: ${BUNDLE_PATH} ($(du -h "$BUNDLE_PATH" | cut -f1))"
echo ""

# ── Step 2: Run pipeline FROM the bundle (deterministic baseline) ─
echo "[2/4] Running pipeline from bundle to produce baseline rankings..."
mkdir -p "$BASELINE_SNAP"
$PYTHON run_screen.py \
  --replay-bundle "$BUNDLE_PATH" \
  --as-of-date "$DATE" \
  --data-dir /tmp \
  --output "${BASELINE_SNAP}/${DATE}/screen_output.json" \
  --decision-mode phase2 \
  --snapshot-dir "$BASELINE_SNAP"

if [ ! -f "${BASELINE_SNAP}/${DATE}/rankings.csv" ]; then
  echo "ERROR: Baseline rankings not produced at ${BASELINE_SNAP}/${DATE}/rankings.csv"
  exit 1
fi
echo "Baseline rankings produced."
echo ""

# ── Step 3: Copy baseline rankings + write metadata to golden/ ───
echo "[3/4] Copying baseline to ${GOLDEN_DIR}/..."
mkdir -p "$GOLDEN_DIR"
cp "${BASELINE_SNAP}/${DATE}/rankings.csv" "${GOLDEN_DIR}/rankings.csv"

# Write baseline_meta.json sidecar for freshness tracking
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
THRESHOLDS_ID=$($PYTHON -c "
from scripts.replay_diff import DiffThresholds
print(DiffThresholds().thresholds_id)
" 2>/dev/null || echo "unknown")

$PYTHON -c "
import json
meta = {
    'as_of_date': '${DATE}',
    'created_at': '${DATE}',
    'git_sha': '${GIT_SHA}',
    'thresholds_id': '${THRESHOLDS_ID}',
}
with open('${GOLDEN_DIR}/baseline_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
    f.write('\n')
print('Wrote baseline_meta.json')
"
echo "Baseline ready: ${GOLDEN_DIR}/rankings.csv + baseline_meta.json"
echo ""

# ── Step 4: Upload bundle to GitHub Release ──────────────────────
echo "[4/4] Uploading bundle to GitHub Release (tag: golden-baseline)..."

# Delete existing release if present (to replace the bundle)
if gh release view golden-baseline >/dev/null 2>&1; then
  echo "  Existing golden-baseline release found — deleting to replace..."
  gh release delete golden-baseline --yes --cleanup-tag 2>/dev/null || true
fi

gh release create golden-baseline \
  "$BUNDLE_PATH" \
  --title "Golden replay baseline" \
  --notes "As-of date: ${DATE}. Replay bundle for PR regression testing."

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. git add ${GOLDEN_DIR}/rankings.csv ${GOLDEN_DIR}/baseline_meta.json"
echo "  2. git commit -m 'Add golden baseline for ${DATE}'"
echo "  3. git push"
echo ""
echo "The replay-regression workflow will now gate PRs against this baseline."
