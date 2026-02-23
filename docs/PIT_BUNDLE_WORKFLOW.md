# PIT Bundle Workflow

Point-in-time (PIT) bundle workflow for reproducible backtests and research evals.

## Pipeline Overview

```
1. Build CTgov PIT caches   (per-date trial_records snapshots)
2. Build 13F PIT caches     (per-date institutional holdings)
3. Build PIT bundles         (clinical + catalyst + coinvest features per date)
4. Run screen from bundles   (decision engine ranking from frozen features)
5. Evaluate forward returns  (IC, L/S, excess returns vs benchmark)
```

## Step-by-Step Commands

### 1. CTgov PIT Cache Backfill

```bash
python3 tools/backfill_ctgov_pit_history.py \
    --date-from 2020-03-31 --date-to 2025-12-31 \
    --cadence monthly --out-root cache/ctgov --resume
```

### 2. 13F PIT Cache Warm

```bash
# Single date
python3 tools/warm_13f_cache.py --as-of-date 2025-06-30

# Batch (quarterly)
python3 tools/warm_13f_cache.py \
    --date-from 2020-03-31 --date-to 2025-12-31 --cadence quarterly
```

### 3. Build PIT Bundles

```bash
# Batch build from existing caches
python3 scripts/build_pit_bundle.py --batch \
    --bundle-root data/bundles/PIT \
    --pit-mode strict \
    --coinvest-carry-forward last_available

# Single date
python3 scripts/build_pit_bundle.py \
    --as-of-date 2025-06-30 \
    --bundle-root data/bundles/PIT \
    --pit-mode strict
```

### 4. Run Screen from Bundles

```bash
# Default (pinned production ruleset)
python3 scripts/run_screen_from_bundle.py \
    --bundle-root data/bundles/PIT \
    --out-root data/snapshots_pit

# With custom ruleset
python3 scripts/run_screen_from_bundle.py \
    --bundle-root data/bundles/PIT \
    --out-root data/snapshots_pit_candidate \
    --ruleset-path production_data/decision_rulesets/my_candidate.json
```

### 5. Evaluate Forward Returns

```bash
python3 scripts/eval_forward_returns.py \
    --snapshot-root data/snapshots_pit \
    --price-csv production_data/price_history.csv \
    --horizons 5,20,63 --top-k 20 --cost-bps 30 \
    --anchor-mode next_trading_day --benchmark XBI \
    --long-short-deciles --component-eval \
    --out-dir output/eval_pit
```

## Research Scripts

Located in `scripts/research/` (not for production):

| Script | Purpose |
|--------|---------|
| `compare_coinvest_modes.py` | 3-way coinvest signal diagnosis (default/off/contra) |
| `component_ic_attribution.py` | Standalone per-signal IC attribution |
| `regenerate_snapshots.py` | Batch re-run of run_screen.py for historical dates |
| `rerank_snapshots.py` | Re-rank existing snapshots through a different ruleset |

## Output Artifacts

- `data/bundles/PIT/{date}/manifest.json` — build provenance + SHA-256 per component
- `data/snapshots_pit/{date}/rankings.csv` — PIT-frozen rankings
- `output/eval_*/summary.json` — IC, excess return, L/S metrics
- `output/eval_*/component_eval_summary.md` — per-component signal attribution
