# Checklist v2 vs final_score — run blocker (2026-06-24)

**Status:** BLOCKED in cloud agent (missing production artifacts). **Operator action on WSL/host.**

Ranker IC on `final_score` stays **UNMEASURED** until this battery runs on a host with PIT snapshots and price history.

## Prerequisites (host)

| Artifact | Path |
| --- | --- |
| PIT snapshot archive | `data/snapshots_pit_v2/` (or `--snapshot-dir` override) |
| Price history | `production_data/price_history.csv` |
| Research panel (full Checklist v2) | built to `output/signals/research_panel.csv` |

## Commands (run on WSL after prerequisites present)

```bash
# Unified setup (recommended)
bash tools/run_operator_host_setup.sh

# Or one-shot research battery only
bash tools/run_research_host_battery.sh

# Or step-by-step:
# 1. Build research panel (if stale)
python3 scripts/research/build_signal_research_panel.py --no-parquet

# 2. Full Checklist v2 battery (Queues A–C)
python3 scripts/research/checklist_v2_rerun.py

# 3. Ranker IC on production final_score (Spec 100)
python3 tools/measure_final_score_ic_spec100.py \
  --start-date 2024-01-01 --end-date 2026-06-18 \
  --snapshot-dir data/snapshots_pit_v2

# 4. Spec 105 live QA artifact
python3 tools/verify_expectation_coverage_spec105.py --as-of-date $(date +%Y-%m-%d) --write
```

## Cloud verification (2026-06-24)

- `data/snapshots_pit_v2/`: absent
- `production_data/price_history.csv`: absent
- `measure_final_score_ic_spec100.py --dry-run`: 0 snapshots found
- `build_signal_research_panel.py`: `FileNotFoundError` on price_history

## Output location (after successful host run)

- Checklist v2: `output/checklist_v2_rerun/`
- Spec 100 IC: `output/dem_ranker_phase_2b_final_score_ic_summary.json` (default field)
- Spec 105: `artifacts/spec105/{date}_coverage.json`

## Governance

Research-only. Does not modify ranker/selector/sizing/production scoring. Results feed promotion evidence only.
