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
# 1. Build research panel (if stale)
python3 scripts/research/build_signal_research_panel.py --no-parquet

# 2. Full Checklist v2 battery (Queues A–C)
python3 scripts/research/checklist_v2_rerun.py

# 3. Ranker IC on production final_score (Spec 100)
python3 tools/measure_final_score_ic_spec100.py \
  --start-date 2024-01-01 --end-date 2026-06-18 \
  --snapshot-dir data/snapshots_pit_v2

# 4. Optional: rank IC backtest default (final_score)
python3 run_rank_ic_backtest.py --signal final_score
```

## Cloud verification (2026-06-24)

- `data/snapshots_pit_v2/`: absent
- `production_data/price_history.csv`: absent
- `measure_final_score_ic_spec100.py --dry-run`: 0 snapshots found
- `build_signal_research_panel.py`: `FileNotFoundError` on price_history

## Output location (after successful host run)

- Checklist v2: `output/checklist_v2_rerun/`
- Spec 100 IC: `output/dem_ranker_phase_2b_final_score_ic_summary.json` (default field)

## Governance

Research-only. Does not modify ranker/selector/sizing/production scoring. Results feed promotion evidence only.
