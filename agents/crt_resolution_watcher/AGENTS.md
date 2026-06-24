# AGENTS.md — CRT Resolution Watcher

## Dependencies

- CRT pipeline (Step 5m in daily production) creates resolution files
- Postmortem agent may create resolution records

## Downstream consumers

- Asymmetry score (via event_move_table → EPD implied_vs_realized)
- Event analyst agent (reads postmortem/resolution records)
- Calibration agent (reads CRT calibration rollup)

## Self-learning (Rule 12)

Resolution/calibration disagreements → `.learnings/LEARNINGS.md` with `Promotion-lane: spec`.
