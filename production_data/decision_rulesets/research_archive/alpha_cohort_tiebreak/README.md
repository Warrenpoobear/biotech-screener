# Research Archive: Alpha Cohort Tiebreak Sweep

**Date**: 2026-03-04
**Baseline ruleset**: v1.8.3 / ID `82982998` (v1.8.3_buffer30_candidate.json)
**Author**: research sweep via run_audited_backtest.py

## Hypothesis

`alpha_cohort_pct` is unique per ticker (no 0.1-ceiling ties unlike `alpha_raw`) and could
serve as a secondary sort contribution within near-ties. Tested as a small positive-only blend:

    delta = alpha_cohort_tiebreak_weight * alpha_cohort_pct

subtracted from `effective_comp_rank` (higher pct → sorts earlier).

## Sweep

**Window**: OOS 2020-03-31 – 2024-12-31 (282 dates, strict)
**Snapshot root**: `data/snapshots_reranked_baseline_oos`
**Setup**: `--rerank`, buffer=30, top-k=20, cost=30bps, horizons=84+126

| Ruleset file                        | Weight |
|-------------------------------------|--------|
| research_ac_tb_v183_w0p005.json     | 0.005  |
| research_ac_tb_v183_w0p01.json      | 0.010  |
| research_ac_tb_v183_w0p02.json      | 0.020  |

## Results

| Weight | 126d Net | 126d Δ | 84d Net | 84d Δ |
|--------|----------|--------|---------|-------|
| 0.000 (baseline) | 4.749% | — | 4.085% | — |
| 0.005 | 4.749% | +0.000pp | 4.085% | +0.000pp |
| 0.010 | 4.702% | −0.047pp | 4.077% | −0.008pp |
| 0.020 | 4.695% | −0.054pp | 4.065% | −0.020pp |

## Verdict

**ARCHIVED — do not promote.**

Signal is completely flat at w=0.005 (zero measurable effect over 282 dates) and monotonically
negative from w=0.01 onward. The monotone degradation with weight confirms there is no
"right weight" — the signal has no portfolio-level edge. `alpha_cohort_tiebreak_weight` remains
in the DecisionRuleset schema at default 0.0 (schema only, no behavior).

Promotion threshold was: 126d net ≥ +0.20pp AND 84d net ≥ −0.05pp. No weight comes close.

## Audited backtest output

`output/ac_tiebreak_sweep/` — one subdirectory per weight + `compare.md` summary.
