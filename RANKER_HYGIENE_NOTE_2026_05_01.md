# Ranker Hygiene Note — 2026-05-01

Short hygiene patch landed alongside the vNext design + first-candidate work. **No production ranker / selector behavior changes.**

## What changed

### 1. Spearman tied-constant bug fixed (1 file, 1 function)
`scripts/research/ees_validation_table.py:_spearman` previously used competitive ranking (1..n by stable-sort order) without a tied-constant guard. When all `xs` (or all `ys`) are equal — which happens on **incomplete production days** where `inst_delta_z` falls back to `0.0` for every row — the function returned a numerically meaningless rank correlation rather than `None`.

Fix: added explicit `len(set(xs)) <= 1 or len(set(ys)) <= 1` guard before ranking.

### 2. Tests added
`tests/test_ees_validation_table.py` — 4 new focused tests:
- `test_spearman_tied_constant_x_returns_none`
- `test_spearman_tied_constant_y_returns_none`
- `test_spearman_both_constant_returns_none`
- `test_spearman_normal_inputs_unaffected` (regression: monotonic + imperfect cases still work)

All 16 tests pass (was 12).

## What surfaced this fix

The 2026-04-30 ranker dominance audit observed an apparent "regime" on 2026-04-07/08/11/12: ρ(coinvest_score_z, final_score) spiked from ~+0.88 to ~+0.96, ρ(inst_delta_z, final_score) flipped from ~+0.27 to ~-0.74, top-30 jaccard with coinvest-only-top-30 jumped from ~57% to ~87%.

Investigation showed those 4 days were **incomplete production runs**:
- 43–46 files in snapshot dir vs 63–67 on normal days
- `institutional_summary.json` and `institutional_summary_delta.json` missing
- All 297 `inst_delta_z` values = 0.0 (sd=0)
- `ACTION.json`, `inputs_manifest.json`, `drift_report.json`, `audit/` all missing

When `institutional_summary_delta.json` is absent, rankings.csv emits `inst_delta_z=0.0` for every row (graceful fallback). The ranker then falls back to ordering by `coinvest_score_z + financial_score` alone, which produces the high coinvest-correlation pattern we initially mistook for a regime.

The "ρ(inst_delta_z, final_score) = -0.74" reading on those days was a **numerical artifact** of computing rank correlation against a constant variable. With a properly-handled Spearman (returning `None` on tied-constant inputs), no apparent correlation surfaces — which is the correct behavior.

## Three findings worth pinning

1. **The regime was a data outage, not a model regime.** Watchdog phase-2 recovery (commit referenced in `project_watchdog_recovery_restored_2026_04_24.md`) addressed the underlying cron-failure modes. Verified no recurrences through 2026-04-30.

2. **Dominance-audit conclusion is robust to the outage.** Excluding the 4 outage days, the recent regime (2026-04-13 → 2026-04-30, 15 clean snapshots) gives:
   - Median ρ(coinvest_score_z, selector_score) = +0.887
   - Median ρ(coinvest_score_z, final_score) = +0.882
   - Median Q2 jaccard (coinvest-only top-30 vs production top-30) = 57%
   - The qualitative finding ("coinvest dominates broad ordering, ranker re-ranks within") holds with or without the outage days. The fake regime did NOT change the headline conclusion.

3. **The Spearman bug is now fixed in the validation table module only.** Other `spearman_ic` helpers across `scripts/research/` were not audited in this pass. The canonical `pit_backtest_ees_v2.py:_spearman_ic` is correct (uses average ranks + explicit `len(set(...))<3` degeneracy check). A future hygiene sweep should check the remaining helpers — most are research scripts scope-bounded to their own questions, but if any is a downstream consumer of validation-table outputs, the same bug pattern could affect their interpretation of degenerate snapshots.

## Forward-return test status

The production-vs-coinvest-only top-30 forward-return test (the natural follow-up to the dominance audit) is **now unblocked** but **not run in this patch**. Will be initiated in a separate change once this hygiene commit is clean.

## Out of scope for this patch

- No production ranker / selector / sizing changes
- No EES promotion (lane closed per `ees_v3_structural_failure_2026_04_30.md`)
- No Form 4 / 13F / options / catalyst work
- No sweep of the other 25+ Spearman implementations

## Future audit checklist (for whoever next investigates a "regime" spike)

When you see Spearman(coinvest_score_z, final_score) ≥ +0.95, or Q2 jaccard
spiking to 80%+:

1. Count files in `data/snapshots/<date>/` (normal day = 60+)
2. Check for `institutional_summary.json` AND `institutional_summary_delta.json`
3. Compute `sd` of `inst_delta_z` across all 297 rows. If ≈ 0, the producer didn't run.
4. If outage confirmed: exclude the day, do NOT label it as a regime.

See memory: `incomplete_production_run_fallback_2026_05_01.md`.
