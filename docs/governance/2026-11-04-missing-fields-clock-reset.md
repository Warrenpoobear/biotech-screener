# Missing-field fixes (health coverage, CT.gov refresh, inst_delta_z transition guard) — out-of-sample clock reset

**Date:** 2026-11-04
**Classification:** Tier 3 overall (highest affected tier governs).
- CT.gov source refresh + `ctgov_content_freshness` gate: Tier 3. It changes clinical/catalyst inputs and adds a blocking production gate.
- `inst_delta_z` registry-transition guard: Tier 3 surface (`run_screen.py` scoring inputs), but no ranking change under the live ruleset (§3).
- Health price-coverage denominator: Tier 1 tooling.

**Authorization:** operator (Darren Schulz), in session 2026-11-04. The operator chose to fix all four gaps found in the 2026-11-03 snapshot and approved the health metric fix, the universe status updates, the CT.gov refresh in cron, and zeroing `inst_delta_z` on a roster change.
**Protocol:** `docs/FORWARD_VALIDATION_PROTOCOL.md` §1 — an authorized bug fix during the NO_MODEL_CHANGE window resets the out-of-sample clock.
**Hash rotation:** none. `model_hash 827c35a9ed3ee6e1` covers `ranker_engine.py`, `selector_engine.py` and `decision_engine.py`, none of which change. Ruleset `8887576e` is unchanged.

---

## 1. Defects

| # | Defect | Evidence | Effect |
|---|---|---|---|
| M1 | Health price coverage counted retired names and the `_XBI_BENCHMARK_` sentinel | `tools/build_data_collection_health.py::_check_market_data` used every `universe.json` entry; the production gate `check_market_data_coverage` already excludes both | `data_collection_health` FAIL at 94.4% (337/357) while the gate passed (342/342) |
| M2 | `production_data/trial_records.json` never refreshed | newest `last_update_posted` = 2026-09-14; `cron_data_refresh.sh stage_ctgov` only ran `warm_caches`, which PIT-filters the stale file into a new `trial_records_<date>.json` daily; `ctgov_state` files 2026-10-19..11-03 byte-identical | diff-based CTGov catalyst events = 0 every day since **2026-08-07**; `source_freshness` passed because it ages the cache by filename date |
| M3 | Registry-transition guard left `inst_delta_z` live | 2026-11-03: current summary 30 managers vs prior (2026-08-13) 49; guard blanked only `inst_flow_*` diagnostics; 189/298 tickers net exits, mean net −1.34, 78.5% nonzero (coverage guard did not trip) | contaminated `inst_delta_z` in rankings.csv and every downstream reader (§3) |
| M4 | Massive (Polygon) options data stopped 2026-05-08 | `logs/massive_options_fetch.log`: `MASSIVE_S3_ACCESS_KEY_ID ... must be set`; backfill log: S3 403 | `pre_event_put_call_ratio` 0% since 05-22; `opt_put_call_skew` capped at the 80-symbol Tastytrade limit since 05-13; `top60_chains=0`. **Not fixed here** — needs the operator to restore credentials/entitlement. Overlay-only fields; no scoring effect under the live ruleset |

## 2. Changes (branch `fix/missing-fields-2026-11-04`)

1. `1da651eb` — M1: health denominator matches the production gate. *Superseded:* `main` fixed the same defect independently in `8f6e7823` (2026-11-11), which also excludes `benchmark`-status rows. The merge of `main` into this branch (2026-11-12) resolved `tools/build_data_collection_health.py` to `main`'s version.
2. `378a199d` — M2:
   - `stage_ctgov` runs `collect_ctgov_data.py` before `warm_caches`. It writes to a seeded temp copy and swaps it in on success; on failure it keeps the previous file.
   - New blocking gate `ctgov_content_freshness`: FAIL when the newest `last_update_posted` on or before as_of is more than 7 days old.
3. `53ccc078` — M3: `_apply_inst_registry_transition_guard` runs after the z-score and zeroes `inst_delta_z` when the manager counts differ. Raw `inst_delta_*` counts are kept. *Amended 2026-11-12:* after #635 made a missing `inst_delta_z` blank rather than a neutral 0, the guard was changed to write `""` and set `has_inst_delta = "False"`, so the IC panel does not count contaminated rows as measured zeros. No ranking change: the live ruleset does not read `inst_delta_z`.

**Universe statuses.** The approved status updates were already on `origin/main` when this branch was cut:
- APGE, CRNX, ATAI and TBPH are `excluded_acquired`.
- BHVN is `active`.

Production still showed them otherwise because the live checkout ran 19 commits behind `origin/main` (deploy_freshness WARN). Deploying `main` is the fix.

**RNA is deliberately left `excluded_acquired`.**
- The symbol now belongs to Atrium Therapeutics (spun off from Avidity on 2026-02-26). Its price history splices Avidity (~$72) onto Atrium (~$8) at 2026-02-27, and the universe CIK is Avidity's.
- Reactivating it would feed a false −88% move into long-window price features.
- It needs a separate change that sets a series start date and the Atrium CIK.

GLPG (renamed Lakefront Biotherapeutics) stays retired until its US symbol is confirmed. Audit: `artifacts/universe_hygiene/stale_price_audit_2026_11_04/`.

## 3. Ranking impact

- **M3: none.** Under the live ruleset, `sort_anchor = selector_score`, so the sort key is `prefix + (-final_score, missing_count, ev_tiebreak, ticker)` (`decision_engine.py:2006`). The institutional sort contribution is computed but not used. The selector weight on `inst_delta_z` is 0.0, and the live ranker (`minimal_v2`) uses only `coinvest_score_z` and `financial_score`.
  - **Readers that do change:**
    - the non-blocking rebalance-plan artifact (Top-30 → Top-20 prune by `inst_delta_z`; it falls back to DEM order on ties)
    - shadow and IC trackers
    - the Event EV expectation overlay
  - Correction: an earlier session note said the pairwise ranker consumes `inst_delta_z`. That applies only to the legacy `minimal` feature set, not the live `minimal_v2`.
- **M2: expected.** Fresh trial records change clinical and catalyst inputs, and so can change eligibility and buckets. The size of the change was **not measured before merge**. Measure it on the first post-deploy run against the prior day.
- **M1: none** (health report only).

## 4. Clock reset

M2 changes production inputs, so the out-of-sample clock resets. `oos_clock_start` becomes **the first production run that uses the merged code**, and mandate SM-20260629-001 restarts at 0/20. The 2026-11-02 start set on 2026-11-01 had collected two runs (11-02, 11-03), both pre-fix, and they no longer count.

**Deploy steps (operator):**
1. Merge the PR.
2. Bring the live checkout to `origin/main` between production runs (never while the 16:30 ET cron is in flight).
3. Run the data refresh once by hand (`tools/cron_data_refresh.sh ctgov`) so the new gate passes on the first run.
4. Set `artifacts/forward_validation/CANDIDATE.json` `oos_clock_start` to that run's date, with an `oos_clock_reset_note` pointing to this memo.

**Risk:** `ctgov_content_freshness` is blocking. If CT.gov collection fails for more than 7 days, production stops promoting snapshots until it is fixed. The collector's coverage-regression guard (exit 2, keeps the previous file) is the most likely trigger.
