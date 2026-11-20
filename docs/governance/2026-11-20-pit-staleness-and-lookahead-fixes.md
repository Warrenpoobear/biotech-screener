# Point-in-time staleness and look-ahead fixes — input change, clock reset (2026-11-20)

**Date:** 2026-11-20
**Classification:** Tier 3 (highest affected tier). Point-in-time logic, a production gate that can block snapshot promotion, and catalyst-override timing. Not a ranker, selector, sizing, or `final_score` formula change.
**Authorization:** operator request in this session: open one pull request with these fixes and follow the governance process for a model-input change. Merging the PR is the authorization. This change does not merge itself.
**Protocol:** `docs/FORWARD_VALIDATION_PROTOCOL.md` §1. An authorized bug fix during the NO_MODEL_CHANGE window resets the out-of-sample clock.
**Hash rotation:** none. `model_hash a9a430154568e242` (`ast-v2`) fingerprints `ranker_engine.py`, `selector_engine.py`, `decision_engine.py`, `common/ranker_mode.py`, and the `A4_SELECTOR_CONFIG` assignment in `run_screen.py`. None of those change. Ruleset `8887576e` is unchanged.
**Base:** `main` at `ab0ad565` (includes #578 short-interest report dates, #613 stage and catalyst rows, #579 dead-but-unretired universe check).

Claude Code review is required before merge. Tests that assert this behavior are Tier 3.

---

## Clock

`oos_clock_start` becomes the first promoted production run on the merged code. Mandate SM-20260629-001 restarts at 0/20. The 2026-11-17 start no longer counts once this merges. Do not edit `artifacts/forward_validation/CANDIDATE.json` here. Record the new start after that run, with `oos_clock_reset_note` pointing at this memo, as in `2026-11-04-missing-fields-clock-reset.md` §4.

The clock resets because the batch changes what can be scored and which coinvest signals are eligible. Display-only stage labels and the 13F gate do not, by themselves, change a score on a clean day. They ride along with the input fixes so the clock resets once.

---

## 1. Alert-stale prices are excluded, not silently scored

`compute_momentum_from_price_history` already logged per-ticker gaps and aborted only when `BIOTECH_STRICT_PRICE_STALENESS=1`. The default still wrote momentum from those closes and left the names in the scored set.

**Choice: exclude, and record the exclusion. Do not fail the run by default.**

A single name whose last close is past the alert threshold should not stop the book. Failing the run leaves the previous snapshot in place, so every name is then served from an older run. The existing alert threshold (`PRICE_STALENESS_ALERT_DAYS = 10`) is the line at which the close should not be trusted. Names beyond it are removed from Module 1's active set before Modules 2–5 and listed on `run_metadata.price_staleness_exclusions`, `metadata.json`, and `run_manifest.json` (`ticker`, `last_close`, `gap_days`, `threshold_days`, `action`). Momentum is not written from the stale close.

Gaps above the warn threshold (4 days) and at or under 10 days are still scored and still logged. That split was already the meaning of the two thresholds. `BIOTECH_STRICT_PRICE_STALENESS=1` still aborts the run instead of excluding.

Names the corporate-actions registry says are dead as of the run date stay out of the check, as before. A severe name that Module 1 was not going to score is not written on the manifest.

## 2. Prebuilt coinvest signals filed after as_of_date are dropped

`_refresh_coinvest_filing_age` recomputed age for `coinvest_signals.json` and clamped a future `latest_filed_at` at 0, which left the signal in the file and marked it fresh. The holdings fallback already skips a holder whose `filed_at` is after the run date.

The prebuilt path now deletes those ticker signals and logs them. The dropped tickers are stored on `coinvest_audit.prebuilt_future_filings_dropped` when any are removed. Entries with no `latest_filed_at` stay as built: the file does not carry an older filing this loader can fall back to, and the previous code left those rows alone. A non-dict value (the flat file's `period_of_report` string, if the wrapper was not peeled off) is not treated as a signal.

## 3. The 13F refresh gate FAILs when most managers are ahead of coinvest

`check_sec_13f_refresh_pending` was WARN at the existing share (`warn_share = 0.5`): most selected managers in the PIT cache have `period_of_report` newer than `coinvest_signals.json`. That status is now FAIL, so the daily manifest is FAIL and the snapshot is not promoted. Rebuilding coinvest stays a manual governed step.

Unreadable coinvest file, or a file with no `period_of_report`: still WARN. Those cases do not show that the cache is newer. A missing cache index stays PASS and is left to `sec_13f_cache`.

## 4. An unreadable corporate-actions file fails closed

`_dead_tickers_as_of` returned an empty set on any read error, so a corrupt registry was treated as "nobody is dead" and acquired names could be scored.

**A missing file stays an empty set.** That is the same rule as `load_actions()` and as `check_dead_not_retired` in #579 when it is handed an empty registry: a data directory without a registry is not a registry that failed to load. **A file that exists but cannot be parsed raises.** Scoring does not continue. #579 already lets a bad JSON propagate out of the maintenance report (only `ImportError` is caught there). This lines `run_screen` up with that. `tools/build_universe_maintenance.py` is not edited. Split-factor loading (`_load_run_corporate_actions`) is not edited; split handling is a separate change.

## 5. Catalyst overrides without source_filed_at are skipped

`module_5_composite_v3` applied an override whenever `source_filed_at` was missing, and only rejected a date after `as_of_date`. A missing, blank, or unparseable `source_filed_at` is now skipped and logged (`Catalyst override skipped for TICKER: ...`). A future date is still skipped.

**Rows in the file the loader reads** (`production_data/catalyst_overrides.json`), as of this change:

| Ticker | source_filed_at | Dropped on a 2026-11-20 run? |
|---|---|---|
| CELC | 2024-11-07 | no |
| COGT | 2026-11-07 (#613) | no |
| IMCR | 2026-11-07 (#613) | no |

Zero production rows lack `source_filed_at`. `data/catalyst_overrides.json` is not in the repo. `production_data/crt_manual_overrides.json` is the CRT outcome log (#613 added AVBP). It is not read by this loader and is not filtered here.

## 6. Stage overrides load inside the run and honor effective_from

`_load_development_stage_overrides` ran at import against the live file, with no as-of, so a historical run applied whatever rows were on disk that day. The map is now loaded in `save_validation_snapshot` for that run's `as_of_date`. The import-time map is empty.

**Rows without `effective_from` are not applied.** An undated override has no availability date. Applying it on every historical as-of is look-ahead. The skip is logged (`missing_effective_from`), the same rule as a catalyst override with no `source_filed_at`. The file-level `as_of_date` is not used as a substitute: #613 moved that stamp to 2026-11-07 for the whole document, and borrowing it would have dated the April rows as November.

The seven rows already in the file had a review date on `added_by`. That date is now `effective_from`, so the rows #613 just added keep applying from the day they were recorded:

| Ticker | stage | effective_from | source of the date |
|---|---|---|---|
| MESO | commercial | 2026-04-28 | `spec_068_audit_2026-04-28` |
| VCEL | commercial | 2026-04-28 | same |
| HALO | commercial | 2026-04-28 | same |
| MLYS | nda_bla | 2026-04-28 | same |
| INSM | commercial | 2026-11-07 | `operator_review_2026-11-07` (#613) |
| IMCR | commercial | 2026-11-07 | same (#613) |
| ANRO | phase_2 | 2026-11-07 | same (#613) |

A run on or after 2026-11-07 applies all seven. The column is display-only: it does not feed archetype, tier, stage bucket, scoring, ranker, selector, or eligibility. A run before a row's `effective_from` does not show that override.

---

## What this does not change

- Ranker, selector, sizing, `final_score` construction, ruleset `8887576e`, `model_hash`.
- `collect_short_interest.py` (#578).
- Split adjustment and `_load_run_corporate_actions` (separate change).
- `check_dead_not_retired` (#579). It stays read-only and does not retire names.
- CRT manual overrides.

## Not measured here

No full `run_screen` control-versus-treatment was run. On a date with no alert-stale scored name and no coinvest `latest_filed_at` after that date, the scored set and the coinvest inputs are the same as before this code. The first post-merge production run is the measurement: compare its Top-30 and the manifest's `price_staleness_exclusions` to the prior promoted snapshot.
