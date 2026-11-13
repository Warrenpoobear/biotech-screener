# Historical runs read today's market data — regeneration tooling fix (2026-11-13)

**Classification:** `RESEARCH_TOOL_FIX / BACKTEST_CORRECTION`. Tier 2 research tooling.
**Model effect:** none. No ranker, selector, eligibility, sizing, ruleset or production-run change. `run_screen.py` is not modified. No out-of-sample clock effect.
**Status of the evidence below:** pseudo-PIT, research only. Not accepted evidence for a model decision until the operator says so.

## What was wrong

`run_screen.py --as-of-date <past date>` loads `production_data/market_data.json` as it stands on the day the command is run. There is no date handling on that file. Every historical snapshot ever regenerated (`data/snapshots_pit_v2`, its backups and variants, `data/snapshots_pit_gap_2026q3`) therefore carries, on every past date:

- today's `market_cap`, `price`, `avg_volume`, `52w_high/low`, `volatility_90d`, `returns_1m/3m`, `short_percent`, `beta`.

`module_2_financial_v2.py` divides cash and burn by `market_cap` (`cash_to_mcap`, `burn_to_mcap`). A company whose market cap grew after the snapshot date gets a lower historical `financial_score`. The `minimal_v2` ranker weights `financial_score` negatively, so on regenerated history it favoured companies that later grew.

Observed directly: `market_cap_mm` for MDGL is 10059.1 in every month from 2020 to 2026 in both `snapshots_pit_v2` and a current-model regeneration. Its market cap in January 2020 was about $1.3B.

## Second defect: historical runs write into the checkout

A historical `run_screen.py` run also rewrites files relative to the checkout it runs in. A two-date trial in the production checkout on 2026-11-13 changed:

| File | Effect |
|---|---|
| `data/state/blind_spot_streak.json` | replaced with `{}` |
| `data/expression_decision_log.jsonl` | that date's existing rows removed, new rows appended |
| `artifacts/readiness/history.jsonl` | row appended |
| `production_data/ctgov_state/<hash>/manifest.json` | entries added (also through hardlinked staging dirs) |
| `production_data/catalyst_events_<date>.json`, `catalyst_events_vnext_<date>.json`, `run_log_<date>.json` | overwritten |
| `artifacts/bioshort_watch/latest_status.json` | rewritten |

All were repaired the same day except the overwritten dated files for 2026-06-26 and 2020-01-31, for which no earlier copy existed.

## What this change does

`scripts/research/regenerate_pit_v2_snapshots.py` gains four protections, all on by default:

- `--pit-market-data`: stages a `market_data.json` built for each date by `common/pit_market_data.py`.
  - Price-derived fields come from closes on or before the date.
  - Market cap = share count from the latest SEC fact *filed* on or before the date × that day's close, restated for splits between the two.
  - Share counts that cannot be used are handled by rule and labelled in `market_cap_source` (see the module docstring): foreign filers reporting ordinary shares, counts above 1.6× today's, and pre-listing counts. The fallback is today's market cap scaled by the price path, which removes the price look-ahead but still carries today's share count.
  - `short_percent`, `short_ratio`, `float_shares` and `beta` have no point-in-time source and are set to None.
  - If the file cannot be built the date fails. It never falls back to today's file silently.
- `--isolate-side-outputs`: passes `--output-dir <out-dir>/_side_outputs/<date>/` so Module 3 side outputs stay out of the data dir.
- `--protect-live-state`: copies the protected files aside before each date and restores any that changed; stops if a daily production lock file exists.
- `--fresh-staging`: removes a date's staging dir before staging, so hardlinks to superseded input files are not reused.

Also: an archived `data/snapshots/<date>/inputs` dir is used only when it holds `financial_records.json`. Archives written from 2026-04-20 do not, and the script previously failed preflight on those dates.

## Evidence

Two full 80-month regenerations with the current model (ruleset `8887576e`, commit `65829ced`, current `production_data`, PIT 13F staging), run in an isolated copy of the checkout. Same code and inputs; only the market data differs.

Top-30 by `actionable_rank`, equal weight, 2020-01-31 → 2026-06-26, 79 periods, JBIO excluded, split-adjusted prices, net of 50 bps one-way on actual turnover, XBI gross (+75.7% cumulative):

| | Today's market data (`snapshots_pit_v3`) | Point-in-time market data (`snapshots_pit_v4`) |
|---|---|---|
| Full period | +1.64 pt/period, t = 2.99, +472.4% | +0.68 pt/period, t = 1.47, +193.1% |
| Historical (before 2024-10-01), 56 periods | +1.18 pt, t = 1.84 | +0.54 pt, t = 0.89 |
| From 2024-10-01, 22 periods | +2.86 pt, t = 2.65 | +1.03 pt, t = 1.62 |
| Max drawdown (relative to XBI) | −44.6% (−15.6%) | −44.9% (−23.7%) |
| Mean one-way turnover | 17% | 21% |

Monthly rank correlation with next-period return, eligible names:

| Field | Today's market data | Point-in-time market data |
|---|---|---|
| `financial_score` | −0.067 (t = −6.9) | −0.001 (t = −0.1) |
| `financial_score`, ranker cohort only | −0.060 (t = −3.2) | +0.021 (t = 1.1) |
| `selector_score` | +0.015 (t = 1.0) | +0.033 (t = 2.1) |
| `coinvest_score_z` | +0.006 (t = 0.4) | +0.022 (t = 1.3) |

Selector-only Top-30 (same eligible set, ordered by `selector_score`) against the current rule, both replayed on the recorded rows:

| History | Selector-only minus current rule |
|---|---|
| Today's market data | −0.60 pt/period (t = −1.69) |
| Point-in-time market data | +0.62 pt/period (t = +1.64) |

Share-count coverage in the point-in-time history: 78% of ticker-months from filed share counts, 20% from the price-scaled fallback, 2% from the first post-listing filing.

End-to-end check of this script (three months, isolated copy): the Top-30 matches `snapshots_pit_v4` name for name on 2020-01-31, 2023-06-30 and 2026-06-26; side outputs land under the out dir; the protected files are byte-identical after the run.

## Limits

- Pseudo-PIT, not true PIT. Survivorship is unchanged: the price file holds no ticker that stopped trading before 2025.
- 20% of ticker-months still carry today's share count through the fallback.
- ABVX is wrong in the fallback: the split registry treats its 2025 price move as a split.
- January and February 2020 have no PIT 13F cache and use current holdings.
- The selector settings and the ranker were chosen with this period in view.
- `universe.json` and other inputs are current files filtered by date, as before.

## Not decided here

- Whether the "+2.34pp/mo, t = 2.57" statements and the historical `financial_score` evidence move to Deprecated (separate draft PR).
- Any change to the ranker. That is a governed model change with a clock reset.
- Whether `run_screen.py` itself should refuse to read today's `market_data.json` for a past date. That file is Tier 3 and frozen; this change stays in the research script.
