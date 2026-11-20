# Decisions recorded before the selector-only clock starts (2026-11-15)

**Status:** all four sections are decided (operator instruction, 2026-11-15). Sections 3 and 4 were written as proposals and **adopted as proposed by the operator later the same day**; they are in force through `production_data/AGENTIC_ACCOUNT_RULES.md` Rule 4 and `docs/FORWARD_VALIDATION_PROTOCOL.md` §4.

**Why now:** `oos_clock_start` for the selector-only order has not been recorded. It is expected to be 2026-11-16, after the first promoted selector-only run (see the addendum in `2026-11-13-selector-only-production-switch.md`). Anything decided after the clock starts either resets it or gets argued with the results in view.

No scoring, selection, order, sizing or eligibility logic changes in this record.

## 1. A routine quarterly 13F refresh is not a clock reset

`production_data/coinvest_signals.json` is still built from Q2 2026 holdings (period 2026-06-30, built 2026-10-30). Q3 2026 filings are due 2026-11-16; on 2026-11-13 the point-in-time cache held Q3 filings for 3 of the 30 roster managers. The rebuild is a manual, governed step and will change the Top-30 within days of the clock starting.

Rule: rebuilding `coinvest_signals.json` from a new quarter's filings is a **routine input refresh and does not reset the out-of-sample clock**, provided all of the following hold.

- The manager roster (`production_data/manager_registry.json`) is unchanged.
- The builder (`scripts/build_coinvest_features_from_13f.py`) and the selector code are unchanged.
- The rebuild uses only filings with a filing date on or before the rebuild date.
- The rebuild date, the period of report and the number of managers with a new filing are written to `.claude/rules/operational-state.md`.

A change to the roster, the builder, the CUSIP map logic or the staleness gate is not routine. It is a governed change and resets the clock, as on 2026-11-01.

The same reading applies to the other scheduled input refreshes the pipeline already performs daily (prices, trial records, short interest, options). New data arriving is what the forward test observes.

## 2. The capture checks the order mode

The candidate `model_hash` covers `ranker_engine.py`, `selector_engine.py` and `decision_engine.py`. It does not cover the `--ranker-mode` dispatch in `run_screen.py`, so the switch to `selector_only` changed the Top-30 without changing the hash.

`tools/run_forward_validation.py` now compares the snapshot's recorded mode (`metadata.json`, `ranker_mode`) with `CANDIDATE.json.ranker_mode`:

- a mismatch, or a snapshot that records no mode while the candidate pins one, is a hard data-quality failure (`ranker_mode_match`), so the capture is not mandate-eligible;
- the capture records `ranker_mode`, `candidate_ranker_mode` and `ranker_mode_match`;
- a candidate without the field pins nothing, and captures written before the field existed are not excluded by it.

Checked on real data: the 2026-11-13 staged selector-only snapshot matches the candidate; the 2026-11-12 pairwise snapshot does not. The hash scheme itself is unchanged, so there is no hash rotation.

## 3. Hard-exit thresholds for the agentic account (adopted 2026-11-15)

`production_data/AGENTIC_ACCOUNT_RULES.md` Rule 4 liquidates the account at −2 pp against XBI. On the selector-only history (`snapshots_pit_v5` plus live month-ends, 84 months, net of 50 bps), monthly excess return has a mean of +1.14 pt and a standard deviation of 4.83 pt. Measured at month-ends from every possible entry month:

| Relative performance vs XBI touched within | −2 pp | −5 pp | −10 pp | −15 pp | −20 pp |
|---|---|---|---|---|---|
| 3 months of entry | 39% | 23% | 9% | 4% | 1% |
| 6 months of entry | 47% | 29% | 11% | 6% | 0% |
| 12 months of entry | 53% | 37% | 15% | 4% | 0% |

So the current trigger fires in about half of all 12-month holding periods through ordinary variation. In this history the six months after a first touch of −2 pp averaged +8.1 pp of excess return and were positive 89% of the time (36 cases), so the rule sold ahead of recoveries. Daily observation would touch each level more often than these month-end figures.

Replacement, adopted by the operator as proposed:

| Trigger (per-lot, XBI-anchored, as measured today) | Action |
|---|---|
| ≤ −10 pp | Review: no new buys until the operator has looked at it |
| ≤ −15 pp | Full liquidation to cash |

These are in-sample, survivorship-biased figures, and the account is a pilot. Rule 4 now carries these levels. The former "liquidate immediately, no delay" band (−5 pp) is removed: liquidation at −15 pp follows operator confirmation.

## 4. What counts as failure for the forward test (adopted 2026-11-15)

`docs/FORWARD_VALIDATION_PROTOCOL.md` §4 already fixes the pass side: at least 20 non-overlapping weekly windows, directional gate t ≥ 1.65 (one-tailed), confirmation |t| ≥ 1.96. Two things are missing.

**There is no fail rule.** If the true edge is the no-hindsight backtest estimate (about +1.1 pt a month, roughly +0.25 pt a week against a weekly standard deviation near 2.3 pt), the expected t-statistic after 20 windows is about 0.5 and the chance of clearing 1.65 is roughly one in eight. "Not passed at 20" will be the usual outcome even if the model works, so it cannot be the failure signal. Adopted:

- Stop and review if, at any checkpoint of 20, 40 or 60 windows, mean weekly excess is negative with t ≤ −1.65.
- Otherwise keep accruing to 60 windows before reading "not significant" as evidence against the model.

**There is no rollback rule for selector-only against the shadow ranker.** The shadow ledger (`tools/selector_only_shadow.py`) records both baskets. Adopted: review reverting to `pairwise_minimal` only if, after at least 20 windows, selector-only trails the ranker basket with t ≤ −1.65 on the paired weekly difference. A smaller gap is noise at this sample size.

Changing §4 of the protocol is a Tier 4 decision; the operator made it on 2026-11-15 with zero forward windows observed for the selector-only order. §4 now carries these rules, plus a read-out at 60 windows.

## 5. The candidate hash covers the selector weights and the order mode (2026-11-15)

Section 2 closed the order-mode gap in the capture. The hash itself still left the selector's block weights outside: `A4_SELECTOR_CONFIG` is defined in `run_screen.py`, which `ast-v1` does not read, so the weights could change with the hash unchanged.

`HASH_SCHEME` is now `ast-v2`: the three engine files as before, plus `common/ranker_mode.py` whole and the `A4_SELECTOR_CONFIG` assignment in `run_screen.py` (only that assignment's AST, so unrelated edits to the file do not move the hash). The candidate is re-registered as `a9a430154568e242`; `827c35a9ed3ee6e1` is kept under `prior_model_hashes` and is still reproducible with `compute_model_hash_ast_v1()`. This is the same model with a wider fingerprint, not a new freeze. Rotation logged in `governance/HASH_ROTATIONS.md`.

Still outside the hash: the rest of `run_screen.py`, including which config object is passed to the selector; the ruleset file (tracked as `ruleset_hash`); input data.

## Evidence these decisions rest on (scratch research, 2026-11-13 to 2026-11-15; not committed)

- Selector-only against XBI, trailing periods through 2026-11-12: three months −15.6% against +2.9%; since 2020 +1.18 pt a month (t = 2.22).
- August and September 2026 autopsy: selection failure from co-investment concentration; no pre-window screen recovers more than 2.5 pt. `artifacts/autopsy/top30_aug_sep_2026_postmortem/`.
- Caps, diversifiers and filing-age decay variants: none beats the current selector.
- Manager roster: the current 30 are at the 98th percentile of random 30-of-63 lists (+1.57 against an average +1.10 pt a month); a point-in-time roster picked on trailing 36-month returns adds +0.12 pt (t = 0.48) over using all managers.

All of it is pseudo point-in-time, survivorship-biased and in-sample for the selector settings. It is context for these decisions, not evidence for promotion.
