# Spec Ticket Template — inst_delta_z Backtest Sweep / Shadow Variant

**TEMPLATE / DRAFT — disposition-dependent. Use only if operator selects Option B (weight reduction) or the shadow-variant sub-option of Option C from the parent memo.**

**Status:** Draft. Not filed against any ticket system. File this as the next available SPEC- ID once the operator decides whether to actually run a sweep.

---

## Title

`SPEC-XXX: inst_delta_z weight sensitivity sweep` (Option B follow-up)
— or —
`SPEC-XXX: inst_delta_z=0 shadow portfolio variant` (Option C sub-option)

## Owner

TBD (operator-assigned)

## Trigger

`INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md` §5 items 3 & 4. Choosing between Options A/B/C requires an IR-impact estimate that does not currently exist.

## Goal

Quantify the impact of reducing or zeroing the `inst_delta_z` ranker weight on:
- Information ratio (IR) of the production strategy
- Sharpe
- Max drawdown
- Top-30 turnover
- Top-30 jaccard vs current production

…over the historical window where IC dashboard evidence shows inst_delta_z anti-predictive.

## Scope (Option B variant — backtest sweep)

### Inputs

- Historical snapshots: `data/snapshots/2026-02-19/` → `data/snapshots/2026-04-30/` (covers IC dashboard window + 1 month of pre-decision validation)
- Ranker config: copy of current `ranker_engine.py` RankerSignalSpec, parameterized on `inst_delta_z` weight
- Sweep grid: `[0.0, 0.10, 0.20, 0.30, 0.45]` (0.45 = baseline)
- All other weights and selector logic FROZEN (do NOT co-vary other signals during this sweep)

### Outputs

- One row per (date × weight) of: top-30 list, fwd 5-day return, fwd 20-day return, jaccard vs baseline
- Aggregated table: weight × {mean fwd 5d, mean fwd 20d, IR vs XBI, Sharpe, max DD, mean turnover, mean jaccard}
- Recommendation NOT embedded in artifact (governance-side decision)

### Deliverable

- `artifacts/inst_delta_z_sweep_<date>/sweep_results.csv` (raw)
- `artifacts/inst_delta_z_sweep_<date>/sweep_summary.md` (table + plain-text findings)
- `BACKTEST_SWEEP_INST_DELTA_Z_WEIGHT_<date>.md` at repo root summarizing methodology + cross-link to parent memo

## Scope (Option C sub-option — shadow variant)

### Inputs

- Live snapshots from the date the variant is configured forward
- Configuration: a parallel ranker pass with `inst_delta_z` weight = 0.0, same selector, same sizing
- Discipline: backtest never modifies production. Variant runs alongside production via `coinvest_shadow`-style daily artifacts.

### Outputs

- Daily comparator artifact: `artifacts/inst_delta_z_zero_shadow/<date>.json` with both variants' top-30, jaccard, fwd returns once matured
- Weekly rollup file matching `coinvest_shadow/summary.md` shape

### Deliverable

- New tool: `tools/build_inst_delta_zero_shadow.py` (parallels `tools/build_coinvest_shadow.py` if such exists)
- Cron entry to fire after production each day; sleep-cliff considerations apply (≤17:45 ET cron slot)
- `INST_DELTA_Z_ZERO_SHADOW_BOOTSTRAP_<date>.md` documenting the variant's configuration + start date

## Out of scope

- Any change to production ranker, selector, decision_engine, or active ruleset. Sweep and shadow are read-only relative to production.
- Promotion of any new weight to production. Promotion is a separate governance decision.
- Reconciliation of the SOUL.md / ranker_engine.py weight-string drift (separate ticket).
- Sweep over the selector or 91-180d bucket weights (those have different evidence bases and need their own tickets).

## Pitfalls (from CCFT principles + skills/biotech-screener-audit)

1. **Don't co-vary other parameters.** Sweep ONE weight; everything else frozen. Multi-parameter sweeps confound attribution.
2. **Don't pick the window post-hoc.** Use the IC dashboard window already cited in the parent memo (2026-02-19 → 2026-04-01 plus 20d validation tail). Do NOT cherry-pick a "best" window.
3. **Don't modify production rankings.csv to test new weights.** Run the parameterized ranker against frozen historical inputs. Backtest never modifies production.
4. **Don't include the 4 outage days (2026-04-07/08/11/12) without flagging them.** `RANKER_HYGIENE_NOTE_2026_05_01.md:21-29` documents these — `inst_delta_z` was constant=0 across all rows on those days, which makes weight sensitivity meaningless.
5. **Don't conclude from one weight grid point.** Five-point sweep is the minimum; finer grid (0.0, 0.05, ..., 0.45) is preferred if compute allows.
6. **Don't run as a real production tool without the preview-then-apply contract.** Step 1: write the script. Step 2: bash -n / read-through. Step 3: run on a single snapshot in a sandbox dir. Step 4: run on full window with operator approval.

## Estimated effort

- Option B sweep: ~1 day for the script + ~1 day to validate + write-up. Compute is cheap (35 snapshots × 5 weights, all CSV-side ranker logic).
- Option C shadow: ~1 day to stand up + ongoing daily cron. First useful comparison ~30 days post-bootstrap.

## Cross-links

- Parent memo: `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md`
- Shared-regime check: `INST_DELTA_Z_SHARED_REGIME_CHECK_2026_05_03.md`
- Governance log template: `INST_DELTA_Z_GOVERNANCE_LOG_TEMPLATE_2026_05_03.md`
- IC dashboard source: `tools/build_ic_dashboard.py`
- Existing shadow-variant pattern reference: `artifacts/coinvest_shadow/` (for shape + cadence)
- CCFT principles: `CLAUDE.md` (canonical, complete, frozen, timestamped data)

---

_This is a template. Filing it as a real spec ticket should happen ONLY after the operator has selected Option B or the Option C shadow sub-option in the governance log entry._
