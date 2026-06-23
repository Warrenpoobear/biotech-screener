# PIT Gap Forward Return Assembly — Implementation Spec

**Document type:** SPECIFICATION — no code, no outputs, no production changes  
**Status:** SPEC_PENDING_OPERATOR_APPROVAL  
**Governance:** Production model freeze ACTIVE  
**Authorized by:** Method decision memo (PR #384, merged 2026-06-22)

**Verdict options (operator fills in after review):**
- [ ] `SPEC_APPROVED_PENDING_IMPLEMENTATION`
- [ ] `SPEC_BLOCKED_AMBIGUOUS_METHOD`
- [ ] `SPEC_BLOCKED_PRODUCTION_RISK`
- [ ] `SPEC_BLOCKED_DATA_INTEGRITY_RISK`

---

## 1. Purpose and Scope

This spec authorizes and constrains a fresh implementation of PIT gap-period forward
return assembly. It covers:

- **Method A (primary):** Same-archive basis. 5d and 20d returns only.
- **Method B (sensitivity):** Single May 7 archive basis. All horizons including 60d.
  Outputs labeled `SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE`.

It does not authorize Method C (external provider), PR #382 code reuse, or any
production file modification.

---

## 2. Allowed Inputs

The implementation MAY read the following and ONLY the following:

| Source | Path | Purpose |
|--------|------|---------|
| Gap snapshot rankings | `data/snapshots/YYYY-MM-DD/rankings.csv` for dates 2026-01-16 to 2026-05-07 | Top-30 tickers, `actionable_rank`, `target_weight_pct` |
| pit_archive price files | `data/pit_archives/YYYY-MM-DD/price_history.csv` | Anchor and forward prices |
| pit_archive manifests | `data/pit_archives/YYYY-MM-DD/manifest.json` | SHA256 integrity check |

### Allowed fields from rankings.csv

- `ticker` — required
- `actionable_rank` — required; use only rows where `actionable_rank` is an integer 1–30
- `target_weight_pct` — optional; carry through for output annotation only

No other fields from `rankings.csv` may be read or used in any computation.

---

## 3. Forbidden Inputs

The implementation MUST NOT read, import, or reference:

- Any file in `data/snapshots/` other than `rankings.csv`
- Any production scoring module: `ranker_engine.py`, `selector_engine.py`,
  `decision_engine.py`, `run_screen.py`, `run_phase2_snapshot_delta.py`,
  `final_score*.py`, `sizing*.py`, or any file they import
- `data/universe_prices.csv` or `data/indices_prices.csv`
- `data/snapshots/_forward_returns_panel.csv` (the live panel; must not be read or modified)
- Any external API, network socket, or subprocess that fetches live or historical data
- Any code from the quarantined PR #382 branch (`research/pit-gap-assembly-2026-06-22`)
  or its commit range

---

## 4. Output Files and Quarantine Locations

All outputs are quarantined research artifacts. They must not be written to any
production data directory.

| Output | Path | Method | Quarantine status |
|--------|------|--------|-------------------|
| Method A panel | `artifacts/audit/gap_panel_method_a_YYYY-MM-DD.csv` | A | QUARANTINED until operator review |
| Method B panel | `artifacts/audit/gap_panel_method_b_sensitivity_YYYY-MM-DD.csv` | B | QUARANTINED; SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE |
| Validation report | `artifacts/audit/gap_assembly_validation_YYYY-MM-DD.md` | Both | Markdown; may be committed |

The date suffix `YYYY-MM-DD` is the run date (not a snapshot date).

**Output CSV files are gitignored** (`artifacts/audit/*` is in `.gitignore`).
Only the validation report markdown may be committed.

### Required header comment in Method B outputs

Every Method B CSV must include a header row before the column names:

```
# SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE
# Method B: single May 7 archive basis. Not primary results. Do not override Method A.
```

---

## 5. Method A Algorithm (Pseudocode)

```
METHOD A — Same-Archive Basis

INPUT:
  gap_dates = sorted canonical YYYY-MM-DD dirs in data/snapshots/ where
              2026-01-16 <= date <= 2026-05-07 AND rankings.csv exists

FOR each snap_date IN gap_dates:

  # 5.1 Resolve archive
  IF data/pit_archives/snap_date/ exists:
    arch_date = snap_date
    is_fallback = False
  ELSE:
    arch_date = most recent date < snap_date that exists in data/pit_archives/
    is_fallback = True
    IF no such date: SKIP snap_date, log warning

  # 5.2 Integrity check (see §6.7)
  Run manifest SHA256 check on arch_date/price_history.csv
  Log result (PASS / STALE_MANIFEST / MISSING_MANIFEST); do not abort on STALE_MANIFEST

  # 5.3 Load prices
  prices = {ticker: {date_str: close}} from arch_date/price_history.csv
  sorted_trading_dates = sorted union of all date_str values across all tickers

  # 5.4 Load top-30
  top30 = rows from snap_date/rankings.csv where actionable_rank in [1..30]

  # 5.5 XBI anchor (see §6.6)
  xbi_anchor_close, xbi_anchor_date = resolve_anchor("XBI", snap_date, prices, sorted_trading_dates)
  FOR h IN [1, 3, 5, 20]:
    xbi_return[h] = compute_return("XBI", xbi_anchor_date, h, prices, xbi_anchor_close, sorted_trading_dates)
    # if xbi_anchor_close is None: xbi_return[h] = None

  FOR each stock IN top30:
    ticker = stock.ticker

    # 5.6 ATXS exclusion (see §6.4)
    IF ticker == "ATXS" AND snap_date > "2026-01-23":
      emit row: anchor_close=null, all returns=null, atxs_excluded=true
      CONTINUE

    anchor_close, anchor_date = resolve_anchor(ticker, snap_date, prices, sorted_trading_dates)

    FOR h IN [1, 3, 5, 20]:
      ret[h] = compute_return(ticker, anchor_date, h, prices, anchor_close, sorted_trading_dates)
      excess[h] = ret[h] - xbi_return[h]  IF both not None, ELSE None

    # 60d: DO NOT COMPUTE. Method A makes no 60d claims.

    emit row with:
      snap_date, ticker, actionable_rank, target_weight_pct,
      archive_date=arch_date, archive_fallback=is_fallback,
      anchor_date, anchor_close, atxs_excluded=false,
      actual_return_1d, actual_return_3d, actual_return_5d, actual_return_20d,
      xbi_return_1d, xbi_return_3d, xbi_return_5d, xbi_return_20d,
      excess_return_1d, excess_return_3d, excess_return_5d, excess_return_20d,
      forward_complete_5d=(actual_return_5d is not None),
      forward_complete_20d=(actual_return_20d is not None)

# 5.7 Sub-functions

resolve_anchor(ticker, snap_date, prices, sorted_dates):
  ticker_prices = prices.get(ticker, {})
  IF snap_date IN ticker_prices: RETURN (ticker_prices[snap_date], snap_date)
  candidates = [d for d in sorted_dates if d <= snap_date and d IN ticker_prices]
  IF candidates: RETURN (ticker_prices[max(candidates)], max(candidates))
  RETURN (None, None)

compute_return(ticker, anchor_date, horizon, prices, anchor_close, sorted_dates):
  IF anchor_date is None OR anchor_close is None: RETURN None
  idx = sorted_dates.index(anchor_date)  # ValueError -> None
  fwd_idx = idx + horizon
  IF fwd_idx >= len(sorted_dates): RETURN None
  fwd_date = sorted_dates[fwd_idx]
  fwd_close = prices.get(ticker, {}).get(fwd_date)
  IF fwd_close is None: RETURN None
  RETURN (fwd_close - anchor_close) / anchor_close
```

---

## 6. Method B Algorithm (Pseudocode)

```
METHOD B — Single May 7 Archive Basis (SENSITIVITY ONLY)

PRECONDITION:
  latest_arch = "2026-05-07"
  ASSERT data/pit_archives/2026-05-07/price_history.csv EXISTS
  Run manifest SHA256 check on latest_arch; log result

  prices = load ALL prices from data/pit_archives/2026-05-07/price_history.csv
  sorted_trading_dates = sorted union of all date_str values across all tickers

INPUT:
  gap_dates = same as Method A

FOR each snap_date IN gap_dates:

  # 6.1 Load top-30 (same as Method A §5.4)
  top30 = rows from snap_date/rankings.csv where actionable_rank in [1..30]

  # 6.2 XBI anchor (uses single archive; same resolve_anchor as Method A)
  xbi_anchor_close, xbi_anchor_date = resolve_anchor("XBI", snap_date, prices, sorted_trading_dates)
  FOR h IN [1, 3, 5, 20, 60]:
    xbi_return[h] = compute_return("XBI", xbi_anchor_date, h, prices, xbi_anchor_close, sorted_trading_dates)

  FOR each stock IN top30:
    ticker = stock.ticker

    # 6.3 ATXS exclusion (same rule as Method A)
    IF ticker == "ATXS" AND snap_date > "2026-01-23":
      emit row: anchor_close=null, all returns=null, atxs_excluded=true
      CONTINUE

    anchor_close, anchor_date = resolve_anchor(ticker, snap_date, prices, sorted_trading_dates)

    FOR h IN [1, 3, 5, 20, 60]:
      ret[h] = compute_return(ticker, anchor_date, h, prices, anchor_close, sorted_trading_dates)
      excess[h] = ret[h] - xbi_return[h]  IF both not None, ELSE None

    emit row with:
      snap_date, ticker, actionable_rank, target_weight_pct,
      archive_date="2026-05-07", archive_fallback=false,
      anchor_date, anchor_close, atxs_excluded=false,
      actual_return_1d .. actual_return_60d,
      xbi_return_1d .. xbi_return_60d,
      excess_return_1d .. excess_return_60d,
      forward_complete_5d, forward_complete_20d, forward_complete_60d,
      sensitivity_label="SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE"

NOTE: Method B output must be written to a SEPARATE file from Method A.
Method B results must never be mixed with Method A results in any summary or table
without explicit labeling.
```

---

## 7. Required Validation Checks

All checks must run before any output is written. Failures must be logged.
A validation report markdown must be produced regardless of outcome.

### 7.1 Archive Date Resolution

For each gap snapshot:
- Confirm that a pit_archive exists on or before the snapshot date.
- Log any snapshot where no archive is found (should be zero per feasibility memo).
- Log fallback count (expected: 3 — for 2026-04-20, 2026-04-24, 2026-04-25).

### 7.2 Same-Archive Adjustment Basis (Method A only)

For Method A, confirm that every return in a given row uses prices from the same
`archive_date`. A row where `anchor_date` and any forward date resolve to prices
from different archive directories is a bug — fail loudly.

*(For Method B, this check is replaced by confirming all prices came from 2026-05-07.)*

### 7.3 May 7 Archive Basis (Method B only)

Confirm `data/pit_archives/2026-05-07/price_history.csv` exists and is readable.
Confirm `last_date` in that archive is 2026-05-07 (sample the max date from the file).
Log the result.

### 7.4 ATXS Exclusion

For every snapshot dated after 2026-01-23:
- Confirm ATXS rows have `anchor_close = null` and all return fields null.
- Confirm ATXS rows have `atxs_excluded = true`.

For every snapshot dated on or before 2026-01-23:
- Confirm ATXS has a non-null anchor (ATXS was still trading).

### 7.5 Anchor Coverage Threshold

For each snapshot, count tickers with non-null `anchor_close`.
- **PASS threshold:** ≥ 28 of 30 (allows for ATXS exclusion post-acquisition + 1 slack).
- Flag any snapshot below 28 as LOW_COVERAGE. Do not silently drop.
- Aggregate: all 87 expected snapshots should be ≥ 28.

*(The 2026-01-19 MLK Day snapshot may have no trading prices and legitimately produce
0 anchors. Document separately; do not count as a threshold failure if prices are
genuinely absent from all pit_archives for that date.)*

### 7.6 XBI Coverage

For each snapshot, confirm XBI has a non-null `anchor_close`.
- XBI must be present in every archive (100% expected per feasibility memo).
- Flag any snapshot where XBI anchor is null.

### 7.7 Continuity Flagging

For each archive loaded, check every ticker's consecutive-day price pairs.
Flag any pair where `|price[t+1] / price[t] - 1| > 0.50`.

- **Do not treat flags as automatic errors.** Biotech binary events routinely exceed 50%.
- Log all flags with ticker, date pair, and magnitude.
- Include a binary-event review note in the validation report: "These flags require
  manual review to confirm they represent real events (readouts, acquisitions) rather
  than price data errors."

### 7.8 Manifest SHA256 Handling

For each archive used:
- Read `manifest.json` and compare `files.price_history.csv.sha256` against the actual file.
- **PASS:** hashes match.
- **STALE_MANIFEST:** hashes differ. Log the archive date and both hashes.
  Per PR #383 archive-ceiling note, this is expected for all early archives rebuilt
  on 2026-04-10. Do not abort — log and continue.
- **MISSING_MANIFEST:** no `manifest.json`. Log and continue.

Aggregate: report total PASS / STALE_MANIFEST / MISSING_MANIFEST counts in the
validation report.

### 7.9 No Production File Modification

The implementation must not write to, read from (beyond allowed inputs §2), or import:
- Any file in `data/snapshots/` except `rankings.csv`
- Any file in `data/pit_archives/` except `price_history.csv` and `manifest.json`
- Any production scoring module
- `data/snapshots/_forward_returns_panel.csv`

A pre-run check should confirm the script's import list contains no production module.
This can be enforced by inspection (not automated) before the implementation PR is opened.

---

## 8. Acceptance Thresholds

Before any conclusion may be drawn from output, the following thresholds must pass.

### Method A

| Check | Threshold | Consequence if failed |
|-------|-----------|----------------------|
| 5d snapshot coverage | ≥ 40 snapshots with ≥ 28 non-null anchors AND non-null 5d return | No 5d IC conclusion |
| 20d snapshot coverage | ≥ 25 snapshots with ≥ 28 non-null anchors AND non-null 20d return | No 20d IC conclusion |
| 60d | Not computed | No 60d conclusion under any circumstance from Method A |
| XBI coverage | 100% of snapshots with anchor also have XBI anchor | Fail assembly |

### Method B

| Check | Threshold | Consequence if failed |
|-------|-----------|----------------------|
| 60d snapshot coverage | ≥ 20 snapshots with ≥ 28 non-null anchors AND non-null 60d return | Report zero 60d coverage; do not claim 60d sensitivity result |
| Single-archive basis | All rows confirm `archive_date = "2026-05-07"` | Fail assembly |
| Method B label | All output rows carry `sensitivity_label` field | Fail assembly |

---

## 9. Explicit Non-Goals

The implementation MUST NOT:

- Modify, read (other than as specified), or import any ranker, selector, sizing,
  `final_score`, gate, snapshot generator, portfolio construction, or production pipeline file
- Fetch live data from any external source (yfinance, IEX, Tiingo, Alpaca, Polygon,
  SEC EDGAR, ClinicalTrials.gov, or any other)
- Make any API or network call
- Write output to `data/snapshots/_forward_returns_panel.csv` or any file in `data/snapshots/`
- Produce any language that constitutes a trading recommendation, buy/sell signal,
  or investment action
- Conclude that the production model freeze should be lifted, modified, or overridden
- Treat Method B outputs as primary evidence or compare them to Method A without explicit
  `SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE` labeling
- Reuse code from PR #382 (`research/pit-gap-assembly-2026-06-22` branch)

---

## 10. Review Checklist (Operator Completes Before Implementation Approval)

All items must be confirmed before implementation proceeds.

**Inputs and scope:**
- [ ] Allowed input list (§2) is complete and sufficient for the algorithm
- [ ] Forbidden input list (§3) covers all production modules and data paths at risk
- [ ] Output paths (§4) are in `artifacts/audit/` (gitignored) and not in `data/`

**Algorithm correctness:**
- [ ] Method A pseudocode (§5) correctly implements same-archive basis
- [ ] Method A excludes 60d at every step (no 60d column, no 60d compute)
- [ ] Method B pseudocode (§6) correctly uses single May 7 archive for all price lookups
- [ ] Both methods handle ATXS exclusion consistently
- [ ] `resolve_anchor` fallback to prior trading date is correct for market holidays
- [ ] `compute_return` correctly counts trading days (not calendar days)

**Validation completeness:**
- [ ] All 9 validation checks (§7) are included and correctly described
- [ ] STALE_MANIFEST handling (§7.8) matches the PR #383 archive-ceiling finding
- [ ] Continuity flagging (§7.7) includes the binary-event review requirement

**Acceptance thresholds:**
- [ ] Method A 5d threshold (≥40 snapshots) is appropriate given known coverage
- [ ] Method A 20d threshold (≥25 snapshots) is appropriate given known coverage
- [ ] Method B 60d threshold (≥20 snapshots) is appropriate given projected coverage
- [ ] No-60d-conclusion for Method A is explicit and unambiguous

**Non-goals and governance:**
- [ ] All 8 non-goal categories (§9) are enforced as written
- [ ] Method B `SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE` label requirement is clear
- [ ] No production file modification path exists in the spec

**Implementation readiness:**
- [ ] PR #382 quarantine is noted and code-reuse prohibition is explicit
- [ ] Fresh implementation branch requirement is stated
- [ ] Spec is complete enough to write code without ambiguity

---

## 11. Required Verdict

Operator selects one before implementation begins:

- [ ] **`SPEC_APPROVED_PENDING_IMPLEMENTATION`** — All checklist items confirmed.
  Authorize a fresh implementation branch. Script name: TBD (not `assemble_gap_forward_returns.py`
  — that name is associated with the quarantined PR #382).

- [ ] **`SPEC_BLOCKED_AMBIGUOUS_METHOD`** — One or more pseudocode steps are unclear
  or inconsistent. Describe the ambiguity below and return spec for revision.

- [ ] **`SPEC_BLOCKED_PRODUCTION_RISK`** — A path exists by which the implementation
  could touch production files or scoring modules. Describe below.

- [ ] **`SPEC_BLOCKED_DATA_INTEGRITY_RISK`** — A validation check is missing or
  insufficient to catch a real data integrity failure. Describe below.

*Operator notes:*

```
[operator fills in verdict and any notes here]
```

---

**Prepared:** 2026-06-22  
**Next action:** Operator reviews checklist and records verdict above.  
**Implementation:** Only after `SPEC_APPROVED_PENDING_IMPLEMENTATION` is recorded here.
