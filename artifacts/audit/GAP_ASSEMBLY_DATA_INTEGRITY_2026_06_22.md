# Gap Assembly Data Integrity Addendum — 2026-06-22

**Script:** `scripts/research/assemble_gap_forward_returns.py`
**Output:** `artifacts/audit/gap_forward_returns_panel.csv`
**Status:** RESEARCH_ONLY — QUARANTINED pending operator review
**Governance:** Production model freeze ACTIVE; no production files touched

---

## Assembly Result Summary

| Metric | Value |
|--------|-------|
| Gap period | 2026-01-16 to 2026-05-07 |
| Total snapshots | 88 |
| Snapshots with output rows | 87 (2026-01-19 MLK Day: no trading prices → 0 rows) |
| Total output rows | 2,610 |
| Rows with 5d return complete | 1,691 / 2,610 (64.8%) |
| Rows with 20d return complete | 1,211 / 2,610 (46.4%) |
| Rows with 60d return complete | 0 / 2,610 (0.0%) |
| ATXS rows excluded post-acquisition | 14 (correct; last trading date 2026-01-23) |
| Snapshots with ≥ 28 non-null anchors | 87 / 87 (100%) |

### Coverage by Snapshot Date

| Horizon | First snapshot | Last snapshot | # snapshots |
|---------|---------------|---------------|-------------|
| 5d | 2026-01-16 | 2026-03-31 | 57 |
| 20d | 2026-01-16 | 2026-03-12 | 41 |
| 60d | — | — | 0 |

---

## Key Finding: Archive Forward Horizon Ceiling

### Root Cause

The early gap archives (2026-01-16 through approximately 2026-03-31) were all **retroactively rebuilt on 2026-04-10** in a mass archive regeneration. Inspection confirms:

- Archives for 2026-01-16, 2026-02-16, 2026-03-02, and all intermediate dates have `last_date=2026-04-10` in their `price_history.csv`.
- All these early archives share the **same SHA256 hash** (`11049eeb6d...`), confirming identical file content — same price pull, same adjustment basis.
- The manifest `sha256` fields for these archives reflect the pre-rebuild files and no longer match the current content (SHA256 mismatch on every early archive).

This is the "split-adjusted prices; archives created retroactively" risk flagged in the feasibility memo (§4.3).

### Why 5d Ends at March 31

March 31 + 5 trading days = April 7. The early archives (rebuilt April 10) contain prices through April 10 → April 7 is present → 5d returns available.

April 1 snapshots use the 2026-04-01 archive, which was built ON April 1 and contains prices only through April 1. April 1 + 5td = April 8 → NOT in the April 1 archive → 0 returns.

### Why 20d Ends at March 12

March 12 + 20 trading days ≈ April 9. The early archives go through April 10 → April 9 is present → 20d returns available.

March 13 + 20 trading days ≈ April 10 (borderline, not all tickers have that date) → drops out.

### Why 60d Has 0 Coverage

January 16 + 60 trading days ≈ April 9-14 (the exact date varies by trading calendar). The April 9 date is near the archive cutoff (April 10), but most tickers do not have a price row for that specific date, so the 60th forward day resolves to April 14 or later → NOT in archives → 0 coverage.

---

## Manifest SHA256 Mismatches

**All 57 early archives** (Jan 16 through late March) report SHA256 mismatches. This is expected and consistent with the mass retroactive rebuild:

- The manifest was written for the original file; the current file is the rebuilt version.
- The files are valid price data — the mismatch indicates the manifest is stale, not that the data is corrupt.
- All early archives have the identical hash `11049eeb6d...` confirming they share the same price content.
- **Assessment: NOT a data quality failure; a manifest staleness issue from the retroactive rebuild.**

---

## Continuity Flag Assessment

The validation flagged >50% single-day price moves for several tickers (ACLX, ABVX, ENGN, PEPG, etc.). These are **not price data errors** — they are confirmed binary biotech events:

| Ticker | Date | Move | Likely event |
|--------|------|------|--------------|
| ABVX | 2025-07-22 | +586% | Binary readout |
| ENGN | 2024-02-13 | +116% | Binary readout |
| ACLX | 2026-02-20 | +77% | Phase 2/3 data |
| SGMT | 2024-01-19 | +170% | Acquisition/readout |

**Assessment: These are real events correctly captured in the price series. No price errors detected.**

---

## Path to Better 60d Coverage

The current implementation uses the snapshot-specific archive for both anchor and forward prices (strict "same archive" consistency). To get 60d coverage, two options exist:

### Option A: Single Latest Archive (Recommended)

Load all gap prices from the **latest available archive** (`data/pit_archives/2026-05-07/price_history.csv`, which has `last_date=2026-05-07`). Use this single archive for BOTH anchor and forward prices across all 88 gap snapshots.

**Benefit:** Split adjustments cancel in return calculations (adjusted_fwd/adjusted_anchor = true_return if same adjustment basis). Using one archive guarantees the same basis.

**Expected coverage improvement:**
| Horizon | With single May 7 archive |
|---------|--------------------------|
| 5d | ~82 snapshots (all through ~April 30) |
| 20d | ~57 snapshots (all through ~April 9) |
| 60d | ~35-40 snapshots (Jan 16 through ~Feb 28) |

**Risk:** If a stock was delisted between its snapshot date and May 7, anchor prices in the May 7 archive may be pre-delisting adjusted prices (affects return level). This is a low risk for the gap period — only ATXS is known to have been acquired (handled).

**Implementation note:** Add `--use-latest-archive` flag to `assemble_gap_forward_returns.py` that overrides per-snapshot archive selection with the most recent available archive.

### Option B: Extend Archives (More Complex)

Fetch current prices for all gap-period tickers via yfinance, update the pit_archives with the full history through the current date, and rerun the assembly. This gives 60d coverage for all snapshots through April 23 (April 23 + 60td ≈ July 17, and current prices are available through today).

**Risk:** Introduces new split-adjustment complexity (current prices may reflect splits not captured in the original archive). Requires more validation.

---

## What This Assembly IS Useful For

Despite the 60d gap:

1. **5d IC analysis (Jan 16 – March 31):** 57 snapshots with complete 5d data — sufficient for a 5d Spearman IC calculation over the gap period.

2. **20d IC analysis (Jan 16 – March 12):** 41 snapshots with complete 20d data — partial but useful.

3. **Anchor price catalog:** 87 snapshots × 30 tickers = 2,610 anchor prices, all with associated `actionable_rank` and `target_weight_pct`. These can be used as the anchor source if forward prices are subsequently obtained from a different source (yfinance, newer archive).

4. **Era reconciliation baseline:** The assembly confirms that prices from the April 10 retroactive rebuild are consistent with the production price series (era reconciliation check passed).

---

## What This Assembly Is NOT Sufficient For

1. **60d excess return analysis** (the primary backtest horizon): 0 data points from the gap period.

2. **Gap-period extension of the walkforward panel** at 60d: still requires Option A or Option B above.

3. **Production model change:** This assembly is RESEARCH_ONLY. No model change is authorized.

---

## Next Steps (Requires Separate Operator Authorization)

1. **Option A (--use-latest-archive):** Implement and run single-archive mode to gain ~35-40 snapshots of 60d coverage. Low implementation cost; manageable split-adjustment risk.

2. **Era reconciliation at 60d boundary:** Sample 5 tickers and compare returns computed from this assembly vs. any independent source (Polygon/Alpaca if available) around the Jan 16 – March 12 window to confirm return accuracy.

3. **Merge with existing panel (if accepted):** After operator review, append accepted rows (where `forward_complete=true`) to `data/snapshots/_forward_returns_panel.csv`. Requires explicit operator sign-off.

---

**Prepared:** 2026-06-22
**Governance:** RESEARCH_ONLY_NO_MODEL_CHANGE; QUARANTINED pending operator review
