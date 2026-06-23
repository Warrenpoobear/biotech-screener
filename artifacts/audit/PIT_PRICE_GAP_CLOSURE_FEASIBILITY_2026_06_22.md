# PIT Price Gap Closure Feasibility Audit — 2026-06-22

**VERDICT:** `PASS_PIT_PRICE_GAP_CLOSURE_PATH_IDENTIFIED_NO_MODEL_CHANGE`

**Status:** `RESEARCH_ONLY_NO_MODEL_CHANGE`
**Production model freeze:** ACTIVE
**Audit scope:** Read-only; no production files modified

---

## Pre-Report Answers

| Question | Answer |
|----------|--------|
| Gap period | 2026-01-16 to 2026-05-07 |
| Canonical gap snapshot directories found | 88 (YYYY-MM-DD format only) |
| Gap snapshot dirs with pit_archive coverage | 85/88 (96.6%) |
| Gap snapshot dirs without pit_archive (require fallback) | 3 (2026-04-20, 2026-04-24, 2026-04-25) |
| Unique tickers in top-30 during gap period | 118 |
| Total ticker-date pairs (top-30 × snapshot dates) | 2,640 |
| Ticker-date pairs covered by existing pit_archives | 2,626 / 2,640 = 99.5% |
| Uncoverable pairs (acquired ticker, by design) | 14 (all ATXS post-acquisition) |
| External price provider needed | NO — gap is closeable from existing cached sources |
| XBI coverage for gap dates | 100% (85/88 direct from archives; 3 via prior-archive fallback) |
| Production files modified | NO |

---

## 1. Cached Price Sources Survey

### 1.1 All Identified Price Sources

| Source | Format | Date Range | Tickers | Gap Coverage? |
|--------|--------|-----------|---------|--------------|
| `data/universe_prices.csv` | Wide (date index, ticker cols) | 2023-12-22 to **2026-01-15** | 307 | NO — ends day before gap starts |
| `data/indices_prices.csv` | Wide (date index) | 2023-12-22 to **2026-01-15** | XBI, SPY | NO — ends day before gap starts |
| `data/daily_prices.csv` | Long (date, ticker, adj_close) | 2022-01-03 to 2024-12-31 | ~25 | NO — historical only, pre-gap |
| `data/prices.db` (SQLite) | Long (date, ticker, close) | 1982-02-16 to **2026-04-17** | ~350 | PARTIAL — covers Jan 16 to Apr 17 only |
| `data/diag/price_cache.csv` | Long (date, ticker, close) | 2025-11-14 to 2026-02-13 | subset | PARTIAL — covers partial gap only |
| `data/diag/price_cache_2y.csv` | Long (date, ticker, close) | 2024-01-31 to 2026-01-26 | subset | PARTIAL — covers Jan 16-26 only |
| `data/pit_archives/YYYY-MM-DD/price_history.csv` | Long (date, ticker, close, open, high, low, volume) | 1982-02-16 to **2026-05-07** (varies by archive) | ~345-352 per archive | **YES — primary source; full gap coverage** |
| `data/snapshots/_forward_returns_panel.csv` | Long (snap_date, ticker, anchor_close, returns) | 2026-05-08 to 2026-06-17 | ~350/snap | NO — starts after gap ends |

### 1.2 Primary Source: pit_archives

The `data/pit_archives/` directory contains 463 dated subdirectories (2020-01-03 to 2026-06-22). Each contains a `price_history.csv` with full OHLCV price history in long format (`date, ticker, close, open, high, low, volume`). This is the **same format** as loaded by `scripts/eval_forward_returns.py::load_price_series()`.

**Key structural property:** Each `pit_archives/YYYY-MM-DD/price_history.csv` archive contains prices FROM earliest history TO the snapshot date (and often a few days further, as archives are generated retroactively). This means:
- The archive for each gap snapshot date contains that date's closing price (the PIT anchor).
- The archive for subsequent dates contains the forward prices needed to compute 1d, 5d, 20d returns.
- No look-ahead contamination exists as long as the **backtest uses the snapshot-date close as anchor** and subsequent dates' prices as forward closes — both are genuine time-series data, not modeled/reconstructed values.

---

## 2. Gap Coverage Analysis

### 2.1 Canonical Gap Snapshot Count

The gap period (2026-01-16 to 2026-05-07) contains **88 canonical YYYY-MM-DD snapshot directories** in `data/snapshots/`. (The backtest audit referenced "111 snapshots failed closed" — the discrepancy reflects that the backtest script counted expected trading-day slots rather than actual snapshot directories present on disk.)

### 2.2 pit_archive Coverage by Snapshot Date

| Category | Count | Notes |
|----------|-------|-------|
| Gap snapshots WITH own pit_archive | 85 | `data/pit_archives/YYYY-MM-DD/price_history.csv` exists |
| Gap snapshots WITHOUT own pit_archive | 3 | 2026-04-20, 2026-04-24, 2026-04-25 |
| Fallback source for missing 3 | Prior pit_archive T-1 data | 2026-04-20 → 2026-04-17 archive; 2026-04-24, 2026-04-25 → 2026-04-23 archive (Apr 22 prices) |

For the 3 missing-archive dates, the nearest prior pit_archive contains prices for Apr 17 (342 tickers), Apr 22 (342 tickers) respectively — sufficient for full top-30 coverage.

**Market-closed dates (holidays):** 2026-01-19 (MLK Day) has a canonical snapshot directory but the pit_archive for that date contains no Jan 19 price row. The T-1 fallback (Jan 16 close) is appropriate as the trade-entry price. This pattern applies to other market holidays in the gap.

### 2.3 Ticker-Date Coverage (Top-30 × Snapshot Dates)

Using pit_archives with T-1 holiday fallbacks:

| Metric | Value |
|--------|-------|
| Total ticker-date pairs (top-30 per snapshot) | 2,640 |
| Covered by pit_archives + fallbacks | 2,626 (99.5%) |
| Missing (ATXS only) | 14 |
| Other unexplained missing | 0 |

**ATXS explanation:** ATXS (Acelyrin) was acquired by Bristol-Myers Squibb; last trading date was 2026-01-23. ATXS appears in the top-30 rankings for 14 subsequent gap snapshots (2026-01-26 through 2026-02-16) but has no price data after its acquisition. This is structurally unresolvable from any source — the ticker ceased to trade. Forward returns for ATXS post-acquisition should be marked as `null` / excluded, not filled. **This is not a data gap; it is a corporate action edge case.**

---

## 3. XBI Coverage for Gap Dates

XBI is present in `data/pit_archives/YYYY-MM-DD/price_history.csv` for all 88 canonical gap snapshot dates:

| Coverage method | Count |
|----------------|-------|
| XBI exact date in archive | 67/88 |
| XBI T-1 fallback (within same archive, market holiday) | 18/88 |
| XBI from prior-date archive | 3/88 (for the 3 missing-archive dates) |
| **Total XBI coverage** | **88/88 = 100%** |

`data/indices_prices.csv` (the Era 1 XBI source) ends at 2026-01-15 and does NOT cover the gap. The pit_archives are the sole XBI source for the gap period.

---

## 4. Recommended Gap Closure Path

### 4.1 Verdict: No External Provider Required

The pit_archives already contain all necessary individual ticker and XBI closing prices for the gap period. The closure path is a **pure assembly operation** using existing cached data — no API calls, no external data acquisition, and no production pipeline changes are needed.

### 4.2 Recommended Approach: assemble_gap_forward_returns.py

**Step 1 — Identify gap snapshot dates with rankings:**
Iterate over `data/snapshots/YYYY-MM-DD/rankings.csv` for dates 2026-01-16 through 2026-05-07.

**Step 2 — For each gap snapshot, resolve the pit_archive source:**
```
if pit_archives/snap_date/ exists:
    use pit_archives/snap_date/price_history.csv
else:
    find nearest prior pit_archives/ dir
    use that archive's price_history.csv
```

**Step 3 — Extract anchor close (PIT price at snapshot date):**
```
anchor_close = price_history[ticker][snap_date]
if no row for snap_date (market holiday):
    anchor_close = price_history[ticker][max(date < snap_date)]
```

**Step 4 — Extract forward closes (1d, 5d, 20d, 60d):**
For each forward horizon, look up `price_history[ticker][snap_date + N trading days]`. Trading days are determined from the set of dates present in the archive for that ticker.

**Step 5 — Compute returns and write to panel:**
Output format to match `data/snapshots/_forward_returns_panel.csv`:
`snap_date, ticker, anchor_date, anchor_close, actual_return_1d, actual_return_5d, ..., xbi_return_5d, excess_return_5d, forward_complete`

**Step 6 — Append to or join with existing _forward_returns_panel.csv** (currently covers 2026-05-08 to 2026-06-17).

### 4.3 Look-Ahead Risk Assessment

| Risk vector | Assessment |
|-------------|-----------|
| Prices retroactively adjusted for post-gap splits | LOW — pit_archives were generated at or shortly after snapshot dates; no evidence of retroactive split-adjustment in sampled tickers |
| Future price data contaminating anchor_close | NONE — anchor is the snapshot-date close; it is genuinely PIT |
| Forward prices not available at snapshot time | NOT APPLICABLE — forward returns are computed ex-post; this is the correct backtest methodology |
| Archive prices sourced from live fetch at creation time | LOW RISK — prices are yfinance historical data pulled on the archive creation date; these represent the same prices the market observed |

**Split-adjustment caveat:** The pit_archives were created retroactively (e.g., the 2026-01-16 archive was `archived_at: 2026-04-10T14:09:50Z`). Prices for dates before the archive creation date may reflect split adjustments applied between the as-of date and the archive creation date. This is a standard yfinance behavior (adjusted close). It does NOT constitute look-ahead for ranking purposes (the production model uses scores, not prices), but the **return computation must use consistent price series** — both anchor and forward prices from the same archive (same adjustment basis). The scripts/eval_forward_returns.py approach of loading all prices from a single price_history.csv satisfies this requirement.

---

## 5. Provider Path Assessment (For Reference)

Since the recommended path requires NO external provider, this section documents why alternatives are unnecessary:

| Provider | Status | Look-ahead risk | Cost | Notes |
|----------|--------|----------------|------|-------|
| **pit_archives (existing)** | AVAILABLE — full gap coverage | NONE (PIT archive) | $0 | **RECOMMENDED** |
| yfinance | OPERATIONAL (recovered post-SIP-2026-003) | LOW (historical fetch) | $0 | Backup if archive gaps found during assembly |
| IEX Cloud | SCRIPT READY (`scripts/iex_cloud_price_download.py`) | LOW (historical endpoint) | $9/mo Starter | Use only if pit_archives have unexplained holes after assembly |
| Alpaca historical | BLOCKED — paper-tier subscription lacks `/v2/stocks/bars` | LOW | $0–paid | Requires subscription upgrade; not needed given archive coverage |
| Broker export | FEASIBLE but manual | NONE | $0 | Last resort; no advantage over pit_archives |

---

## 6. Minimum Validation Checks Required Before Backtest Extension

Before using the assembled gap returns in any performance conclusion:

1. **Price continuity check:** For each ticker, verify no sudden price jump > 50% between consecutive trading days (would indicate a split-adjustment inconsistency between archives).

2. **Coverage completeness check:** Confirm each gap snapshot has ≥ 28/30 top-30 tickers with a non-null anchor_close (threshold excludes ATXS but flags unexpected gaps).

3. **XBI correlation sanity:** On a 5-day return basis, confirm gap-period XBI returns are plausible vs. known market history (XBI Jan–May 2026 performance).

4. **ATXS exclusion confirmation:** Verify all ATXS rows have `anchor_close = null` and are excluded from return calculations after 2026-01-23.

5. **Archive integrity hash check:** For each pit_archive used, verify `price_history.csv` SHA256 against `manifest.json` before reading. (manifests are present and contain SHA256 for all files.)

6. **Era reconciliation check:** Ensure returns computed from gap pit_archives use the same adjustment convention as Era 1 (universe_prices.csv) and Era 2 (_forward_returns_panel.csv anchor_close). Sample 5 overlapping dates around 2026-01-15 to confirm price levels are consistent.

7. **No production model touch:** Assembly script must be placed in `scripts/research/` and must not import or modify any ranker, selector, sizing, final_score, or portfolio construction module.

---

## 7. ATXS Corporate Action Note

ATXS (Acelyrin Inc.) traded through 2026-01-23 and was acquired by Bristol-Myers Squibb (deal completed circa Jan 24, 2026). The last recorded price in all pit_archives is 2026-01-23. ATXS appears in 14 gap-period top-30 rankings after that date, reflecting a lag in the universe maintenance process. The correct treatment for backtest purposes is:

- For snapshots on or before 2026-01-23: compute returns normally.
- For snapshots after 2026-01-23: ATXS anchor_close = null; exclude from portfolio return average (portfolio N = 29 for those snapshots).

This is structurally correct and consistent with how delisted tickers are handled elsewhere in the backtest framework.

---

## 8. Confirmation of No Production File Modifications

This audit was conducted read-only. The following pre-existing modified files were observed but not touched:

- `artifacts/audit/cross_signal_forward_shadow/buckets.jsonl`
- `artifacts/audit/inst_delta_forward_shadow/checkpoints.jsonl`
- `data/expression_decision_log.jsonl`
- `artifacts/audit/SCIENTIFIC_CARTOGRAPHY_OPERATIONAL_REVIEW_2026_06_22.md`

No ranker, selector, sizing, final_score, portfolio, or pipeline production files were read, modified, or staged during this audit.

---

## Summary

The 2026-01-16 to 2026-05-07 PIT price gap is **closeable entirely from existing cached sources** (`data/pit_archives/`), with the following profile:

- **85/88** gap snapshot dates have a direct pit_archive
- **3/88** missing-archive dates have a usable prior-date archive as T-1 fallback
- **99.5%** of top-30 ticker-date pairs are covered
- **ATXS** accounts for all 14 uncoverable pairs (acquired Jan 23, 2026; no post-acquisition prices exist by definition)
- **XBI** is covered at 100% for all 88 gap snapshot dates
- **No external API needed**, no live price fetch required, no model changes required

The recommended closure path is a research-only assembly script that reads pit_archives, computes forward returns, and outputs a gap panel for appending to the existing `_forward_returns_panel.csv`. This does not constitute a production model change.

---

**Verdict:** `PASS_PIT_PRICE_GAP_CLOSURE_PATH_IDENTIFIED_NO_MODEL_CHANGE`
**Prepared:** 2026-06-22
**Auditor:** Claude (read-only audit agent)
