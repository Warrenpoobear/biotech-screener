# Universe Hygiene Audit Report

**Audit timestamp:** 2026-11-09T13:30:59.995943+00:00
**Reference date:** 2026-06-28
**Stale price cutoff:** 2026-06-15 (10 trading days before 2026-06-28)

---

## Classification

```
UNIVERSE_HYGIENE_AUDIT
COVERAGE_DIAGNOSTIC
NO_MODEL_CHANGE
NO_RANKER_CHANGE
NO_SELECTOR_CHANGE
NO_SIZING_CHANGE
NO_TRADING_CHANGE
```

---

## Executive Verdict

Model universe: **357 tickers**.
- Active/valid: **325**
- Already flagged delisted/inactive: **8** (APLS, GLPG, KALV, ACLX, DAWN, FOLD, TERN)
- Stale price (>10 trading days): **1** (no price since before 2026-06-15)
- Price data missing entirely: **0** (placeholder `_XBI_BENCHMARK_` entry)
- Pending / needs review: **23**

XBI ETF: **157 holdings** (source: SPDR_LIVE_XLSX (https://www.ssga.com/us/en/institutional/etfs/library-content/pr...)
IBB ETF: **242 holdings** (source: BLACKROCK_VARNISH_API_portfolioId=239699 (iShares Biotechnology ETF, asOfDate=20...)

Missing from model: **12 XBI** / **5 IBB** candidates
High-priority new candidates (XBI small/mid pure-play): **11**

**No production files were modified.** All findings are proposals only.

---

## Data Sources

- **Model universe:** `production_data/universe.json` — 357 entries
- **Split-adjusted prices:** `production_data/price_history_split_adj.csv` — clean ticker count varies
- **Raw prices:** `production_data/price_history.csv`
- **XBI holdings:** SPDR_LIVE_XLSX (https://www.ssga.com/us/en/institutional/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx) @ 2026-11-09T13:31:02.126777+00:00
- **IBB holdings:** BLACKROCK_VARNISH_API_portfolioId=239699 (iShares Biotechnology ETF, asOfDate=20260626) @ 2026-11-09T13:31:02.848867+00:00

---

## Current Model Universe Summary

| Metric | Value |
| --- | --- |
| Total tickers | 357 |
| Active (valid price) | 325 |
| Delisted / inactive | 8 |
| Active but stale price | 1 |
| Price data missing | 0 |
| Pending / needs review | 23 |

---

## XBI Coverage Section

XBI ETF (157 holdings) vs model (357 tickers).

- Tickers in XBI not in model: **12**
- Core biotech candidates (XBI small/mid): **11**

### Missing XBI Names

| ticker | name | missing_classification |
| --- | --- | --- |
| PURR | HYPERLIQUID STRATEGIES | MISSING_LOW_RELEVANCE |
| NKTR | NEKTAR THERAPEUTICS | MISSING_CORE_BIOTECH_CANDIDATE |
| COAG | HEMAB THERAPEUTICS HOLDINGS | MISSING_CORE_BIOTECH_CANDIDATE |
| IMMX | IMMIX BIOPHARMA INC | MISSING_CORE_BIOTECH_CANDIDATE |
| ACHV | ACHIEVE LIFE SCIENCES INC | MISSING_CORE_BIOTECH_CANDIDATE |
| AVTX | AVALO THERAPEUTICS INC | MISSING_CORE_BIOTECH_CANDIDATE |
| MNPR | MONOPAR THERAPEUTICS INC | MISSING_CORE_BIOTECH_CANDIDATE |
| STTK | SHATTUCK LABS INC | MISSING_CORE_BIOTECH_CANDIDATE |
| AKTS | AKTIS ONCOLOGY INC | MISSING_CORE_BIOTECH_CANDIDATE |
| OVID | OVID THERAPEUTICS INC | MISSING_CORE_BIOTECH_CANDIDATE |
| FTH | FAETH THERAPEUTICS INC | MISSING_CORE_BIOTECH_CANDIDATE |
| CBIO | CRESCENT BIOPHARMA INC | MISSING_CORE_BIOTECH_CANDIDATE |


---

## IBB Coverage Section

IBB ETF (242 holdings) vs model (357 tickers).

- Tickers in IBB not in model: **5**
- Core biotech candidates (IBB): **2**

### Missing IBB Names

| ticker | name | missing_classification |
| --- | --- | --- |
| LKFT | LAKEFRONT BIOTHERAPEUTICS ADR NV | MISSING_ADR_OR_FOREIGN |
| AKE | AKERO THERAPEUTICS CVR | MISSING_LOW_RELEVANCE |
| ADRO | CHINOOK THERAPEUTICS INC | MISSING_CORE_BIOTECH_CANDIDATE |
| THRD | THIRD HARMONIC BIO INC | MISSING_NEEDS_MANUAL_REVIEW |
| CRGX | CARGO THERAPEUTICS INC | MISSING_CORE_BIOTECH_CANDIDATE |


---

## High-Priority Quarantine Candidates

Tickers in model flagged as stale or inactive that should be reviewed for removal:

| ticker | company_name | model_classification | status | last_price |
| --- | --- | --- | --- | --- |
| APLS | Apellis Pharmaceuticals, Inc. | DELISTED_OR_INACTIVE | delisted | 2026-05-15 |
| GLPG | Lakefront Biotherapeutics NV | DELISTED_OR_INACTIVE | delisted | 2026-06-26 |
| KALV | KalVista Pharmaceuticals Inc | DELISTED_OR_INACTIVE | delisted | 2026-06-12 |
| SGMO | Sangamo Therapeutics, Inc. | DELISTED_OR_INACTIVE | delisted | 2026-07-01 |
| _XBI_BENCHMARK_ |  | ACTIVE_BUT_STALE_PRICE | benchmark | N/A |
| ACLX |  | DELISTED_OR_INACTIVE | delisted | 2026-04-29 |
| DAWN |  | DELISTED_OR_INACTIVE | delisted | 2026-04-24 |
| FOLD |  | DELISTED_OR_INACTIVE | delisted | 2026-04-27 |
| TERN |  | DELISTED_OR_INACTIVE | delisted | 2026-05-07 |


---

## Stale / Inactive Model Names

Staleness cutoff: last price before 2026-06-15 (10 trading days before 2026-06-28).

### Delisted / Inactive (7 tickers)

| Ticker | Company | Status | Last Price | Note |
| --- | --- | --- | --- | --- |
| APLS | Apellis Pharmaceuticals, Inc. | delisted | 2026-05-15 | Already flagged |
| GLPG | Lakefront Biotherapeutics NV | delisted | 2026-06-26 | Already flagged |
| KALV | KalVista Pharmaceuticals Inc | delisted | 2026-06-12 | Already flagged |
| SGMO | Sangamo Therapeutics, Inc. | delisted | 2026-07-01 | Already flagged |
| ACLX |  | delisted | 2026-04-29 | Already flagged |
| DAWN |  | delisted | 2026-04-24 | Already flagged |
| FOLD |  | delisted | 2026-04-27 | Already flagged |
| TERN |  | delisted | 2026-05-07 | Already flagged |

### Price Data Notes

- `_XBI_BENCHMARK_` — placeholder entry in universe.json, no price data (expected — it is a benchmark, not a trading ticker)
- `ATXS`, `CVAC`, `MRSN` — price data exists in CSV but these tickers are NOT in universe.json (orphaned price rows, likely historical)
- `IBB` — IBB ETF benchmark price row in price CSV, not a model constituent

---

## Identifier Conflicts

_No identifier conflicts detected._


---

## Known Corporate-Action Issue Names Status

| Ticker | Status |
| --- | --- |
| RNA | ACTIVE_VALID | status=excluded_acquired | last_price=2026-10-30 | name=Atrium Therapeutics, Inc. |
| GOSS | ACTIVE_VALID | status=active | last_price=2026-11-06 | name=Gossamer Bio, Inc. |
| REPL | ACTIVE_VALID | status=active | last_price=2026-11-06 | name=Replimune Group, Inc. |
| ACLX | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-04-29 | name= |
| APLS | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-05-15 | name=Apellis Pharmaceuticals, Inc. |
| DAWN | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-04-24 | name= |
| FOLD | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-04-27 | name= |
| GLPG | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-06-26 | name=Lakefront Biotherapeutics NV |
| KALV | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-06-12 | name=KalVista Pharmaceuticals Inc |
| TERN | DELISTED_OR_INACTIVE | status=delisted | last_price=2026-05-07 | name= |

**Notes:**
- `RNA` — Ticker was re-assigned. Old `RNA` (Avidity Biosciences) was acquired by Novartis. New `RNA` in model = Atrium Therapeutics (spun off). Status: active, price current. **Monitor: ensure this is the intended entity.**
- `GOSS` — Gossamer Bio; status: active, price current.
- `REPL` — Replimune Group; status: active, price current.
- `ACLX`, `DAWN`, `FOLD`, `TERN` — all status=delisted with None company name. Confirm delisting and purge data.
- `APLS` — Apellis Pharmaceuticals; status=delisted; last price 2026-05-15.
- `GLPG` — Lakefront Biotherapeutics NV; status=delisted.
- `KALV` — KalVista Pharmaceuticals; status=delisted; last price 2026-06-12.

---

## Recommended Universe Actions

### 1. Confirm and clean delisted tickers (7 tickers)
All 7 delisted tickers (`APLS`, `GLPG`, `KALV`, `ACLX`, `DAWN`, `FOLD`, `TERN`) are already flagged `status=delisted` in universe.json. No immediate action required, but their price history should be frozen and they should be excluded from all model scoring.

### 2. Investigate GOSS price staleness
GOSS (Gossamer Bio) shows `status=active` but price data in split-adjusted CSV may be stale — verify current price feed.

### 3. RNA ticker re-assignment watch
The ticker `RNA` now maps to Atrium Therapeutics (new post-spinoff entity). Ensure backtest data before the Novartis acquisition of old RNA (Avidity) is not contaminating model signals for the new entity.

### 4. ETF coverage gap candidates
If ETF data was successfully fetched: evaluate 11 XBI core biotech candidates and 2 IBB core biotech candidates for potential addition in the next universe refresh cycle.

### 5. Orphaned price rows
ATXS, CVAC, MRSN have price history rows but are not in universe.json. These are likely historical tickers that were removed. Consider cleaning the price CSV.

### 6. Corrupt price CSV rows
The price CSVs contain ~39 rows with malformed ticker names (contain 'TICKER\n' pattern). These appear to be DataFrame repr strings accidentally appended. The audit filtered them out — but the underlying data write should be fixed.

---

## Risks and Caveats

- ETF holdings may be from a stale snapshot if live fetch failed. Use `--fetch-current-etf-holdings` with a valid internet connection.
- XBI and IBB hold different subsets. Coverage gaps vs XBI are more actionable for small-cap pure-play biotech.
- The 10-trading-day stale threshold is a heuristic. Some tickers may have legitimate data gaps due to halts or low volume.
- RNA ticker reuse across two different companies is a known risk; the audit flags it but cannot automatically determine correctness.
- Large-cap biopharma exclusion list (`LARGE_CAP_EXCLUSIONS`) is hardcoded; verify it reflects current policy.

---

## Governance Conclusion

This audit is **read-only and proposals-only**. No production files were modified.

```
UNIVERSE_HYGIENE_AUDIT     ✓
COVERAGE_DIAGNOSTIC        ✓
NO_MODEL_CHANGE            ✓
NO_RANKER_CHANGE           ✓
NO_SELECTOR_CHANGE         ✓
NO_SIZING_CHANGE           ✓
NO_TRADING_CHANGE          ✓
```

---

## Next Validation Steps

1. Re-run with fresh XBI/IBB live fetch once network/auth is confirmed.
2. For each `MISSING_CORE_BIOTECH_CANDIDATE`: manual review of market cap, pipeline stage, liquidity before any add proposal.
3. For delisted tickers (ACLX, DAWN, FOLD, TERN — missing company name): source company name and final delisting date for archival.
4. Investigate and fix root cause of malformed price CSV rows (`TICKER\n` pattern).
5. Confirm RNA entity mapping with explicit audit trail.
6. Review KALV delisting date (last price 2026-06-12, very recent).
