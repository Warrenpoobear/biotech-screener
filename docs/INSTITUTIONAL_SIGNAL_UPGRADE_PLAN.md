# Institutional Signal + Risk Gates Upgrade Plan

## Overview

This document describes the upgrade to implement deterministic, audit-ready institutional signal scoring with risk gates, liquidity scoring, and governance audit trails.

---

## Current File Inventory

### Core Files

| File | Purpose |
|------|---------|
| `institutional_signal_report.py` | Main signal scoring and report generation |
| `institutional_alerts_CORRECTED.py` | Alternative/backup alerting implementation |

### Input Data Paths

| Path | Description |
|------|-------------|
| `production_data/holdings_snapshots.json` | 13F holdings data (current + prior quarters) |
| `production_data/market_data.json` | Market data including price, volume, market cap |
| `production_data/financial_data.json` | Financial metrics (Cash, Assets, Liabilities, R&D) |
| `production_data/financial_records.json` | Same schema as financial_data.json |
| `production_data/manager_registry.json` | Manager CIK to name mapping |

---

## Detected Schema Fields

### holdings_snapshots.json

```json
{
  "<TICKER>": {
    "market_cap_usd": <int>,
    "holdings": {
      "current": {
        "<MANAGER_CIK>": {
          "quarter_end": "YYYY-MM-DD",
          "state": "KNOWN|NOT_HELD",
          "shares": <int>,
          "value_kusd": <int>,
          "put_call": ""
        }
      },
      "prior": { /* same structure */ }
    },
    "filings_metadata": {
      "<MANAGER_CIK>": {
        "quarter_end": "YYYY-MM-DD",
        "accession": "<SEC_ACCESSION>",
        "total_value_kusd": <int>,
        "filed_at": "ISO_DATETIME",
        "is_amendment": <bool>
      }
    }
  }
}
```

### market_data.json

```json
[
  {
    "ticker": "<TICKER>",
    "price": <float>,
    "market_cap": <int>,
    "enterprise_value": <int>,
    "shares_outstanding": <int>,
    "float_shares": <int>,
    "avg_volume": <int>,             // 20-day average volume (shares)
    "avg_volume_90d": <float>,       // 90-day average volume (shares)
    "beta": <float>,
    "short_percent": <float>,
    "short_ratio": <float>,
    "52w_high": <float>,
    "52w_low": <float>,
    "high_90d": <float>,
    "low_90d": <float>,
    "volatility_90d": <float>,
    "returns_1m": <float>,
    "returns_3m": <float|null>,
    "exchange": "<EXCHANGE>",
    "sector": "<SECTOR>",
    "industry": "<INDUSTRY>",
    "collected_at": "YYYY-MM-DD"
  }
]
```

### financial_data.json / financial_records.json

```json
[
  {
    "ticker": "<TICKER>",
    "cik": "<CIK>",
    "Assets": <int>,
    "Assets_date": "YYYY-MM-DD",
    "CurrentAssets": <int>,
    "CurrentAssets_date": "YYYY-MM-DD",
    "Liabilities": <int>,
    "Liabilities_date": "YYYY-MM-DD",
    "CurrentLiabilities": <int>,
    "CurrentLiabilities_date": "YYYY-MM-DD",
    "ShareholdersEquity": <int>,
    "ShareholdersEquity_date": "YYYY-MM-DD",
    "Cash": <int>,
    "Cash_date": "YYYY-MM-DD",
    "Revenue": <int>,
    "Revenue_date": "YYYY-MM-DD",
    "COGS": <int>,                   // optional
    "COGS_date": "YYYY-MM-DD",
    "R&D": <int>,
    "R&D_date": "YYYY-MM-DD",
    "NetIncome": <int>,
    "NetIncome_date": "YYYY-MM-DD",
    "collected_at": "YYYY-MM-DD"
  }
]
```

---

## Proposed Field Mappings

### ADV (Average Dollar Volume) Calculation

**Priority order for direct ADV fields:**
1. `adv_usd_20d`
2. `adv_20d_usd`
3. `avg_volume_usd`
4. `avg_dollar_volume_20d`
5. `adv_usd`

**Fallback calculation (if no direct ADV field):**
```
ADV_USD = avg_volume * price
```

**Volume field candidates (in priority order):**
1. `avg_volume_20d`
2. `avg_volume`
3. `volume_avg_30d`
4. `volume`
5. `avg_volume_90d`

**Price field candidates (in priority order):**
1. `price`
2. `current_price`
3. `close`

### Market Cap Mapping

**Field candidates (in priority order):**
1. `market_cap`
2. `marketCap`
3. `market_cap_usd`

### Runway Months Calculation

**If `runway_months` field not present, compute from:**
```
burn_rate = R&D (quarterly) * 4  // annualized if quarterly
           OR NetIncome if negative * -1

runway_months = Cash / (burn_rate / 12)
```

**Fields used:**
- `Cash` - available cash
- `R&D` - R&D expense (proxy for burn)
- `NetIncome` - if negative, indicates cash burn

---

## New Files to Create

| File | Purpose |
|------|---------|
| `risk_gates.py` | Fail-closed risk gate module |
| `liquidity_scoring.py` | Tiered liquidity scoring with audit logging |
| `production_data/params_archive/` | Directory for frozen parameter snapshots |
| `tests/test_risk_gates.py` | Unit tests for risk gates |
| `tests/test_liquidity_scoring.py` | Unit tests for liquidity scoring |
| `tests/test_institutional_signal.py` | Unit tests for signal scoring |
| `tests/test_integration_institutional.py` | End-to-end determinism tests |

---

## Constants to Define

### Risk Gate Thresholds

```python
ADV_MINIMUM = 500_000          # $500K minimum ADV
PRICE_MINIMUM = 2.00           # $2.00 penny stock threshold
MARKET_CAP_MINIMUM = 50_000_000  # $50M micro cap floor
RUNWAY_MINIMUM_MONTHS = 6      # 6 months cash runway minimum
```

### Liquidity Scoring Tiers

| Tier | Market Cap Range | ADV Threshold (USD) |
|------|------------------|---------------------|
| Micro | < $300M | $750,000 |
| Small | $300M - $2B | $2,000,000 |
| Mid | $2B - $10B | $5,000,000 |
| Large | >= $10B | $10,000,000 |

### Scoring Components

| Component | Range | Formula |
|-----------|-------|---------|
| ADV Score | 0-70 | Linear: 0 at 0, 70 at 2x tier threshold |
| Spread Score | 0-30 | Linear: 30 at <=50bps, 0 at >=400bps |
| Penny Stock Cap | max 10 | If price < $2.00 |

---

## Output Schema Additions

### Signal Score Output (enhanced)

```json
{
  "ticker": "<TICKER>",
  "mgrs": <int>,
  "pct_change": <float>,
  "net_buyers": <int>,
  "signal_score": <int>,
  "net_flow_kusd": <float>,
  "conviction_avg": <float>,
  "new": ["<MANAGER_NAME>", ...],
  "increased": ["<MANAGER_NAME>", ...],
  "decreased": ["<MANAGER_NAME>", ...],
  "passes_gates": <bool>,
  "risk_flags": ["<FLAG>", ...],
  "gate_results": {
    "adv_usd": <float>,
    "price": <float>,
    "market_cap": <float>,
    "runway_months": <float>
  },
  "score_version": "<VERSION>",
  "parameters_hash": "<SHA256_PREFIX>"
}
```

### Risk Flags

- `ADV_UNKNOWN` - Cannot compute ADV (fail-closed)
- `LOW_LIQUIDITY` - ADV below minimum threshold
- `PENNY_STOCK` - Price below $2.00
- `MICRO_CAP` - Market cap below minimum
- `CASH_RISK` - Runway below 6 months
- `WIDE_SPREAD` - Spread > 400bps (if available)

---

## Audit Trail Schema

### Run Audit Entry (JSONL)

```json
{
  "timestamp": "ISO_DATETIME",
  "pipeline_version": "<VERSION>",
  "score_version": "<VERSION>",
  "parameters_hash": "<SHA256>",
  "inputs": {
    "holdings_hash": "<SHA256>",
    "market_data_hash": "<SHA256>",
    "financial_data_hash": "<SHA256>"
  },
  "output_hash": "<SHA256>",
  "dry_run": <bool>,
  "resume": <bool>,
  "as_of_date": "YYYY-MM-DD"
}
```

### Checkpoint Schema

```json
{
  "stage_name": "<STAGE>",
  "completed_at": "ISO_DATETIME",
  "input_hashes": {...},
  "output_hash": "<SHA256>",
  "parameters_hash": "<SHA256>",
  "score_version": "<VERSION>"
}
```

---

## Dependency Graph

```
SUBTASK 0: Inventory (this doc)
    |
    v
SUBTASK 1: Risk Gates (risk_gates.py)
    |
    +-----> SUBTASK 2: Liquidity Scoring
    |
    v
SUBTASK 3: Signal Scoring Updates
    |
    v
SUBTASK 4: Report Generation Updates
    |
    v
SUBTASK 5: Pipeline Orchestration
    |
    v
SUBTASK 6: Integration Tests
```

---

## Migration Notes

1. **Backwards Compatibility**: Old reports without risk gates will continue to work; risk gates are additive.

2. **Parameter Versioning**: First version will be `1.0.0`. Parameter changes bump minor version.

3. **Determinism**: All timestamps derived from `as_of_date`, no `datetime.now()` in scoring.

4. **Fail-Closed Philosophy**: Unknown ADV = rejected, not assumed to pass.

---

## Review Checklist

- [ ] All field mappings verified against actual data files
- [ ] Thresholds reviewed with domain expertise
- [ ] Test fixtures created for edge cases (GOSS-proof scenarios)
- [ ] Audit log format approved
- [ ] Parameter archive directory created

---

*Document created: 2026-01-11*
*Author: Institutional Signal Upgrade Project*
