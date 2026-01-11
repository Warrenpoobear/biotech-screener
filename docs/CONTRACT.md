# Backtest Harness + 12-Quarter 13F Extraction Contract

## Overview

This document defines the data contracts, file formats, and determinism rules
for the historical 13F extraction and backtest harness system.

## 1. Snapshot Storage Locations

### Holdings History Directory
```
production_data/holdings_history/
├── manifest.json                    # Run metadata + per-quarter index
├── holdings_2025-12-31.json        # Q4 2025 snapshot
├── holdings_2025-09-30.json        # Q3 2025 snapshot
├── holdings_2025-06-30.json        # Q2 2025 snapshot
├── holdings_2025-03-31.json        # Q1 2025 snapshot
├── ... (up to 12 quarters)
└── meta_2025-12-31.json            # Optional per-quarter metadata
```

### Backtest Output Directory
```
backtests/
├── backtest_results.json           # Full backtest results (canonical JSON)
├── BACKTEST_SUMMARY.txt            # Human-readable summary
└── BACKTEST_PER_QUARTER_TABLE.txt  # Per-quarter tabular results
```

## 2. File Schemas

### 2.1 Per-Quarter Holdings Snapshot

**Filename pattern:** `holdings_{YYYY-MM-DD}.json`

```json
{
  "_schema": {
    "version": "13f_holdings_snapshot_v1",
    "quarter_end": "2025-09-30",
    "prior_quarter_end": "2025-06-30",
    "created_by": "extract_13f_history.py"
  },
  "_governance": {
    "run_id": "abc123...",
    "score_version": "v1",
    "parameters_hash": "...",
    "input_lineage": [...]
  },
  "tickers": {
    "TICKER": {
      "market_cap_usd": 1234567890,
      "holdings": {
        "current": {
          "CIK_10_DIGIT": {
            "quarter_end": "2025-09-30",
            "state": "KNOWN",
            "shares": 100000,
            "value_kusd": 5000,
            "put_call": ""
          }
        },
        "prior": {
          "CIK_10_DIGIT": {
            "quarter_end": "2025-06-30",
            "state": "KNOWN",
            "shares": 80000,
            "value_kusd": 4000,
            "put_call": ""
          }
        }
      },
      "filings_metadata": {
        "CIK_10_DIGIT": {
          "quarter_end": "2025-09-30",
          "accession": "0001234567-25-012345",
          "total_value_kusd": 50000000,
          "filed_at": "2025-11-14T00:00:00",
          "is_amendment": false
        }
      }
    }
  },
  "managers": {
    "CIK_10_DIGIT": {
      "name": "Manager Name",
      "aum_b": 10.5,
      "style": "concentrated_conviction"
    }
  },
  "stats": {
    "tickers_count": 50,
    "managers_count": 9,
    "total_positions": 200
  },
  "warnings": []
}
```

### 2.2 Manifest File

**Filename:** `manifest.json`

```json
{
  "_schema": {
    "version": "13f_manifest_v1"
  },
  "run_id": "abc123...",
  "params": {
    "quarter_end": "2025-12-31",
    "quarters": 12,
    "manager_registry_hash": "...",
    "universe_hash": "...",
    "cusip_map_hash": "..."
  },
  "quarters": [
    {
      "quarter_end": "2025-12-31",
      "prior_quarter_end": "2025-09-30",
      "filename": "holdings_2025-12-31.json",
      "sha256": "...",
      "tickers_count": 50,
      "managers_count": 9,
      "warnings_count": 0
    }
  ],
  "input_hashes": [
    {"path": "manager_registry.json", "sha256": "..."},
    {"path": "universe.json", "sha256": "..."},
    {"path": "cusip_static_map.json", "sha256": "..."}
  ]
}
```

### 2.3 Prices CSV Format

**Filename:** `prices.csv` or `daily_prices.csv`

```csv
date,ticker,adj_close
2025-01-02,XBI,98.50
2025-01-02,REGN,750.25
2025-01-03,XBI,99.10
2025-01-03,REGN,755.00
...
```

**Rules:**
- `date` in YYYY-MM-DD format
- `ticker` uppercase, no spaces
- `adj_close` decimal with up to 6 decimal places
- Sorted by (date ASC, ticker ASC)
- Must include benchmark ticker (default: XBI)

### 2.4 Backtest Results

**Filename:** `backtest_results.json`

```json
{
  "_schema": {
    "version": "backtest_results_v1"
  },
  "_governance": {
    "run_id": "...",
    "score_version": "v1",
    "parameters_hash": "...",
    "manifest_hash": "...",
    "prices_hash": "..."
  },
  "params": {
    "holdings_history_dir": "production_data/holdings_history",
    "prices_csv": "backtests/prices.csv",
    "benchmark": "XBI",
    "horizons": [30, 60, 90],
    "topk": [10, 25],
    "min_managers": 4
  },
  "quarters_used": 10,
  "per_quarter": [
    {
      "quarter_end": "2024-09-30",
      "signal_date": "2024-10-01",
      "topk_10": {
        "tickers": ["TICK1", "TICK2", ...],
        "scores": [50, 48, ...],
        "horizons": {
          "30": {
            "mean_return": 0.0523,
            "median_return": 0.0412,
            "benchmark_return": 0.0234,
            "mean_excess": 0.0289,
            "hit_rate": 0.70
          }
        }
      },
      "turnover_vs_prior": 0.30
    }
  ],
  "aggregate": {
    "topk_10": {
      "horizons": {
        "30": {
          "mean_return": 0.0456,
          "median_return": 0.0380,
          "mean_excess": 0.0210,
          "hit_rate": 0.65,
          "quarters_count": 10
        }
      }
    }
  }
}
```

## 3. Determinism Rules

### 3.1 JSON Canonicalization
- All dict keys sorted recursively
- Floats rounded to 6 decimal places (or use string representation)
- No NaN or Infinity values (raise error)
- Lists NOT reordered by serializer (caller must sort)
- Output ends with single trailing newline
- Encoding: UTF-8, no BOM

### 3.2 Sorting Requirements
- Tickers: alphabetical ascending
- Manager CIKs: string sort ascending (preserves leading zeros)
- Quarters: chronological descending (newest first in processing, but can be stored either order - document choice)
- Holdings within ticker: sorted by CIK

### 3.3 Date Handling
- All dates as ISO format strings: `YYYY-MM-DD`
- Quarter ends: March 31, June 30, September 30, December 31
- Prior quarter calculation: subtract 3 months, adjust to valid quarter end
- No `datetime.now()` in outputs - use explicit `as_of_date` parameter

### 3.4 Hashing
- SHA256 for all file hashes
- Canonical JSON (no indent, sorted keys) before hashing objects
- 64-character hex digest (full) or 16-character (truncated for run_id)

### 3.5 Tie-Breaking for Rankings
When scores are equal:
1. Higher score (desc)
2. More managers (desc)
3. Ticker alphabetically (asc)

### 3.6 Forward Return Calculation
- Entry date: first trading day AFTER quarter_end
- Exit date: first trading day >= (entry_date + horizon_days)
- Return = (exit_price / entry_price) - 1
- Excess return = ticker_return - benchmark_return
- If no price data available: exclude ticker from that quarter's analysis

## 4. Error Handling

### 4.1 Fail-Closed Behavior
- Missing input files: raise exception, do not proceed
- Invalid JSON: raise exception with clear message
- Schema validation failures: raise exception

### 4.2 Partial Success (Extraction Only)
- If a manager's filing cannot be retrieved: record warning, continue
- If CUSIP cannot be mapped: skip that holding, record warning
- Warnings collected in `warnings` array per snapshot

## 5. CLI Interfaces

### extract_13f_history.py
```bash
python scripts/extract_13f_history.py \
  --quarter-end 2025-12-31 \
  --quarters 12 \
  --manager-registry production_data/manager_registry.json \
  --universe production_data/universe.json \
  --cusip-map production_data/cusip_static_map.json \
  --out-dir production_data/holdings_history/ \
  --mode live
```

### backtest_institutional_signal.py
```bash
python scripts/backtest_institutional_signal.py \
  --holdings-history-dir production_data/holdings_history \
  --prices-csv backtests/prices.csv \
  --benchmark XBI \
  --horizons 30,60,90 \
  --topk 10,25 \
  --min-managers 4 \
  --score-version v1 \
  --out backtests/backtest_results.json
```

## 6. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial contract definition |
