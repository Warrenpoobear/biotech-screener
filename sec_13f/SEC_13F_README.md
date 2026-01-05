# SEC 13F Provider — Elite Manager Co-Investment Signals

**Wake Robin Biotech Alpha System**

This module fetches and aggregates 13F holdings from elite biotech hedge funds
to generate co-investment signals.

## Components

| File | Purpose |
|------|---------|
| `elite_managers.py` | Registry of ~13 elite biotech managers with CIKs, tiers, weights |
| `cusip_resolver.py` | CUSIP→ticker mapping with local cache + OpenFIGI fallback |
| `edgar_13f.py` | SEC EDGAR fetcher/parser for 13F-HR filings |
| `aggregator.py` | Combines holdings across managers into conviction signals |
| `test_sec_13f.py` | 31 tests covering all components |

## Installation

```
wake_robin/
└── providers/
    └── sec_13f/
        ├── __init__.py
        ├── elite_managers.py
        ├── cusip_resolver.py
        ├── edgar_13f.py
        └── aggregator.py
tests/
└── test_sec_13f.py
```

## Quick Start

### 1. Fetch a single manager's holdings

```python
from wake_robin.providers.sec_13f.edgar_13f import get_manager_holdings

# Baker Bros (CIK 1074999)
result = get_manager_holdings('1074999')

print(f"Total: ${result['summary']['total_value']:,.0f}")
for h in result['holdings'][:10]:
    print(f"  {h['ticker']}: ${h['value']:,.0f}")
```

### 2. Get aggregated signals across all elite managers

```python
from wake_robin.providers.sec_13f.aggregator import ElitePositionAggregator
from wake_robin.providers.sec_13f.elite_managers import get_tier_1_managers

# Tier 1 only for speed
agg = ElitePositionAggregator(managers=get_tier_1_managers())

# Top signals (held by 2+ managers)
signals = agg.get_top_signals(n=20, min_overlap=2)

for s in signals:
    print(f"{s.ticker}: score={s.conviction_score:.0f}, holders={s.holders}")
```

### 3. CLI testing

```powershell
# Single manager holdings
python edgar_13f.py 1074999

# Aggregated signals
python aggregator.py
```

## Signal Output

The `AggregatedSignal` provides:

```python
{
    'ticker': 'VRTX',
    'issuer_name': 'VERTEX PHARMACEUTICALS',
    'overlap_count': 4,        # Managers holding
    'tier_1_count': 3,         # Tier 1 managers
    'conviction_score': 82.5,  # 0-100 score
    'holders': ['Baker Bros', 'RA Capital', 'Perceptive', 'BVF'],
    'managers_adding': ['Baker Bros'],  # Increased this quarter
    'new_positions': [],                 # New this quarter
}
```

## Point-in-Time Safety

- `filing_date` = when 13F was filed (public knowledge date)
- `report_date` = quarter-end the holdings represent

For backtesting, filter by `filing_date`:

```python
from datetime import date

signals = agg.compute_signals(
    as_of_date=date(2025, 11, 15)  # Only use filings filed by this date
)
```

## Conviction Score (0-100)

Factors:
- **Overlap** (40 pts max): 8 points per manager holding
- **Tier weighting** (30 pts max): Higher for Tier 1 concentrated managers
- **Position size** (20 pts max): Larger % of portfolio = more conviction
- **Momentum** (10 pts max): Bonus for new positions and additions

## Elite Manager Tiers

| Tier | Managers | Weight | Description |
|------|----------|--------|-------------|
| 1 | Baker Bros, RA Capital, Perceptive, BVF, EcoR1 | 1.0x | Pure biotech specialists, 20+ year track records |
| 2 | OrbiMed, Redmile, Deerfield, Farallon, Citadel | 0.7x | Excellent healthcare allocators |
| 3 | Avoro, Venrock HCP, Cormorant | 0.4x | Notable but smaller/narrower |

## CUSIP Resolution

The CUSIP cache builds over time:
1. Hardcoded mappings for common biotech CUSIPs (~40 names)
2. Local cache (`data/cusip_cache.json`) persists resolved CUSIPs
3. OpenFIGI API for unknowns (rate limited, requires internet)

For determinism, the cache ensures same CUSIP always returns same ticker.

## Integration with Scoring Modules

Wire into Module 3 (Catalyst) or create Module 6:

```python
def get_elite_signal_score(ticker: str, signals: dict) -> float:
    """
    Convert elite conviction to 0-100 score for composite.
    """
    signal = signals.get(ticker)
    if not signal:
        return 50.0  # Neutral
    
    return signal.conviction_score
```

## Tests

```powershell
pytest test_sec_13f.py -v
# 31 passed
```

## Next Steps

1. Run `python aggregator.py` to fetch live data
2. Spot-check top signals against your intuition
3. Integrate into screener's composite scoring
4. Add to backtest framework
