# Walk-Forward Validation Report

Generated: 2026-02-12T15:03:28
Ruleset: dcdcccc8 | Tier filter: ['A', 'B'] | Top-K: 20 | Snapshots: 22 | Date range: 2024-01-31 to 2025-10-31

## Tier Separation (60d forward returns)

| Tier | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| A | 104 | +10.89 | +3.89 | 55.4 | -25.41 |
| B | 369 | +8.75 | -1.73 | 46.2 | -28.06 |
| C | 585 | +1.07 | -2.18 | 45.3 | -27.94 |
| D | 2864 | +17.59 | +2.14 | 51.9 | -32.44 |

## Band Separation (60d forward returns)

| Band | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| L | 729 | +6.17 | -1.78 | 46.0 | -27.34 |
| M | 266 | +2.49 | +0.00 | 49.6 | -28.23 |
| S | 55 | +3.30 | -3.41 | 46.0 | -29.31 |
| XS | 2872 | +17.48 | +2.00 | 51.8 | -32.45 |

## Catalyst Strength Distribution (60d forward returns)

| Strength | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|----------|---|--------|----------|------------|---------------|
| near | 504 | +21.52 | +5.54 | 55.1 | -30.12 |
| mid | 361 | +16.02 | +3.95 | 54.3 | -28.98 |
| far | 619 | +17.72 | -0.21 | 49.4 | -29.05 |
| missing | 2438 | +11.86 | -0.40 | 49.4 | -32.25 |

## Cost Analysis

### Gross vs Net Tier Separation (60d forward returns)

| Tier | N | Gross Mean % | Net Mean % | Gross Hit % | Net Hit % |
|------|---|-------------|-----------|------------|----------|
| A | 104 | +10.89 | +6.90 | 55.4 | 49.5 |
| B | 369 | +8.75 | +5.61 | 46.2 | 42.2 |
| C | 585 | +1.07 | +1.07 | 45.3 | 45.3 |
| D | 2864 | +17.59 | +17.59 | 51.9 | 51.9 |

### Cost Distribution (portfolio positions)

- Portfolio rows: 404
- Median round-trip cost: 406.0 bps
- Mean round-trip cost: 385.7 bps
- P95 round-trip cost: 420.0 bps
- Median participation: 9.32%
- P95 participation: 105.06%
- Low-ADV positions (< $500K): 0
- At $50M AUM: median 406.0 bps round-trip

## Eligible vs Ineligible (60d forward returns)

| Group | N | Mean % | Median % | Hit Rate % |
|-------|---|--------|----------|------------|
| Eligible | 957 | +4.88 | -1.69 | 46.7 |
| Ineligible | 2862 | +17.59 | +2.14 | 51.9 |

## Rescued by Relative Gate (60d)

Tickers where abs drawdown breached but XBI-relative did not. Survived via AND gate (require_both=True).

| Group | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|-------|---|--------|----------|------------|---------------|
| Rescued | 11 | +13.95 | +17.46 | 63.6 | -20.62 |
| Clean eligible | 932 | +5.23 | -1.62 | 47.0 | -26.76 |

Spread (rescued - clean): +19.08pp

## Gate Pressure

Share of dev tickers within +/-5pp of each gate threshold (panel-wide aggregate).

| Metric | Value |
|--------|-------|
| DD abs near gate | 12.2% |
| DD rel near gate | 9.5% |
| Optionality near A-floor | 10.0% |
| Rescued share | 1.0% |

Rescued: 11 / 1058 eligible

## Stability

- Mean top-25 overlap (Jaccard): 70.0%
- Mean position turnover: 30.0%
- Date transitions: 21

## Coverage

- Panel rows: 3922 | With fwd returns: 3819 (97.4%) | With max-DD: 3922 (100.0%)

