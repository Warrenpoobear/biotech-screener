# Walk-Forward Validation Report

Generated: 2026-02-12T12:32:11
Ruleset: dcdcccc8 | Tier filter: ['A', 'B'] | Top-K: 20 | Snapshots: 22 | Date range: 2024-01-31 to 2025-10-31

## Tier Separation (60d forward returns)

| Tier | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| A | 91 | +11.26 | +3.73 | 54.5 | -25.32 |
| B | 376 | +9.03 | -1.69 | 47.3 | -28.04 |
| C | 591 | +0.96 | -2.32 | 45.0 | -27.91 |
| D | 2864 | +17.59 | +2.14 | 51.9 | -32.44 |

## Band Separation (60d forward returns)

| Band | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| L | 826 | +5.80 | -1.78 | 46.0 | -27.46 |
| M | 180 | +2.42 | +0.66 | 51.0 | -28.98 |
| S | 46 | +1.53 | -1.16 | 48.8 | -26.53 |
| XS | 2870 | +17.51 | +2.00 | 51.8 | -32.45 |

## Catalyst Strength Distribution (60d forward returns)

| Strength | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|----------|---|--------|----------|------------|---------------|
| near | 435 | +21.07 | +5.56 | 55.5 | -29.92 |
| mid | 360 | +16.98 | +4.50 | 55.6 | -29.10 |
| far | 1590 | +13.44 | +0.80 | 50.8 | -29.74 |
| missing | 1537 | +12.92 | -1.34 | 47.7 | -33.49 |

## Cost Analysis

### Gross vs Net Tier Separation (60d forward returns)

| Tier | N | Gross Mean % | Net Mean % | Gross Hit % | Net Hit % |
|------|---|-------------|-----------|------------|----------|
| A | 91 | +11.26 | +7.30 | 54.5 | 48.9 |
| B | 376 | +9.03 | +5.82 | 47.3 | 42.8 |
| C | 591 | +0.96 | +0.96 | 45.0 | 45.0 |
| D | 2864 | +17.59 | +17.59 | 51.9 | 51.9 |

### Cost Distribution (portfolio positions)

- Portfolio rows: 403
- Median round-trip cost: 406.0 bps
- Mean round-trip cost: 386.7 bps
- P95 round-trip cost: 420.0 bps
- Median participation: 8.39%
- P95 participation: 112.8%
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

- Mean top-25 overlap (Jaccard): 69.9%
- Mean position turnover: 30.1%
- Date transitions: 21

## Coverage

- Panel rows: 3922 | With fwd returns: 3819 (97.4%) | With max-DD: 3922 (100.0%)

