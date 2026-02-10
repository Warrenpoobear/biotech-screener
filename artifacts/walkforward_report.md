# Walk-Forward Validation Report

Generated: 2026-02-09T22:42:15
Ruleset: d3cdf5c8 | Tier filter: ['A', 'B'] | Top-K: 20 | Snapshots: 22 | Date range: 2024-01-31 to 2025-10-31

## Tier Separation (60d forward returns)

| Tier | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| A | 65 | +8.78 | +4.12 | 61.5 | -25.95 |
| B | 526 | +22.99 | -1.40 | 47.8 | -28.75 |
| C | 716 | +9.42 | -1.47 | 47.1 | -28.76 |
| D | 2615 | +14.12 | +1.69 | 51.7 | -32.45 |

## Band Separation (60d forward returns)

| Band | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| L | 776 | +22.42 | -1.12 | 48.3 | -28.07 |
| M | 434 | +5.58 | +0.30 | 50.4 | -29.16 |
| S | 86 | +0.37 | -8.74 | 40.0 | -29.52 |
| XS | 2626 | +13.95 | +1.55 | 51.5 | -32.48 |

## Eligible vs Ineligible (60d forward returns)

| Group | N | Mean % | Median % | Hit Rate % |
|-------|---|--------|----------|------------|
| Eligible | 1206 | +15.03 | -0.94 | 48.2 |
| Ineligible | 2613 | +14.12 | +1.69 | 51.7 |

## Stability

- Mean top-25 overlap (Jaccard): 79.7%
- Mean position turnover: 20.3%
- Date transitions: 21

## Coverage

- Panel rows: 3922 | With fwd returns: 3819 (97.4%) | With max-DD: 3922 (100.0%)

