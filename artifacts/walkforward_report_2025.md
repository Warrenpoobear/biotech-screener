# Walk-Forward Validation Report

Generated: 2026-02-10T06:56:19
Ruleset: d3cdf5c8 | Tier filter: ['A', 'B'] | Top-K: 20 | Snapshots: 10 | Date range: 2025-01-31 to 2025-10-31

## Tier Separation (60d forward returns)

| Tier | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| A | 26 | +19.45 | +24.08 | 69.2 | -23.75 |
| B | 182 | +28.38 | +11.74 | 64.2 | -26.55 |
| C | 249 | +19.01 | +13.82 | 66.1 | -24.28 |
| D | 1351 | +29.51 | +14.84 | 65.2 | -29.43 |

## Band Separation (60d forward returns)

| Band | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| L | 283 | +26.20 | +14.43 | 67.7 | -24.59 |
| M | 151 | +14.62 | +8.29 | 60.6 | -25.98 |
| S | 22 | +33.56 | +25.59 | 68.2 | -26.84 |
| XS | 1352 | +29.49 | +14.83 | 65.3 | -29.42 |

## Eligible vs Ineligible (60d forward returns)

| Group | N | Mean % | Median % | Hit Rate % |
|-------|---|--------|----------|------------|
| Eligible | 444 | +22.81 | +13.84 | 65.5 |
| Ineligible | 1349 | +29.51 | +14.84 | 65.2 |

## Stability

- Mean top-25 overlap (Jaccard): 84.6%
- Mean position turnover: 15.4%
- Date transitions: 9

## Coverage

- Panel rows: 1808 | With fwd returns: 1793 (99.2%) | With max-DD: 1808 (100.0%)

