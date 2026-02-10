# Walk-Forward Validation Report

Generated: 2026-02-10T13:22:06
Ruleset: a2ea153c | Tier filter: ['A', 'B'] | Top-K: 20 | Snapshots: 22 | Date range: 2024-01-31 to 2025-10-31

## Tier Separation (60d forward returns)

| Tier | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| A | 59 | +6.72 | -1.81 | 47.4 | -26.03 |
| B | 273 | +3.70 | -1.86 | 45.0 | -28.11 |
| C | 484 | +0.11 | -2.37 | 44.7 | -26.83 |
| D | 3106 | +17.30 | +2.09 | 51.8 | -32.22 |

## Band Separation (60d forward returns)

| Band | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|---|--------|----------|------------|---------------|
| L | 557 | +2.85 | -2.04 | 43.9 | -26.60 |
| M | 191 | +3.11 | +0.88 | 51.6 | -27.62 |
| S | 61 | -6.45 | -8.74 | 41.1 | -30.09 |
| XS | 3113 | +17.20 | +1.88 | 51.8 | -32.23 |

## Catalyst Strength Distribution (60d forward returns)

| Strength | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|----------|---|--------|----------|------------|---------------|
| near | 435 | +21.07 | +5.56 | 55.5 | -29.92 |
| mid | 360 | +16.98 | +4.50 | 55.6 | -29.10 |
| far | 644 | +18.20 | -0.15 | 49.5 | -29.07 |
| missing | 2483 | +11.90 | -0.63 | 49.2 | -32.24 |

## Eligible vs Ineligible (60d forward returns)

| Group | N | Mean % | Median % | Hit Rate % |
|-------|---|--------|----------|------------|
| Eligible | 717 | +1.89 | -1.96 | 45.0 |
| Ineligible | 3102 | +17.30 | +2.09 | 51.8 |

## Stability

- Mean top-25 overlap (Jaccard): 72.3%
- Mean position turnover: 27.7%
- Date transitions: 21

## Coverage

- Panel rows: 3922 | With fwd returns: 3819 (97.4%) | With max-DD: 3922 (100.0%)

