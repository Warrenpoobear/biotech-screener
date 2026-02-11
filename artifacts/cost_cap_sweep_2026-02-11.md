# Cost Impact Cap Calibration Sweep

**Date**: 2026-02-11
**Sweep dimension**: `cost_impact_cap_bps` only (all else held constant)
**Baseline**: disabled (enable_cost_haircut=False), ruleset 34bb662d
**Buckets**: (400, 1.0), (1000, 0.85), (2000, 0.70), floor=0.55
**Window**: 2025-01-01 to 2025-12-31 (10 snapshots, 43 unique tickers)

---

## Sweep Table

| Cap (bps) | Bind% | P10 | P50 | P90 | None% | Mild% | Heavy% | Floor% | Band chg | Δ20d | Δ60d |
|-----------|-------|-----|-----|------|-------|-------|--------|--------|----------|------|------|
| 500 | 24.6% | 264 | 514 | 1020 | 28.4% | 43.7% | 27.9% | **0.0%** | 0 | +0.83 | +1.87 |
| **1000** | **2.7%** | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |
| 1500 | 1.1% | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |
| 2000 | 0.5% | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |
| 3000 | 0.0% | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |
| 5000 | 0.0% | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |
| 10000 | 0.0% | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |
| uncapped | 0.0% | 264 | 514 | 1888 | 28.4% | 43.7% | 22.4% | 5.5% | 0 | +0.70 | +1.61 |

*Δ columns are vs disabled baseline (20d=+7.19%, 60d=+22.97%).*

### Column definitions

- **Bind%**: % of costed position-snapshots where impact hits the cap (from `cost_telemetry`)
- **P10/P50/P90**: percentiles of `est_cost_bps` (round-trip) across portfolio positions
- **None/Mild/Heavy/Floor%**: share of positions in each bucket (1.00x / 0.85x / 0.70x+step / 0.55x+step)
- **Band chg**: count of band differences vs disabled baseline
- **Δ20d/Δ60d**: change in mean weighted residual return vs disabled baseline (pp)

---

## Key Findings

### 1. Clear elbow at cap=1000

The sweep reveals two distinct regimes:

- **cap < 1000** (degenerate): High binding (24.6% at 500) compresses the cost
  distribution, eliminating the floor bucket entirely. All tail-illiquid names
  are pulled below 2000 bps round-trip, removing the maximum penalty tier.

- **cap >= 1000** (stable plateau): Binding drops to <=2.7% and all metrics
  (bucket shares, P90, residual, turnover) are identical from 1000 through
  uncapped. The cap simply stops mattering.

### 2. Performance is cap-insensitive above the elbow

From cap=1000 through uncapped (1e9), every metric is byte-for-byte identical:
+1.61pp 60d residual, 28/44/22/6% bucket distribution, zero band changes vs
baseline. This means the cost model's natural impact estimates (without capping)
already fall within the bucket thresholds — the cap only matters for the 2-3 most
illiquid names at cap=500.

### 3. Cap=500's +0.26pp advantage is noise

Cap=500 shows +1.87pp vs +1.61pp at 60d. This 0.26pp gap comes from 3 snapshots
(Aug-Oct 2025) where 1-2 floor-bucket names at cap>=1000 drag slightly. With
N=10 snapshots, this is well within sampling noise. More importantly, cap=500
loses the floor bucket entirely — the feature can't maximally penalize truly
untradeable names (NAUT at $0.6M ADV, IKT at $0.8M ADV).

### 4. The impact model is well-behaved

The fact that uncapped performance equals cap=1000 confirms the square-root
impact model (`0.10 * sqrt(participation) * 10000`) produces realistic estimates
for this universe. There is no need for aggressive capping to prevent blow-ups.

---

## Chosen Cap: 1000 bps

**`cost_impact_cap_bps = 1000`** is the recommended production value.

Rationale: It is the smallest cap that (a) passes the <20% binding threshold
(2.7%), (b) preserves all 4 bucket tiers including the floor for truly illiquid
names, and (c) sits on the performance plateau — identical to all higher caps
through uncapped. The default of 200 caused 71.6% cap binding (severe
degeneracy); 1000 reduces this to 2.7% while providing a safety ceiling against
model blow-ups if the universe ever includes sub-$100K ADV names.

---

## Promotion Recommendation

The cost-aware sizing feature is ready for promotion with these parameters:

| Parameter | Value | Basis |
|-----------|-------|-------|
| `enable_cost_haircut` | `True` | Backtest shows +1.61pp 60d residual |
| `cost_impact_cap_bps` | `1000` | Sweep elbow; 2.7% binding, full bucket spread |
| `cost_haircut_buckets` | `(400, 1000, 2000)` | Biotech P30/P70/P90 percentile breaks |
| `cost_haircut_floor_mult` | `0.55` | Default (unchanged) |

**Expected production behavior**:
- 5 of 19 A+B names (26%) get no haircut (ADV > ~$75M)
- 6 names (32%) get mild 0.85x haircut
- 6 names (32%) get 0.70x + band step-down
- 2 names (11%) get 0.55x floor + band step-down (NAUT, IKT)
- Eligibility: unchanged
- Membership: unchanged (same 19 names in portfolio)
- Turnover: unchanged (34.6% position, 28.4% weight)

**Remaining gate before promotion**: None from a data/calibration standpoint.
Coverage is 100%, bucket spread is healthy, cap is calibrated, performance is
positive. The feature can be promoted via `bump_ruleset.py` whenever the operator
is ready.
