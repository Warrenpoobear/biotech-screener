# SFO Liquidity Architecture Skill

## Purpose

Reference for the Wake Robin Liquidity Architecture - a deterministic, multi-engine modeling stack for a Gen3-Gen5 single-family office. This is NOT a generic asset-allocation framework. Every modeling decision is downstream of the four-line principle.

---

## Operator Context

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin (Holland, MI)
- **SFO modeling authority**: 30+ years institutional investment management. Built the North Dakota Legacy Fund ($6B sovereign wealth fund) from inception. Managed $14B+ across pension, insurance, and sovereign wealth pools as Deputy CIO/Interim CIO. Designed asset allocation, spending policy (~2-3% of fund), and rebalancing frameworks for state investment boards.
- **PE pacing expertise**: Extensive manager due diligence across private equity, private real estate, infrastructure, and timber. CAIA-credentialed with direct experience evaluating GP terms, commitment pacing, and capital call management.
- **Wake Robin context**: Wake Robin is a real estate investment and community development company (multifamily housing, 55+ age-qualified communities, 60+ communities, 16,000+ apartment homes). The SFO Liquidity Architecture models Wake Robin's multi-entity family office structure.

## The Four-Line Principle (Load-Bearing)

```
NAV is not liquidity.
Appraisal value is not spending capacity.
Development / land value is not distributable income.
OpCo value is not automatically portfolio liquidity.
```

This principle governs every phase of work. It is not a preface to skim past.

---

## Architecture Overview

**Repo**: `Warrenpoobear/asset-allocation`
**Current Phase**: 23 (PE real-data commitment input layer)
**Tests**: 386 passing
**Stack**: Python 3.12, pydantic v2, numpy, pandas, pyarrow

### Seven Layers (dependency order)

| Layer | Status | Description |
| --- | --- | --- |
| 3.1 Entity | Not yet built | Multi-entity SFO chart (LLCs, trusts, individuals) |
| 3.2 Account/Position | Partial | Per-account holdings, asset-class taxonomy |
| 3.3 Cash-flow | Not yet built | Entity-by-entity quarterly forecast |
| 3.4 PE Pacing | Shipped (Phases 1,7,8) | Commitment, call, distribution, NAV projection |
| 3.5 RE + OpCo | Not yet built | Stabilized RE, development RE, operating companies |
| 3.6 Liquidity | Partial (Phase 8) | Illiquidity overlay; full tier system not built |
| 3.7 Allocation/Policy | Shipped (Phases 1-11) | Weights, bands, rebalance, spending, scenarios |

---

## Quarterly Ledger (The Spine)

**File**: `src/aa_model/integration/ledger.py`

Every module produces or consumes rows on the quarterly ledger. It is the central object.

### Schema

| Column | Type | Description |
| --- | --- | --- |
| quarter | Period[Q] | e.g. 2026Q2 |
| bucket | str | public_equity, public_bond, cash, pe_buyout |
| flow_type | str | Canonical ordering below |
| amount_usd | float | Signed dollar impact |
| nav_start_usd | float | Bucket NAV before this flow |
| nav_end_usd | float | Bucket NAV after this flow |
| source | str | Producing module name |
| run_id | str | Manifest run ID |

### Canonical Intra-Quarter Flow Ordering

1. `inflow` - external contributions
2. `return` - mark-to-market on liquid buckets
3. `pe_call` - capital deployed into PE
4. `pe_distribution` - capital returned from PE
5. `pe_nav_mark` - PE NAV growth + yield
6. `spend` - withdrawals
7. `rebalance` - intra-portfolio transfer (sums to zero)

### Invariants

- Per-row: `nav_end_usd == nav_start_usd + amount_usd`
- Chain: within each (run_id, bucket) chain, start = prior end
- Rebalance is zero-sum per quarter
- Total NAV conservation: moves only via market P&L and external cash
- No NaN in amount, nav_start, nav_end
- Determinism: identical inputs produce byte-identical ledger.parquet

---

## Spending Rules

**ABC**: `src/aa_model/spending/base.py`

| Rule | Description |
| --- | --- |
| flat_real | Fixed real spending, inflation-adjusted |
| smoothing | Weighted average of prior spend and current NAV-implied |
| Owl (Guyton-Klinger) | Guardrail rules with prosperity/capital preservation triggers |

### Spending Base (Phase 12/12.5)

Configurable denominator for spending rate calculation. Critical because total NAV materially overstates spendable resources for this household.

### Liquidity Coverage

**File**: `src/aa_model/liquidity/coverage.py`

Coverage ratios, reserve floor (18 months default), shortfall frequency.

---

## Governance Rules (Do Not Violate)

1. Ledger is sole state spine - no sidecars, no hidden state
2. CMA baseline immutable; scenarios are perturbations
3. No implementation before design lock (docs commit first)
4. MODEL_DOCUMENTATION.md updated for any behavior change
5. Identical inputs produce byte-identical ledger.parquet
6. No overwriting run directories
7. PROJECT_SCOPE.md is authoritative for reference architecture

---

## Configuration

| Key | Default | Notes |
| --- | --- | --- |
| governance.size_usd | 100,000,000 | Sizing only |
| solver.preferred | clarabel | Fallback: scs, osqp |
| liquidity.floor_months | 18 | Cash + ST bonds reserve |
| pe.sleeve_target_pct | 0.25 | PE share of total |
| rebalance.frequency | quarterly | Aligns with ledger |

---

## Source Files

| Component | File |
| --- | --- |
| Quarterly Ledger | `src/aa_model/integration/ledger.py` |
| Orchestrator | `src/aa_model/integration/orchestrator.py` |
| Manifest | `src/aa_model/integration/manifest.py` |
| Allocation (stub) | `src/aa_model/allocation/stub.py` |
| Riskfolio Adapter | `src/aa_model/allocation/riskfolio_adapter.py` |
| cvxportfolio Adapter | `src/aa_model/allocation/cvxportfolio_adapter.py` |
| Liquidity Overlay | `src/aa_model/allocation/liquidity_overlay.py` |
| Spending Rules | `src/aa_model/spending/rules.py` |
| Owl Adapter | `src/aa_model/spending/owl_adapter.py` |
| Spending Base | `src/aa_model/spending/spending_base.py` |
| Liquidity Coverage | `src/aa_model/liquidity/coverage.py` |
| TA Model | `src/aa_model/pe/ta_model.py` |
| STAIRS Adapter | `src/aa_model/pe/stairs_adapter.py` |
| Run Script | `scripts/run_sfo_study.py` |
