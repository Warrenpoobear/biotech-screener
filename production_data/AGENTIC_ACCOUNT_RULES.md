# Agentic Account Operational Rules

**Account**: 802349084 (Robinhood agentic)  
**Confirmed**: 2026-06-24 (Rule 4 changed 2026-11-15)  
**Purpose**: Live test case for Claude-managed biotech model portfolio operations

---

## Rule 1 — Rebalance Trigger

| Condition | Action |
|-----------|--------|
| Monday market open | Weekly rebalance |
| Any position drifts >25% from target | Off-cycle rebalance |
| T+1 gap blocks >50% of buys | Defer to Tuesday open |

Equal-weight target = total equity / number of positions.

---

## Rule 2 — Entry/Exit on Roster Changes

| Event | Action |
|-------|--------|
| New ticker enters top-30 | Add at next weekly rebalance |
| Ticker drops out of top-30 | Exit at next weekly rebalance |
| Ticker drops below rank 40 | Exit within session discovered |
| Binary catalyst within 5 days | Flag for manual review — do not auto-act |

---

## Rule 3 — Position Sizing

- **Current rule (account < $5,000)**: Equal weight
- **Switch to model weight**: When account exceeds $5,000 (use `target_weight_pct` from `data/snapshots/<date>/rankings.csv`)
- **Exception**: If model weight spread >3x between highest and lowest position, escalate for manual decision

---

## Rule 4 — Governance Hard Exit

| Trigger | Action |
|---------|--------|
| Drawdown vs XBI ≤ −10pp | Review: no new buys and no rebalance until the operator has reviewed model health |
| Drawdown vs XBI ≤ −15pp | Full liquidation to cash, market orders, within session, after operator confirmation |

Drawdown is measured per lot against XBI from each lot's own fill date (the convention in the `biotech-governance-check` skill). If it cannot be measured, the gate is UNVERIFIED: do not trade, ask the operator.

Re-entry after hard exit requires explicit operator instruction. Not automatic.

Check drawdown gate at the start of every rebalance session via `get_portfolio`.

*Changed 2026-11-15 (operator decision; `docs/governance/2026-11-15-pre-clock-decisions.md` §3). The earlier triggers were −2pp (liquidate) and −5pp (liquidate immediately). On the selector-only history, −2pp was touched in about half of all 12-month holding periods through ordinary variation, and the six months after a first touch averaged +8.1pp of excess return. The new levels are touched in about 15% (−10pp) and 4% (−15pp) of 12-month periods on month-end observations. Those figures are in-sample and survivorship-biased.*

---

## Rule 5 — IRA vs Agentic Coordination

- **Agentic account**: Follows model top-30 rotation per Rules 1–4. Weekly scheduled rebalancing.
- **IRAs (••••0727, ••••0174)**: Long-term conviction holds. Manual rebalance on explicit operator instruction only — never on a schedule.
- **Overlap**: A position can be held in both an IRA and the agentic account simultaneously.

---

## Robinhood API Constraints (Operational)

| Constraint | Detail |
|------------|--------|
| T+1 settlement | Same-session sell proceeds not available as buying power until next business day |
| Rate limit | ~16–20 orders/minute; batch orders in groups of 8–10 |
| Order type | Dollar-based fractional orders: GFD only (no GTC) |
| Minimum order | $1.00 for dollar-based orders |
| ABVX | Sell-only restriction — exclude from all buy lists |

---

## Rebalance Session Checklist

1. `get_portfolio` → verify buying power; confirm drawdown gate clear (Rule 4)
2. `get_equity_positions` → current holdings and market values
3. Load `data/snapshots/<date>/rankings.csv` → check for roster changes (Rule 2)
4. Compute equal-weight target = total equity / positions (Rule 3)
5. Execute sells first (opens buying power)
6. Execute buys in batches of 8–10 (rate limit)
7. Note any blocked orders (sub-$1 minimum, restrictions)
