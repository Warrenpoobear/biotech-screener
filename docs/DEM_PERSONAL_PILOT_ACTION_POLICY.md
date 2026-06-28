# DEM Personal Pilot Action Policy

**Status**: PILOT_ACTION_READY  
**Created**: 2026-06-28  
**Account**: 802349084 (Robinhood agentic)  
**Classification**: PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_RANKER_CHANGE /
NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_AUTONOMOUS_TRADING

---

## Standing Status Flags

```
PILOT_ACTION_READY
FROZEN_MODEL
CONTROLLED_PERSONAL_CAPITAL
NO_AUTONOMOUS_TRADING
NO_MODEL_CHANGE
FORWARD_VALIDATION_CONTINUES
```

Change any flag only by writing a dated addendum to this document with operator
initials. No flag is silently overridden.

---

## 1. When do I rebalance?

| Trigger | Timing |
|---------|--------|
| Weekly scheduled | Monday market open |
| Drift off-cycle | Any position drifts >25% from equal-weight target |
| T+1 block | If >50% of required buys are blocked by settlement gap → defer to Tuesday open |

Rebalance requires a complete pre-trade checklist pass (Section 5).
If checklist fails, **no rebalance that session**.

---

## 2. What basket do I own?

| Component | Definition |
|-----------|------------|
| **Action basket** | Top-30 by `actionable_rank`, equal weight |
| **Shadow bench** | Ranks 31–60 (diagnostic pool — not the portfolio) |
| **No Top-60 portfolio** | Never equal-weight all 60 names |
| **No Top-10 concentration** | No manual overweight of any name |
| **No discretionary overweight** | All positions equal weight until $5K, then model weight |

Switch to model weight (`target_weight_pct` from rankings.csv) when account
value exceeds $5,000. If model weight spread exceeds 3x between highest and
lowest position, escalate for operator decision before switching.

---

## 3. What names are ineligible?

| Name / Condition | Rule |
|------------------|------|
| ABVX | Sell-only restriction — **exclude from all buy lists** |
| `eligible=0` in snapshot | Exclude from basket; note in blotter |
| Rank < 40 (discovered intra-session) | Exit within session |
| Rank drops 31–39 | Exit at next weekly rebalance |
| Binary catalyst within 5 days | Flag for manual review — **do not auto-act** |
| Unavailable to trade (thin market, halt, etc.) | Hold cash drag; document in blotter |

No substitution from ranks 31–60 for an unavailable name without explicit
operator approval noted in the blotter.

---

## 4. What happens if data quality fails?

Pre-trade checklist items 1–6 are **hard gates**. Any failure → no rebalance.
Document the failure in `artifacts/live_pilot/ACTION_CARD_{date}.md` with reason.

If data quality fails on two consecutive Mondays, escalate — do not wait silently.

---

## 5. Pre-trade checklist

Run `tools/build_personal_pilot_action_card.py --as-of-date YYYY-MM-DD` before
any rebalance. Every item must show PASS.

| # | Check | Fail action |
|---|-------|-------------|
| 1 | Price coverage ≥ 95% of Top-30 names | BLOCK — no rebalance |
| 2 | XBI endpoint price available | BLOCK — no rebalance |
| 3 | Split-adjusted price source active | BLOCK — no rebalance |
| 4 | Model hash unchanged from last run | BLOCK — no rebalance |
| 5 | Ruleset hash unchanged from last run | BLOCK — no rebalance |
| 6 | Rankings snapshot complete (actionable_rank ≥ 30 eligible names) | BLOCK — no rebalance |
| 7 | EES-false count recorded | WARN only |
| 8 | Repeat-offender count recorded | WARN only |
| 9 | Ranks 31–60 replacement bench recorded | INFO only |
| 10 | Estimated cost recorded | INFO only |

---

## 6. EES policy

The EES behaves like insurance — it helps in failure windows but can create drag
in rally/success weeks. **Do not hard-veto EES-false names until shadow
substitution evidence matures** (gate: ≥20 completed windows, t-stat ≥ 1.5 on
delta, per `docs/SHADOW_GUARD_PROMOTION_GATES.md`).

| Condition | Action |
|-----------|--------|
| EES-False alone | **WARN only** — include in basket |
| EES-False + repeat-offender flag | **OPERATOR REVIEW** — pause that name's trade; operator decides |
| EES-False + repeat-offender + rank 31–60 replacement available | **SHADOW SUBSTITUTION** — track in blotter, not automatic |

Record EES status in every blotter row. Never claim EES is "proven alpha" until
shadow gate is met.

---

## 7. Repeat-offender policy

A repeat offender is any name flagged by `tools/stress_wrapper_monitor.py`
as appearing in the bottom-5 contributors by `contrib_to_xs` in ≥2 of the last
12 completed windows.

| Condition | Action |
|-----------|--------|
| Repeat-offender flag alone | **WARN only** — include in basket |
| Repeat-offender + EES-False | **OPERATOR REVIEW** (see Section 6) |
| Repeat-offender still improving in current window | Override flag with operator note in blotter |

Repeat-offender shadow promotion gate: ≥10 rolling forward windows of evidence
beyond YTD seed, per `docs/SHADOW_GUARD_PROMOTION_GATES.md` Monitor A.

---

## 8. Unavailable name handling

If a name in Top-30 cannot be traded (halted, sub-$1 fill, API rejection):

1. Record in blotter with `action=UNAVAILABLE` and `reason` detail.
2. Hold the allocated weight as cash drag — do not substitute automatically.
3. If unavailability persists ≥2 consecutive Mondays, flag for operator decision.
4. Cash drag must be reported in every weekly action card until resolved.

---

## 9. Position and portfolio limits

**Sizing**
- Equal weight only (1 / number of held positions)
- No name overweight
- No manual adds outside Top-30
- No manual deletions unless documented as unavailable or ineligible

**Cash drag**
- Acceptable if caused by unavailability; document in blotter
- Not acceptable as a strategic choice — deploy capital unless a gate blocks it

---

## 10. Drawdown rails (relative vs XBI)

Compute relative drawdown as: `(pilot_return − XBI_return)` since inception or
since last account reset.

| Threshold | Action |
|-----------|--------|
| ≤ −5pp | **Review** — no new capital added; read action card; identify cause |
| ≤ −7.5pp | **Pause** — no new capital; hold current positions; do not rebalance |
| ≤ −10pp | **Freeze pilot** — liquidate to cash; run biotech-autopsy; operator must explicitly re-authorize |

These thresholds apply to the **relative** drawdown, not absolute P&L.
XBI endpoint price must be confirmed at pre-trade checklist step 2.

Hard exit (≤ −2pp from `AGENTIC_ACCOUNT_RULES.md` Rule 4) supersedes all
thresholds above and triggers immediate full liquidation.

---

## 11. Max capital at risk

Operator sets dollar limit in `production_data/portfolio_policy.json` under
`pilot_max_capital_usd`. Default before explicit set: current account value only
(no additional capital added beyond what's already deployed).

No leverage. No margin. No options overlay without separate operator authorization.

---

## 12. When to stop adding capital

Do not add new cash to the pilot account if:
- Any drawdown rail (Section 10) is active
- Pre-trade checklist fails two consecutive weeks
- Forward validation falls below bootstrap 10th percentile for 4+ consecutive weeks
- Operator freeze is in effect for any reason

---

## 13. Blotter requirement

Every rebalance session — regardless of whether trades execute — must produce a
blotter update. If no trades are made (hold decision), write a row with
`action=HOLD` and the reason.

Blotter path: `artifacts/live_pilot/dem_personal_pilot_blotter.csv`

The blotter is **append-only**. Never delete or edit prior rows.

---

## 14. Four tracking portfolios

Every week, compare these four in the action card:

| Portfolio | Purpose |
|-----------|---------|
| **Theoretical Top-30** | What the model says; benchmark for implementation quality |
| **Actual account** | What was actually held (from blotter) |
| **EES-guarded shadow** | Top-30 with EES-false removed, bench fill (shadow only) |
| **Repeat-offender shadow** | Top-30 with repeat-offenders removed, bench fill (shadow only) |

If actual lags theoretical → **execution problem**.  
If theoretical lags XBI → **model behavior problem**.  
If shadow guards beat raw Top-30 → review promotion gate status.

---

## 15. Net-of-cost reporting

Report gross and net excess return in every action card:

```
gross Top-30 excess vs XBI
estimated cost drag (transaction costs + bid/ask + slippage)
net Top-30 excess
actual account excess
implementation gap = actual − theoretical
```

Use estimated costs until actual trade confirmations are logged. Mark as
`ESTIMATED` until confirmed in blotter.

---

## 16. Regime policy

Regime evidence is **mixed and definition-dependent**. Regime signals remain
**interpretation-only**:

- Do not use regime label as a ranking input
- Do not use regime label as a sizing input
- Do not adjust position count or weights based on current regime
- Include regime label in action card as informational context only

Conservative default: treat every week as regime-neutral until a separate,
statistically validated regime-sizing rule is approved.

---

## Revision log

| Date | Change |
|------|--------|
| 2026-06-28 | Initial version |
