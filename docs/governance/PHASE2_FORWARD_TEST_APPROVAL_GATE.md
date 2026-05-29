# Phase 2 Forward Paper Test — Governance Approval Gate

**Status:** Not approved  
**Prerequisite:** Review of Phase 1 diagnostic PR (#308)  
**Production impact:** None unless separately approved  
**Purpose:** Decide whether to launch a forward paper test of portfolio transition policies.

---

## 1. Background

Phase 1 created a diagnostic-only harness and report on branch `phase1-portfolio-diagnostic` (PR #308, now merged to main).

### Phase 1 Finding

* **Quarterly was the leading tested policy** in the historical diagnostic.
* Quarterly average return: **+127.4%**
* Weekly proxy average return: **+115.6%**
* **Quarterly gap over weekly: +11.8pp**
* **The mechanism remains unresolved** after timeline spell-tracking corrected earlier false entry-timing claims.
* No production code, ranking/scoring logic, daily pipeline, or deployment behavior changed.

### Critical Note

**Phase 1 does not authorize implementation.** The finding is diagnostic-only and lacks forward validation, transaction-cost modeling, and causal explanation.

---

## 2. Decision Required

Governance must decide whether to approve a **Phase 2 forward paper test**.

Approval means:

* Track policies prospectively on live data.
* **Do not execute trades.**
* **Do not change production.**
* **Do not modify selector/ranker/scoring logic.**
* **Do not treat Phase 1 as deployment validation.**
* Mark all artifacts **paper-only** and diagnostic.

Approval does **not** mean:
* Quarterly is ready to deploy.
* The mechanism is proven.
* Any policy change is justified.

---

## 3. Proposed Phase 2 Scope

### Forward Paper Test Period

* **Start:** First approved trading day after governance sign-off.
* **Minimum duration:** 60 trading days.
* **Preferred duration:** 90 trading days.
* **Review checkpoints:** 30, 60, and 90 trading days.

### Policies to Track

**Required:**
1. Current shadow/advisory behavior (daily fresh top-30).
2. Weekly trade packet proxy.
3. Quarterly rebalance proxy.
4. Static inception hold.
5. Delisting/liquidity-only hold.

**Optional (only if explicitly approved):**
* HOLD_30 comparison.
* Monthly rebalance proxy.

---

## 4. Required Metrics

### Daily Tracking

* Holdings by policy.
* Entry and exit dates.
* Position weights.
* Turnover.
* Gross return.
* XBI return.
* Alpha vs XBI.
* Drawdown (peak-to-trough).
* Staleness vs current top-30 (rank drift).
* Names added / removed per rebalance.
* Realized and unrealized contribution by ticker.
* Price availability and missing-data events.

### Weekly Tracking

* Estimated transaction cost (borrow, slippage, bid-ask).
* ADV / liquidity constraints.
* Bid-ask spread proxy (where available).
* Concentration and right-tail dependency.
* Exposure to near-term catalysts.

---

## 5. Guardrails

### Phase 2 is Paper-Only

**Not allowed:**
* No live trades.
* No production rebalance changes.
* No persistent live portfolio state.
* No selector/ranker/scoring changes.
* No daily pipeline changes.
* No deployment recommendation before test completion.
* No "optimal policy" language.

**Allowed:**
* Read-only paper portfolio artifacts.
* Diagnostic reports.
* Governance summaries.
* Forward monitoring dashboards (if explicitly marked paper-only).

---

## 6. Success Criteria

### Phase 2 May Be Considered Promising If

* Quarterly or another persistent policy beats weekly/current behavior **net of estimated transaction costs**.
* Outperformance is **not driven by one ticker only**.
* Drawdown does **not materially worsen** versus XBI or current shadow behavior.
* Turnover reduction is **meaningful**.
* Staleness versus current top-30 remains **acceptable**.
* Results remain **interpretable after attribution**.

### Phase 2 Fails or Remains Inconclusive If

* Advantage disappears **after transaction costs**.
* Performance depends on **one or two names**.
* Quarterly **underperforms** in the forward window.
* Staleness becomes **excessive**.
* Missing-data or execution assumptions **dominate results**.
* **Attribution remains unresolved**.

---

## 7. Stop Conditions

Stop or pause the paper test if:

* Data source alignment breaks.
* Daily snapshots stop or become incomplete.
* Price data becomes stale.
* XBI benchmark data is unavailable.
* The test **accidentally affects production behavior**.
* Artifacts are **not clearly marked paper-only**.
* **Governance scope is exceeded**.

---

## 8. Required Deliverables

If approved, Phase 2 should produce:

1. `artifacts/portfolio_policy_forward_test/README.md` (paper-only disclaimer)
2. Daily policy holdings files
3. Daily policy performance files
4. Transaction-cost estimate file
5. 30-day checkpoint memo
6. 60-day checkpoint memo
7. 90-day final memo

**All artifacts must be marked:**

> **Paper-only. No production trading instruction.**

---

## 9. Governance Decision Options

Governance may choose one of:

**A. Approve Phase 2 forward paper test**
* Proceed with paper tracking per the scope above.
* Requires explicit sign-off on duration and metrics.

**B. Request changes to the Phase 2 design**
* Specify which policies, which metrics, or which guardrails to modify before launch.

**C. Defer until Phase 1 PR review is complete**
* Wait for full PR review before committing to Phase 2.
* Recommended if Phase 1 diagnostic findings need stakeholder buy-in first.

**D. Reject Phase 2 and keep rankings advisory-only**
* Accept Phase 1 as informational but do not pursue further testing.
* Keep current shadow/advisory behavior unchanged.

**E. Require additional historical diagnostics before forward testing**
* Run additional Phase 1 scenarios or sensitivity analyses.
* Specify which policies or cohorts to re-test before Phase 2 launch.

---

## 10. Recommended Decision

### Recommended Path

> **Defer Phase 2 launch until the Phase 1 diagnostic PR (#308) is formally reviewed.**

After review, if the PR is accepted as diagnostic-only and governance agrees the signal is worth monitoring, **approve a paper-only Phase 2 forward test with no production changes.**

### Rationale

* Phase 1 findings are promising but mechanism is unproven.
* Forward validation will clarify whether the historical advantage persists on live data.
* Paper testing incurs no execution risk and informs governance without commitment.
* Transaction costs and staleness need measurement; historical harness cannot model these.

### Next Steps (if approved)

1. Governance approves Phase 2 scope and reviews Phase 1 PR.
2. Create Phase 2 tracking harness (read-only, paper-only).
3. Launch tracking on agreed start date.
4. Deliver checkpoint memos at 30, 60, and 90 trading days.
5. Governance reviews final memo and decides on next action (Phase 3, deployment, or archive).

---

**Status: Awaiting governance review and decision.**

*Last updated: 2026-05-29*  
*Prepared by: Phase 1 Diagnostic Harness (diagnostic-only)*
