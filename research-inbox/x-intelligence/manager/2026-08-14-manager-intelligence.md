---
type: x-intelligence
subtype: manager-intelligence
captured_at: 2026-08-14 16:15 ET
week_of: 2026-08-14
pattern: pre_13f, position_claim
sources: SEC EDGAR + 13F aggregators (direct check), Decision Impact Ledger (carried entries), manager_registry.json v3.2, event_alert_watchlist.json
tickers: VRDN, ORKA, RVMD, REPL, AUTL
mapped_managers: Fairmount Funds Management, Deep Track Capital, Logos Global Management, EcoR1 Capital, Perceptive Advisors
manager_tier: elite_core
source_class: rumor
confidence: low-med (carried, unchanged this week)
confirming_monitor: SEC Filing Monitor (Q2 2026 13F-HR / Form 4)
lead_time_estimate: pending — Q2 13F deadline is today (Aug 14); none of Fairmount/Deep Track/Logos had filed as of ~4pm ET direct check
ladder_stage: x_post
status: unverified
---

## Status note

X read was fully capped this week — monthly quota exhausted across every OR-batched query (journalists, KOLs, BioPharmIQ). This is the third consecutive degraded week (week of Jul 31: quota exhausted; week of Aug 7: OR-batch queries returned empty; this week: hard cap, zero results). No new items landed in the x-intelligence/biotech content library since Aug 7, and zero new `manager_signal` rows were added to the Decision Impact Ledger this week. There is also no populated Investors roster in the roster registry doc — that gap has been standing since setup.

This digest is a carry-forward status check against the manager registry, watchlist, and today's Q2 13F deadline — not a fresh discovery run. Treating it as such: no new patterns are manufactured below.

## Convergence

None identified this week. No independent multi-source clustering to report — X read capped, no fresh captures.

## Divergence

None identified this week.

## Pre-13F watch (RUMOR — unverified)

All five items below are carried from last week (Aug 7) with **no new corroborating evidence**. Confidence is unchanged; week counts increment because the underlying claim has not yet been confirmed or refuted by a filing.

**1. VRDN — Fairmount + Deep Track Q2 pre-13F sizing — WEEK 3**
- Mapped managers: Fairmount Funds Management (elite_core, Tier 1), Deep Track Capital (elite_core, Tier 1)
- Basis (unchanged since Aug 7): Fairmount 14.04% + Deep Track 5.4M sh per prior (Q1 2026) filings; Aug 6 earnings showed 95.8% Q/Q revenue growth and 95% of ~2,000 prescribing physicians engaged post-LUMVOA launch; conference call Aug 7.
- Read: strong commercial launch data is the kind of event that could pull either manager to add into strength — this is an interpretation, not a fact; nothing new this week supports or refutes it.
- Confidence: med (unchanged — capped by lack of fresh source this week).
- Confirming monitor + lead: SEC Filing Monitor, Q2 2026 13F-HR. Direct EDGAR/aggregator check this afternoon (~4pm ET) shows both managers' most recent indexed 13F is still Q1 2026 (filed 5/15/26) — no Q2 filing visible yet. Deadline is today; resolution likely trickles in through the weekend.
- Process flag: VRDN is NOT on the coinvest `event_alert_watchlist` — flag for watchlist-addition review given the two-manager convergence on this name.

**2. ORKA — Fairmount post-Ph2a sizing — WEEK 4**
- Mapped manager: Fairmount Funds Management (elite_core, Tier 1)
- Basis (unchanged): ~19.5% ORKA post-Jul 1 preferred conversion; Ph2a primary endpoint hit (EVERLAST-A).
- Note: flagged last week to "close out if Q2 13F Aug 14 confirms no change" — 13F not yet filed as of this check, so held open one more week rather than closed prematurely.
- Confidence: med.
- Confirming monitor + lead: SEC Filing Monitor, Q2 2026 13F-HR — pending, same filing gap as above.

**3. RVMD — Logos Q2 post-NDA sizing — WEEK 3**
- Mapped manager: Logos Global Management (elite_core, Tier 1)
- Basis (unchanged): Logos was the #1 disclosed hold at $194.5M in the Q1 2026 13F; daraxonrasib NDA accepted Jul 22.
- Confidence: low (unchanged).
- Confirming monitor + lead: SEC Filing Monitor, Q2 2026 13F-HR. Logos's most recent indexed 13F is also still Q1 2026 as of this afternoon's check.

**4. REPL — EcoR1 Q2 persistence — WEEK 2**
- Mapped manager: EcoR1 Capital (elite_core)
- Basis (unchanged): EcoR1 confirmed Q1 2026 re-entry (1M sh, EDGAR 0000935836-26-000270); FDA accelerated approval (TUDRIQEV) Aug 6.
- Confidence: low.
- Confirming monitor + lead: SEC Filing Monitor, Q2 2026 13F-HR — pending.

**5. AUTL — Perceptive Advisors — WEEK 2**
- Mapped manager: Perceptive Advisors (elite_core)
- Basis (unchanged): $250M senior credit facility (initial $75M drawn) + warrants to 3.5M ADSs at $1.9314/ADS, per MSWPulse Aug 3. Equity position size unknown pre-13F.
- Confidence: low.
- Confirming monitor + lead: SEC Filing Monitor, Q2 2026 13F-HR / Form 4 (warrant exercise) — pending.
- Process flag: AUTL is NOT on the coinvest `event_alert_watchlist` — still flagged for watchlist-addition review (carried from last week, unresolved).

## Manager-position claims (to verify)

Decomposed ownership/concentration components underlying the five items above (component-level per the routine's decomposition rule — each verifies independently of the launch/event narrative):

| Manager | Ticker | Claim | Verification path |
|---|---|---|---|
| Fairmount | VRDN | POSITION CLAIM — holds ~14.04% | SEC Filing Monitor — most-recent 13F + applicable 13D/13G/Form 4 first |
| Deep Track | VRDN | POSITION CLAIM — holds 5.4M sh | Same |
| Fairmount | ORKA | POSITION CLAIM — ~19.5% post-conversion | Same |
| Logos | RVMD | POSITION CLAIM — $194.5M, #1 disclosed hold (Q1 13F) | Same |
| EcoR1 | REPL | POSITION CLAIM — 1M sh (Q1 2026 re-entry) | Same |
| Perceptive | AUTL | POSITION CLAIM — equity/warrant exposure, share-equivalent size UNKNOWN pre-13F | Same; weakest-sourced of the six |

None of the six is confirmed by a Q2 2026 13F-HR as of this afternoon's direct check — Fairmount, Deep Track, and Logos each still show Q1 2026 as their latest indexed filing. Overall confidence stays capped at the weakest of the three legs (Interpretation = low on all six) per the three-leg discipline; no concentration figure above has been reconstructed against a current portfolio denominator this week.

## Weakly sourced / widely repeated

- **AUTL / Perceptive equity sizing** — single-source (MSWPulse), no independent corroboration this week, ticker not on watchlist. Caution-list status until a second source or a filing appears.
- No new widely-repeated-but-thin items surfaced this week — X capped, so there was no fresh repetition to assess.

## Verification steps

- SEC Filing Monitor: confirm/deny Fairmount, Deep Track, and Logos Q2 2026 13F-HRs as they post. Deadline is today; direct EDGAR/aggregator check at ~4pm ET showed none of the three filed yet — expect a filing-day rush over the next several hours and through the weekend.
- SEC Filing Monitor: confirm EcoR1's REPL Q2 persistence and Perceptive's AUTL equity/warrant sizing via Q2 13F-HR or Form 4.
- Watchlist owner: review VRDN and AUTL for addition to `event_alert_watchlist.json` — both carry manager-convergence or Tier-1-adjacent signal quality but sit outside current coverage.
- X read capacity: three consecutive degraded/capped weeks (Jul 31 quota exhausted, Aug 7 empty OR-batches, Aug 14 full monthly cap) — structural gap, not a one-off. Recommend evaluating a higher read tier or accepting a monitoring-only cadence until the quota resets.
