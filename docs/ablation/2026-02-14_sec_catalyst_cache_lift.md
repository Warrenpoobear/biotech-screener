# 2026-02-14 Catalyst Cache Lift Attribution (post-filter / hard-gated)

## What changed (screen run after rebuilding caches for 2026-02-14)
- Catalyst events: 1099 -> 1301 (+202)
- Dev specific_days: 133/183 (72.7%) -> 141/183 (77.0%)  [8 dev tickers flipped no_upcoming -> specific_days]
- A-tier count: 33 -> 37 (+4)
- Phase-2 health: OK (turnover 15.0%), delta +3/-3 names

## Source mix (post-cache run)
- SEC_8K_FILING: 206  (primary driver)
- CTGOV_CALENDAR: 1084
- FDA_CALENDAR: 11
- Multi-form: 0 net impact (15 multi-form events were LOW/HALF_YEAR and blocked by merge-time hard gate)

## Quality check (post-cache run)
- by_confidence: HIGH=145, MED=255, LOW=901
  - LOW share is dominated by CTGOV items; incremental 8-K events are predominantly MED/HIGH.
- by_date_precision: DAY=929, RANGE=232, QUARTER=22, HALF_YEAR=118
  - DAY-dominant; HALF_YEAR not driving tier changes due to hard gate.

## Interpretation
The 2026-02-14 uplift is attributable to refreshed SEC 8-K caches (206 events). Multi-form contributed no decision impact in this run due to the confidence/precision hard gate.
