# AGENTS.md — Options Watch Agent (Phase 2)

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — data sources, thresholds, and eligibility gates
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) if they exist

## Daily sequence

### Step 1: Build watchlist (max 30 names)

Priority order (fill until cap):
1. Hard-catalyst names already in the review queue (`review_queue.csv`)
2. Names in the current trade plan (`trade_plan.csv`)
3. Names in shadow positions (`positions/{date}.json`)
4. Names surfaced by `catalyst_delta` (`artifacts/catalyst_delta/{date}_delta.json`)
5. A-tier names with `catalyst_days <= 30`

### Step 2: Eligibility gate (per name)

A name is eligible for alerting ONLY if ALL three hold:
- `opt_has_data == 1` in rankings.csv
- `opt_liquidity_ok == 1` in rankings.csv
- `opt_use_for_judgment == YES` in rankings.csv

Skip names that fail any gate. Note skipped names in the suppressed section.

### Step 3: Read options state

For each eligible watchlist name, read from rankings.csv:
- `opt_atm_iv`, `opt_front_iv`, `opt_back_iv`
- `opt_term_slope`, `opt_put_call_skew`, `opt_rr_25d`
- `opt_iv_regime`, `opt_event_premium`
- `actual_implied_move_pctile`, `atm_iv_change_5d`

### Step 4: Apply alert thresholds

| Code | Condition | Level |
|------|-----------|-------|
| `EVENT_PREMIUM` | `opt_event_premium == YES` or `opt_term_slope <= -0.10` | flag |
| `IV_RAMP_HIGH` | `atm_iv_change_5d >= 0.10` | high |
| `IV_RAMP_MED` | `0.05 <= atm_iv_change_5d < 0.10` | medium |
| `IV_FALLING` | `atm_iv_change_5d <= -0.05` | note |
| `SURFACE_MOVE_HIGH` | `actual_implied_move_pctile >= 0.80` | high |
| `SURFACE_MOVE_MED` | `0.60 <= actual_implied_move_pctile < 0.80` | medium |
| `DRIFT_RISK_HIGH` | percentile >= 0.85 or IV ramp >= 0.12 | high |
| `DRIFT_RISK_MED` | percentile >= 0.65 or IV ramp >= 0.06 | medium |
| `EXTREME_SKEW` | `abs(opt_rr_25d) >= 0.15` | flag |

### Step 5: Suppress / demote

- Suppress if `opt_iv_regime == EXTREME` and name is NOT hard-catalyst or trade-plan
- Suppress if snapshot coverage/credentials are broken
- Note all suppressions in the output

### Step 6: Priority score

| Condition | Points |
|-----------|--------|
| SURFACE_MOVE_HIGH | +2 |
| SURFACE_MOVE_MED | +1 |
| IV_RAMP_HIGH | +2 |
| IV_RAMP_MED | +1 |
| **Cap** | **+3 max** |

The cap prevents shadow surface signals from dominating the watch.

### Step 7: Write output

- `artifacts/options_watch/{date}_watch.json` (schema `options_watch.v1`)
- `artifacts/options_watch/{date}_watch.md`

### Step 8: Report

Count of flagged names, one line per flag with model context.
Only report names with priority_score >= 1.

## Memory

Write daily notes to `memory/YYYY-MM-DD.md`:
- Watchlist size and composition
- Flags raised (ticker, code, priority)
- Suppressed names and reasons
- Data gaps (missing opt_* fields, stale chains)

## Self-learning (Rule 12)

Recurring surface false-positive pattern → LRN.

## Red lines

- Do not edit `.py` files, scoring logic, rulesets, or manifest
- Do not generate trade recommendations or weight suggestions
- Do not modify options overlay parameters or surface signal weights
- Do not auto-promote names into the review queue
- Do not alert when freshness/credentials are not confirmed
- Report and wait for human review
