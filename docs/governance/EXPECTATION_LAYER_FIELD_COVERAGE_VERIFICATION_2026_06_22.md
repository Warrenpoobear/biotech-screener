# Expectation Layer Field Coverage Verification

**Date verified:** 2026-06-23  
**Snapshot:** `data/snapshots/2026-06-23/rankings.csv` (291 rows)  
**Verdict: PASS**

---

## Background

The expectation layer (EES v2) requires four market-data fields from `rankings.csv`
as inputs: `short_interest_pct`, `close_price`, `market_cap_mm`, and `priced_move_pct`.
A plumbing change surfaced these fields from the screener pipeline into `rankings.csv`
(estimated 80%→95% weighted feature coverage). This document verifies they are wired
end-to-end and flowing correctly. A fifth field, `insider_net_buy_value_90d`, is a
diagnostic pass-through that must NOT feed into scoring.

---

## Field Presence in rankings.csv

All five fields are declared in `run_screen_columns.py` (lines 190–197) and present
in today's snapshot.

| Field | In rankings.csv | n non-empty / 291 | Coverage | Notes |
|---|---|---|---|---|
| `short_interest_pct` | PRESENT | 289 / 291 | 99.3% | 2 missing: small-cap edge cases |
| `close_price` | PRESENT | 291 / 291 | 100.0% | |
| `market_cap_mm` | PRESENT | 291 / 291 | 100.0% | |
| `priced_move_pct` | PRESENT | 254 / 291 | 87.3% | Expected gap: tickers without options or near-term catalysts have no implied move |
| `insider_net_buy_value_90d` | PRESENT | 291 / 291 | 100.0% | Diagnostic pass-through only — scoring lane CLOSED 2026-04-05 |

---

## EES Model Consumption

`event_ev/expectation_error_model.py` (lines 177–190) reads all four required fields
directly from the rankings row dict via `_safe_float(row.get(...))`.

The `expectation_error_overlay.json` from today's snapshot confirms active consumption:

```json
"features_used": {
  "inputs": {
    "priced_move_pct": 33.33,
    "short_interest_pct": 0.1089,
    "market_cap_mm": 3685.0,
    "close_price": 21.60,
    ...
  }
}
```

- `n_scored`: 291 / 291 rows (100%)
- All four required fields appear in `features_used.inputs` for every scored row
- `slippage_penalty_score` (consumes `close_price` + `market_cap_mm`) populated correctly
- `crowding_bias_score` (consumes `short_interest_pct`) populated correctly
- `base_rate_gap_score` + `timing_decay_risk_score` (consume `priced_move_pct`) populated correctly

---

## Insider Pass-Through Integrity

`insider_net_buy_value_90d` is present in `rankings.csv` as a diagnostic column.
Verification that it does NOT feed scoring:

- `run_screen_columns.py` comment (line 194): *"insider_net_buy_value_90d is a diagnostic
  pass-through only — the scoring lane was closed 2026-04-05 and is NOT reopened by this
  column appearing in rankings.csv."*
- `expectation_error_model.py` does not reference `insider_net_buy_value_90d` anywhere
- `features_used.inputs` in the EES overlay never includes `insider_net_buy_value_90d`
- `final_score`, `composite_score`, and `expectation_error_score` are all computed
  without this field

Raw value range in today's snapshot: −490M to +104M (median −103K). Values are
raw dollar figures, confirming the field is ingested as-is with no normalization
applied to scoring.

---

## Remaining Gap

`insider_net_buy_value_90d` has 100% population (up from the ~0% noted at freeze time).
This reflects the enrichment layer now running correctly. The **scoring lane remains
closed** — this is expected and correct. No action required.

`priced_move_pct` at 87.3% is the designed floor: tickers with no near-term catalyst
or no options chain legitimately have no implied move. The EES model handles `None`
gracefully via `_safe_float`.

---

## Verdict

**PASS.** All four required expectation fields are wired from `rankings.csv` into the
EES v2 model and producing scores for all 291 rows. Insider pass-through is present but
correctly isolated from scoring. No corrective action required.
