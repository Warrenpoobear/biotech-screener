# PIT Guardrail Validation

Validated 2026-02-01 against `trial_records_date = 2026-01-30`.

## Paths tested

| Mode | as_of_date | age_days | Result |
|------|-----------|----------|--------|
| **Clean** | 2026-02-01 | +2 | `confidence_level=HIGH`, `catalyst_mode=full`, 460 events detected |
| **Degrade** | 2026-01-28 | -2 | `pit_violation` logged, `catalyst_mode=corporate_only_due_to_pit_violation`, 11 corporate events only, `effective_weights.catalyst=0.0000`, `catalyst_confidence_gated` flag set |
| **Strict** | 2026-01-28 | -2 | `PITViolationError` raised at `module_3_catalyst.py:630`, pipeline aborted, no output JSON written |

## Key behaviours confirmed

- Strict mode raises and stops before any scoring or ranking occurs.
- Degrade mode suppresses all trial-records-derived signals (diff, calendar, activity proxy) and halves corporate catalyst scores with confidence downgraded to LOW.
- Position sizing is pure post-processing with zero feedback into ranking or suppression logic.
