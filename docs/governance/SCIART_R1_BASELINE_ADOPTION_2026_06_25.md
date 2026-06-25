# Sci-Cart R1 — Baseline Adoption Record (2026-06-25)

**Status:** PROVISIONAL — operator may ratify or revise  
**Governance:** READ_ONLY_DIAGNOSTIC — no scoring changes

## Decision

For Phase 13 implementation continuity, the repo treats **`artifacts/scientific_cartography/2026-06-23-postfix/`** as the working diagnostic baseline pending explicit operator ratification.

## Evidence

| Metric | Pre-fix | Post-fix |
|--------|---------|----------|
| Mis-normalization rate | 35.1% | 13.8% |
| Ticker linkage | 0% | 97.9% |
| Fix commits | — | `697c0b83`, `38edb0ab` |

## Operator ratification checklist

- [ ] Promote `2026-06-23-postfix/` as active artifacts directory
- [ ] Confirm normalizer fixes are authoritative on host cron runs
- [ ] Authorize Phase 13.2+ tooling (sample review, mechanism design memo)
- [ ] Complete R4 worksheet verdicts; summarize with:
  `python3 tools/sciart_normalization_sample_review.py --summarize --worksheet docs/governance/SCIART_PHASE13_2_NORMALIZATION_SAMPLE_REVIEW_2026_06_25.md`

## Implementation note

Phase 13.1 (`trial_records.json` lookup) and Phase 13.3 (confidence floor decoupling) are **already merged** in `scientific_cartography/build/asset_indication_builder.py`. Remaining work is R4 manual review worksheet and R6 mechanism coverage design.
