# Ruleset Release Summary

Generated: 2026-02-10T13:23:28

| Field | Value |
|-------|-------|
| Baseline ruleset | `f6c99132` (v1.2.0_candidate.json) |
| Candidate a_floor | `0.6` |
| Panel rows | 1808 |
| a_floor sweep | [0.45, 0.5, 0.55, 0.58, 0.6, 0.62, 0.65] |

## Parameter Changes

| Parameter | Baseline | Candidate | Delta |
|-----------|----------|-----------|-------|
| `tier_a_optionality_floor` | 0.55 | 0.6 | +0.05 |

## Objective Improvement

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d return (%) | 9.41 | 10.50 | +1.09 |
| CD median 60d return (%) | 7.98 | 7.73 | -0.26 |
| AB-CD separation (%) | 1.43 | 2.78 | +1.35 |
| AB median max-DD (%) | -20.63 | -19.83 | +0.80 |
| Mean A-count / date | 3.00 | 1.70 | -1.30 |
| Top-25 overlap (%) | 71.40 | 70.10 | -1.30 |
| Mean turnover (%) | 28.60 | 29.90 | +1.30 |

## Tier Distribution

| Tier | Baseline % | Candidate % | Delta |
|------|-----------|------------|-------|
| A | 1.7 | 0.9 | -0.8 |
| B | 4.1 | 4.0 | -0.1 |
| C | 7.1 | 8.0 | +0.9 |
| D | 87.1 | 87.1 | +0.0 |

## Per-Tier 60d Return Detail (Candidate)

| Tier | Count | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|-------|--------|----------|------------|---------------|
| A | 17 | +13.41 | +9.88 | 58.8 | -27.09 |
| B | 72 | +30.59 | +11.01 | 67.6 | -21.05 |
| C | 144 | +10.43 | +7.73 | 61.2 | -21.97 |
| D | 1575 | +29.37 | +15.31 | 65.6 | -29.28 |

## Walkforward Panel Coverage

- Total rows: 3922
- With forward returns: 3819 (97.4%)
- With forward max-DD: 3922 (100.0%)
- Top-25 overlap (Jaccard): 72.3%

## Scoring

- Objective score: **-3.1683**
- Candidates evaluated: 49
- Candidates passing constraints: 36
- Constraints: A% >= 3.0%, turnover <= 50.0%, separation > 0
- Status: **PASS**

## QA Checklist

- [ ] Calibration report reviewed (`artifacts/calibration_report.md`)
- [ ] Walkforward report reviewed (`artifacts/walkforward_report.md`)
- [ ] `bump_ruleset.py --from-json artifacts/candidate_overrides.json` executed
- [ ] Contract tests pass (`pytest tests/test_decision_engine_contract.py`)
- [ ] Replay regression pass (`pytest tests/test_decision_engine_replay_regression.py`)
- [ ] Golden records refreshed if needed (`scripts/refresh_goldens.py`)
- [ ] Changelog finalized (remove [DRAFT] marker)
- [ ] `promote_ruleset.py <candidate_id>` executed

