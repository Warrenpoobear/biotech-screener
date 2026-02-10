# Ruleset Release Summary

Generated: 2026-02-10T11:37:24

| Field | Value |
|-------|-------|
| Baseline ruleset | `f6c99132` (v1.2.0_candidate.json) |
| Candidate a_floor | `0.6` |
| Panel rows | 1808 |
| a_floor sweep | [0.4, 0.45, 0.5, 0.52, 0.55, 0.58, 0.6, 0.65] |

## Parameter Changes

| Parameter | Baseline | Candidate | Delta |
|-----------|----------|-----------|-------|
| `tier_a_optionality_floor` | 0.55 | 0.6 | +0.05 |

## Objective Improvement

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d return (%) | 9.41 | 10.00 | +0.58 |
| CD median 60d return (%) | 7.98 | 7.47 | -0.52 |
| AB-CD separation (%) | 1.43 | 2.53 | +1.10 |
| AB median max-DD (%) | -20.63 | -19.83 | +0.80 |
| Mean A-count / date | 3.00 | 2.30 | -0.70 |
| Top-25 overlap (%) | 71.40 | 69.50 | -1.90 |
| Mean turnover (%) | 28.60 | 30.50 | +1.90 |

## Tier Distribution

| Tier | Baseline % | Candidate % | Delta |
|------|-----------|------------|-------|
| A | 1.7 | 1.3 | -0.4 |
| B | 4.1 | 3.9 | -0.2 |
| C | 7.1 | 7.7 | +0.6 |
| D | 87.1 | 87.1 | +0.0 |

## Per-Tier 60d Return Detail (Candidate)

| Tier | Count | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|-------|--------|----------|------------|---------------|
| A | 23 | +15.92 | +2.20 | 52.2 | -26.63 |
| B | 71 | +30.00 | +11.42 | 71.4 | -20.94 |
| C | 139 | +10.33 | +7.47 | 60.5 | -21.87 |
| D | 1575 | +29.37 | +15.31 | 65.6 | -29.28 |

## Walkforward Panel Coverage

- Total rows: 3922
- With forward returns: 3819 (97.4%)
- With forward max-DD: 3922 (100.0%)
- Top-25 overlap (Jaccard): 72.3%

## Scoring

- Objective score: **-3.4183**
- Candidates evaluated: 8
- Candidates passing constraints: 6
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

