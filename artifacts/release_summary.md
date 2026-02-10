# Ruleset Release Summary

Generated: 2026-02-09T22:45:56

| Field | Value |
|-------|-------|
| Baseline ruleset | `d3cdf5c8` (v2_phase2_default.json) |
| Candidate a_floor | `0.58` |
| Panel rows | 3922 |
| Sweep range | [0.4, 0.45, 0.5, 0.52, 0.55, 0.58, 0.6, 0.65] |

## Parameter Changes

| Parameter | Baseline | Candidate | Delta |
|-----------|----------|-----------|-------|
| `tier_a_optionality_floor` | 0.55 | 0.58 | +0.03 |

## Objective Improvement

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d return (%) | 0.00 | 0.00 | +0.00 |
| CD median 60d return (%) | 1.03 | 0.79 | -0.24 |
| AB-CD separation (%) | -1.03 | -0.79 | +0.24 |
| AB median max-DD (%) | -25.39 | -25.38 | +0.01 |
| Mean A-count / date | 3.00 | 2.70 | -0.30 |
| Top-25 overlap (%) | 82.50 | 79.60 | -2.90 |
| Mean turnover (%) | 17.50 | 20.40 | +2.90 |

## Tier Distribution

| Tier | Baseline % | Candidate % | Delta |
|------|-----------|------------|-------|
| A | 1.7 | 1.5 | -0.2 |
| B | 13.4 | 12.9 | -0.5 |
| C | 18.3 | 18.9 | +0.6 |
| D | 66.7 | 66.7 | +0.0 |

## Per-Tier 60d Return Detail (Candidate)

| Tier | Count | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|-------|--------|----------|------------|---------------|
| A | 60 | +8.37 | +4.88 | 60.0 | -26.73 |
| B | 504 | +9.23 | -0.42 | 48.5 | -28.68 |
| C | 743 | +19.81 | -1.56 | 46.8 | -28.73 |
| D | 2615 | +14.12 | +1.69 | 51.7 | -32.45 |

## Walkforward Panel Coverage

- Total rows: 3922
- With forward returns: 3819 (97.4%)
- With forward max-DD: 3922 (100.0%)
- Top-25 overlap (Jaccard): 79.7%

## Tradeoffs

- Lower stability: overlap 79.6% vs baseline 82.5%

## Scoring

- Objective score: **-8.4041**
- Candidates evaluated: 8
- Candidates passing constraints: 0
- Constraints: A% >= 3.0%, turnover <= 50.0%, separation > 0
- Status: **FAIL**
  - A-tier 1.5% < min 3.0%
  - separation -0.79 <= 0

## QA Checklist

- [ ] Calibration report reviewed (`artifacts/calibration_report.md`)
- [ ] Walkforward report reviewed (`artifacts/walkforward_report.md`)
- [ ] `bump_ruleset.py --from-json artifacts/candidate_overrides.json` executed
- [ ] Contract tests pass (`pytest tests/test_decision_engine_contract.py`)
- [ ] Replay regression pass (`pytest tests/test_decision_engine_replay_regression.py`)
- [ ] Golden records refreshed if needed (`scripts/refresh_goldens.py`)
- [ ] Changelog finalized (remove [DRAFT] marker)
- [ ] `promote_ruleset.py <candidate_id>` executed

