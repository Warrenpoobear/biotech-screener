# Ruleset Release Summary

Generated: 2026-02-10T08:18:02

| Field | Value |
|-------|-------|
| Baseline ruleset | `d3cdf5c8` (v2_phase2_default.json) |
| Candidate ruleset | `eb833c56` (v1.2.0_candidate.json) |
| Panel rows | 1808 |
| a_floor sweep | [0.4, 0.45, 0.5, 0.52, 0.55, 0.58, 0.6, 0.65] |
| catalyst_near sweep | [30, 45, 60, 90, 120] |

## Parameter Changes

| Parameter | Baseline | Candidate | Delta |
|-----------|----------|-----------|-------|
| `tier_a_optionality_floor` | 0.55 | 0.6 | +0.05 |
| `catalyst_near_days` | 90 | 120 | +30 |

## Objective Improvement

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d return (%) | 13.85 | 15.27 | +1.41 |
| CD median 60d return (%) | 14.42 | 14.30 | -0.13 |
| AB-CD separation (%) | -0.57 | 0.97 | +1.54 |
| AB median max-DD (%) | -23.03 | -22.98 | +0.05 |
| Mean A-count / date | 2.60 | 2.90 | +0.30 |
| Top-25 overlap (%) | 84.30 | 86.60 | +2.30 |
| Mean turnover (%) | 15.70 | 13.40 | -2.30 |

## Tier Distribution

| Tier | Baseline % | Candidate % | Delta |
|------|-----------|------------|-------|
| A | 1.4 | 1.6 | +0.2 |
| B | 10.1 | 9.5 | -0.6 |
| C | 13.8 | 14.2 | +0.4 |
| D | 74.7 | 74.7 | +0.0 |

## Per-Tier 60d Return Detail (Candidate)

| Tier | Count | Mean % | Median % | Hit Rate % | Mean Max-DD % |
|------|-------|--------|----------|------------|---------------|
| A | 29 | +15.04 | +15.31 | 62.1 | -25.96 |
| B | 172 | +31.80 | +15.22 | 67.5 | -26.01 |
| C | 256 | +17.55 | +11.87 | 64.6 | -24.49 |
| D | 1351 | +29.51 | +14.84 | 65.2 | -29.43 |

## Walkforward Panel Coverage

- Total rows: 1808
- With forward returns: 1793 (99.2%)
- With forward max-DD: 1808 (100.0%)
- Top-25 overlap (Jaccard): 84.6%

## Scoring

- Objective score: **-5.9245**
- Candidates evaluated: 40
- Candidates passing constraints: 12
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

