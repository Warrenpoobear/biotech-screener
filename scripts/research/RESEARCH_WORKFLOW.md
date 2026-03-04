# Research Submission Workflow

This document describes the standard workflow for evaluating a new decision-engine
ruleset candidate and promoting it to production.

---

## Overview

```
Experiment design
      ↓
Build candidate ruleset JSON
      ↓
submit_research.py → Audited Backtest (IS + OOS)
      ↓
VERDICT: PROMOTE | ARCHIVE | NEEDS_MORE
      ↓
 PROMOTE → promote_ruleset.py
 ARCHIVE → move to research archive
 NEEDS_MORE → iterate
```

---

## Step 1: Build candidate ruleset

Copy the current active ruleset and modify the parameters you want to test:

```bash
cp production_data/decision_rulesets/v1.8.3_buffer30_candidate.json \
   production_data/decision_rulesets/my_candidate.json
# Edit my_candidate.json with your changes
```

Key fields to consider modifying:
- `calendar_alpha_sort_weight` — sort tilt strength
- `rebalance_buffer_ranks` — rebalance stability
- `enable_clinical_sizing` — clinical position sizing
- `enable_calendar_alpha_sort` — calendar alpha sort toggle

---

## Step 2: Submit for evaluation

```bash
python3 scripts/research/submit_research.py \
    --ruleset production_data/decision_rulesets/my_candidate.json \
    --name "v1.9.0_my_experiment"
```

This runs:
- Preflight batch on current snapshots (strict mode)
- Re-rank through candidate ruleset
- eval_forward_returns at horizons 84d + 126d, top-K=20, cost=30bps
- Baseline auto-discovered from manifest (active ruleset → most recent audited run)
- Writes results to `output/audited_backtests/v1.9.0_my_experiment/`

**Exit codes:**
| Code | Verdict | Meaning |
|------|---------|---------|
| 0 | PROMOTE | Meets both primary (126d IC) and guardrail (84d IC) bars vs baseline |
| 1 | ARCHIVE | Does not meet promotion bars |
| 2 | NEEDS_MORE | Insufficient dates or no baseline for comparison |

---

## Step 3: Review the verdict

```bash
cat output/audited_backtests/v1.9.0_my_experiment/VERDICT.md
```

The VERDICT.md shows:
- IC comparison: candidate vs baseline at 84d and 126d
- Net return delta
- Turnover impact
- t-statistics and confidence intervals
- Recommended promote/archive command

---

## Step 4: Promote or archive

**If PROMOTE:**
```bash
python3 scripts/promote_ruleset.py \
    --ruleset production_data/decision_rulesets/my_candidate.json \
    --gate-summary "$(cat output/audited_backtests/v1.9.0_my_experiment/VERDICT.md | head -5)"
```

**If ARCHIVE:**
```bash
mv production_data/decision_rulesets/my_candidate.json \
   production_data/decision_rulesets/research_archive/my_candidate.json
```

---

## Advanced options

```bash
# Use explicit baseline instead of auto-discovery:
python3 scripts/research/submit_research.py \
    --ruleset my_candidate.json \
    --name my_run \
    --baseline-dir output/audited_backtests/oos_2020_2024_runA/

# Dry run (print what would execute):
python3 scripts/research/submit_research.py \
    --ruleset my_candidate.json \
    --name my_run \
    --dry-run

# Custom snapshot root (e.g. OOS data):
python3 scripts/research/submit_research.py \
    --ruleset my_candidate.json \
    --name my_run_oos \
    --snapshot-root data/snapshots_reranked_baseline_oos
```

---

## Promotion bars (current thresholds)

| Metric | Promotion requirement |
|--------|-----------------------|
| 126d IC (primary) | ΔIC ≥ 0 vs baseline |
| 84d IC (guardrail) | ΔIC ≥ -0.003 vs baseline (guardrail, not primary) |
| Minimum dates | ≥ 50 evaluated dates for PROMOTE/ARCHIVE verdict |

These thresholds are defined in `run_audited_backtest._compute_verdict()`.

---

## File locations

| File | Purpose |
|------|---------|
| `output/audited_backtests/{name}/VERDICT.json` | Machine-readable verdict |
| `output/audited_backtests/{name}/VERDICT.md` | Human-readable verdict |
| `output/audited_backtests/{name}/eval/summary.json` | Eval statistics |
| `output/audited_backtests/{name}/AUDIT.md` | Full audit trail |
