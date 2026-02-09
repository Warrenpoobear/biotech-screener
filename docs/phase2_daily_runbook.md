# Phase-2 Daily Runner — Ops Runbook

## Overview

The `phase2-daily` GitHub Actions workflow runs every weekday at 14:00 UTC.
It executes the full screening pipeline in `--decision-mode phase2 --strict`,
then evaluates the health gate. Outcomes:

| Exit code | Health status | GitHub behavior |
|-----------|--------------|-----------------|
| 0 | OK | Green check |
| 2 | WARN | Yellow annotation (job passes, no email) |
| 1 | FAIL | Red X (job fails, emails repo watchers) |

## Health reasons reference

### FAIL reasons (exit 1 — immediate action required)

| Reason | Meaning | Likely cause |
|--------|---------|-------------|
| `optionality_broken` | Dev-stage optionality coverage < 80% | Catalyst feed is stale or broken — clinical_optionality_pct not populated |
| `catalyst_broken` | Catalyst coverage below minimum | Catalyst events file missing or empty for the as-of-date |
| `pipeline_error` | run_screen.py crashed before producing health JSON | Code bug, missing dependency, or data file corruption |
| `ruleset_mismatch` | Loaded ruleset ID != pinned Phase-2 ID | Wrong default ruleset or config drift |

### WARN reasons (exit 2 — review, usually benign)

| Reason | Meaning | Typical action |
|--------|---------|---------------|
| `no_a_tier_regime` | Zero A-tier securities but optionality feed is healthy (coverage >= 80%) | Sparse-catalyst month — expected ~10% of the time. No action unless persistent. |
| `high_turnover` | Portfolio turnover exceeds warning threshold | Check if major index rebalance or data correction caused unusual churn |
| `weight_drift` | L1 weight change exceeds threshold vs prior snapshot | Review which names moved and why |

## First 3 things to check on FAIL

1. **Inputs bundle**: Did the data hydrate step succeed? Check the `Verify inputs`
   step log — missing files or hash mismatches mean stale or corrupt data.
2. **Catalyst coverage**: Open the snapshot's `phase2_health.json` →
   `metrics.dev_optionality_coverage_pct`. Below 80% triggers `optionality_broken`.
3. **Pipeline log**: Download the `phase2-output` artifact, read `phase2_daily.log`
   for the full traceback or error message.

## How to rerun for a specific date

1. Go to **Actions → phase2-daily → Run workflow**
2. Enter the date in `as_of_date` (e.g. `2026-02-07`)
3. Click **Run workflow**

Or via CLI:

```bash
gh workflow run phase2-daily -f as_of_date=2026-02-07
```

## How to rerun locally

```bash
python run_phase2_daily.py --as-of-date 2026-02-07
echo $?  # 0=OK, 1=FAIL, 2=WARN
```

Log file: `output/phase2_daily.log`
Health JSON: `data/snapshots/2026-02-07/phase2_health.json`

## Updating pinned thresholds

Health thresholds are pinned in `production_data/phase2_health_thresholds/v1.json`
(ID: `26f0d3d2`). To recalibrate:

```bash
python run_phase2_health_calibration.py
# Review output/phase2_health_threshold_recommendation.txt
# Copy new thresholds to production_data/phase2_health_thresholds/v1.json
# Regenerate manifest: python verify_inputs.py --generate
```

## Updating the inputs manifest

After any change to core production_data files:

```bash
python verify_inputs.py --generate
# Commit production_data/inputs_manifest.json
```
