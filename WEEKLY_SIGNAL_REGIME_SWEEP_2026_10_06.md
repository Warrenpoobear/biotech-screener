# Weekly Signal Shared-Regime Sweep — 2026-10-06

## Status
**No signals at WARN/ALERT this week.** Shared-regime comparator probes were NOT run — none were triggered per the skill's run criteria.

## Facts

```json
{
  "schema": "weekly_signal_regime_sweep.v1",
  "as_of_date": "2026-10-06",
  "latest_ic_dashboard_found": "artifacts/ic_dashboard/2026-08-14_dashboard.json",
  "dashboard_as_of_date": "2026-08-14",
  "dashboard_staleness_days": 53,
  "load_bearing_signals_tracked": ["score_rank_pct", "inst_delta_z"],
  "signals": {
    "score_rank_pct": {
      "mean_ic": 0.0182,
      "hit_rate": 0.8158,
      "n_dates": 38,
      "health": "WEAK",
      "latest_ic": 0.0049,
      "threshold_bands": {"healthy": ">=0.03", "weak": "[0.0, 0.03)", "warn": "[-0.03, 0.0)", "alert": "<-0.03"}
    },
    "inst_delta_z": {
      "mean_ic": 0.1342,
      "hit_rate": 1.0,
      "n_dates": 26,
      "health": "HEALTHY",
      "latest_ic": 0.0315
    }
  },
  "overall_attention": "LOW",
  "signals_checked_for_regime_probe": 0,
  "signals_flagged_warn_or_alert": [],
  "comparator_probes_run": [],
  "findings": {
    "anti_correlated_signal_specific": [],
    "shared_regime": [],
    "insufficient_data": [],
    "healthy_no_probe_needed": ["score_rank_pct (WEAK, not WARN/ALERT)", "inst_delta_z (HEALTHY)"]
  }
}
```

## Fleet-level summary

| Signal | Health | Mean IC | Hit Rate | Action |
|---|---|---|---|---|
| score_rank_pct | WEAK | +0.0182 | 81.6% | No probe — WEAK is above WARN threshold (>=0.0), does not trigger regime check |
| inst_delta_z | HEALTHY | +0.1342 | 100% | No probe — healthy |

Only two signals are currently tracked in `tools/build_ic_dashboard.py:SIGNALS` — `clinical_optionality_pct_dev` and `clinical_score_v2_z` were removed/closed per Spec 111 (2026-06-12) and prior governance (2026-04). There is no third load-bearing signal active in the dashboard to check.

Neither tracked signal is at WARN (mean_ic < 0.0) or ALERT (mean_ic < -0.03) this cycle. Per skill criteria (`signal-shared-regime-check` runs only on `ic_health_monitor` SIGNAL_ALERT/WARN), **no comparator probe was warranted or executed**. No `<date>_shared_regime_check.json` artifact was produced this week — there is nothing to diagnose.

## Anomaly flagged (separate from regime-check scope)

**IC dashboard staleness: 53 days.** The most recent `*_dashboard.json`/`history.jsonl` entry is dated 2026-08-14; no dashboard has been regenerated since, despite today being 2026-10-06. This sweep used the last available dashboard as instructed ("latest IC dashboard"), but the underlying `build_ic_dashboard.py` refresh job appears to not have run for ~7.5 weeks. This is a monitoring-pipeline health issue, not a signal-regime finding — flagging for operator visibility since IC health conclusions drawn today may not reflect current market conditions.

- Confidence on "no WARN/ALERT signals" conclusion: HIGH, conditioned on the (stale) dashboard being the only available data. Confidence this reflects *current* (Oct 2026) signal health: LOW — flagged, not inferred.

## What this memo does NOT answer
- Whether `score_rank_pct` or `inst_delta_z` health has changed in the 53 days since the last dashboard build (unknown — dashboard refresh is stale).
- Why the dashboard refresh job stopped running after 2026-08-14 (out of scope for this skill; recommend separate cron/pipeline health check).

## Recommended action (operator decision required)
1. Investigate why `build_ic_dashboard.py` / its scheduling cron has not produced output since 2026-08-14.
2. Once a current dashboard exists, re-run this sweep against it.
3. No signal weight or ruleset changes are warranted from this cycle's data — no WARN/ALERT signals observed.

## Provenance
- Source dashboard: `artifacts/ic_dashboard/2026-08-14_dashboard.json`, `artifacts/ic_dashboard/history.jsonl` (last 5 entries)
- Threshold logic: `tools/build_ic_dashboard.py:234-242` (IC_HEALTHY=0.03, IC_WARN=0.0, IC_ALERT=-0.03)
- Skill reference: `~/.hermes/skills/data-science/signal-shared-regime-check/SKILL.md`
- Idempotency check: prior sweep found at `WEEKLY_SIGNAL_REGIME_SWEEP_2026_08_09.md` (58 days prior to this run — outside the 5-day dedup window)
- Diagnose-only. No production files modified.
