# Weekly Signal Regime Sweep — 2026-05-03

**Status:** Diagnose-only weekly sweep per `signal-shared-regime-check` skill (Sunday cron).
**Type:** Read-only IC inventory; no production code or config touched.
**Source dashboard:** `artifacts/ic_dashboard/2026-05-01_dashboard.json` (latest available — no run since Friday)

---

## Question

For each load-bearing signal currently at WARN or ALERT in the rolling IC dashboard, is the failure shared-regime (lane broken) or signal-specific (lane healthy, signal broken)?

## Inventory

| Signal | health | mean_ic | hit_rate | latest_ic | n | window |
|---|---|---|---|---|---|---|
| score_rank_pct | WEAK | +0.0011 | 0.389 | -0.0316 | 36 | 2026-02-18 → 2026-04-01 |
| clinical_optionality_pct_dev | HEALTHY | +0.1074 | 0.917 | -0.0371 | 36 | 2026-02-18 → 2026-04-01 |
| inst_delta_z | **ALERT** | -0.1006 | 0.086 | -0.0092 | 35 | 2026-02-19 → 2026-04-01 |

Thresholds in dashboard: HEALTHY≥0.03, WEAK≥0.00, WARN≥-0.03, ALERT<-0.03. Dashboard `attention: HIGH`.

## Triggered checks

**Eligible (WARN/ALERT, per user filter):** `inst_delta_z` only.

**Out of scope per filter:**
- `score_rank_pct` (WEAK, not WARN/ALERT — borderline at +0.0011, hit rate 0.389; flag for next week if it crosses below 0)
- `clinical_optionality_pct_dev` (HEALTHY)

## Result for inst_delta_z

**No new evidence to publish.** A fidelity-verified comparator check against `coinvest_score_z` already exists from 2026-05-03 over the IDENTICAL window the dashboard reports today:

- Existing memo: `INST_DELTA_Z_SHARED_REGIME_CHECK_2026_05_03.md`
- Existing artifact: `artifacts/ic_dashboard/2026-05-03_shared_regime_check.json`
- Window: 2026-02-19 → 2026-04-01, n=35, horizon=20 — matches `inst_delta_z.per_date` in the 2026-05-01 dashboard exactly
- Headline finding: `coinvest_score_z` mean IC +0.1033 (HEALTHY) over the same window; Spearman ρ on the two IC trajectories = **−0.5712** (p=0.0003). Lane is healthy. Failure is signal-specific and structurally anti-correlated with the lane's healthy sibling.

Re-running the comparator would reproduce that artifact byte-equivalently because:
1. The dashboard window has not advanced (latest `per_date[-1]` is still 2026-04-01 — 20-day forward-return maturity gates this; new dates won't appear until forward returns settle).
2. The price source (`production_data/price_history.csv`) is the same.
3. The comparator (`coinvest_score_z`) is the same.

This is a no-op week for inst_delta_z by design. Re-publication would manufacture an artifact, not produce new evidence.

## Summary table

| Signal | Check | Verdict | Memo |
|---|---|---|---|
| inst_delta_z | vs coinvest_score_z | anti_correlated_signal_specific | `INST_DELTA_Z_SHARED_REGIME_CHECK_2026_05_03.md` (existing, current) |
| score_rank_pct | not run (WEAK, not WARN/ALERT) | — | — |
| clinical_optionality_pct_dev | not run (HEALTHY) | — | — |

## Implications

- The disposition lane on inst_delta_z (parent memo `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md`) is unchanged by this sweep.
- The next material event for inst_delta_z is the ~2026-05-15 13F refresh, which resets the delta input. The `−0.57` IC anti-correlation is structural (level vs delta), so a refresh may reset magnitude but not necessarily sign. Sweep should re-run the day the dashboard window actually advances.
- `score_rank_pct` mean IC is sitting at +0.0011 with a 38.9% hit rate — it is one bad week from crossing into WARN. Watch for a status change in next Friday's dashboard; if it crosses, the natural comparator is `selector_score` (the production sort anchor itself) or its constituent block scores.
- No new finding to escalate; no governance log entry required.

## What this sweep does NOT prove

- Does not re-test inst_delta_z over a new window (the dashboard window has not advanced).
- Does not evaluate `score_rank_pct` against any comparator — explicitly out of scope per WARN/ALERT filter.
- Does not address whether `coinvest_score_z`'s HEALTHY reading on the same window is itself robust to bootstrap/LOSO (Checklist v2 already says coinvest is 3/5 standalone — see CLAUDE.md trust buckets).

## Provenance

- Skill: `signal-shared-regime-check` (weekly sweep mode)
- Latest dashboard: `artifacts/ic_dashboard/2026-05-01_dashboard.json` (mtime 2026-05-01)
- Reused artifact: `artifacts/ic_dashboard/2026-05-03_shared_regime_check.json`
- Parent memo (still current): `INST_DELTA_Z_SHARED_REGIME_CHECK_2026_05_03.md`
- Governance memo (open disposition): `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md`
- Methodology source: `tools/build_ic_dashboard.py:117-173`
- Generated: 2026-05-03 (Sunday cron, 14:00 ET)
