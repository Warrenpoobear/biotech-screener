# CI and production-pipeline hardening — 2026-11-20

**Classification:** Tier 2 engineering with a Tier 3 promotion-gate effect. No ranker, selector, sizing, `final_score`, or weight change. Scoped production-model freeze is unchanged.

## Decision

Fail closed on three production-promotion paths, and make the GitHub daily workflow refresh `production_data/trial_records.json` before it warms the CT.gov cache and runs the screen.

1. A split-adjusted price refresh whose status is not `APPENDED` or `RETROACTIVE_CHANGE` is gate `split_adj_refresh` = FAIL and the run returns before the screen. `RETROACTIVE_CHANGE` stays a warning and still promotes.
2. `run_screen.py --strict` exit 1 stays gate `phase2_health` = FAIL even when `rankings.csv` exists, so the snapshot is not promoted.
3. A same-day rerun skips only when the promoted manifest's `input_hash` matches the current prices (raw and split-adjusted), `trial_records.json`, `coinvest_signals.json`, and the operator override files. A missing hash does not skip.
4. `.github/workflows/phase2-daily-production.yml` runs `collect_ctgov_data.py` (temp file, swap on success) after input hydration and before `warm_caches.py --sources ctgov`. A failed collect keeps the previous file. The existing `ctgov_content_freshness` gate still FAILs promotion when the newest `last_update_posted` is more than 7 days old.

CI-only companions: `requirements.lock` regenerated with `pip-compile --allow-unsafe --generate-hashes`; pytest and agent-workflow install from that lock; replay-regression exits 1 when the golden-baseline download fails; `detect-secrets` excludes `artifacts/monitoring/*.json` (the `model_hash` string is not a credential). `scripts/webhook_receiver.py` rejects POSTs unless `WEBHOOK_SHARED_SECRET` is set and presented, and it binds `127.0.0.1` by default.

## Evidence

- `tools/check_agent_workflow.py` `dependency_lock` compared `requirements.txt` pins (`firecrawl-py==4.31.0`, `langgraph==1.2.7`, `orjson==3.11.9`, `pytest==9.1.1`, `yfinance==1.5.1`) to older lock pins.
- `artifacts/monitoring/daily_2026_11_09.json` field `model_hash` = `827c35a9ed3ee6e1` is the frozen model id from `docs/governance/2026-11-04-missing-fields-clock-reset.md`. The snapshot file was not edited.
- Replay download failure was `exit 0` plus `skip=true` in `.github/workflows/replay-regression.yml`, so later steps were skipped and the job stayed green.
- GitHub Actions run [44749423118](https://github.com/Warrenpoobear/biotech-screener/actions/runs/44749423118) (2026-11-20, `phase2-daily-production`, head `c11c8180`) logged:
  - `CTGov content freshness gate: FAIL — trial_records.json newest last_update_posted=2026-11-03 is 17d old (>7d)`
  - `Hard options coverage: FAIL — hard=105, opt_cov=0.0%, straddle_cov=0.0%, reviewable=0`
  - `RESULT: FAIL — snapshot NOT promoted`
- Committed `production_data/trial_records.json` still has newest `last_update_posted` = 2026-11-03 (21,981 trials). Last content commit that collected trials is `82ba7e7b` (2026-11-04); `c2c353b9` (2026-11-11) only renamed a ticker. `tools/cron_data_refresh.sh` `stage_ctgov` does call `collect_ctgov_data.py`, but that script is the operator-host cron. The Actions workflow checked out git, hydrated inputs, and warmed `cache/ctgov/trial_records_2026-11-20.json` from the stale source (cache gate PASS in the same run). `warm_caches.warm_ctgov` copies the source; it does not fetch CT.gov. The 2026-11-15 host refresh described in `docs/governance/2026-11-15-regulatory-calendar-refresh.md` never landed on `main`, so Actions has been screening a file frozen after 2026-11-03.
- `check_options_coverage` is WARN-only. The promotion block is `check_hard_options_coverage`: FAIL when hard-row `opt_atm_iv` coverage is under 40% (`GateConfig.hard_options_coverage_fail_pct`). Coverage was 0% on 105 hard rows.

## Reasoning

A green CI job that did not run, a WARN that promotes a `--strict` health FAIL, and a manifest skip that ignores replaced inputs all hide a bad production day. The CT.gov gate was already correct; the workflow never refreshed the file the gate reads. Refreshing in the workflow, in the same order as `stage_ctgov` (collect, then warm), is the missing step. The collector's coverage-regression exit is left intact: a bad fetch must not replace a good file.

## Boundary

- No change to scoring, ranks, weights, selector, sizer, or `final_score`.
- `hard_options_coverage` thresholds are unchanged. **Operator decision:** production will keep refusing to promote while hard-catalyst IV coverage is 0%. Restoring the options feed (Tastytrade metrics and/or the Massive S3 entitlement noted as missing since 2026-05-08 in the 2026-11-04 memo) is what clears that gate. Loosening it is a separate decision and was not done here.
- This record does not reset `oos_clock_start`. A successful post-merge collect changes clinical inputs the screen reads. Whether that reset is required is an operator call under `docs/FORWARD_VALIDATION_PROTOCOL.md`, same as the 2026-11-04 CT.gov refresh.
- Webhook auth applies to `scripts/webhook_receiver.py` only. Set `WEBHOOK_SHARED_SECRET` before starting it. The process refuses to start when the variable is unset.
