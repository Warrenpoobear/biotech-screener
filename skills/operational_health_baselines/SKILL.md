# Operational Health Baselines

SLA baselines for agent artifact freshness and CI health. Used by fleet heartbeat, Herald health check, and failure-pattern prevention rules (F-2026-005, F-2026-006).

## When to use

- Diagnosing dark pipelines (Herald, daily production chain)
- Escalating stale artifacts past threshold
- Writing LRNs for recurring ops outages
- Closing stalled-loop verdicts in `.learnings/memory.md`

## Artifact freshness SLAs

| Lane | Path pattern | Cadence | Stale threshold | Dark fail |
|------|--------------|---------|-----------------|-----------|
| Herald classified | `data/press_releases/classified/classified_{date}.jsonl` | daily_premarket | 2d | 7d |
| Herald deduped | `data/press_releases/deduped/deduped_{date}.jsonl` | daily_premarket | 2d | 7d |
| Ops digest | `artifacts/ops_digest/{date}_digest.json` | daily_after_production | 2d | — |
| IC dashboard | `artifacts/ic_dashboard/{date}_dashboard.json` | daily_after_production | 2d | — |
| Shadow monitor | `artifacts/shadow_monitor/{date}_monitor.json` | daily_after_production | 2d | — |
| Production QA | `artifacts/production_qa/{date}_report.json` | daily_after_production | 2d (after 21:00 ET) | — |
| CTgov cache | `cache/ctgov/trial_records_{date}.json` | daily_premarket | 2d | — |
| CTgov diff | `artifacts/ctgov_daily/{date}_diff.json` | daily_premarket | 2d | — |
| Snapshot | `data/snapshots/{date}/rankings.csv` | daily_after_production | 2d | — |

**Content date rule:** prefer `YYYY-MM-DD` embedded in filenames over file mtime (git checkout must not mask staleness).

## Health check commands

```bash
python3 tools/fleet_ops_status.py
python3 tools/fleet_ops_status.py --write
python3 tools/agent_heartbeat_checks.py --json
python3 tools/herald_health_check.py --stdout
python3 tools/weekly_skills_digest.py
python3 tools/audit_learnings.py
```

## Heartbeat escalation (phase 4+)

Default cron path is **artifact-only** — no LLM unless operator sets `HEARTBEAT_LLM_ESCALATE=1`.

| Artifact | Purpose |
|----------|---------|
| `artifacts/heartbeat/{date}_receipt.md` | Daily fleet verdict |
| `artifacts/heartbeat/{date}_escalation.json` | Structured anomalies (ops_supervisor reads first) |
| `artifacts/heartbeat/{date}_anomalies.md` | Human-readable fallback |
| `artifacts/fleet_ops/{date}_status.json` | Weekly/daily operator triage (`fleet_ops_status --write`) |

## Escalation ladder

| Condition | Action |
|-----------|--------|
| Herald classified > 2d stale | WARN in heartbeat; run Herald recovery sequence |
| Herald classified ≥ 7d dark | FAIL; file/update F-2026-005 pattern |
| CI red > 5 consecutive weekdays | Operator escalation; file F-2026-006 |
| ops_supervisor RED | Read `artifacts/ops_supervisor/{date}_supervisor.md`; append LRN if recurring |
| Memory > 7d behind artifacts | `STALE_MEMORY` in heartbeat — LLM agent not writing notes |

## Herald recovery (operator host)

```bash
# Automated minimal recovery (recommended)
python3 tools/herald_recovery.py --as-of-date $(date +%Y-%m-%d)

# Or via health check
python3 tools/herald_health_check.py --recover

# Dry-run / full pipeline
python3 tools/herald_recovery.py --dry-run --full --digest
bash tools/herald_recovery.sh --full --digest
```

Manual step-by-step (if needed):

```bash
TODAY=$(date +%Y-%m-%d)
python3 tools/fetch_company_press_releases.py --as-of-date $TODAY
python3 tools/dedupe_press_releases.py --input data/press_releases/releases_${TODAY}.jsonl
python3 tools/classify_press_releases.py --input data/press_releases/deduped/deduped_${TODAY}.jsonl
python3 scripts/build_news_digest.py --window evening --as-of-date $TODAY
python3 tools/herald_health_check.py --stdout
```

## Self-learning integration

When an outage matches a row above:

1. Append `.learnings/LEARNINGS.md` with `Promotion-lane: skill` (ops/plumbing) or `spec` (scoring)
2. Search `docs/FAILURE_PATTERN_LIBRARY.md` for existing pattern ID
3. After fix + 14d zero recurrence → close stalled-loop row in `memory.md`

## Governance

Tier 0 — observability only. Does not change scoring, ranker, or production rulesets.
