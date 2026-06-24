# TOOLS.md — Herald Agent Commands

Canonical news pipeline: fetch → dedupe → classify → digest. Deterministic (no LLM for collection/classification by default).

## Environment

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
source .env 2>/dev/null   # SMTP_USER, SMTP_PASSWORD, XAI_API_KEY (optional)
export PYTHONPATH=.
```

## Daily pipeline (manual recovery)

**Preferred:** `python3 tools/herald_recovery.py --as-of-date YYYY-MM-DD` or `python3 tools/herald_health_check.py --recover`

Replace `YYYY-MM-DD` with the as-of date (usually today on weekdays).

```bash
AS_OF=YYYY-MM-DD

# 1. Fetch all universe tickers
python3 tools/fetch_company_press_releases.py --as-of-date "$AS_OF"

# 2. Dedupe (required for supervisor done predicate)
python3 tools/dedupe_press_releases.py \
  --input "data/press_releases/releases_${AS_OF}.jsonl"

# 3. Classify (input: deduped file — writes classified/classified_${AS_OF}.jsonl)
python3 tools/classify_press_releases.py \
  --input "data/press_releases/deduped/deduped_${AS_OF}.jsonl"

# 4. Digest (morning / midday / evening windows)
python3 scripts/build_news_digest.py --window morning --as-of-date "$AS_OF"
python3 scripts/build_news_digest.py --window midday --as-of-date "$AS_OF"
python3 scripts/build_news_digest.py --window evening --as-of-date "$AS_OF"
```

## Health check (host cron / operator triage)

```bash
python3 tools/herald_health_check.py
python3 tools/herald_health_check.py --as-of-date YYYY-MM-DD
python3 tools/herald_health_check.py --json
```

Writes `artifacts/herald/health_check_YYYY-MM-DD.json`. Exit 0 = HEALTHY, 1 = WARN, 2 = FAIL (dark).

## Source health only (no fetch)

```bash
python3 tools/fetch_company_press_releases.py --health-check
```

## Single-ticker debug

```bash
python3 tools/fetch_company_press_releases.py --as-of-date YYYY-MM-DD --ticker MRNA
```

## Done predicate (supervisor + heartbeat)

Both must exist for `YYYY-MM-DD`:

- `data/press_releases/deduped/deduped_YYYY-MM-DD.jsonl`
- `data/press_releases/classified/classified_YYYY-MM-DD.jsonl`

If dedupe exists but classify failed, re-run classify only (step 3).

## Cron (authoritative on WSL host)

| Time (ET) | Job |
| --- | --- |
| 07:30 | Herald-only fetch (pre-morning) |
| 08:00 | Morning digest |
| 14:00 | `cron_data_refresh.sh herald` (fetch + classify) |
| 14:35 | Herald agent heartbeat |
| 15:00 | Midday digest |
| 17:30 | Production pipeline (includes Herald step 5l.5) |
| 18:00 | Evening digest |

## Daily working set

1. `python3 tools/herald_health_check.py` — pipeline status
2. `data/press_releases/fetch_state.json` — per-ticker fetch state
3. `data/press_releases/health_YYYY-MM-DD.json` — fetch health artifact
4. `artifacts/news_digest/biotech_news_digest_YYYY-MM-DD_*.json` — digest outputs
5. `artifacts/news_digest/delivery_log.jsonl` — send failures

## Red lines

- Do not modify rankings, scoring, rulesets, or production snapshots
- Do not `git push` or change tracked production data without operator approval
- Do not feed unverified social media into digests
- When fetch fails for >10% of tickers: report `FETCH_DEGRADED`, do not silence
