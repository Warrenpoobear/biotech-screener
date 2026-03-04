# Phase-2 Daily Production Runbook

**Active ruleset:** `fb0af0ac` — v1.8.1 alpha_modifier_off
**Engine:** v1.3.0
**Last updated:** 2026-03-05

---

## 1. Daily command (single entrypoint)

```bash
python3 tools/run_daily_production.py --as-of-date YYYY-MM-DD
```

No manual pre-warm steps. The runner is self-sufficient:

| Step | What it does |
|---|---|
| 1 | Incremental `price_history.csv` refresh via Yahoo Finance |
| 1.5 | Warm `sec_8k`, `ctgov`, `sec_13f` caches (idempotent — skips if already fresh) |
| 2 | Run screen → staging dir |
| 2.5 | Create PIT price anchor from staging `rankings.csv` (before gates) |
| 3 | Data integrity audit |
| 4 | Evaluate all gates |
| 5 | Build run manifest + atomic promotion to `data/snapshots/YYYY-MM-DD/` |
| 6 | (opt-in) PIT price backfill via `--price-pit-backfill` |

If `market_data.json` is stale the runner auto-calls `collect_market_data.py` before aborting.

**Escape hatches:**

```bash
# Skip all PIT warming (CI handles externally):
python3 tools/run_daily_production.py --as-of-date DATE --skip-pit-warm

# Use prior ctgov cache if today's isn't available yet (WARN instead of FAIL):
python3 tools/run_daily_production.py --as-of-date DATE --allow-date-fallback

# Override warm sources (empty string = skip warm entirely):
python3 tools/run_daily_production.py --as-of-date DATE --warm-sources "ctgov,sec_13f"
```

---

## 2. Exit codes

| Code | Meaning |
|---|---|
| 0 | All gates PASS — snapshot promoted |
| 1 | Hard gate FAIL — snapshot stays in staging, not promoted |
| 2 | Soft gate WARN — snapshot promoted but flagged |

---

## 3. Gate reference and expected daily status

### Hard gates (FAIL = snapshot not promoted)

| Gate | Normal | FAIL condition |
|---|---|---|
| `xbi_staleness` | PASS | XBI price gap > 3 trading days |
| `ctgov_cache` | PASS | ctgov cache absent after warm attempt |
| `inputs_present` | PASS | `market_data.json` or `universe.json` missing |
| `market_data_schema` | PASS | Required fields absent/malformed |
| `market_data_staleness` | PASS | `market_data.json` age > 3d (auto-refresh fires first) |
| `market_data_coverage` | PASS | < 90% of universe tickers have market data |
| `screen` | PASS | `run_screen.py` exits non-zero |
| `audit` | PASS | `data_integrity_audit.py` exits non-zero |
| `sort_contrib_sanity` | PASS | Any sort contrib > 50 or non-finite |
| `exposure_missingness` | PASS | > 25% eligible tickers missing an exposure |
| `risk_concentration` | PASS | > 50% top-K weight has catalyst ≤ 7d |

### Soft gates (WARN = promoted with flag)

These should be **PASS on a healthy day** after 2026-03-05:

| Gate | Normal | WARN condition |
|---|---|---|
| `drift_monitoring` | PASS | > 30% top-20 turnover vs prior (day-1 after promotion is normal) |
| `ruleset_health` | PASS | Rank drift vs baseline exceeds 3× warn factor |
| `sec_13f_cache` | PASS | 13F warm failed or < 80% coverage |
| `price_pit_cache` | PASS | Price anchor missing for as-of date |
| `pit_bundle_health` | PASS | ctgov or 13F prerequisite missing |
| `institutional_summary` | PASS | Sidecar missing or low coverage |
| `institutional_delta` | PASS | Delta sidecar not written |
| `cache_health` | PASS | sec_8k or ctgov cache marked bad |
| `forward_eval` | PASS (cold-start until horizons fill) | Rolling IC below floor |
| `pnl_attribution` | **WARN expected** (see §4) | Price data unavailable for attribution period |
| `risk_concentration` | **WARN expected** (see §4) | > 50% top-K high-beta or drawdown positions |

---

## 4. Known persistent WARNs (not actionable)

### `pnl_attribution` — price lag

```
PnL coverage 0.0% below 80.0%
```

PnL attribution computes returns between T-1 and T positions, requiring closing prices for both dates. During the burn-in period we're running for dates ahead of available Yahoo Finance data, so `n_priced=0`. This resolves automatically once the runner operates in a T+1 workflow (e.g. running Monday evening for Monday's close).

**This is not a model failure.** The turnover and position-action breakdown still populate correctly.

### `risk_concentration` — portfolio composition signal

```
high_risk=55% > 50% WARN
```

55% of top-20 positions have beta ≥ 1.5 or drawdown ≤ -30%. This reflects structural biotech characteristics (high beta is sector-wide). The 50% warn threshold is intentionally conservative to surface real crowding events. **Review if this rises above 70%** or if `stacked_wt` (catalyst ≤ 7d AND high-risk) becomes non-zero.

---

## 5. Burn-in results (new self-sufficient runner)

Gate key: ✓ PASS · ! WARN · ✗ FAIL · — not run

| Gate | 03-03 pre-fix | 03-04 pre-fix | **03-05 new runner** |
|---|:---:|:---:|:---:|
| `cache_health` | ✓ | ✓ | ✓ |
| `drift_monitoring` | ✓ | ✓ | ✓ |
| `ruleset_health` | ✓ | ✓ | ✓ |
| `sec_13f_cache` | ! | ! | **✓** |
| `price_pit_cache` | ! | ! | **✓** |
| `pit_bundle_health` | ! | ! | **✓** |
| `institutional_summary` | ! | ! | **✓** |
| `institutional_delta` | ! | ! | **✓** |
| `pnl_attribution` | ! | ! | ! (price lag — expected) |
| `forward_eval` | — | — | ✓ cold-start |

03-06 aborted: XBI staleness FAIL — no new Yahoo data available yet for that week.

---

## 6. Ruleset health and rollback

### Monitor

```bash
tail -10 artifacts/ruleset_health_history.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    print(d.get('as_of_date'), d.get('status'), d.get('detail','')[:70])
"
```

**Rollback trigger:** 3 consecutive `ruleset_health` WARN days.

### Rollback to last known good

```bash
python3 tools/promote_ruleset.py --rollback --reason "3 consecutive ruleset_health WARNs"
```

Then update `PHASE2_PINNED_RULESET_ID` in both `run_screen.py` and `scripts/run_phase2_snapshot_delta.py` to match the new active ID.

### Check active ruleset

```bash
python3 -c "
import json
m = json.load(open('production_data/decision_rulesets/manifest.json'))
r = next(r for r in m['rulesets'] if r['status']=='active')
print(r['id'], r['file'], r['description'])
"
```

---

## 7. Artifact locations

| Artifact | Path |
|---|---|
| Daily snapshots | `data/snapshots/YYYY-MM-DD/` |
| Run manifest | `data/snapshots/YYYY-MM-DD/run_manifest.json` |
| Rankings | `data/snapshots/YYYY-MM-DD/rankings.csv` |
| Institutional sidecar | `data/snapshots/YYYY-MM-DD/institutional_summary.json` |
| PnL attribution | `data/snapshots/YYYY-MM-DD/pnl_attribution.json` |
| Ruleset manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion receipts | `production_data/decision_rulesets/promote_receipt_*.json` |
| Ruleset health history | `artifacts/ruleset_health_history.jsonl` |
| PIT price cache | `data/caches/price_pit/PIT/YYYY-MM-DD/` |
| 13F PIT cache | `data/caches/sec_13f/PIT/YYYY-MM-DD/` |
| ctgov cache | `cache/ctgov/trial_records_YYYY-MM-DD.json` |
| Fallback manifest (CI) | `output/run_manifest.json` |

---

## 8. Common failures and fixes

### XBI staleness FAIL
```
XBI last=2026-03-02, as_of=2026-03-06, gap=4 trading days
```
Yahoo Finance data usually appears 15–30 min after 4 pm ET. Re-run after market close, or run for the prior trading day.

### ctgov cache FAIL (after warm)
```
CTGov cache gate: FAIL — PIT cache missing after warm
```
CTGov API is down or network error. Retry manually:
```bash
python3 warm_caches.py --as-of-date DATE --sources ctgov
```
Or use `--allow-date-fallback` to fall back to prior day's cache (WARN instead of FAIL).

### Screen FAIL
```
Screen FAILED (exit 1)
```
Run `run_screen.py` manually with the same args to see the full traceback. Most common causes: missing data dependency, PIT violation, or DE schema mismatch.

### Market data stale + auto-refresh failure
```
Market data stale — auto-refreshing ...
Market data refresh FAILED (exit 1)
```
Run manually:
```bash
python3 collect_market_data.py --universe production_data/universe.json
```
Then re-run the daily runner.

---

## Evaluation pitfalls

### A/B comparisons must rerank both legs

When comparing two rulesets using `eval_forward_returns.py`, **both candidate and
baseline must be reranked through their respective rulesets** via
`scripts/research/rerank_snapshots.py`. Never compare reranked candidate snapshots
against raw `data/snapshots/` — those contain a mix of historical rulesets and will
produce misleading IC deltas (discovered 2026-03-03, v1.8.2 promotion).

Correct workflow:
```bash
# Rerank both legs
python3 scripts/research/rerank_snapshots.py --ruleset <candidate.json> --out-root /tmp/reranked_candidate
python3 scripts/research/rerank_snapshots.py --ruleset <baseline.json>  --out-root /tmp/reranked_baseline

# Eval both against same price data
python3 scripts/eval_forward_returns.py --snapshot-root /tmp/reranked_candidate --out-dir /tmp/eval_candidate ...
python3 scripts/eval_forward_returns.py --snapshot-root /tmp/reranked_baseline  --out-dir /tmp/eval_baseline  ...
```

`eval_ruleset.py --rerank-only` handles this internally (reranks both legs on the fly)
and is safe for gate verdicts, but does not compute forward returns.

---

## 9. Weekly health packet

### Generating

```bash
python3 tools/weekly_health_packet.py                          # latest snapshot
python3 tools/weekly_health_packet.py --as-of-date 2026-03-07  # specific date
```

Output: `output/health_packets/health_YYYY-MM-DD.md` + `.json`

In CI (`phase2-daily-production.yml`), the packet is generated automatically after every
successful run and uploaded as a `health-packet-YYYY-MM-DD` artifact (180-day retention).
The markdown is also appended to the GitHub Actions step summary.

### Action threshold policy (what requires a response)

| Signal | Condition | Action |
|--------|-----------|--------|
| Gate FAIL | Any gate in FAIL | Investigate immediately |
| Drift WARN streak | ≥ 2 consecutive WARN days | Review drift report; consider rollback |
| Ruleset rollback flag | `recommend_rollback: true` in `ruleset_health.json` | Run rollback check (`scripts/promote_ruleset.py --rollback`) |
| Turnover spike | This-week > 2.5× trailing 4-week avg **and** > 5% | Check for composition change or data anomaly |
| Cache health | Explicit `warn` or `fail` (not `unknown`) | Re-warm cache; check `cache_health.json` |

**Not actionable**: `unknown` cache health (old snapshots), single-day drift WARN,
fresh-start turnover (shown as `—`).

### Relaxed mode

Pass `--relaxed` to generate a packet for a non-production run (e.g., a backtest
snapshot or a manual re-run). The output gets a prominent `⚠ RELAXED MODE` banner
and is not suitable for governance review. Relaxed packets are never uploaded by CI.
