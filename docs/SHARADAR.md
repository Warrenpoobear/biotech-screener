# Sharadar (direct API): shadow data feed

**Status:** SHADOW ONLY. This is ingestion infrastructure (Tier 2). No production scoring path reads it.
Adopting Sharadar as a production price, financial or 13F source is a source change under the
NO_MODEL_CHANGE forward-validation window (`docs/FORWARD_VALIDATION_PROTOCOL.md` §1). It needs
operator authorization and a parity check against the current sources first.
`tests/test_sync_sharadar_shadow.py::TestShadowIsolation` enforces the boundary.

## Access

| | |
|---|---|
| Service | `https://api.sharadar.com/v1.0` (direct Sharadar, **not** Nasdaq Data Link) |
| Key | `SHARADAR_API_KEY` in `.env`, sent as the `x-api-key` header |
| Subscription | Bundle with full history (verified 2026-11-04: SEP back to 2005, SF1, SF2, SF3A, DAILY, ACTIONS) |
| Rate limit | `x-ratelimit-limit: 5000` requests per window (response headers) |
| Licence | Internal team use. Check with Jason before using the data externally or sharing the key. |

## Code

| File | Role |
|---|---|
| `common/sharadar_client.py` | Client with retries, offset pagination and ticker batching. The key never goes on a URL. |
| `tools/sync_sharadar_shadow.py` | Mirrors universe tables into `cache/sharadar/` |
| `backtest/sharadar_provider.py` | `SharadarReturnsProvider.from_api` now uses the direct client |
| `sec_13f/scripts/sync_sharadar_sep.py` | Legacy SEP mirror, moved off Nasdaq Data Link |

## Cache layout (`cache/sharadar/`, gitignored)

| Path | Contents |
|---|---|
| `raw/<TABLE>/<stamp>_<mode>.jsonl.gz` | Immutable slices, exactly as received |
| `manifest.jsonl` | One line per slice: `fetched_at_utc`, filters, tickers, rows, sha256 |
| `curated/<TABLE>.csv.gz` | Latest-known view, upserted by primary key (the newest `lastupdated` wins) |
| `state.json` | Per-table tickers and the incremental watermark |
| `status_latest.json` | Summary of the last run |

| Table | Key | Incremental on | Tickers |
|---|---|---|---|
| TICKERS | table, permaticker | full snapshot | universe |
| SEP | ticker, date | `lastupdated` (5d overlap) | universe |
| SFP | ticker, date | `lastupdated` (5d overlap) | XBI, IBB, SPY |
| DAILY | ticker, date | `lastupdated` (5d overlap) | universe |
| SF1 | ticker, dimension, calendardate, date | `lastupdated` (5d overlap) | universe |
| ACTIONS | ticker, date, action, contraticker | `date` (45d overlap) | universe |
| SF3A | ticker, date | `date` (150d overlap) | universe |

The universe is every ticker in `production_data/universe.json`, whatever its status (no
survivorship filter). A ticker new to the universe gets its own backfill from `--backfill-start`
(default 2010-01-01).

## Running it

```bash
set -a; source .env; set +a
python3 tools/sync_sharadar_shadow.py                        # all tables
python3 tools/sync_sharadar_shadow.py --tables SEP,SFP
tools/cron_data_refresh.sh sharadar                          # cron wrapper (not part of `all`)
```

Exit codes: `0` ok, `1` at least one table failed (the others still ran), `2` auth failure (the run aborts).

## Pitfalls (verified live 2026-11-04)

1. **`count` does not reveal truncation.** A query with more than 10,000 rows returns `count=10000`,
   so the vendor README's `len(data) < count` check never fires. The client treats a full page as
   "maybe more" and pages on with `offset`.
2. **The curated view is not point-in-time.** Sharadar re-stamps `lastupdated` on historical rows.
   For example, as-reported VKTX SF1 rows from 2015 were re-stamped 2026-10-29. To build an as-of
   view, replay `raw/` slices up to the date using `manifest.jsonl` `fetched_at_utc`. That timestamp
   is each row's data_available_timestamp in this cache.
3. **SF1 `date` is the SEC filing date (datekey).** For as-of work, use the `ARQ`/`ARY`/`ART`
   dimensions joined on `date`. `MRQ`/`MRY`/`MRT` include restatements and are keyed to `reportperiod`
   (about two months of look-ahead).
4. **Bad or missing keys still work for DJIA sample names** (e.g. AMGN). Test a key with a
   non-sample ticker such as VKTX: an invalid key returns `403 Exceeds free tier`.
5. **DAILY `marketcap`/`ev` are in $ millions.**
6. ACTIONS and SF3A have no `lastupdated` column, so they re-pull a trailing date window.
