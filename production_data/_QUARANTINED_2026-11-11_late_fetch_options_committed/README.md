# Quarantined: late-fetched options snapshots that were already committed (2026-11-11)

**Do not use as point-in-time data.**

Each file's embedded `metadata.fetch_timestamp` is after the as-of date in its filename. tastytrade market metrics are current-state only, so the contents reflect the fetch time, not the labelled date. Reading them as of their label would violate the CCFT rule `data_available_timestamp <= as_of_date`.

| File | Labelled as-of | Actually fetched (America/New_York) |
|---|---|---|
| `options_snapshot_2026-06-26.json` | 2026-06-26 | 2026-06-28 19:36 |
| `options_snapshot_2026-07-01.json` | 2026-07-01 | 2026-07-02 08:23 |
| `options_snapshot_2026-07-03.json` | 2026-07-03 | 2026-07-05 15:34 |
| `options_snapshot_2026-07-13.json` | 2026-07-13 | 2026-07-17 10:26 |
| `options_snapshot_2026-07-14.json` | 2026-07-14 | 2026-07-17 10:21 |
| `options_snapshot_2026-07-27.json` | 2026-07-27 | 2026-07-28 08:17 |
| `options_snapshot_2026-08-05.json` | 2026-08-05 | 2026-08-06 08:43 |

07-13 and 07-14 (fetched 07-17) are clear look-ahead. 07-01, 07-27 and 08-05 were fetched before the next session opened and are borderline, but are quarantined under the same strict rule.

Moved here with `git mv` so history is preserved. Companion quarantines (untracked, prod checkout only): `_QUARANTINED_2026-11-01_catchup_options/`, `_QUARANTINED_2026-11-11_late_fetch_options/`.

Root cause: `tools/collect_options_snapshot.py` accepted any `--date`. Fixed in the same PR with a guard that refuses any as-of date other than today in America/New_York. Research loaders that glob `production_data/options_snapshot_*.json` (top level only) no longer see these files.
