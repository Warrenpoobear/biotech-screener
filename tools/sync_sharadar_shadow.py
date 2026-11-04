#!/usr/bin/env python3
"""Sharadar shadow cache sync.

Mirrors Sharadar tables for the screener universe into ``cache/sharadar/``:

    raw/<TABLE>/<stamp>_<mode>.jsonl.gz   immutable, as-received slices (PIT record)
    manifest.jsonl                        one line per slice: fetched_at_utc, filters, rows, sha256
    curated/<TABLE>.csv.gz                latest-known view, upserted by primary key
    state.json                            per-table tickers + incremental watermark
    status_latest.json                    last run summary

SHADOW ONLY. Nothing in the production scoring path reads this cache (enforced
by tests/test_sync_sharadar_shadow.py::TestShadowIsolation). Adopting any of it
as a production input is a price/feature-source change under the
NO_MODEL_CHANGE forward-validation window and needs operator authorization.

PIT notes
    * data_available_timestamp for every row = the slice's ``fetched_at_utc``.
    * The curated view is NOT point-in-time: Sharadar re-stamps ``lastupdated``
      on historical rows (e.g. as-reported SF1 rows from 2015 re-stamped
      2026-10-29). Reconstruct an as-of view from raw slices + manifest.
    * SF1 ``date`` is the SEC filing date (datekey). Use ARQ/ARY/ART dimensions
      joined on ``date`` for as-of work, never MRQ/``reportperiod``.

Usage:
    python3 tools/sync_sharadar_shadow.py                  # all tables
    python3 tools/sync_sharadar_shadow.py --tables SEP,SFP
    python3 tools/sync_sharadar_shadow.py --tickers VKTX,AMGN --cache-root /tmp/x
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.sharadar_client import SharadarAuthError, SharadarClient, SharadarError  # noqa: E402

DEFAULT_CACHE_ROOT = REPO_ROOT / "cache" / "sharadar"
DEFAULT_UNIVERSE = REPO_ROOT / "production_data" / "universe.json"
DEFAULT_BACKFILL_START = "2010-01-01"
BENCHMARK_TICKERS = ("XBI", "IBB", "SPY")


@dataclass(frozen=True)
class TableSpec:
    key: Tuple[str, ...]
    incremental_field: Optional[str]  # None => full snapshot every run
    overlap_days: int = 0
    ticker_source: str = "universe"  # "universe" | "benchmarks"


# Order matters: TICKERS first so the security master is fresh for the rest.
TABLE_SPECS: Dict[str, TableSpec] = {
    "TICKERS": TableSpec(key=("table", "permaticker"), incremental_field=None),
    "SEP": TableSpec(key=("ticker", "date"), incremental_field="lastupdated", overlap_days=5),
    "SFP": TableSpec(
        key=("ticker", "date"), incremental_field="lastupdated", overlap_days=5, ticker_source="benchmarks"
    ),
    "DAILY": TableSpec(key=("ticker", "date"), incremental_field="lastupdated", overlap_days=5),
    "SF1": TableSpec(
        key=("ticker", "dimension", "calendardate", "date"), incremental_field="lastupdated", overlap_days=5
    ),
    # ACTIONS / SF3A carry no lastupdated column: re-pull a trailing date window.
    "ACTIONS": TableSpec(key=("ticker", "date", "action", "contraticker"), incremental_field="date", overlap_days=45),
    # 13F summaries fill in over the 45-day filing window plus late amendments.
    "SF3A": TableSpec(key=("ticker", "date"), incremental_field="date", overlap_days=150),
}


@dataclass
class SyncResult:
    exit_code: int
    tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------- io


def _iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def _gzip_bytes(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:
        gz.write(payload)
    return buf.getvalue()


def load_universe_tickers(path: Path) -> List[str]:
    """Every ticker in the universe file, any status (no survivorship filter)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("tickers", data) if isinstance(data, dict) else data
    out = set()
    for e in entries:
        t = e.get("ticker") if isinstance(e, dict) else e
        if isinstance(t, str) and t.strip():
            out.add(t.strip().upper())
    return sorted(out)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- planning


def _plan(
    spec: TableSpec,
    tickers: Sequence[str],
    table_state: Optional[Dict[str, Any]],
    backfill_start: str,
    end_date: str,
) -> List[Tuple[str, List[str], Dict[str, str], Optional[Tuple[str, str]]]]:
    """[(mode, tickers, filters, date_range)] for one table.

    ``date_range`` is passed separately (not as ``date.gte`` filters) so the
    client can split a truncated request along it.
    """
    tickers = sorted(set(tickers))
    if spec.incremental_field is None:
        return [("snapshot", tickers, {}, None)]

    known = set((table_state or {}).get("tickers", []))
    watermark = (table_state or {}).get("watermark")
    if not watermark:
        known = set()
    old = [t for t in tickers if t in known]
    new = [t for t in tickers if t not in known]

    plan: List[Tuple[str, List[str], Dict[str, str], Optional[Tuple[str, str]]]] = []
    if old:
        since = (pd.Timestamp(watermark) - pd.Timedelta(days=spec.overlap_days)).strftime("%Y-%m-%d")
        if spec.incremental_field == "date":
            plan.append(("incremental", old, {}, (since, end_date)))
        else:
            plan.append(("incremental", old, {f"{spec.incremental_field}.gte": since}, (backfill_start, end_date)))
    if new:
        plan.append(("backfill", new, {}, (backfill_start, end_date)))
    return plan


# ------------------------------------------------------------------- writing


def _write_raw_slice(
    cache_root: Path, table: str, mode: str, now: datetime, rows: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    blob = _gzip_bytes(payload)
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = cache_root / "raw" / table
    path = raw_dir / f"{stamp}_{mode}.jsonl.gz"
    n = 1
    while path.exists():  # never overwrite an existing slice
        n += 1
        path = raw_dir / f"{stamp}_{mode}_{n}.jsonl.gz"
    _atomic_write_bytes(path, blob)
    return str(path.relative_to(cache_root)), hashlib.sha256(blob).hexdigest()


def _to_str_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = [None if (v is None or (isinstance(v, float) and v != v)) else str(v) for v in df[col]]
    return df


def _upsert_curated(cache_root: Path, table: str, spec: TableSpec, rows: List[Dict[str, Any]]) -> int:
    path = cache_root / "curated" / f"{table}.csv.gz"
    new = _to_str_frame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new

    key = list(spec.key)
    for k in key:
        if k not in combined.columns:
            combined[k] = None
    # Newest lastupdated wins; ties go to the most recent fetch (appended last).
    if "lastupdated" in combined.columns:
        combined = combined.sort_values("lastupdated", kind="mergesort", na_position="first")
    combined = combined.drop_duplicates(subset=key, keep="last")
    combined = combined.sort_values(key, kind="mergesort").reset_index(drop=True)
    combined = combined[key + [c for c in combined.columns if c not in key]]

    csv_text = combined.to_csv(index=False, lineterminator="\n")
    _atomic_write_bytes(path, _gzip_bytes(csv_text.encode("utf-8")))
    return len(combined)


def _new_watermark(spec: TableSpec, previous: Optional[str], rows: List[Dict[str, Any]]) -> Optional[str]:
    if spec.incremental_field is None:
        return None
    values = [str(r.get(spec.incremental_field))[:10] for r in rows if r.get(spec.incremental_field)]
    candidates = values + ([previous] if previous else [])
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------- sync


def run_sync(
    *,
    client: Any,
    cache_root: Path,
    tables: Iterable[str],
    universe_tickers: Sequence[str],
    now: datetime,
    backfill_start: str = DEFAULT_BACKFILL_START,
) -> SyncResult:
    cache_root = Path(cache_root)
    tables = [t.upper() for t in tables]
    unknown = [t for t in tables if t not in TABLE_SPECS]
    if unknown:
        raise ValueError(f"unknown Sharadar table(s): {', '.join(unknown)}")

    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = cache_root / "state.json"
    state: Dict[str, Any] = _load_json(state_path, {})
    result = SyncResult(exit_code=0)
    fetched_at = _iso(now)
    end_date = now.astimezone(timezone.utc).strftime("%Y-%m-%d")

    for table in tables:
        spec = TABLE_SPECS[table]
        tickers = list(BENCHMARK_TICKERS) if spec.ticker_source == "benchmarks" else list(universe_tickers)
        plan = _plan(spec, tickers, state.get(table), backfill_start, end_date)
        fetched: List[Tuple[str, List[str], Dict[str, Any], List[Dict[str, Any]]]] = []
        try:
            for mode, batch, filters, date_range in plan:
                rows = client.fetch(table, tickers=batch, filters=filters, date_range=date_range)
                recorded = dict(filters)
                if date_range is not None:
                    recorded["date.gte"], recorded["date.lte"] = date_range
                fetched.append((mode, batch, recorded, rows))
        except SharadarAuthError as e:
            result.tables[table] = {"status": "auth_error", "error": str(e)}
            result.exit_code = 2
            print(f"[sharadar] {table}: AUTH ERROR — aborting run: {e}", flush=True)
            break
        except (SharadarError, OSError) as e:
            result.tables[table] = {"status": "error", "error": str(e)}
            result.exit_code = max(result.exit_code, 1)
            print(f"[sharadar] {table}: ERROR {e}", flush=True)
            continue

        all_rows: List[Dict[str, Any]] = []
        with open(cache_root / "manifest.jsonl", "a", encoding="utf-8", newline="\n") as mf:
            for mode, batch, filters, rows in fetched:
                rel, sha = _write_raw_slice(cache_root, table, mode, now, rows)
                entry = {
                    "table": table,
                    "mode": mode,
                    "fetched_at_utc": fetched_at,
                    "filters": filters,
                    "tickers": batch,
                    "rows": len(rows),
                    "path": rel,
                    "sha256": sha,
                }
                mf.write(json.dumps(entry, sort_keys=True) + "\n")
                all_rows.extend(rows)

        curated_rows = _upsert_curated(cache_root, table, spec, all_rows) if all_rows else None
        prev = state.get(table, {})
        state[table] = {
            "tickers": sorted(set(prev.get("tickers", [])) | set(tickers)),
            "watermark": _new_watermark(spec, prev.get("watermark"), all_rows),
            "last_success_utc": fetched_at,
        }
        _atomic_write_json(state_path, state)
        result.tables[table] = {"status": "ok", "rows": len(all_rows), "requests": len(plan)}
        print(
            f"[sharadar] {table}: ok — {len(all_rows)} rows fetched"
            + (f", curated {curated_rows}" if curated_rows is not None else ""),
            flush=True,
        )

    _atomic_write_json(
        cache_root / "status_latest.json",
        {"run_at_utc": fetched_at, "exit_code": result.exit_code, "tables": result.tables},
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", default=",".join(TABLE_SPECS), help="comma list (default: all)")
    ap.add_argument("--tickers", default=None, help="comma list overriding the universe file")
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--backfill-start", default=DEFAULT_BACKFILL_START)
    args = ap.parse_args(argv)

    tickers = (
        sorted({t.strip().upper() for t in args.tickers.split(",") if t.strip()})
        if args.tickers
        else load_universe_tickers(args.universe)
    )
    try:
        client = SharadarClient()
    except SharadarAuthError as e:
        print(f"[sharadar] {e}", file=sys.stderr)
        return 2

    result = run_sync(
        client=client,
        cache_root=args.cache_root,
        tables=[t for t in args.tables.split(",") if t.strip()],
        universe_tickers=tickers,
        now=datetime.now(timezone.utc),
        backfill_start=args.backfill_start,
    )
    print(f"[sharadar] done — exit {result.exit_code}", flush=True)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
