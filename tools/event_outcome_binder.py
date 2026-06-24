"""Event-outcome binder for prediction-shadow ledgers.

Joins decision-time prediction rows (currently the clinical-transmission
shadow ledger) to resolved catalyst outcomes under
``data/snapshots/resolutions/``. Writes a sidecar JSONL of bound rows; never
mutates the source ledger.

The binder is read-only and idempotent. Each invocation rebuilds the sidecar
from scratch by re-reading both the ledger and the resolution index, so a
partial or interrupted run is safe to retry.

Match strategy (per changed_name row):

    1. exact   — same ticker + same catalyst date as the row's
                 ``expected_date`` field (when present).
    2. windowed — same ticker, catalyst date within ±7 days of
                 ``expected_date``. Catalyst dates routinely slip by a few
                 days between shadow capture and resolution.
    3. none    — no candidate; row is dropped from the sidecar.

Old shadow rows written before catalyst_id/expected_date were added to the
schema only have (ticker, event_type, days_to_event, as_of_date). The binder
reconstructs an effective expected_date as ``as_of_date + days_to_event``
for those rows.

Usage:
    python tools/event_outcome_binder.py
    python tools/event_outcome_binder.py --dry-run
    python tools/event_outcome_binder.py \\
        --ledger artifacts/clinical_transmission_shadow.jsonl \\
        --out artifacts/clinical_transmission_shadow_resolved.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LEDGER = REPO_ROOT / "artifacts" / "clinical_transmission_shadow.jsonl"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "clinical_transmission_shadow_resolved.jsonl"
RESOLUTIONS_DIR = REPO_ROOT / "data" / "snapshots" / "resolutions"

WINDOW_DAYS = 7
BINDER_VERSION = 1

# These are the outcome buckets we treat as "resolved enough to bind". The
# tracker also writes PENDING / NEEDS_REVIEW; PENDING is filtered out, but
# NEEDS_REVIEW is kept so downstream can see the binder caught it.
TERMINAL_OUTCOMES: frozenset[str] = frozenset({"HIT", "MISS", "DELAYED", "MIXED", "NEEDS_REVIEW"})

SKIP_RESOLUTION_FILES = frozenset(
    {
        "calibration_summary.json",
        "manual_overrides.json",
        "watchlist_current.json",
    }
)


@dataclass
class ResolutionRow:
    ticker: str
    catalyst_date: str  # ISO
    outcome: str
    resolution_date: Optional[str] = None
    outcome_detail: Optional[str] = None
    price_t_minus_1: Optional[float] = None
    price_t_plus_5: Optional[float] = None
    catalyst_type: Optional[str] = None
    source_path: str = ""

    @property
    def realized_return(self) -> Optional[float]:
        if self.price_t_minus_1 in (None, 0) or self.price_t_plus_5 is None:
            return None
        try:
            return (float(self.price_t_plus_5) - float(self.price_t_minus_1)) / float(self.price_t_minus_1)
        except (TypeError, ValueError, ZeroDivisionError):
            return None


@dataclass
class BindResult:
    n_ledger_entries: int = 0
    n_changed_names: int = 0
    n_bound: int = 0
    n_match_exact: int = 0
    n_match_windowed: int = 0
    n_unresolved: int = 0
    n_no_expected_date: int = 0
    bound_rows: List[Dict[str, Any]] = field(default_factory=list)


def load_resolution_index(resolutions_dir: Path = RESOLUTIONS_DIR) -> Dict[str, List[ResolutionRow]]:
    """Build ticker -> list[ResolutionRow], sorted by catalyst_date."""
    index: Dict[str, List[ResolutionRow]] = {}
    if not resolutions_dir.is_dir():
        return index
    for f in resolutions_dir.rglob("*.json"):
        if f.name in SKIP_RESOLUTION_FILES:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ticker = d.get("ticker")
        catalyst_date = d.get("catalyst_date")
        outcome = d.get("outcome")
        if not (ticker and catalyst_date and outcome):
            continue
        if outcome not in TERMINAL_OUTCOMES:
            continue
        try:
            source_path = str(f.relative_to(REPO_ROOT))
        except ValueError:
            source_path = str(f)
        row = ResolutionRow(
            ticker=ticker,
            catalyst_date=catalyst_date,
            outcome=outcome,
            resolution_date=d.get("resolution_date"),
            outcome_detail=d.get("outcome_detail"),
            price_t_minus_1=d.get("price_t_minus_1"),
            price_t_plus_5=d.get("price_t_plus_5"),
            catalyst_type=d.get("catalyst_type"),
            source_path=source_path,
        )
        index.setdefault(ticker, []).append(row)
    for rows in index.values():
        rows.sort(key=lambda r: r.catalyst_date)
    return index


def _safe_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def reconstruct_expected_date(changed_name: Dict[str, Any], as_of_date: Optional[str]) -> Optional[str]:
    """Best effort recovery of expected_date for ledger rows predating the schema add."""
    explicit = changed_name.get("expected_date")
    if explicit:
        return explicit
    days_to = changed_name.get("days_to_event")
    base = _safe_date(as_of_date)
    if days_to is None or base is None:
        return None
    try:
        return (base + timedelta(days=int(days_to))).isoformat()
    except (TypeError, ValueError):
        return None


def find_match(
    ticker: str,
    expected_date: Optional[str],
    index: Dict[str, List[ResolutionRow]],
    window_days: int = WINDOW_DAYS,
) -> Tuple[Optional[ResolutionRow], str, Optional[int]]:
    """Return (match, match_type, distance_days). match_type in {exact, windowed, none}."""
    rows = index.get(ticker)
    if not rows:
        return None, "none", None
    target = _safe_date(expected_date)
    if target is None:
        return None, "none", None

    best: Optional[ResolutionRow] = None
    best_distance: Optional[int] = None
    for row in rows:
        cd = _safe_date(row.catalyst_date)
        if cd is None:
            continue
        dist = abs((cd - target).days)
        if dist == 0:
            return row, "exact", 0
        if dist <= window_days and (best_distance is None or dist < best_distance):
            best = row
            best_distance = dist
    if best is not None and best_distance is not None:
        return best, "windowed", best_distance
    return None, "none", None


def iter_ledger(ledger_path: Path) -> Iterable[Dict[str, Any]]:
    if not ledger_path.is_file():
        return
    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def bind(
    ledger_path: Path = DEFAULT_LEDGER,
    resolutions_dir: Path = RESOLUTIONS_DIR,
    window_days: int = WINDOW_DAYS,
) -> BindResult:
    index = load_resolution_index(resolutions_dir)
    bound_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = BindResult()

    for entry in iter_ledger(ledger_path):
        result.n_ledger_entries += 1
        as_of = entry.get("as_of_date")
        for cn in entry.get("changed_names", []) or []:
            result.n_changed_names += 1
            ticker = cn.get("ticker")
            if not ticker:
                continue
            expected_date = reconstruct_expected_date(cn, as_of)
            if expected_date is None:
                result.n_no_expected_date += 1
                continue
            match, match_type, dist = find_match(ticker, expected_date, index, window_days)
            if match is None:
                result.n_unresolved += 1
                continue
            row = {
                "binder_version": BINDER_VERSION,
                "bound_at": bound_at,
                "shadow_as_of_date": as_of,
                "ticker": ticker,
                "event_type": cn.get("event_type"),
                "expected_date": expected_date,
                "catalyst_id": cn.get("catalyst_id"),
                "match_type": match_type,
                "match_distance_days": dist,
                "resolution": {
                    "outcome": match.outcome,
                    "resolution_date": match.resolution_date,
                    "outcome_detail": match.outcome_detail or None,
                    "realized_return": match.realized_return,
                    "catalyst_type": match.catalyst_type,
                    "source_path": match.source_path,
                },
            }
            result.bound_rows.append(row)
            result.n_bound += 1
            if match_type == "exact":
                result.n_match_exact += 1
            elif match_type == "windowed":
                result.n_match_windowed += 1
    return result


def write_sidecar(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), sort_keys=True))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="Source shadow ledger JSONL.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Sidecar JSONL to write.")
    parser.add_argument(
        "--resolutions-dir",
        default=str(RESOLUTIONS_DIR),
        help="Resolution records root (default: data/snapshots/resolutions).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=WINDOW_DAYS,
        help="Max ±days for windowed catalyst-date match (default: %(default)s).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write sidecar — print stats only.")
    args = parser.parse_args(argv)

    result = bind(
        ledger_path=Path(args.ledger),
        resolutions_dir=Path(args.resolutions_dir),
        window_days=args.window_days,
    )

    summary = {
        "ledger": args.ledger,
        "out": args.out if not args.dry_run else None,
        "n_ledger_entries": result.n_ledger_entries,
        "n_changed_names": result.n_changed_names,
        "n_bound": result.n_bound,
        "n_match_exact": result.n_match_exact,
        "n_match_windowed": result.n_match_windowed,
        "n_unresolved": result.n_unresolved,
        "n_no_expected_date": result.n_no_expected_date,
        "window_days": args.window_days,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    exec_id = None
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "event_outcome_binder",
            "Bind shadow ledger to CRT resolutions",
            inputs={"ledger": args.ledger, "window_days": args.window_days},
            outputs=summary,
            success=True,
        )
        if exec_id and result.n_changed_names > 0:
            bind_rate = result.n_bound / result.n_changed_names
            attach_outcome_verdict(
                exec_id,
                was_correct=bind_rate >= 0.5,
                evidence=f"bound {result.n_bound}/{result.n_changed_names} changed_names",
            )
    except Exception:
        pass

    if not args.dry_run:
        write_sidecar(Path(args.out), result.bound_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
