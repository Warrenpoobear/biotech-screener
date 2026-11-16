#!/usr/bin/env python3
"""Refresh universe coverage status for active biotech tickers.

This is a data-quality refresh: tickers that lack local market/company/clinical
coverage are marked pending_data_collection instead of remaining silently active.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERIC_COMPANY_NAMES = {"healthcare", "biotechnology", "biotech", "unknown"}


def _json_safe(obj):
    """Replace NaN/inf floats with None, recursively."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _dumps_strict(obj) -> str:
    return json.dumps(_json_safe(obj), indent=2, allow_nan=False) + "\n"


def _ticker(row: dict[str, Any]) -> str:
    return str(row.get("ticker") or "").strip().upper()


def _is_active_status(row: dict[str, Any]) -> bool:
    return str(row.get("status", "active") or "active").strip().lower() in {"", "active"}


def _is_refreshable_status(row: dict[str, Any]) -> bool:
    return str(row.get("status", "active") or "active").strip().lower() in {
        "",
        "active",
        "pending_data_collection",
        "pending_coverage",
    }


def _is_biotech(row: dict[str, Any]) -> bool:
    market_data = row.get("market_data") if isinstance(row.get("market_data"), dict) else {}
    text = " ".join(
        str(value or "")
        for value in (
            row.get("sector"),
            market_data.get("sector"),
            market_data.get("industry"),
        )
    ).lower()
    return "biotech" in text or "biotechnology" in text


def _real_company_name(row: dict[str, Any]) -> str | None:
    market_data = row.get("market_data") if isinstance(row.get("market_data"), dict) else {}
    ticker = _ticker(row).lower()
    for value in (
        row.get("company"),
        market_data.get("company_name"),
        row.get("company_name"),
        row.get("name"),
    ):
        if not value:
            continue
        name = str(value).strip()
        if not name:
            continue
        if name.lower() in GENERIC_COMPANY_NAMES or name.lower() == ticker:
            continue
        return name
    return None


def _market_data_status(row: dict[str, Any]) -> str:
    market_data = row.get("market_data") if isinstance(row.get("market_data"), dict) else {}
    has_price = market_data.get("price") not in (None, "")
    has_market_cap = market_data.get("market_cap") not in (None, "") or row.get("market_cap") not in (None, "")
    return "covered" if has_price and has_market_cap else "pending"


def _financial_status(row: dict[str, Any]) -> str:
    financial_data = row.get("financial_data") if isinstance(row.get("financial_data"), dict) else {}
    if financial_data:
        return "covered"
    # Some legacy rows carry flattened financial fields.
    for key in ("Cash", "Assets", "Liabilities", "NetIncome", "Revenue"):
        if row.get(key) not in (None, ""):
            return "covered"
    return "pending"


def _trial_ticker_sets(trial_records: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    trial_tickers = set()
    intervention_tickers = set()
    for record in trial_records:
        if not isinstance(record, dict):
            continue
        ticker = _ticker(record)
        if not ticker:
            continue
        trial_tickers.add(ticker)
        if record.get("interventions"):
            intervention_tickers.add(ticker)
    return trial_tickers, intervention_tickers


def _coverage_status(row: dict[str, Any], trial_tickers: set[str], intervention_tickers: set[str]) -> dict[str, str]:
    ticker = _ticker(row)
    return {
        "company_name": "covered" if _real_company_name(row) else "pending",
        "market_data": _market_data_status(row),
        "financials": _financial_status(row),
        "clinical_trials": "covered" if ticker in trial_tickers else "pending",
        "scientific_cartography": "covered" if ticker in intervention_tickers else "pending",
    }


def refresh_universe(
    universe: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
    as_of_date: str,
    *,
    finalize_collection: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return refreshed universe rows and a summary report."""
    trial_tickers, intervention_tickers = _trial_ticker_sets(trial_records)
    refreshed = deepcopy(universe)
    pending_collection_tickers = []
    pending_coverage_tickers = []
    promoted_tickers = []
    refreshable_biotech_count = 0

    for row in refreshed:
        ticker = _ticker(row)
        if not ticker or not _is_refreshable_status(row) or not _is_biotech(row):
            continue

        refreshable_biotech_count += 1
        coverage = _coverage_status(row, trial_tickers, intervention_tickers)
        pending_reasons = [
            name
            for name in ("company_name", "market_data", "clinical_trials", "scientific_cartography")
            if coverage[name] == "pending"
        ]

        if not pending_reasons:
            if str(row.get("status", "")).strip().lower() in {"pending_data_collection", "pending_coverage"}:
                promoted_tickers.append(ticker)
            row["status"] = "active"
            row.pop("status_reason", None)
            row["coverage_status"] = coverage
            row["coverage_refreshed_as_of"] = as_of_date
            continue

        collection_complete = (
            finalize_collection
            and set(pending_reasons).issubset({"clinical_trials", "scientific_cartography"})
            and coverage["company_name"] == "covered"
            and coverage["market_data"] == "covered"
        )
        if collection_complete:
            row["status"] = "pending_coverage"
            row["status_reason"] = "coverage_unavailable:" + ",".join(pending_reasons)
            for reason in pending_reasons:
                coverage[reason] = "unavailable"
            pending_coverage_tickers.append(ticker)
        else:
            row["status"] = "pending_data_collection"
            row["status_reason"] = "coverage_pending:" + ",".join(pending_reasons)
            pending_collection_tickers.append(ticker)
        row["coverage_status"] = coverage
        row["coverage_refreshed_as_of"] = as_of_date

    report = {
        "as_of_date": as_of_date,
        "refreshable_biotech_count": refreshable_biotech_count,
        "marked_pending_count": len(pending_collection_tickers) + len(pending_coverage_tickers),
        "marked_pending_tickers": sorted(pending_collection_tickers + pending_coverage_tickers),
        "pending_collection_count": len(pending_collection_tickers),
        "pending_collection_tickers": sorted(pending_collection_tickers),
        "pending_coverage_count": len(pending_coverage_tickers),
        "pending_coverage_tickers": sorted(pending_coverage_tickers),
        "promoted_active_count": len(promoted_tickers),
        "promoted_active_tickers": sorted(promoted_tickers),
    }
    return refreshed, report


def status_changes(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Tickers whose ``status`` differs between two universe lists, in ticker order."""
    old = {_ticker(row): str(row.get("status", "active") or "active") for row in before if _ticker(row)}
    changes = []
    for row in after:
        ticker = _ticker(row)
        if not ticker or ticker not in old:
            continue
        new_status = str(row.get("status", "active") or "active")
        if new_status != old[ticker]:
            changes.append(
                {
                    "ticker": ticker,
                    "prior_status": old[ticker],
                    "new_status": new_status,
                    "status_reason": str(row.get("status_reason") or ""),
                }
            )
    return sorted(changes, key=lambda change: change["ticker"])


def audit_lines(changes: list[dict[str, str]], as_of_date: str, operator: str, timestamp: str) -> list[str]:
    """One ``universe_audit_log.jsonl`` line per status change (same shape as apply_ticker_change.py)."""
    lines = []
    for change in changes:
        reason = f"coverage refresh as of {as_of_date}"
        if change["status_reason"]:
            reason += f": {change['status_reason']}"
        lines.append(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "action": "coverage_refresh",
                    "ticker": change["ticker"],
                    "prior_status": change["prior_status"],
                    "new_status": change["new_status"],
                    "reason": reason,
                    "operator": operator,
                }
            )
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--universe-path", type=Path, default=REPO_ROOT / "production_data" / "universe.json")
    parser.add_argument("--trial-records-path", type=Path, default=REPO_ROOT / "production_data" / "trial_records.json")
    parser.add_argument(
        "--finalize-collection",
        action="store_true",
        help="Mark post-fetch clinical/cartography gaps as pending_coverage instead of pending collection.",
    )
    parser.add_argument("--apply", action="store_true", help="Write refreshed universe.json. Defaults to dry-run.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional path for refresh report JSON.")
    parser.add_argument(
        "--audit-log-path",
        type=Path,
        default=None,
        help="Audit log appended on --apply (default: universe_audit_log.jsonl beside the universe file).",
    )
    parser.add_argument("--operator", default="operator", help="Name recorded in the audit log.")
    args = parser.parse_args()

    universe = json.loads(args.universe_path.read_text(encoding="utf-8"))
    trial_records = json.loads(args.trial_records_path.read_text(encoding="utf-8"))
    refreshed, report = refresh_universe(
        universe,
        trial_records,
        args.as_of_date,
        finalize_collection=args.finalize_collection,
    )

    changes = status_changes(universe, refreshed)
    report["status_changes"] = changes
    report["dry_run"] = not args.apply
    if args.apply:
        # NaN-safe: run_screen loads universe.json with orjson, which rejects NaN.
        args.universe_path.write_text(_dumps_strict(refreshed), encoding="utf-8")
        if changes:
            audit_path = args.audit_log_path or args.universe_path.with_name("universe_audit_log.jsonl")
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            existing = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            new_lines = audit_lines(changes, args.as_of_date, args.operator, stamp)
            audit_path.write_text(existing + "\n".join(new_lines) + "\n", encoding="utf-8")
            report["audit_log_entries"] = len(new_lines)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
