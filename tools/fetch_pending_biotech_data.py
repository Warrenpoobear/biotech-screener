#!/usr/bin/env python3
"""Fetch missing market, financial, and CTGov data for pending biotech tickers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_ctgov_data import _fetch_all_trials, get_trials_for_ticker  # noqa: E402
from collect_financial_data import get_cik_from_ticker, get_company_facts  # noqa: E402
from tools.refresh_eligible_biotech_universe import refresh_universe  # noqa: E402


def _ticker(row: dict[str, Any]) -> str:
    return str(row.get("ticker") or "").strip().upper()


def _pending_tickers(universe: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            _ticker(row)
            for row in universe
            if _ticker(row) and row.get("status") in {"pending_data_collection", "pending_coverage"}
        }
    )


def _company_search_names(row: dict[str, Any]) -> list[str]:
    """Return sponsor search names from universe company metadata."""
    market_data = row.get("market_data") if isinstance(row.get("market_data"), dict) else {}
    candidates = [
        row.get("company_name"),
        row.get("company"),
        market_data.get("company_name"),
        row.get("name"),
    ]
    names = []
    seen = set()
    for value in candidates:
        if not value:
            continue
        name = str(value).strip()
        if not name or name.lower() in {"healthcare", "biotechnology", "biotech", "unknown"}:
            continue
        for variant in (name, _strip_company_suffix(name)):
            if not variant or variant.lower() in seen:
                continue
            seen.add(variant.lower())
            names.append(variant)
    return names


def _strip_company_suffix(name: str) -> str:
    """Strip common corporate suffixes without reducing to a generic first word."""
    import re

    current = name.strip()
    suffixes = [
        r",?\s+incorporated\.?$",
        r",?\s+inc\.?$",
        r",?\s+corporation$",
        r",?\s+corp\.?$",
        r",?\s+limited$",
        r",?\s+ltd\.?$",
        r",?\s+plc$",
        r",?\s+s\.?a\.?$",
        r",?\s+se$",
        r",?\s+ag$",
        r",?\s+n\.?v\.?$",
    ]
    for _ in range(4):
        for pattern in suffixes:
            stripped = re.sub(pattern, "", current, flags=re.IGNORECASE).strip(" ,.-")
            if stripped != current and stripped:
                current = stripped
                break
        else:
            return current
    return current


def _fetch_trials_for_universe_row(row: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    """Fetch CTGov trials via sponsor-specific company-name searches."""
    seen = set()
    trials = []
    for name in _company_search_names(row):
        for trial in _fetch_all_trials({"query.spons": name}, ticker):
            nct_id = trial.get("nct_id")
            if nct_id and nct_id not in seen:
                seen.add(nct_id)
                trials.append(trial)
    return trials


def _fetch_market_data(ticker: str, as_of_date: str) -> dict[str, Any] | None:
    """Fetch market/company metadata from yfinance."""
    import yfinance as yf

    as_of = date.fromisoformat(as_of_date)
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    hist = stock.history(start=as_of - timedelta(days=120), end=as_of + timedelta(days=1))

    if hist is not None and not hist.empty:
        price = float(hist["Close"].iloc[-1])
        volume_avg_30d = int(hist["Volume"].tail(30).mean()) if "Volume" in hist else None
        high_52 = float(hist["High"].max()) if "High" in hist else None
        low_52 = float(hist["Low"].min()) if "Low" in hist else None
    else:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        volume_avg_30d = info.get("averageVolume")
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")

    if (
        price in (None, "")
        and info.get("marketCap") in (None, "")
        and not (info.get("longName") or info.get("shortName"))
    ):
        return None

    return {
        "price": price,
        "market_cap": info.get("marketCap"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "volume_avg_30d": volume_avg_30d,
        "52_week_high": high_52,
        "52_week_low": low_52,
        "pe_ratio": info.get("trailingPE"),
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "collected_at": as_of_date,
        "source": "yfinance",
    }


def _fetch_financial_data(row: dict[str, Any], ticker: str, as_of_date: str) -> dict[str, Any] | None:
    """Fetch financial facts from SEC, with yfinance info fallback."""
    cik = row.get("cik") or get_cik_from_ticker(ticker)
    if cik:
        facts = get_company_facts(str(cik).zfill(10), ticker)
        if facts:
            facts["collected_at"] = as_of_date
            return facts

    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        record = {
            "ticker": ticker,
            "cik": cik,
            "cash": info.get("totalCash"),
            "debt": info.get("totalDebt"),
            "net_debt": (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0),
            "revenue_ttm": info.get("totalRevenue"),
            "assets": info.get("totalAssets"),
            "liabilities": info.get("totalLiabilities"),
            "equity": info.get("totalStockholderEquity"),
            "currency": info.get("currency", "USD"),
            "collected_at": as_of_date,
            "source": "yfinance_info",
        }
        return record if any(record.get(key) is not None for key in ("cash", "debt", "revenue_ttm", "assets")) else None
    except Exception:
        return None


def _merge_trials(
    existing: list[dict[str, Any]], fetched: list[dict[str, Any]], as_of_date: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge fetched trials by (ticker, nct_id), preserving existing order."""
    existing_keys = {
        (_ticker(record), record.get("nct_id")): record
        for record in existing
        if isinstance(record, dict) and _ticker(record) and record.get("nct_id")
    }
    merged = list(existing)
    added = 0
    already_present = 0
    for record in fetched:
        if not isinstance(record, dict) or not _ticker(record) or not record.get("nct_id"):
            continue
        merged_record = dict(record)
        merged_record["collected_at"] = as_of_date
        key = (_ticker(merged_record), merged_record["nct_id"])
        if key in existing_keys:
            already_present += 1
            continue
        existing_keys[key] = merged_record
        merged.append(merged_record)
        added += 1
    return merged, added, already_present


def _apply_market_data(row: dict[str, Any], market_data: dict[str, Any]) -> None:
    current = row.get("market_data") if isinstance(row.get("market_data"), dict) else {}
    current.update({key: value for key, value in market_data.items() if value not in (None, "")})
    row["market_data"] = current
    company_name = market_data.get("company_name")
    if company_name:
        row["name"] = company_name
        row["company_name"] = company_name
    if market_data.get("market_cap") not in (None, ""):
        row["market_cap"] = market_data["market_cap"]
    if market_data.get("sector"):
        row["sector"] = row.get("sector") or market_data["sector"]


def _apply_financial_data(row: dict[str, Any], financial_data: dict[str, Any]) -> None:
    row["financial_data"] = financial_data
    for src, dest in (
        ("Cash", "Cash"),
        ("Assets", "Assets"),
        ("Liabilities", "Liabilities"),
        ("NetIncome", "NetIncome"),
        ("Revenue", "Revenue"),
        ("cash", "Cash"),
        ("assets", "Assets"),
        ("liabilities", "Liabilities"),
        ("revenue_ttm", "Revenue"),
    ):
        if financial_data.get(src) is not None:
            row[dest] = financial_data[src]


def fetch_pending_data(
    universe: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
    as_of_date: str,
    *,
    market_fetcher: Callable[[str, str], dict[str, Any] | None] = _fetch_market_data,
    financial_fetcher: Callable[[dict[str, Any], str, str], dict[str, Any] | None] = _fetch_financial_data,
    trial_fetcher: Callable[[str], list[dict[str, Any]]] = get_trials_for_ticker,
    fallback_trial_fetcher: Callable[[dict[str, Any], str], list[dict[str, Any]]] = _fetch_trials_for_universe_row,
    sleep_seconds: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fetch and merge data for pending rows."""
    pending = _pending_tickers(universe)
    report: dict[str, Any] = {
        "as_of_date": as_of_date,
        "pending_tickers_before": pending,
        "market_success": [],
        "financial_success": [],
        "trial_success": [],
        "trial_records_added": 0,
        "trial_records_updated": 0,
        "errors": {},
    }
    fetched_trials = []

    rows_by_ticker = {_ticker(row): row for row in universe if _ticker(row)}
    for index, ticker in enumerate(pending, start=1):
        row = rows_by_ticker[ticker]
        try:
            market = market_fetcher(ticker, as_of_date)
            if market:
                _apply_market_data(row, market)
                report["market_success"].append(ticker)
        except Exception as exc:
            report["errors"].setdefault(ticker, []).append(f"market:{exc}")

        try:
            financial = financial_fetcher(row, ticker, as_of_date)
            if financial:
                _apply_financial_data(row, financial)
                report["financial_success"].append(ticker)
        except Exception as exc:
            report["errors"].setdefault(ticker, []).append(f"financial:{exc}")

        try:
            trials = trial_fetcher(ticker)
            if not trials:
                trials = fallback_trial_fetcher(row, ticker)
            if trials:
                fetched_trials.extend(trials)
                report["trial_success"].append(ticker)
        except Exception as exc:
            report["errors"].setdefault(ticker, []).append(f"trials:{exc}")

        if sleep_seconds and index < len(pending):
            time.sleep(sleep_seconds)

    merged_trials, added, updated = _merge_trials(trial_records, fetched_trials, as_of_date)
    report["trial_records_added"] = added
    report["trial_records_updated"] = updated

    refreshed_universe, refresh_report = refresh_universe(
        universe,
        merged_trials,
        as_of_date,
        finalize_collection=True,
    )
    report["refresh_report"] = refresh_report
    return refreshed_universe, merged_trials, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--universe-path", type=Path, default=REPO_ROOT / "production_data" / "universe.json")
    parser.add_argument("--trial-records-path", type=Path, default=REPO_ROOT / "production_data" / "trial_records.json")
    parser.add_argument(
        "--report-path", type=Path, default=REPO_ROOT / "artifacts" / "universe_refresh" / "pending_fetch_report.json"
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--apply", action="store_true", help="Write fetched universe/trial data. Defaults to dry-run.")
    args = parser.parse_args()

    universe = json.loads(args.universe_path.read_text(encoding="utf-8"))
    trial_records = json.loads(args.trial_records_path.read_text(encoding="utf-8"))
    refreshed_universe, merged_trials, report = fetch_pending_data(
        universe,
        trial_records,
        args.as_of_date,
        sleep_seconds=args.sleep_seconds,
    )

    report["dry_run"] = not args.apply
    if args.apply:
        args.universe_path.write_text(json.dumps(refreshed_universe, indent=2) + "\n", encoding="utf-8")
        args.trial_records_path.write_text(json.dumps(merged_trials, indent=2) + "\n", encoding="utf-8")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
