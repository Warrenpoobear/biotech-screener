#!/usr/bin/env python3
"""Apply a ticker change (same company, new symbol) across production_data.

A ticker change touches a dozen files, several of them untracked and rewritten by
cron (price history, market data). Doing it by hand has left half-renamed states
before. This tool relabels every file in one pass, writes each file atomically, and
records the change in corporate_actions.json and universe_audit_log.jsonl.

It is for a plain rename only: the old symbol must be in the universe and the new
symbol must not appear anywhere yet. Ticker REUSE (the new symbol previously belonged
to another company, as with INDV -> SUPN in 2026-11) needs manual handling and is
refused here.

Dry-run by default. Nothing is written without --apply.

Usage:
    python3 tools/apply_ticker_change.py --old ACHV --new QUIT --effective-date 2026-11-10 \\
        --reason "Achieve Life Sciences changed its symbol; Sharadar ACTIONS tickerchangeto 2026-11-10"
    python3 tools/apply_ticker_change.py --old ACHV --new QUIT --effective-date 2026-11-10 \\
        --reason "..." --operator "<name>" --apply

Classification: universe/data maintenance tooling. No model, ranker, selector, sizing
or eligibility logic. The company keeps its CIK, CUSIP, history and scores.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# JSON files whose records carry a "ticker" field.
TICKER_FIELD_FILES: Tuple[str, ...] = (
    "universe.json",
    "trial_records.json",
    "financial_records.json",
    "financial_data.json",
    "short_interest.json",
    "market_data.json",
    "quarterly_burn_history.json",
)
# JSON files keyed by ticker.
TICKER_KEY_FILES: Tuple[str, ...] = ("ipo_dates.json", "quarterly_burn_history.json")
PRICE_FILES: Tuple[str, ...] = ("price_history.csv", "price_history_split_adj.csv")
PIT_DIR = "pit_financials"
REGISTRY = "corporate_actions.json"
AUDIT_LOG = "universe_audit_log.jsonl"
PRODUCTION_LOCKS: Tuple[str, ...] = ("logs/.daily_production.lock", "logs/.daily_production_py.lock")

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


class TickerChangeError(RuntimeError):
    """The change cannot be applied safely."""


def _field_pattern(symbol: str) -> re.Pattern:
    return re.compile(r'("ticker"\s*:\s*)"' + re.escape(symbol) + r'"')


def _key_pattern(symbol: str) -> re.Pattern:
    return re.compile(r'"' + re.escape(symbol) + r'"(\s*:\s*[\{\[])')


def _relabel_json_text(text: str, old: str, new: str, *, fields: bool, keys: bool) -> Tuple[str, int]:
    """Replace the symbol as a "ticker" value and/or as an object key. Formatting is preserved."""
    n = 0
    if fields:
        text, k = _field_pattern(old).subn(lambda m: f'{m.group(1)}"{new}"', text)
        n += k
    if keys:
        text, k = _key_pattern(old).subn(lambda m: f'"{new}"{m.group(1)}', text)
        n += k
    return text, n


def _mentions(text: str, symbol: str) -> int:
    return len(_field_pattern(symbol).findall(text)) + len(_key_pattern(symbol).findall(text))


def _relabel_price_csv(text: str, old: str, new: str) -> Tuple[str, int, int]:
    """Relabel the ticker column. Returns (text, rows relabelled, rows already under `new`)."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return text, 0, 0
    if "ticker" not in header:
        raise TickerChangeError("price file has no 'ticker' column")
    col = header.index("ticker")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    changed = existing = 0
    for row in reader:
        if len(row) > col:
            if row[col] == new:
                existing += 1
            elif row[col] == old:
                row[col] = new
                changed += 1
        writer.writerow(row)
    return out.getvalue(), changed, existing


def _registry_entry(old: str, new: str, effective_date: str, reason: str) -> Dict[str, str]:
    return {
        "ticker": new,
        "action": "ticker_change",
        "effective_date": effective_date,
        "old_ticker": old,
        "new_ticker": new,
        "notes": reason,
    }


def _append_registry_entry(text: str, entry: Dict[str, str]) -> str:
    """Append an action to corporate_actions.json without reformatting the file."""
    data = json.loads(text)
    actions = data.get("actions")
    if not isinstance(actions, list) or not actions:
        raise TickerChangeError("corporate_actions.json has no 'actions' list")
    for a in actions:
        if (
            a.get("action") == "ticker_change"
            and a.get("old_ticker") == entry["old_ticker"]
            and a.get("new_ticker") == entry["new_ticker"]
        ):
            return text  # already recorded
    match = re.search(r"\}(\s*)\]\s*\}\s*$", text)
    if not match:
        raise TickerChangeError("could not find the end of the 'actions' list")
    block = json.dumps(entry, indent=2, ensure_ascii=False)
    indented = "\n".join("    " + line for line in block.split("\n"))
    insert_at = match.start() + 1
    return text[:insert_at] + ",\n" + indented + text[insert_at:]


def plan_ticker_change(
    production_dir: Path,
    old: str,
    new: str,
    effective_date: str,
    reason: str,
    operator: str = "operator",
    timestamp: Optional[str] = None,
) -> Tuple[Dict[Path, str], List[Tuple[Path, Path]], Dict[str, int]]:
    """Work out every change without writing. Returns (file contents, renames, counts)."""
    old, new = old.upper(), new.upper()
    if not _SYMBOL.match(old) or not _SYMBOL.match(new) or old == new:
        raise TickerChangeError(f"bad symbols: {old!r} -> {new!r}")
    datetime.strptime(effective_date, "%Y-%m-%d")
    base = Path(production_dir)
    universe_path = base / "universe.json"
    if not universe_path.exists():
        raise TickerChangeError(f"{universe_path} not found")
    universe_text = universe_path.read_text(encoding="utf-8")
    if not _field_pattern(old).search(universe_text):
        raise TickerChangeError(f"{old} is not in universe.json")

    writes: Dict[Path, str] = {}
    counts: Dict[str, int] = {}
    reuse: List[str] = []

    for name in sorted(set(TICKER_FIELD_FILES) | set(TICKER_KEY_FILES)):
        path = base / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if _mentions(text, new):
            reuse.append(name)
            continue
        new_text, n = _relabel_json_text(
            text, old, new, fields=name in TICKER_FIELD_FILES, keys=name in TICKER_KEY_FILES
        )
        if n:
            writes[path] = new_text
            counts[name] = n

    for name in PRICE_FILES:
        path = base / name
        if not path.exists():
            continue
        new_text, changed, existing = _relabel_price_csv(path.read_text(encoding="utf-8"), old, new)
        if existing:
            reuse.append(name)
        elif changed:
            writes[path] = new_text
            counts[name] = changed

    renames: List[Tuple[Path, Path]] = []
    pit_old, pit_new = base / PIT_DIR / f"{old}.json", base / PIT_DIR / f"{new}.json"
    if pit_new.exists():
        reuse.append(f"{PIT_DIR}/{new}.json")
    elif pit_old.exists():
        text, n = _relabel_json_text(pit_old.read_text(encoding="utf-8"), old, new, fields=True, keys=False)
        writes[pit_new] = text
        renames.append((pit_old, pit_new))
        counts[f"{PIT_DIR}/{old}.json"] = max(n, 1)

    if reuse:
        raise TickerChangeError(
            f"{new} already appears in: {', '.join(sorted(reuse))}. That is ticker reuse or a half-applied "
            "change; resolve it by hand."
        )

    registry_path = base / REGISTRY
    if registry_path.exists():
        text = registry_path.read_text(encoding="utf-8")
        new_text = _append_registry_entry(text, _registry_entry(old, new, effective_date, reason))
        if new_text != text:
            writes[registry_path] = new_text
            counts[REGISTRY] = 1

    audit_path = base / AUDIT_LOG
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = json.dumps(
        {
            "timestamp": stamp,
            "action": "ticker_change",
            "ticker": new,
            "old_ticker": old,
            "new_ticker": new,
            "prior_status": "active",
            "new_status": "active",
            "reason": f"{old} -> {new} effective {effective_date}: {reason}",
            "operator": operator,
        }
    )
    existing_log = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    if existing_log and not existing_log.endswith("\n"):
        existing_log += "\n"
    writes[audit_path] = existing_log + line + "\n"
    counts[AUDIT_LOG] = 1
    return writes, renames, counts


def apply_plan(writes: Dict[Path, str], renames: List[Tuple[Path, Path]]) -> None:
    """Write every file through a temp file, then remove renamed sources."""
    staged: List[Tuple[Path, Path]] = []
    for path, text in writes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".ticker_change.tmp")
        tmp.write_text(text, encoding="utf-8")
        staged.append((tmp, path))
    for tmp, path in staged:
        os.replace(tmp, path)
    for src, _dst in renames:
        if src.exists():
            src.unlink()


def production_lock_held(repo_root: Path) -> Optional[str]:
    for rel in PRODUCTION_LOCKS:
        if (repo_root / rel).exists():
            return rel
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a ticker change across production_data")
    parser.add_argument("--old", required=True, help="Current symbol in the universe")
    parser.add_argument("--new", required=True, help="New symbol")
    parser.add_argument("--effective-date", required=True, help="First trading day under the new symbol (YYYY-MM-DD)")
    parser.add_argument("--reason", required=True, help="What happened and the source that confirms it")
    parser.add_argument("--operator", default="operator", help="Name recorded in the audit log")
    parser.add_argument("--production-dir", type=Path, default=REPO_ROOT / "production_data")
    parser.add_argument("--apply", action="store_true", help="Write the changes. Default is a dry run.")
    args = parser.parse_args(argv)

    try:
        writes, renames, counts = plan_ticker_change(
            args.production_dir, args.old, args.new, args.effective_date, args.reason, args.operator
        )
    except TickerChangeError as exc:
        print(f"REFUSED: {exc}")
        return 2

    print(f"{args.old.upper()} -> {args.new.upper()} effective {args.effective_date}")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    if not args.apply:
        print("Dry run. Nothing written. Re-run with --apply.")
        return 0
    lock = production_lock_held(args.production_dir.resolve().parent)
    if lock:
        print(f"REFUSED: daily production lock present ({lock}). Run this when no production job is running.")
        return 3
    apply_plan(writes, renames)
    print("Applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
