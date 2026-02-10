#!/usr/bin/env python3
"""Audit catalyst data coverage across historical archives.

Extracts each archive, inspects decision_inputs.json + trial_records.json +
rankings.csv to measure catalyst coverage at multiple trial-PCD window sizes.
Produces JSON + Markdown reports showing which archives pass/fail the
calibration DQ gate.

CLI:
  python scripts/audit_archive_catalyst.py \
    --archive-dir data/archives \
    --output-dir output \
    [--dq-threshold 80]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_DQ_THRESHOLD = 80.0
HYPOTHETICAL_WINDOWS = [180, 270, 365]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date | None:
    """Parse YYYY-MM-DD string to date, or None."""
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _extract_archive(tar_path: Path) -> Tuple[Path, Path]:
    """Extract tar.gz to tempdir, return (tmp_dir, archive_root)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="audit_cat_"))
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    date_str = tar_path.name.replace(".tar.gz", "")
    archive_root = tmp_dir / date_str
    if not archive_root.exists():
        subdirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
        if subdirs:
            archive_root = subdirs[0]
    return tmp_dir, archive_root


def _load_json(path: Path) -> Any:
    """Load JSON file or return empty structure."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def _load_eligible_dev_tickers(csv_path: Path) -> Set[str]:
    """Load eligible drug_developer tickers from rankings.csv.

    Older pre-backfill archives may lack 'eligible' or 'archetype' columns;
    returns empty set in that case (archive is skipped gracefully).
    """
    tickers: Set[str] = set()
    if not csv_path.exists():
        return tickers
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "archetype" not in headers or "eligible" not in headers:
            return tickers
        for row in reader:
            if (row.get("archetype", "").strip() == "drug_developer"
                    and row.get("eligible", "").strip() == "1"):
                t = row.get("ticker", "").strip().upper()
                if t:
                    tickers.add(t)
    return tickers


def _tickers_with_catalyst_from_inputs(
    decision_inputs: Dict[str, Dict[str, Any]],
    eligible: Set[str],
) -> Set[str]:
    """Return eligible tickers that have catalyst data in decision_inputs.json."""
    present: Set[str] = set()
    for ticker in eligible:
        entry = decision_inputs.get(ticker, {})
        if "days_to_catalyst" in entry:
            present.add(ticker)
    return present


# ---------------------------------------------------------------------------
# Hypothetical coverage at varying trial windows
# ---------------------------------------------------------------------------

def compute_hypothetical_coverage(
    trials: List[Dict[str, Any]],
    pdufa: List[Dict[str, Any]],
    eligible_tickers: Set[str],
    as_of: date,
    windows: List[int] | None = None,
) -> Dict[int, int]:
    """Count how many eligible tickers would have catalyst at each window size.

    Replicates PIT logic from enrich_archive_inputs.compute_catalyst() with
    varying trial_window_days.  PDUFA dates have no window cap (any future
    PDUFA counts at all windows).
    """
    if windows is None:
        windows = list(HYPOTHETICAL_WINDOWS)

    # Pre-compute PDUFA tickers (any future date, no window cap)
    pdufa_tickers: Set[str] = set()
    for entry in pdufa:
        ticker = entry.get("ticker", "").upper()
        if ticker not in eligible_tickers:
            continue
        pdufa_date = _parse_date(entry.get("pdufa_date", ""))
        if pdufa_date and (pdufa_date - as_of).days > 0:
            pdufa_tickers.add(ticker)

    # Pre-compute trial coverage per ticker: smallest days_away
    # Only P2/P3, first_posted <= as_of, PCD in future
    trial_min_days: Dict[str, int] = {}
    for trial in trials:
        ticker = trial.get("ticker", "").upper()
        if ticker not in eligible_tickers:
            continue
        phase = str(trial.get("phase", "")).upper()
        if phase not in ("PHASE2", "PHASE3", "PHASE 2", "PHASE 3",
                         "PHASE2/PHASE3", "PHASE 2/PHASE 3"):
            continue
        fp = _parse_date(trial.get("first_posted"))
        pcd = _parse_date(trial.get("primary_completion_date"))
        if not fp or not pcd:
            continue
        if fp > as_of:
            continue
        days_away = (pcd - as_of).days
        if days_away <= 0:
            continue
        if ticker not in trial_min_days or days_away < trial_min_days[ticker]:
            trial_min_days[ticker] = days_away

    result: Dict[int, int] = {}
    for w in windows:
        covered = set(pdufa_tickers)  # PDUFA always counts
        for ticker, days in trial_min_days.items():
            if days <= w:
                covered.add(ticker)
        result[w] = len(covered & eligible_tickers)

    return result


# ---------------------------------------------------------------------------
# Single-archive audit
# ---------------------------------------------------------------------------

def audit_single_archive(
    tar_path: Path,
    dq_threshold: float = DEFAULT_DQ_THRESHOLD,
) -> Dict[str, Any]:
    """Audit catalyst coverage for one archive.

    Returns dict with coverage metrics + hypothetical window analysis.
    """
    date_str = tar_path.name.replace(".tar.gz", "")
    result: Dict[str, Any] = {"date": date_str, "status": "ok"}

    as_of = _parse_date(date_str)
    if as_of is None:
        result["status"] = "error"
        result["error"] = f"Cannot parse date from {tar_path.name}"
        return result

    tmp_dir = None
    try:
        tmp_dir, archive_root = _extract_archive(tar_path)
        inputs_dir = archive_root / "inputs"
        csv_path = archive_root / "rankings.csv"

        # Load eligible dev tickers
        eligible = _load_eligible_dev_tickers(csv_path)
        if not eligible:
            result["status"] = "skipped"
            result["reason"] = "no eligible dev tickers (pre-backfill archive?)"
            return result

        result["n_eligible"] = len(eligible)

        # Current coverage from decision_inputs.json
        di_path = inputs_dir / "decision_inputs.json" if inputs_dir.exists() else Path("/nonexistent")
        decision_inputs: Dict[str, Dict[str, Any]] = {}
        if di_path.exists():
            decision_inputs = _load_json(di_path)

        present = _tickers_with_catalyst_from_inputs(decision_inputs, eligible)
        n_present = len(present)
        n_missing = len(eligible) - n_present
        missing_pct = round(n_missing / len(eligible) * 100, 1) if eligible else 0.0

        result["catalyst_present"] = n_present
        result["catalyst_missing"] = n_missing
        result["catalyst_missing_pct"] = missing_pct
        result["passes_dq_gate"] = missing_pct <= dq_threshold

        # Source breakdown among present
        source_counts: Dict[str, int] = {}
        for ticker in present:
            src = decision_inputs.get(ticker, {}).get("catalyst_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        result["source_breakdown"] = source_counts

        # Hypothetical coverage at different windows
        trials = _load_json(inputs_dir / "trial_records.json") if inputs_dir.exists() else []
        pdufa = _load_json(inputs_dir / "pdufa_dates.json") if inputs_dir.exists() else []

        hyp = compute_hypothetical_coverage(trials, pdufa, eligible, as_of)
        hyp_detail: Dict[str, Any] = {}
        for w, count in sorted(hyp.items()):
            w_missing = len(eligible) - count
            w_missing_pct = round(w_missing / len(eligible) * 100, 1)
            hyp_detail[f"{w}d"] = {
                "present": count,
                "missing": w_missing,
                "missing_pct": w_missing_pct,
                "passes_dq": w_missing_pct <= dq_threshold,
            }
        result["hypothetical"] = hyp_detail

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# All-archive audit
# ---------------------------------------------------------------------------

def audit_all_archives(
    archive_dir: Path,
    dq_threshold: float = DEFAULT_DQ_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Audit all .tar.gz archives in directory."""
    archives = sorted(archive_dir.glob("*.tar.gz"))
    results: List[Dict[str, Any]] = []
    for tar_path in archives:
        r = audit_single_archive(tar_path, dq_threshold=dq_threshold)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_audit_report_md(results: List[Dict[str, Any]], dq_threshold: float = DEFAULT_DQ_THRESHOLD) -> str:
    """Markdown table: date | n_eligible | missing_180d | missing_270d | missing_365d | pass/fail."""
    lines: List[str] = []
    lines.append("# Archive Catalyst Coverage Audit")
    lines.append("")
    lines.append(f"DQ threshold: catalyst_missing_pct <= {dq_threshold}%")
    lines.append("")

    # Current coverage table
    lines.append("## Current Coverage (from decision_inputs.json)")
    lines.append("")
    lines.append("| Date | Eligible | Present | Missing | Missing% | DQ |")
    lines.append("|------|----------|---------|---------|----------|----|")

    for r in results:
        if r.get("status") not in ("ok",):
            lines.append(f"| {r['date']} | — | — | — | — | {r.get('status', '?')} |")
            continue
        dq = "PASS" if r.get("passes_dq_gate") else "FAIL"
        lines.append(
            f"| {r['date']} | {r['n_eligible']} | {r['catalyst_present']} "
            f"| {r['catalyst_missing']} | {r['catalyst_missing_pct']}% | {dq} |"
        )
    lines.append("")

    # Hypothetical window table
    lines.append("## Hypothetical Coverage by Trial Window")
    lines.append("")
    window_cols = []
    for r in results:
        if r.get("status") == "ok" and "hypothetical" in r:
            window_cols = sorted(r["hypothetical"].keys())
            break

    if window_cols:
        header = "| Date | Eligible | " + " | ".join(f"Missing {w}" for w in window_cols) + " |"
        sep = "|------|----------| " + " | ".join("---" for _ in window_cols) + " |"
        lines.append(header)
        lines.append(sep)

        for r in results:
            if r.get("status") != "ok":
                continue
            hyp = r.get("hypothetical", {})
            cells = []
            for w in window_cols:
                wd = hyp.get(w, {})
                pct = wd.get("missing_pct", "—")
                dq_tag = "PASS" if wd.get("passes_dq") else "FAIL"
                cells.append(f"{pct}% ({dq_tag})")
            lines.append(f"| {r['date']} | {r['n_eligible']} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def format_audit_json(results: List[Dict[str, Any]], dq_threshold: float = DEFAULT_DQ_THRESHOLD) -> Dict[str, Any]:
    """JSON-serializable audit summary."""
    ok_results = [r for r in results if r.get("status") == "ok"]
    n_pass = sum(1 for r in ok_results if r.get("passes_dq_gate"))
    n_fail = sum(1 for r in ok_results if not r.get("passes_dq_gate"))

    return {
        "dq_threshold": dq_threshold,
        "n_archives": len(results),
        "n_ok": len(ok_results),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "current_gate": {"pass": n_pass, "fail": n_fail},
        "archives": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit catalyst data coverage across historical archives."
    )
    parser.add_argument(
        "--archive-dir", type=str, default=str(DEFAULT_ARCHIVE_DIR),
        help=f"Directory containing .tar.gz archives (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dq-threshold", type=float, default=DEFAULT_DQ_THRESHOLD,
        help=f"DQ gate: max catalyst_missing_pct (default: {DEFAULT_DQ_THRESHOLD})",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"ERROR: Archive directory not found: {archive_dir}", file=sys.stderr)
        return 1

    archives = sorted(archive_dir.glob("*.tar.gz"))
    print(f"Auditing catalyst coverage for {len(archives)} archives")
    print(f"  DQ threshold: {args.dq_threshold}%")
    print()

    results: List[Dict[str, Any]] = []
    for tar_path in archives:
        print(f"  {tar_path.name} ...", end=" ", flush=True)
        r = audit_single_archive(tar_path, dq_threshold=args.dq_threshold)
        results.append(r)

        if r["status"] == "ok":
            dq = "PASS" if r["passes_dq_gate"] else "FAIL"
            print(f"{dq} (missing={r['catalyst_missing_pct']}%, "
                  f"{r['catalyst_present']}/{r['n_eligible']} present)")
        else:
            print(f"{r['status'].upper()}: {r.get('error', r.get('reason', ''))}")

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_data = format_audit_json(results, dq_threshold=args.dq_threshold)
    json_path = out_dir / "archive_catalyst_audit.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
        f.write("\n")

    md_text = format_audit_report_md(results, dq_threshold=args.dq_threshold)
    md_path = out_dir / "archive_catalyst_audit.md"
    with open(md_path, "w") as f:
        f.write(md_text)

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    n_pass = sum(1 for r in ok_results if r.get("passes_dq_gate"))
    n_fail = len(ok_results) - n_pass
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    print()
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} skipped")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
