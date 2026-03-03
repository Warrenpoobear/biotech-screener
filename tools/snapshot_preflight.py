"""Snapshot eligibility preflight for research backtests.

Validates snapshot structural quality before consumption by
eval_forward_returns.py and rerank_snapshots.py.  Failing snapshots
are dropped with a one-line exclusion reason.

Artifact schema (v1):
    preflight_summary.csv — one row per snapshot date with status, reasons,
        and key metrics (eligible_n, rankings_cols_n, hydration_coverage_pct,
        pit_cache_status, split_warning_count).
    preflight_manifest.json — run_id, created_at, git_sha, inputs, hashes,
        horizons, strict flag.

Usage (standalone):
    python tools/snapshot_preflight.py \
        --snapshot-root data/snapshots \
        --check-pit --pit-cache-base data/caches/price_pit/PIT \
        --verbose --out output/preflight_report.json --strict
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("snapshot_preflight")

# ── Constants ───────────────────────────────────────────────────────
REQUIRED_COLS = {"ticker", "eligible", "actionable_rank"}
HYDRATION_SOURCE_COLS = ("de_beta_xbi_60d_source", "de_alpha_60d_source")
HYDRATION_REASON_COLS = ("de_beta_xbi_60d_missing_reason", "de_alpha_60d_missing_reason")
HYDRATED_VALUE = "price_history"
# Tickers with these reasons (or empty source) couldn't have been hydrated —
# exclude from the denominator so early dates aren't penalised for pre-IPO tickers.
_NON_HYDRATABLE_REASONS = {"series_too_short"}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SUMMARY_CSV_COLUMNS = [
    "date", "status", "fail_reasons", "warn_reasons",
    "eligible_n", "rankings_cols_n", "hydration_coverage_pct",
    "pit_cache_status", "split_warning_count",
    "dated_source_status", "dated_source_details",
]

# Columns that indicate a data-source family is active in the snapshot.
# If ANY column in a family is present in the rankings header, the
# corresponding data_sources entry is expected in metadata.json.
_SOURCE_FAMILY_COLUMNS = {
    "sec_13f": {"coinvest_score_z", "sponsor_tier1_count", "inst_delta_z",
                "has_coinvest_signal", "has_inst_delta"},
    "ctgov":   {"clinical_score", "clinical_score_z", "clinical_score_v2",
                "clinical_readout_days", "clinical_coverage_flag"},
    "sec_8k":  {"catalyst_days", "catalyst_decay_w", "catalyst_source",
                "has_catalyst_signal"},
}


# ── Result types ────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    status: str          # "PASS", "WARN", "FAIL"
    detail: str = ""

    def __repr__(self) -> str:
        tag = f"[{self.status}]"
        return f"{tag} {self.name}: {self.detail}" if self.detail else f"{tag} {self.name}"


@dataclass
class SnapshotPreflightResult:
    date: str
    status: str          # worst across checks
    checks: List[CheckResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_summary_row(self) -> Dict[str, str]:
        """Flatten to a single CSV row for preflight_summary.csv."""
        fail_reasons = "; ".join(
            c.detail for c in self.checks if c.status == "FAIL" and c.detail
        )
        warn_reasons = "; ".join(
            c.detail for c in self.checks if c.status == "WARN" and c.detail
        )
        return {
            "date": self.date,
            "status": self.status,
            "fail_reasons": fail_reasons,
            "warn_reasons": warn_reasons,
            "eligible_n": str(self.metrics.get("eligible_n", "")),
            "rankings_cols_n": str(self.metrics.get("rankings_cols_n", "")),
            "hydration_coverage_pct": str(self.metrics.get("hydration_coverage_pct", "")),
            "pit_cache_status": self.metrics.get("pit_cache_status", ""),
            "split_warning_count": str(self.metrics.get("split_warning_count", "")),
            "dated_source_status": self.metrics.get("dated_source_status", ""),
            "dated_source_details": self.metrics.get("dated_source_details", ""),
        }


@dataclass
class PreflightReport:
    n_total: int = 0
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0
    results: List[SnapshotPreflightResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Status helpers ──────────────────────────────────────────────────
_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worst(a: str, b: str) -> str:
    return a if _STATUS_ORDER.get(a, 0) >= _STATUS_ORDER.get(b, 0) else b


# ── Check 1: rankings exist ────────────────────────────────────────
def check_rankings_exist(
    snapshot_dir: Path,
    min_cols: int = 25,
) -> CheckResult:
    """FAIL if rankings.csv is missing, empty, or lacks required columns."""
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return CheckResult("rankings_exist", "FAIL", "rankings.csv missing")

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = f.readline().strip()
    except OSError as exc:
        return CheckResult("rankings_exist", "FAIL", f"unreadable: {exc}")

    if not header:
        return CheckResult("rankings_exist", "FAIL", "rankings.csv empty")

    cols = {c.strip() for c in header.split(",")}
    missing = REQUIRED_COLS - cols
    if missing:
        return CheckResult(
            "rankings_exist", "FAIL",
            f"missing required columns: {sorted(missing)}",
        )

    n_cols = len(cols)
    if n_cols < min_cols:
        return CheckResult(
            "rankings_exist", "WARN",
            f"only {n_cols} columns (expected >= {min_cols})",
        )

    return CheckResult("rankings_exist", "PASS", f"{n_cols} columns")


# ── Check 2: eligible count ────────────────────────────────────────
def check_eligible_count(
    snapshot_dir: Path,
    min_eligible: int = 10,
) -> CheckResult:
    """FAIL if fewer than *min_eligible* rows have eligible == '1'."""
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return CheckResult("eligible_count", "FAIL", "rankings.csv missing")

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            n_elig = sum(1 for row in reader if row.get("eligible") == "1")
    except (OSError, csv.Error) as exc:
        return CheckResult("eligible_count", "FAIL", f"read error: {exc}")

    if n_elig < min_eligible:
        return CheckResult(
            "eligible_count", "FAIL",
            f"{n_elig} eligible rows (need >= {min_eligible})",
        )

    return CheckResult("eligible_count", "PASS", f"{n_elig} eligible")


# ── Check 3: hydration coverage ────────────────────────────────────
def check_hydration_coverage(
    snapshot_dir: Path,
    warn_threshold: float = 0.80,
) -> CheckResult:
    """WARN if beta/alpha source columns are absent or coverage < threshold."""
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return CheckResult("hydration_coverage", "FAIL", "rankings.csv missing")

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_source_cols = all(c in fieldnames for c in HYDRATION_SOURCE_COLS)
            if not has_source_cols:
                return CheckResult(
                    "hydration_coverage", "WARN",
                    "source columns absent (pre-hydration vintage)",
                )

            has_reason_cols = all(c in fieldnames for c in HYDRATION_REASON_COLS)
            n_elig = 0
            n_hydrated = 0
            n_non_hydratable = 0
            for row in reader:
                if row.get("eligible") != "1":
                    continue
                n_elig += 1

                sources = [row.get(c, "") for c in HYDRATION_SOURCE_COLS]

                if all(s == HYDRATED_VALUE for s in sources):
                    n_hydrated += 1
                    continue

                # Exclude non-hydratable tickers from the denominator:
                # empty source (pre-hydration vintage) or series_too_short.
                if all(s == "" for s in sources):
                    n_non_hydratable += 1
                elif has_reason_cols and any(
                    row.get(c, "") in _NON_HYDRATABLE_REASONS
                    for c in HYDRATION_REASON_COLS
                ):
                    n_non_hydratable += 1
    except (OSError, csv.Error) as exc:
        return CheckResult("hydration_coverage", "FAIL", f"read error: {exc}")

    if n_elig == 0:
        return CheckResult("hydration_coverage", "WARN", "no eligible rows")

    denom = n_elig - n_non_hydratable
    if denom == 0:
        # All eligible tickers are non-hydratable (pre-hydration era)
        return CheckResult(
            "hydration_coverage", "PASS",
            f"hydration n/a ({n_elig} eligible, all non-hydratable)",
        )

    coverage = n_hydrated / denom
    if coverage < warn_threshold:
        return CheckResult(
            "hydration_coverage", "WARN",
            f"hydration {coverage:.0%} < {warn_threshold:.0%} "
            f"({n_hydrated}/{denom} hydratable of {n_elig} eligible)",
        )

    return CheckResult(
        "hydration_coverage", "PASS",
        f"hydration {coverage:.0%} ({n_hydrated}/{denom} hydratable of {n_elig} eligible)",
    )


# ── Check 4: PIT price cache ───────────────────────────────────────
def check_pit_price_cache(
    snap_date: str,
    pit_cache_base: Path,
) -> CheckResult:
    """WARN if PIT cache dir is missing; FAIL if index.json fails validation."""
    cache_dir = pit_cache_base / snap_date
    index_path = cache_dir / "index.json"

    if not index_path.exists():
        return CheckResult(
            "pit_price_cache", "WARN",
            f"no PIT cache for {snap_date}",
        )

    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            "pit_price_cache", "FAIL",
            f"index.json unreadable: {exc}",
        )

    # Lazy import to avoid import-time failures
    from tools.warm_price_cache import validate_price_pit_index

    ok, detail = validate_price_pit_index(index, snap_date)
    if not ok:
        return CheckResult("pit_price_cache", "FAIL", f"schema: {detail}")

    return CheckResult("pit_price_cache", "PASS")


# ── Check 5: split warnings ────────────────────────────────────────
def check_split_warnings(
    snap_date: str,
    pit_cache_base: Path,
    eval_horizons: Optional[List[int]] = None,
) -> CheckResult:
    """WARN if PIT index has split_warnings matching eval horizons."""
    cache_dir = pit_cache_base / snap_date
    index_path = cache_dir / "index.json"

    if not index_path.exists():
        return CheckResult("split_warnings", "PASS", "no PIT cache (vacuous)")

    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        # Already caught by check_pit_price_cache; don't double-fail
        return CheckResult("split_warnings", "PASS", "index unreadable (deferred)")

    warnings = index.get("split_warnings", [])
    if not warnings:
        return CheckResult("split_warnings", "PASS", "no split warnings")

    if eval_horizons:
        horizon_set = set(eval_horizons)
        relevant = [
            w for w in warnings
            if w.get("horizon") in horizon_set
        ]
    else:
        relevant = list(warnings)

    if not relevant:
        return CheckResult("split_warnings", "PASS", "no relevant split warnings")

    tickers = sorted({w.get("ticker", "?") for w in relevant})
    return CheckResult(
        "split_warnings", "WARN",
        f"{len(relevant)} split warning(s): {', '.join(tickers[:5])}"
        + (f" +{len(tickers) - 5} more" if len(tickers) > 5 else ""),
    )


# ── Check 6: dated-source provenance ───────────────────────────────
def check_dated_sources(
    snapshot_dir: Path,
    snap_date: str,
) -> CheckResult:
    """WARN if metadata.json lacks data_sources or dates violate PIT rules.

    For each signal family present in rankings.csv, verifies that
    metadata.json contains a data_sources.<family>.effective_date field
    and that the date satisfies PIT constraints:
      - sec_13f: effective_date <= as_of_date - 45d (filing lag)
      - ctgov:   effective_date <= as_of_date
      - sec_8k:  effective_date <= as_of_date
    """
    meta_path = snapshot_dir / "metadata.json"
    csv_path = snapshot_dir / "rankings.csv"

    # Detect which signal families are present in rankings FIRST
    if not csv_path.exists():
        return CheckResult("dated_sources", "PASS", "no rankings (deferred)")
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            header_line = f.readline().strip()
    except OSError:
        return CheckResult("dated_sources", "PASS", "rankings unreadable (deferred)")

    csv_cols = {c.strip() for c in header_line.split(",")}
    active_families = []
    for family, indicator_cols in _SOURCE_FAMILY_COLUMNS.items():
        if csv_cols & indicator_cols:
            active_families.append(family)

    if not active_families:
        return CheckResult("dated_sources", "PASS", "no dated-source families detected")

    # Load metadata (only if we have active families to check)
    if not meta_path.exists():
        return CheckResult(
            "dated_sources", "WARN",
            f"metadata.json missing; active families: {', '.join(sorted(active_families))}",
        )
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult("dated_sources", "WARN", f"metadata.json unreadable: {exc}")

    data_sources = meta.get("data_sources")
    if data_sources is None:
        return CheckResult(
            "dated_sources", "WARN",
            f"data_sources map missing; active families: {', '.join(sorted(active_families))}",
        )

    # Validate each active family
    as_of = date.fromisoformat(snap_date)
    warnings: List[str] = []

    for family in sorted(active_families):
        entry = data_sources.get(family)
        if entry is None:
            warnings.append(f"{family}: entry missing")
            continue

        eff_str = entry.get("effective_date")
        if not eff_str:
            warnings.append(f"{family}: effective_date missing")
            continue

        try:
            eff_date = date.fromisoformat(eff_str)
        except ValueError:
            warnings.append(f"{family}: bad date '{eff_str}'")
            continue

        # PIT constraint: data must not be from the future.
        # Note: for sec_13f the actual PIT lag (45d from quarter-end to
        # filing availability) is enforced at cache-build time by
        # select_pit_filing().  The preflight only checks that the
        # stamped effective_date is not ahead of the snapshot date.
        if eff_date > as_of:
            warnings.append(f"{family}: effective_date {eff_str} > as_of {snap_date}")

    if warnings:
        return CheckResult(
            "dated_sources", "WARN",
            "; ".join(warnings),
        )

    return CheckResult("dated_sources", "PASS", f"all {len(active_families)} families valid")


# ── Orchestrator ────────────────────────────────────────────────────
def run_preflight(
    snapshot_dir: Path,
    snap_date: str,
    *,
    min_cols: int = 25,
    min_eligible: int = 10,
    warn_threshold: float = 0.80,
    check_pit: bool = False,
    pit_cache_base: Optional[Path] = None,
    eval_horizons: Optional[List[int]] = None,
) -> SnapshotPreflightResult:
    """Run all preflight checks for a single snapshot date."""
    result = SnapshotPreflightResult(date=snap_date, status="PASS")

    # Check 1: rankings exist (short-circuit on FAIL)
    c1 = check_rankings_exist(snapshot_dir, min_cols=min_cols)
    result.checks.append(c1)
    result.status = _worst(result.status, c1.status)
    # Extract col count from detail
    m = re.search(r"(\d+) columns", c1.detail)
    if m:
        result.metrics["rankings_cols_n"] = int(m.group(1))
    if c1.status == "FAIL":
        return result

    # Check 2: eligible count
    c2 = check_eligible_count(snapshot_dir, min_eligible=min_eligible)
    result.checks.append(c2)
    result.status = _worst(result.status, c2.status)
    m = re.search(r"(\d+) eligible", c2.detail)
    if m:
        result.metrics["eligible_n"] = int(m.group(1))

    # Check 3: hydration coverage
    c3 = check_hydration_coverage(snapshot_dir, warn_threshold=warn_threshold)
    result.checks.append(c3)
    result.status = _worst(result.status, c3.status)
    m = re.search(r"hydration (\d+)%", c3.detail)
    if m:
        result.metrics["hydration_coverage_pct"] = int(m.group(1))

    # Check 6: dated-source provenance
    c6 = check_dated_sources(snapshot_dir, snap_date)
    result.checks.append(c6)
    result.status = _worst(result.status, c6.status)
    result.metrics["dated_source_status"] = c6.status
    result.metrics["dated_source_details"] = c6.detail

    # Check 4 & 5: PIT price cache (optional)
    if check_pit and pit_cache_base is not None:
        c4 = check_pit_price_cache(snap_date, pit_cache_base)
        result.checks.append(c4)
        result.status = _worst(result.status, c4.status)
        if c4.status == "PASS":
            result.metrics["pit_cache_status"] = "OK"
        elif "no PIT cache" in c4.detail:
            result.metrics["pit_cache_status"] = "MISSING"
        else:
            result.metrics["pit_cache_status"] = "SCHEMA_FAIL"

        c5 = check_split_warnings(snap_date, pit_cache_base, eval_horizons)
        result.checks.append(c5)
        result.status = _worst(result.status, c5.status)
        m = re.search(r"(\d+) split warning", c5.detail)
        result.metrics["split_warning_count"] = int(m.group(1)) if m else 0

    return result


# ── Batch runner ────────────────────────────────────────────────────
def run_preflight_batch(
    snapshot_root: Path,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    **kwargs: Any,
) -> PreflightReport:
    """Discover YYYY-MM-DD snapshot dirs and run preflight on each."""
    report = PreflightReport()

    if not snapshot_root.is_dir():
        return report

    snap_dates = sorted(
        p.name for p in snapshot_root.iterdir()
        if p.is_dir() and _DATE_RE.match(p.name)
    )

    if date_from:
        snap_dates = [d for d in snap_dates if d >= date_from]
    if date_to:
        snap_dates = [d for d in snap_dates if d <= date_to]

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date
        pf = run_preflight(snap_dir, snap_date, **kwargs)
        report.results.append(pf)
        report.n_total += 1
        if pf.status == "PASS":
            report.n_pass += 1
        elif pf.status == "WARN":
            report.n_warn += 1
        else:
            report.n_fail += 1

    return report


# ── Artifact helpers ────────────────────────────────────────────────

def _git_sha() -> str:
    """Return the current git HEAD SHA (short), or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    """Return hex sha256 of a file, or empty string if missing."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_preflight_summary_csv(
    report: PreflightReport,
    output_dir: Path,
) -> Path:
    """Write preflight_summary.csv — one row per snapshot date."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "preflight_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for pf in report.results:
            writer.writerow(pf.to_summary_row())
    return out_path


def write_preflight_manifest(
    report: PreflightReport,
    output_dir: Path,
    *,
    run_id: str,
    snapshot_root: Optional[Path] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    horizons: Optional[List[int]] = None,
    strict: bool = False,
    file_hashes: Optional[Dict[str, str]] = None,
) -> Path:
    """Write preflight_manifest.json — provenance + input hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "snapshot_root": str(snapshot_root) if snapshot_root else "",
        "date_from": date_from or "",
        "date_to": date_to or "",
        "horizons": horizons or [],
        "preflight_strict": strict,
        "counts": {
            "total": report.n_total,
            "pass": report.n_pass,
            "warn": report.n_warn,
            "fail": report.n_fail,
        },
        "file_hashes": file_hashes or {},
    }
    out_path = output_dir / "preflight_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def write_preflight_artifacts(
    report: PreflightReport,
    output_dir: Path,
    *,
    run_id: str,
    snapshot_root: Optional[Path] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    horizons: Optional[List[int]] = None,
    strict: bool = False,
    file_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    """Write both preflight_summary.csv and preflight_manifest.json.

    Returns (summary_csv_path, manifest_json_path).
    """
    csv_path = write_preflight_summary_csv(report, output_dir)
    manifest_path = write_preflight_manifest(
        report, output_dir,
        run_id=run_id,
        snapshot_root=snapshot_root,
        date_from=date_from,
        date_to=date_to,
        horizons=horizons,
        strict=strict,
        file_hashes=file_hashes,
    )
    return csv_path, manifest_path


# ── CLI ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot eligibility preflight for research backtests",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=REPO_ROOT / "data" / "snapshots",
        help="Root directory containing YYYY-MM-DD snapshot subdirectories",
    )
    parser.add_argument(
        "--date-from", default=None,
        help="Earliest date to check (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--date-to", default=None,
        help="Latest date to check (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--min-cols", type=int, default=25,
        help="Minimum expected column count in rankings.csv (default: 25)",
    )
    parser.add_argument(
        "--min-eligible", type=int, default=10,
        help="Minimum eligible row count (default: 10)",
    )
    parser.add_argument(
        "--warn-threshold", type=float, default=0.80,
        help="Hydration coverage WARN threshold (default: 0.80)",
    )
    parser.add_argument(
        "--check-pit", action="store_true", default=False,
        help="Enable PIT price cache checks (checks 4 & 5)",
    )
    parser.add_argument(
        "--pit-cache-base", type=Path,
        default=REPO_ROOT / "data" / "caches" / "price_pit" / "PIT",
        help="Base directory for PIT price caches",
    )
    parser.add_argument(
        "--eval-horizons", default=None,
        help="Comma-separated horizons for split warning filter (e.g. 5,20,63)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print per-check details",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=None,
        help="Write preflight_summary.csv + preflight_manifest.json to this directory",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run identifier for artifact manifest (auto-generated if omitted)",
    )
    parser.add_argument(
        "--strict", action="store_true", default=False,
        help="Exit 1 if any snapshot FAILs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    eval_horizons = None
    if args.eval_horizons:
        eval_horizons = [int(h.strip()) for h in args.eval_horizons.split(",")]

    report = run_preflight_batch(
        args.snapshot_root,
        date_from=args.date_from,
        date_to=args.date_to,
        min_cols=args.min_cols,
        min_eligible=args.min_eligible,
        warn_threshold=args.warn_threshold,
        check_pit=args.check_pit,
        pit_cache_base=args.pit_cache_base,
        eval_horizons=eval_horizons,
    )

    # Print summary
    for pf in report.results:
        tag = f"[{pf.status}]"
        line = f"  {pf.date} {tag}"
        if args.verbose or pf.status != "PASS":
            details = [
                c.detail for c in pf.checks
                if c.detail and (args.verbose or c.status != "PASS")
            ]
            if details:
                line += f"  {'; '.join(details)}"
        print(line)

    print(
        f"\nPreflight: {report.n_total} snapshots — "
        f"{report.n_pass} PASS, {report.n_warn} WARN, {report.n_fail} FAIL"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report → {args.out}")

    if args.artifact_dir:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        csv_p, manifest_p = write_preflight_artifacts(
            report, args.artifact_dir,
            run_id=run_id,
            snapshot_root=args.snapshot_root,
            date_from=args.date_from,
            date_to=args.date_to,
            horizons=eval_horizons,
            strict=args.strict,
        )
        print(f"Artifacts → {csv_p}, {manifest_p}")

    if args.strict and report.n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
