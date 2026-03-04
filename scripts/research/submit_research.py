#!/usr/bin/env python3
"""
Research Submission Workflow — scripts/research/submit_research.py

Thin wrapper around run_audited_backtest.py that:
1. Auto-discovers (or accepts) the active baseline eval summary
2. Runs an audited backtest with standard production defaults
3. Prints the verdict and path to VERDICT.md
4. Exits with code: 0=PROMOTE, 1=ARCHIVE, 2=NEEDS_MORE

Usage:
    python3 scripts/research/submit_research.py \\
        --ruleset production_data/decision_rulesets/candidate.json \\
        --name "v1.9.0_coinvest_contra"

    # With explicit baseline (skip auto-discovery):
    python3 scripts/research/submit_research.py \\
        --ruleset my_candidate.json \\
        --baseline-dir output/audited_backtests/baseline_run/

See scripts/research/RESEARCH_WORKFLOW.md for the full workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.run_audited_backtest import run_audited_backtest

# ---------------------------------------------------------------------------
# Production defaults (matched to current active ruleset parameters)
# ---------------------------------------------------------------------------
_DEFAULT_HORIZONS = [84, 126]
_DEFAULT_TOP_K = 20
_DEFAULT_COST_BPS = 30.0
_DEFAULT_ANCHOR_MODE = "prev_trading_day"
_DEFAULT_BENCHMARK = "XBI"
_DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
_DEFAULT_PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"
_DEFAULT_OUT_ROOT = PROJECT_ROOT / "output" / "audited_backtests"
_MANIFEST_PATH = PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
_AUDITED_ROOT = PROJECT_ROOT / "output" / "audited_backtests"

# Verdict → exit code mapping
_VERDICT_EXIT = {
    "PROMOTE": 0,
    "ARCHIVE": 1,
    "NEEDS_MORE": 2,
}


# ---------------------------------------------------------------------------
# Baseline auto-discovery
# ---------------------------------------------------------------------------

def _load_manifest() -> List[Dict]:
    """Load ruleset manifest, return list of rulesets."""
    if not _MANIFEST_PATH.is_file():
        return []
    try:
        data = json.loads(_MANIFEST_PATH.read_text())
        return data.get("rulesets", [])
    except Exception:
        return []


def _find_active_ruleset_id() -> Optional[str]:
    """Return the active ruleset ID from the manifest."""
    rulesets = _load_manifest()
    for rs in rulesets:
        if rs.get("status") == "active":
            return rs["id"]
    return None


def _find_baseline_summary(baseline_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the baseline eval summary.json.

    If baseline_dir is provided, look there directly.
    Otherwise: find active ruleset ID → search output/audited_backtests/ for
    a VERDICT.json with matching ruleset_id → use its sibling eval/summary.json.

    Returns None if no baseline can be found.
    """
    if baseline_dir is not None:
        # Accept either a run dir or the eval/ subdirectory
        candidate = baseline_dir / "eval" / "summary.json"
        if candidate.is_file():
            return candidate
        candidate2 = baseline_dir / "summary.json"
        if candidate2.is_file():
            return candidate2
        return None

    # Auto-discover via active ruleset
    active_id = _find_active_ruleset_id()
    if not active_id:
        return None

    if not _AUDITED_ROOT.is_dir():
        return None

    # Walk all run dirs, find latest VERDICT.json with matching ruleset_id
    matches = []
    for run_dir in _AUDITED_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        verdict_json = run_dir / "VERDICT.json"
        if not verdict_json.is_file():
            verdict_json = run_dir / "eval" / "VERDICT.json"
        if verdict_json.is_file():
            try:
                vdata = json.loads(verdict_json.read_text())
                if vdata.get("ruleset_id") == active_id:
                    summary = run_dir / "eval" / "summary.json"
                    if summary.is_file():
                        matches.append((verdict_json.stat().st_mtime, summary))
            except Exception:
                pass

    if not matches:
        return None
    # Return the most recently modified match
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def submit(
    *,
    ruleset_path: Path,
    name: str,
    baseline_dir: Optional[Path] = None,
    snapshot_root: Optional[Path] = None,
    price_csv: Optional[Path] = None,
    out_root: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """Submit a research ruleset for evaluation.

    Returns the exit code: 0=PROMOTE, 1=ARCHIVE, 2=NEEDS_MORE.
    """
    snap_root = snapshot_root or _DEFAULT_SNAPSHOT_ROOT
    pcsv = price_csv or _DEFAULT_PRICE_CSV
    out = out_root or _DEFAULT_OUT_ROOT

    baseline_summary = _find_baseline_summary(baseline_dir)
    if baseline_summary is None:
        print(
            "[submit_research] WARNING: No baseline summary found. "
            "Verdict will be NEEDS_MORE (cannot compare without baseline).",
            file=sys.stderr,
        )
        if baseline_dir is not None:
            print(
                f"  Tried: {baseline_dir / 'eval' / 'summary.json'}",
                file=sys.stderr,
            )
    else:
        print(f"[submit_research] Using baseline: {baseline_summary}")

    run_id = name.replace(" ", "_").lower()
    print(f"[submit_research] Submitting: {run_id}")
    print(f"  Ruleset    : {ruleset_path}")
    print(f"  Snapshot   : {snap_root}")
    print(f"  Horizons   : {_DEFAULT_HORIZONS}")
    print(f"  top_k={_DEFAULT_TOP_K}, cost_bps={_DEFAULT_COST_BPS}")
    print(f"  Output dir : {out / run_id}")

    if dry_run:
        print("[submit_research] DRY RUN — not executing backtest.")
        return _VERDICT_EXIT["NEEDS_MORE"]

    rc = run_audited_backtest(
        snapshot_root=snap_root,
        price_csv=pcsv,
        horizons=_DEFAULT_HORIZONS,
        out_root=out,
        ruleset_path=ruleset_path,
        baseline_summary_path=baseline_summary,
        verdict_name=run_id,
        top_k=_DEFAULT_TOP_K,
        cost_bps=_DEFAULT_COST_BPS,
        anchor_mode=_DEFAULT_ANCHOR_MODE,
        benchmark=_DEFAULT_BENCHMARK,
        rerank=True,
        preflight_strict=True,
        relaxed=False,
        run_id=run_id,
    )

    if rc != 0:
        print(f"[submit_research] Backtest runner returned non-zero: {rc}", file=sys.stderr)
        return _VERDICT_EXIT["NEEDS_MORE"]

    # Load and display verdict
    verdict_json = out / run_id / "VERDICT.json"
    verdict_md = out / run_id / "VERDICT.md"

    if not verdict_json.is_file():
        # Some run dirs nest results under a timestamped sub-run-id
        verdict_json = next((out / run_id).glob("*/VERDICT.json"), None)
        verdict_md = next((out / run_id).glob("*/VERDICT.md"), None)

    verdict_str = "NEEDS_MORE"
    if verdict_json and verdict_json.is_file():
        try:
            vdata = json.loads(verdict_json.read_text())
            verdict_str = vdata.get("verdict", "NEEDS_MORE")
            reasons = vdata.get("verdict_reasons", [])
            print(f"\n=== VERDICT: {verdict_str} ===")
            for r in reasons:
                print(f"  • {r}")
        except Exception as exc:
            print(f"[submit_research] Could not parse VERDICT.json: {exc}", file=sys.stderr)

    if verdict_md and verdict_md.is_file():
        print(f"\n→ Full verdict: {verdict_md}")

    return _VERDICT_EXIT.get(verdict_str, _VERDICT_EXIT["NEEDS_MORE"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a candidate ruleset for evaluation against the active baseline. "
            "See scripts/research/RESEARCH_WORKFLOW.md for the full workflow."
        )
    )
    parser.add_argument(
        "--ruleset", type=Path, required=True,
        help="Path to candidate DecisionRuleset JSON",
    )
    parser.add_argument(
        "--name", required=True,
        help="Short name for this research run (used as run_id)",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=None,
        help=(
            "Path to prior audited backtest run directory containing "
            "eval/summary.json. Auto-discovered from manifest if omitted."
        ),
    )
    parser.add_argument(
        "--snapshot-root", type=Path, default=None,
        help=f"Snapshot directory (default: {_DEFAULT_SNAPSHOT_ROOT})",
    )
    parser.add_argument(
        "--price-csv", type=Path, default=None,
        help="Price history CSV (default: production_data/price_history.csv)",
    )
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help=f"Output root (default: {_DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing",
    )
    args = parser.parse_args()

    exit_code = submit(
        ruleset_path=args.ruleset,
        name=args.name,
        baseline_dir=args.baseline_dir,
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        out_root=args.out_root,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
