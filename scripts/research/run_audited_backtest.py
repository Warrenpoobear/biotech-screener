#!/usr/bin/env python3
"""Audited backtest runner — preflight + optional rerank + eval + AUDIT.md.

Single entrypoint that produces a reproducible audit package:
  1. Runs snapshot preflight batch (writes artifacts)
  2. Optionally re-ranks snapshots through a ruleset (with --preflight)
  3. Runs eval_forward_returns.py (with --preflight-strict by default)
  4. Writes output/<run_id>/AUDIT.md summarizing everything

Defaults to STRICT mode (preflight-strict=True).  For archive/OOS data where
all snapshots are WARN-status, add --relaxed to include WARN dates.  Use
--date-manifest to restrict evaluation to a curated date list.

Standard OOS usage (archive snapshots, curated dates):
    python scripts/research/run_audited_backtest.py \
        --snapshot-root data/snapshots_reranked_baseline_oos \
        --ruleset production_data/decision_rulesets/v1.8.3_buffer30_candidate.json \
        --date-manifest output/audited_sets/audited_dates_2020_2024_strict.txt \
        --horizons 84,126 --top-k 20 --cost-bps 30 \
        --rerank --relaxed \
        --out-root output/audited_backtests/my_run

Standard production usage (live snapshots, strict preflight):
    python scripts/research/run_audited_backtest.py \
        --snapshot-root data/snapshots \
        --ruleset production_data/decision_rulesets/v1.8.3_buffer30_candidate.json \
        --horizons 84,126 --top-k 20 --cost-bps 30 \
        --rerank \
        --out-root output/audited_backtests/my_run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.snapshot_preflight import (
    PreflightReport,
    _git_sha,
    _sha256_file,
    run_preflight_batch,
    write_preflight_artifacts,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _collect_file_hashes(
    *,
    ruleset_path: Optional[Path] = None,
    price_csv: Optional[Path] = None,
    snapshot_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Compute sha256 for key input files."""
    hashes: Dict[str, str] = {}
    if ruleset_path:
        hashes["ruleset_json"] = _sha256_file(ruleset_path)
    if price_csv:
        hashes["price_history_csv"] = _sha256_file(price_csv)
    # Universe JSON (if exists alongside snapshot root or at standard location)
    universe_path = PROJECT_ROOT / "production_data" / "universe.json"
    if universe_path.is_file():
        hashes["universe_json"] = _sha256_file(universe_path)
    # The preflight module itself
    preflight_py = PROJECT_ROOT / "tools" / "snapshot_preflight.py"
    if preflight_py.is_file():
        hashes["snapshot_preflight_py"] = _sha256_file(preflight_py)
    return hashes


def _run_rerank(
    snapshot_root: Path,
    out_root: Path,
    ruleset_path: Path,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    preflight: bool = True,
) -> int:
    """Run rerank_snapshots.py as a subprocess.  Returns exit code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "research" / "rerank_snapshots.py"),
        "--snapshot-root", str(snapshot_root),
        "--out-root", str(out_root),
        "--ruleset", str(ruleset_path),
    ]
    if date_from:
        cmd += ["--date-from", date_from]
    if date_to:
        cmd += ["--date-to", date_to]
    if preflight:
        cmd.append("--preflight")
    print(f"  rerank cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def _run_eval(
    snapshot_root: Path,
    price_csv: Path,
    out_dir: Path,
    *,
    horizons: List[int],
    top_k: int = 20,
    cost_bps: float = 30,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_manifest: Optional[Path] = None,
    anchor_mode: str = "prev_trading_day",
    benchmark: str = "XBI",
    preflight: bool = True,
    preflight_strict: bool = True,
    rebalance_buffer_ranks: int = 0,
    turnover_cap: float = 0.0,
    ruleset_path: Optional[Path] = None,
) -> int:
    """Run eval_forward_returns.py as a subprocess.  Returns exit code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_forward_returns.py"),
        "--snapshot-root", str(snapshot_root),
        "--price-csv", str(price_csv),
        "--horizons", ",".join(str(h) for h in horizons),
        "--top-k", str(top_k),
        "--cost-bps", str(cost_bps),
        "--anchor-mode", anchor_mode,
        "--benchmark", benchmark,
        "--out-dir", str(out_dir),
    ]
    if ruleset_path:
        cmd += ["--ruleset", str(ruleset_path)]
    if date_manifest:
        cmd += ["--date-manifest", str(date_manifest)]
    elif date_from:
        cmd += ["--date-from", date_from]
    if not date_manifest and date_to:
        cmd += ["--date-to", date_to]
    if preflight_strict:
        cmd.append("--preflight-strict")
    elif preflight:
        cmd.append("--preflight")
    if rebalance_buffer_ranks > 0:
        cmd += ["--rebalance-buffer-ranks", str(rebalance_buffer_ranks)]
    if turnover_cap > 0:
        cmd += ["--turnover-cap", str(turnover_cap)]
    print(f"  eval cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# ── AUDIT.md writer ─────────────────────────────────────────────────

def _write_audit_md(
    out_dir: Path,
    *,
    run_id: str,
    git_sha: str,
    file_hashes: Dict[str, str],
    report: PreflightReport,
    horizons: List[int],
    top_k: int,
    cost_bps: float,
    snapshot_root: Path,
    eval_snapshot_root: Path,
    price_csv: Path,
    ruleset_path: Optional[Path],
    date_from: Optional[str],
    date_to: Optional[str],
    date_manifest: Optional[Path],
    anchor_mode: str,
    benchmark: str,
    strict: bool,
    relaxed: bool,
    reranked: bool,
    preflight_dir: Path,
    eval_dir: Path,
) -> Path:
    """Write AUDIT.md summarizing the audited backtest run."""
    lines: List[str] = []
    lines.append(f"# Audited Backtest — {run_id}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Git SHA**: `{git_sha}`")
    lines.append(f"- **Created**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"- **Snapshot root**: `{snapshot_root}`")
    if reranked:
        lines.append(f"- **Eval snapshot root** (reranked): `{eval_snapshot_root}`")
    lines.append(f"- **Price CSV**: `{price_csv}`")
    if ruleset_path:
        lines.append(f"- **Ruleset**: `{ruleset_path}`")
    lines.append(f"- **Horizons**: {horizons}")
    lines.append(f"- **Top-K**: {top_k}")
    lines.append(f"- **Cost BPS**: {cost_bps}")
    lines.append(f"- **Anchor mode**: {anchor_mode}")
    lines.append(f"- **Benchmark**: {benchmark}")
    if date_manifest:
        lines.append(f"- **Date manifest**: `{date_manifest}`")
    lines.append(f"- **Date range**: {date_from or '(all)'} to {date_to or '(all)'}")
    mode_str = "relaxed (WARN included)" if relaxed else "strict (WARN excluded)"
    lines.append(f"- **Preflight mode**: {mode_str}")
    lines.append(f"- **Reranked**: {reranked}")
    lines.append("")

    # File hashes
    lines.append("## File Hashes (SHA-256)")
    lines.append("")
    lines.append("| File | SHA-256 |")
    lines.append("|------|---------|")
    for name, sha in sorted(file_hashes.items()):
        display = sha[:16] + "..." if sha else "(missing)"
        lines.append(f"| {name} | `{display}` |")
    lines.append("")

    # Preflight summary
    lines.append("## Preflight Summary")
    lines.append("")
    lines.append(f"- **Total snapshots**: {report.n_total}")
    lines.append(f"- **PASS**: {report.n_pass}")
    lines.append(f"- **WARN**: {report.n_warn}")
    lines.append(f"- **FAIL**: {report.n_fail}")
    n_included = report.n_pass + (0 if strict else report.n_warn)
    n_excluded = report.n_fail + (report.n_warn if strict else 0)
    lines.append(f"- **Included in eval**: {n_included}")
    lines.append(f"- **Excluded from eval**: {n_excluded}")
    lines.append("")

    # Exclusion reasons table
    fail_results = [r for r in report.results if r.status == "FAIL"]
    warn_results = [r for r in report.results if r.status == "WARN"]
    excluded = fail_results + (warn_results if strict else [])
    if excluded:
        # Aggregate reasons
        reason_counts: Dict[str, int] = {}
        for pf in excluded:
            for c in pf.checks:
                if c.status in ("FAIL", "WARN") and c.detail:
                    key = f"[{c.status}] {c.name}"
                    reason_counts[key] = reason_counts.get(key, 0) + 1
        lines.append("### Top Exclusion Reasons")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

        lines.append("### Excluded Dates")
        lines.append("")
        for pf in excluded[:20]:
            details = "; ".join(c.detail for c in pf.checks if c.status != "PASS" and c.detail)
            lines.append(f"- `{pf.date}` [{pf.status}]: {details}")
        if len(excluded) > 20:
            lines.append(f"- ... and {len(excluded) - 20} more")
        lines.append("")

    # PIT cache coverage
    pit_results = [r for r in report.results if "pit_cache_status" in r.metrics]
    if pit_results:
        n_ok = sum(1 for r in pit_results if r.metrics["pit_cache_status"] == "OK")
        n_missing = sum(1 for r in pit_results if r.metrics["pit_cache_status"] == "MISSING")
        n_schema_fail = sum(1 for r in pit_results if r.metrics["pit_cache_status"] == "SCHEMA_FAIL")
        lines.append("## PIT Cache Coverage")
        lines.append("")
        lines.append(f"- OK: {n_ok}")
        lines.append(f"- Missing: {n_missing}")
        lines.append(f"- Schema fail: {n_schema_fail}")
        lines.append("")

    # Split warnings
    split_results = [r for r in report.results if r.metrics.get("split_warning_count", 0) > 0]
    if split_results:
        total_sw = sum(r.metrics["split_warning_count"] for r in split_results)
        lines.append("## Split Warnings")
        lines.append("")
        lines.append(f"- Dates with split warnings: {len(split_results)}")
        lines.append(f"- Total split warnings: {total_sw}")
        lines.append("")

    # Dated-source provenance
    ds_results = [r for r in report.results if "dated_source_status" in r.metrics]
    if ds_results:
        n_ds_pass = sum(1 for r in ds_results if r.metrics["dated_source_status"] == "PASS")
        n_ds_warn = sum(1 for r in ds_results if r.metrics["dated_source_status"] == "WARN")
        lines.append("## Dated-Source Provenance")
        lines.append("")
        lines.append(f"- PASS: {n_ds_pass}")
        lines.append(f"- WARN: {n_ds_warn}")
        if n_ds_warn > 0:
            lines.append("")
            lines.append("### Top dated-source warnings")
            lines.append("")
            warn_details: Dict[str, int] = {}
            for r in ds_results:
                if r.metrics["dated_source_status"] == "WARN":
                    detail = r.metrics.get("dated_source_details", "unknown")
                    warn_details[detail] = warn_details.get(detail, 0) + 1
            for detail, count in sorted(warn_details.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"- ({count}x) {detail}")
        lines.append("")

    # Output paths
    lines.append("## Output Paths")
    lines.append("")
    lines.append(f"- Preflight artifacts: `{preflight_dir}`")
    lines.append(f"- Eval outputs: `{eval_dir}`")
    lines.append(f"- This audit: `{out_dir / 'AUDIT.md'}`")
    lines.append("")

    audit_path = out_dir / "AUDIT.md"
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    return audit_path


# ── Main ────────────────────────────────────────────────────────────

def run_audited_backtest(
    *,
    snapshot_root: Path,
    price_csv: Path,
    horizons: List[int],
    out_root: Path,
    ruleset_path: Optional[Path] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_manifest: Optional[Path] = None,
    top_k: int = 20,
    cost_bps: float = 30,
    anchor_mode: str = "prev_trading_day",
    benchmark: str = "XBI",
    rerank: bool = False,
    preflight_strict: bool = True,
    relaxed: bool = False,
    check_pit: bool = False,
    pit_cache_base: Optional[Path] = None,
    run_id: Optional[str] = None,
    rebalance_buffer_ranks: int = 0,
    turnover_cap: float = 0.0,
) -> int:
    """Run an audited backtest.  Returns 0 on success, non-zero on failure.

    Defaults to strict preflight (WARN dates excluded).  For archive/OOS data
    where all snapshots carry WARN status, pass relaxed=True to include them.
    Use date_manifest to restrict evaluation to a curated allowed-date set.
    """
    # --relaxed overrides the strict default
    effective_strict = preflight_strict and not relaxed

    # Derive date_from/date_to from manifest bounds if not provided
    eff_date_from = date_from
    eff_date_to = date_to
    if date_manifest and date_manifest.is_file():
        raw = date_manifest.read_text().splitlines()
        manifest_dates = sorted(d.strip() for d in raw if d.strip())
        if manifest_dates:
            if eff_date_from is None:
                eff_date_from = manifest_dates[0]
            if eff_date_to is None:
                eff_date_to = manifest_dates[-1]

    run_id = run_id or _make_run_id()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    preflight_dir = run_dir / "preflight"
    eval_dir = run_dir / "eval"
    git_sha = _git_sha()

    print(f"=== Audited Backtest: {run_id} ===")
    print(f"  git SHA: {git_sha}")
    print(f"  snapshot_root: {snapshot_root}")
    print(f"  horizons: {horizons}, top_k: {top_k}, cost_bps: {cost_bps}")
    if rebalance_buffer_ranks > 0 or turnover_cap > 0:
        print(f"  buffer: {rebalance_buffer_ranks}, turnover_cap: {turnover_cap}")
    if date_manifest:
        print(f"  date_manifest: {date_manifest}")
    mode_label = "RELAXED (WARN included)" if relaxed else "strict (WARN excluded)"
    print(f"  preflight: {mode_label}, rerank: {rerank}")
    if relaxed:
        print("  [RELAXED MODE] WARN-status snapshots will be included in eval.")
        print("  Intended for archive/OOS data. Ensure date_manifest provides quality control.")
    print()

    # ── Step 1: Preflight ───────────────────────────────────────────
    print("Step 1/4: Running preflight batch...")
    report = run_preflight_batch(
        snapshot_root,
        date_from=eff_date_from,
        date_to=eff_date_to,
        check_pit=check_pit,
        pit_cache_base=pit_cache_base,
        eval_horizons=horizons,
    )
    file_hashes = _collect_file_hashes(
        ruleset_path=ruleset_path,
        price_csv=price_csv,
        snapshot_root=snapshot_root,
    )
    write_preflight_artifacts(
        report, preflight_dir,
        run_id=run_id,
        snapshot_root=snapshot_root,
        date_from=eff_date_from,
        date_to=eff_date_to,
        horizons=horizons,
        strict=effective_strict,
        file_hashes=file_hashes,
    )
    print(f"  preflight: {report.n_total} snapshots — "
          f"{report.n_pass} PASS, {report.n_warn} WARN, {report.n_fail} FAIL")
    print(f"  artifacts → {preflight_dir}")
    print()

    # ── Step 2: Rerank (optional) ───────────────────────────────────
    eval_snapshot_root = snapshot_root
    if rerank:
        if ruleset_path is None:
            print("ERROR: --rerank requires --ruleset", file=sys.stderr)
            return 1
        print("Step 2/4: Re-ranking snapshots...")
        rerank_out = run_dir / "reranked"
        rc = _run_rerank(
            snapshot_root, rerank_out, ruleset_path,
            date_from=eff_date_from, date_to=eff_date_to,
            preflight=True,
        )
        if rc != 0:
            print(f"ERROR: rerank exited with code {rc}", file=sys.stderr)
            return rc
        eval_snapshot_root = rerank_out
        print(f"  reranked → {rerank_out}")
        print()
    else:
        print("Step 2/4: Rerank skipped (--rerank not set)")
        print()

    # ── Step 3: Eval ────────────────────────────────────────────────
    print("Step 3/4: Running forward-return evaluation...")
    rc = _run_eval(
        eval_snapshot_root, price_csv, eval_dir,
        horizons=horizons,
        top_k=top_k,
        cost_bps=cost_bps,
        date_from=eff_date_from,
        date_to=eff_date_to,
        date_manifest=date_manifest,
        anchor_mode=anchor_mode,
        benchmark=benchmark,
        preflight=True,
        preflight_strict=effective_strict,
        rebalance_buffer_ranks=rebalance_buffer_ranks,
        turnover_cap=turnover_cap,
        ruleset_path=ruleset_path,
    )
    if rc != 0:
        print(f"ERROR: eval exited with code {rc}", file=sys.stderr)
        return rc
    print(f"  eval → {eval_dir}")
    print()

    # ── Step 4: AUDIT.md ────────────────────────────────────────────
    print("Step 4/4: Writing AUDIT.md...")
    audit_path = _write_audit_md(
        run_dir,
        run_id=run_id,
        git_sha=git_sha,
        file_hashes=file_hashes,
        report=report,
        horizons=horizons,
        top_k=top_k,
        cost_bps=cost_bps,
        snapshot_root=snapshot_root,
        eval_snapshot_root=eval_snapshot_root,
        price_csv=price_csv,
        ruleset_path=ruleset_path,
        date_from=eff_date_from,
        date_to=eff_date_to,
        date_manifest=date_manifest,
        anchor_mode=anchor_mode,
        benchmark=benchmark,
        strict=effective_strict,
        relaxed=relaxed,
        reranked=rerank,
        preflight_dir=preflight_dir,
        eval_dir=eval_dir,
    )
    print(f"  audit → {audit_path}")
    print()
    print(f"=== Audited Backtest Complete: {run_dir} ===")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an audited backtest with preflight + optional rerank + eval",
        allow_abbrev=False,  # prevent --preflight silently matching --preflight-strict
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
    )
    parser.add_argument(
        "--ruleset", type=Path, default=None,
        help="Path to DecisionRuleset JSON",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--date-manifest", type=Path, default=None,
        help=(
            "Path to file with one YYYY-MM-DD date per line (strict curated set). "
            "Restricts eval to exactly those dates; derives --date-from/--date-to "
            "from manifest bounds if not set. "
            "Example: output/audited_sets/audited_dates_2020_2024_strict.txt"
        ),
    )
    parser.add_argument(
        "--horizons", type=str, default="63",
        help="Comma-separated forward-return horizons (default: 63)",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=30)
    parser.add_argument(
        "--anchor-mode", default="prev_trading_day",
        choices=["exact", "prev_trading_day", "next_trading_day"],
    )
    parser.add_argument("--benchmark", default="XBI")
    parser.add_argument(
        "--rerank", action="store_true", default=False,
        help="Re-rank snapshots through --ruleset before evaluating",
    )
    parser.add_argument(
        "--relaxed", action="store_true", default=False,
        help=(
            "Allow WARN-status snapshots (non-strict preflight). "
            "Required for archive/OOS data where all snapshots are WARN. "
            "A --date-manifest should be provided to maintain quality control."
        ),
    )
    parser.add_argument(
        "--check-pit", action="store_true", default=False,
        help="Enable PIT price cache checks in preflight",
    )
    parser.add_argument(
        "--pit-cache-base", type=Path,
        default=PROJECT_ROOT / "data" / "caches" / "price_pit" / "PIT",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=PROJECT_ROOT / "output" / "audited_backtests",
        help="Output root directory (default: output/audited_backtests)",
    )
    parser.add_argument(
        "--rebalance-buffer-ranks", type=int, default=0,
        help="Buffer zone around top-K for rebalance (0=disabled)",
    )
    parser.add_argument(
        "--turnover-cap", type=float, default=0.0,
        help="Max turnover fraction per rebalance (0=unlimited)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run identifier (auto-generated if omitted)",
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    rc = run_audited_backtest(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        horizons=horizons,
        out_root=args.out_root,
        ruleset_path=args.ruleset,
        date_from=args.date_from,
        date_to=args.date_to,
        date_manifest=args.date_manifest,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        anchor_mode=args.anchor_mode,
        benchmark=args.benchmark,
        rerank=args.rerank,
        relaxed=args.relaxed,
        check_pit=args.check_pit,
        pit_cache_base=args.pit_cache_base,
        run_id=args.run_id,
        rebalance_buffer_ranks=args.rebalance_buffer_ranks,
        turnover_cap=args.turnover_cap,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
