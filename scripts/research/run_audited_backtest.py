#!/usr/bin/env python3
"""Audited backtest runner — preflight + optional rerank + eval + AUDIT.md + VERDICT.md.

Single entrypoint that produces a reproducible audit package:
  1. Runs snapshot preflight batch (writes artifacts)
  2. Optionally re-ranks snapshots through a ruleset (with --preflight)
  3. Runs eval_forward_returns.py (with --preflight-strict by default)
  4. Writes output/<run_id>/AUDIT.md summarizing everything
  5. Writes output/<run_id>/VERDICT.md + VERDICT.json (signed result page)

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


# ── Verdict ──────────────────────────────────────────────────────────

# Promotion thresholds (pp = percentage points, not fraction)
_PRIMARY_THRESHOLD_PP: float = 0.20    # 126d net Δ must be ≥ this
_GUARDRAIL_THRESHOLD_PP: float = -0.05  # 84d net Δ must be ≥ this
_MIN_DATES_FOR_VERDICT: int = 50        # fewer → NEEDS_MORE regardless


def _load_eval_summary(eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Load summary.json from eval output dir, or None if missing/corrupt."""
    p = eval_dir / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compute_verdict(
    cand: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    *,
    primary_threshold_pp: float = _PRIMARY_THRESHOLD_PP,
    guardrail_threshold_pp: float = _GUARDRAIL_THRESHOLD_PP,
    min_dates: int = _MIN_DATES_FOR_VERDICT,
) -> Dict[str, Any]:
    """Compute a signed verdict dict from eval summary dicts.

    Args:
        cand:      candidate summary.json dict
        baseline:  baseline summary.json dict (None → absolute-only, no delta)

    Returns a dict with schema 'verdict.v1'.
    """
    horizons_raw = cand.get("horizons", [])
    horizons = sorted(int(h) for h in horizons_raw)
    primary_h = max(horizons) if horizons else None
    guardrail_h = sorted(horizons)[-2] if len(horizons) >= 2 else None
    n_evaluated = cand.get("n_evaluated", 0)

    def _h(summary: Dict, h: int) -> Dict[str, Any]:
        return summary.get("by_horizon", {}).get(str(h), {})

    def _pct(val: Optional[float]) -> Optional[float]:
        return round(val * 100, 4) if val is not None else None

    results: Dict[str, Any] = {}
    for h in horizons:
        ch = _h(cand, h)
        row: Dict[str, Any] = {
            "net_pct": _pct(ch.get("mean_net_return")),
            "excess_pct": _pct(ch.get("mean_excess_return")),
            "hedged_pct": _pct(ch.get("mean_hedged_return")),
            "turnover_pct": _pct(ch.get("mean_turnover")),
            "mean_ic": round(ch["mean_ic"], 6) if ch.get("mean_ic") is not None else None,
            "ic_t_stat": round(ch["ic_t_stat"], 3) if ch.get("ic_t_stat") is not None else None,
            "delta_net_pp": None,
        }
        if baseline is not None:
            bh = _h(baseline, h)
            b_net = bh.get("mean_net_return")
            c_net = ch.get("mean_net_return")
            if b_net is not None and c_net is not None:
                row["delta_net_pp"] = round((c_net - b_net) * 100, 4)
        results[str(h)] = row

    # ── Verdict logic ────────────────────────────────────────────────
    reasons: List[str] = []
    verdict: str

    if n_evaluated < min_dates:
        verdict = "NEEDS_MORE"
        reasons.append(f"n_evaluated={n_evaluated} < min {min_dates} dates for verdict")
    elif baseline is None:
        verdict = "NEEDS_MORE"
        reasons.append("no baseline provided — pass --baseline-summary for PROMOTE/ARCHIVE verdict")
    else:
        primary_delta = results.get(str(primary_h), {}).get("delta_net_pp") if primary_h else None
        guardrail_delta = results.get(str(guardrail_h), {}).get("delta_net_pp") if guardrail_h else None

        primary_pass = primary_delta is not None and primary_delta >= primary_threshold_pp
        guardrail_pass = (guardrail_delta is None  # single-horizon: no guardrail check
                         or guardrail_delta >= guardrail_threshold_pp)

        if primary_pass and guardrail_pass:
            verdict = "PROMOTE"
            reasons.append(
                f"{primary_h}d Δ = {primary_delta:+.3f}pp ≥ threshold {primary_threshold_pp:+.2f}pp"
            )
            if guardrail_delta is not None:
                reasons.append(
                    f"{guardrail_h}d Δ = {guardrail_delta:+.3f}pp ≥ guardrail {guardrail_threshold_pp:+.2f}pp"
                )
        else:
            verdict = "ARCHIVE"
            if primary_delta is not None and not primary_pass:
                reasons.append(
                    f"{primary_h}d Δ = {primary_delta:+.3f}pp < threshold {primary_threshold_pp:+.2f}pp"
                )
            if guardrail_delta is not None and not guardrail_pass:
                reasons.append(
                    f"{guardrail_h}d Δ = {guardrail_delta:+.3f}pp < guardrail {guardrail_threshold_pp:+.2f}pp"
                )

    return {
        "schema": "verdict.v1",
        "n_evaluated": n_evaluated,
        "primary_horizon": primary_h,
        "guardrail_horizon": guardrail_h,
        "results": results,
        "thresholds": {
            "primary_delta_pp": primary_threshold_pp,
            "guardrail_delta_pp": guardrail_threshold_pp,
            "min_dates": min_dates,
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


def _write_verdict(
    out_dir: Path,
    verdict: Dict[str, Any],
    *,
    name: str,
    git_sha: str,
    run_id: str,
    candidate_ruleset: Optional[str],
    baseline_ruleset: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    relaxed: bool,
) -> Path:
    """Write VERDICT.md and VERDICT.json to out_dir."""
    v = verdict["verdict"]
    ph = verdict["primary_horizon"]
    gh = verdict["guardrail_horizon"]
    res = verdict["results"]
    thresholds = verdict["thresholds"]

    # ── JSON ────────────────────────────────────────────────────────
    full = dict(verdict)
    full.update({
        "name": name,
        "run_id": run_id,
        "git_sha": git_sha,
        "candidate_ruleset": candidate_ruleset,
        "baseline_ruleset": baseline_ruleset,
        "date_from": date_from,
        "date_to": date_to,
        "relaxed": relaxed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    json_path = out_dir / "VERDICT.json"
    json_path.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")

    # ── Markdown ────────────────────────────────────────────────────
    icon = {"PROMOTE": "🟢", "ARCHIVE": "🔴", "NEEDS_MORE": "🟡"}.get(v, "⚪")
    lines: List[str] = []
    lines += [
        f"# Research Verdict — {name}",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Verdict | **{icon} {v}** |",
        f"| Run ID | `{run_id}` |",
        f"| Git SHA | `{git_sha}` |",
        f"| Candidate | `{candidate_ruleset or '—'}` |",
        f"| Baseline | `{baseline_ruleset or '—'}` |",
        f"| Window | {date_from or '—'} → {date_to or '—'} ({verdict['n_evaluated']} dates) |",
        f"| Mode | {'relaxed' if relaxed else 'strict'} |",
        "",
    ]

    # Results table
    def _fmt(val: Optional[float], suffix: str = "") -> str:
        return f"{val:.4f}{suffix}" if val is not None else "—"

    def _fmt_delta(val: Optional[float]) -> str:
        if val is None:
            return "—"
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.3f}pp"

    primary_label = f"{ph}d _(primary)_" if ph else "—"
    guardrail_label = f"{gh}d _(guardrail)_" if gh else "—"

    lines += [
        "## Results",
        "",
        f"| Horizon | Net% | Excess% | Hedged% | Turnover% | Δ Net (pp) |",
        f"|---------|------|---------|---------|-----------|------------|",
    ]
    for h_str, row in sorted(res.items(), key=lambda x: int(x[0])):
        h_int = int(h_str)
        label = (f"{h_int}d (primary)" if h_int == ph
                 else f"{h_int}d (guardrail)" if h_int == gh
                 else f"{h_int}d")
        delta_str = _fmt_delta(row.get("delta_net_pp"))
        lines.append(
            f"| {label} | {_fmt(row.get('net_pct'), '%')} "
            f"| {_fmt(row.get('excess_pct'), '%')} "
            f"| {_fmt(row.get('hedged_pct'), '%')} "
            f"| {_fmt(row.get('turnover_pct'), '%')} "
            f"| {delta_str} |"
        )
    lines.append("")

    # Thresholds
    lines += [
        "## Thresholds",
        "",
        f"| | Threshold | Result |",
        f"|--|-----------|--------|",
    ]
    if ph:
        ph_delta = res.get(str(ph), {}).get("delta_net_pp")
        ph_pass = ph_delta is not None and ph_delta >= thresholds["primary_delta_pp"]
        ph_icon = "✅" if ph_pass else ("—" if ph_delta is None else "❌")
        lines.append(
            f"| Primary ({ph}d Δ net) | ≥ {thresholds['primary_delta_pp']:+.2f}pp "
            f"| {ph_icon} {_fmt_delta(ph_delta)} |"
        )
    if gh:
        gh_delta = res.get(str(gh), {}).get("delta_net_pp")
        gh_pass = gh_delta is None or gh_delta >= thresholds["guardrail_delta_pp"]
        gh_icon = "✅" if gh_pass else ("—" if gh_delta is None else "❌")
        lines.append(
            f"| Guardrail ({gh}d Δ net) | ≥ {thresholds['guardrail_delta_pp']:+.2f}pp "
            f"| {gh_icon} {_fmt_delta(gh_delta)} |"
        )
    lines.append("")

    # Verdict reasons
    lines += ["## Verdict Reasons", ""]
    for reason in verdict["verdict_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")

    # Next steps
    if v == "PROMOTE":
        lines += [
            "## Next Step",
            "",
            "```bash",
            f"python3 scripts/promote_ruleset.py <ruleset_id> "
            f"--reason \"verdict: {name}\"",
            "```",
            "",
        ]
    elif v == "ARCHIVE":
        lines += [
            "## Next Step",
            "",
            "Move research rulesets to `production_data/decision_rulesets/research_archive/`.",
            "",
        ]

    md_path = out_dir / "VERDICT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


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
    baseline_summary_path: Optional[Path] = None,
    verdict_name: Optional[str] = None,
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
    print("Step 4/5: Writing AUDIT.md...")
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

    # ── Step 5: VERDICT.md ──────────────────────────────────────────
    print("Step 5/5: Writing VERDICT.md...")
    cand_summary = _load_eval_summary(eval_dir)
    # --baseline-summary accepts either path/to/summary.json or path/to/eval/dir/
    baseline_summary: Optional[Dict[str, Any]] = None
    if baseline_summary_path:
        if baseline_summary_path.is_file():
            try:
                baseline_summary = json.loads(
                    baseline_summary_path.read_text(encoding="utf-8")
                )
            except Exception:
                print(f"  WARN: could not parse baseline summary {baseline_summary_path}")
        elif baseline_summary_path.is_dir():
            baseline_summary = _load_eval_summary(baseline_summary_path)

    if cand_summary:
        eff_verdict_name = verdict_name or (
            ruleset_path.stem if ruleset_path else run_id
        )
        verdict = _compute_verdict(cand_summary, baseline_summary)
        verdict_path = _write_verdict(
            run_dir,
            verdict,
            name=eff_verdict_name,
            git_sha=git_sha,
            run_id=run_id,
            candidate_ruleset=str(ruleset_path) if ruleset_path else None,
            baseline_ruleset=str(baseline_summary_path) if baseline_summary_path else None,
            date_from=eff_date_from,
            date_to=eff_date_to,
            relaxed=relaxed,
        )
        v = verdict["verdict"]
        print(f"  verdict → {verdict_path}  [{v}]")
    else:
        print("  WARN: eval summary.json not found — VERDICT.md skipped")
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
    parser.add_argument(
        "--baseline-summary", type=Path, default=None,
        help=(
            "Path to baseline eval summary.json (or its parent dir) for delta "
            "computation in VERDICT.md. Without this, verdict is NEEDS_MORE. "
            "Example: output/audited_backtests/baseline/eval/summary.json"
        ),
    )
    parser.add_argument(
        "--verdict-name", type=str, default=None,
        help=(
            "Human-readable name for VERDICT.md header "
            "(default: ruleset filename stem or run_id)"
        ),
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    rc = run_audited_backtest(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        horizons=horizons,
        out_root=args.out_root,
        ruleset_path=args.ruleset,
        baseline_summary_path=args.baseline_summary,
        verdict_name=args.verdict_name,
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
