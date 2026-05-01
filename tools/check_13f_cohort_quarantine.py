#!/usr/bin/env python3
"""13F cohort-quarantine pre/post diff harness — SKELETON (2026-05-01).

Read-only diagnostic. Compares two production snapshots — the last clean
PRE-refresh snapshot and the first POST-refresh snapshot — to characterize
the impact of a 13F cache update on selector/ranker outputs.

Companion artifact: 13F_COHORT_QUARANTINE_PREP_2026_05_01.md
                    (defines the diff schema, thresholds, and refresh-day
                     checklist this script implements).

NOT FOR PRODUCTION CRON. Runs on demand. Does not modify any production
state, manager registry, or snapshot files.

Guardrails (apply in order, abort on any failure):
  G1 — snapshot completeness (rankings.csv + institutional_summary_delta.json
       present in both pre and post; inst_delta_z has sd > 0.1)
  G2 — producer freshness (institutional_summary.json mtime advanced;
       prior_date in delta JSON advanced from pre to post)
  G3 — distinguish manager-level cause (new managers, registry change)
       vs window-level cause (prior_date roll only)

Diff sections (per artifact §2):
  A. Manager-level diff (registry counts, AUM, new/removed managers)
  B. Coverage diff (tickers_with_signal, signal_coverage_pct)
  C. Per-ticker score diff (coinvest_score_z, inst_delta_z, distributions)
  D. Top-30 churn (Jaccard, entries/exits, rank movement)
  E. Sector / market_cap / stage_bucket skew

Quarantine triggers (per artifact §3): if Top-30 Jaccard < 0.70 OR
manager Δ > 5 OR coverage drop ≥ 10pp → quarantine warranted.

Usage:
    python -m tools.check_13f_cohort_quarantine \\
        --pre-date 2026-05-14 --post-date 2026-05-18 \\
        --output artifacts/13f_diff_2026_05_18.md

Refinement TODOs (deferred to refresh-day or later):
  - Validate §2.A manager-level diff against actual registry shape
  - Calibrate KS-stat thresholds against historical refresh observations
  - Wire stage_bucket diff once Spec 068 changes settle
  - Add JSON output mode for downstream tools
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SNAP_ROOT = PROJECT_ROOT / "data" / "snapshots"
INST_SUMMARY = PROJECT_ROOT / "production_data" / "institutional_summary.json"
MGR_REGISTRY = PROJECT_ROOT / "production_data" / "manager_registry.json"

# Quarantine thresholds (per artifact §3)
TOP30_JACCARD_QUARANTINE = 0.70
TOP30_JACCARD_REVIEW = 0.85
MANAGER_DELTA_THRESHOLD = 5
COVERAGE_DROP_PP_THRESHOLD = 10.0
INST_DELTA_KS_THRESHOLD = 0.30
COINVEST_KS_THRESHOLD = 0.20
INST_DELTA_SD_FLOOR = 0.10  # G1 — below this, inst_delta_z is effectively constant


def _safe_float(v) -> Optional[float]:
    if v in (None, "", "nan", "None", "null"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_rankings(snap_date: str) -> List[dict]:
    p = SNAP_ROOT / snap_date / "rankings.csv"
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_delta_json(snap_date: str) -> Optional[dict]:
    p = SNAP_ROOT / snap_date / "institutional_summary_delta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_summary() -> Optional[dict]:
    if not INST_SUMMARY.exists():
        return None
    return json.loads(INST_SUMMARY.read_text())


def _load_registry() -> Optional[dict]:
    if not MGR_REGISTRY.exists():
        return None
    return json.loads(MGR_REGISTRY.read_text())


def _column_sd(rows: List[dict], col: str) -> Optional[float]:
    vals = [_safe_float(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def _ks_stat(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sample Kolmogorov-Smirnov statistic. Returns None if either sample
    is too small. Pure stdlib — no scipy dependency."""
    a = sorted(a)
    b = sorted(b)
    if len(a) < 5 or len(b) < 5:
        return None
    all_vals = sorted(set(a + b))
    n_a, n_b = len(a), len(b)
    max_diff = 0.0
    for v in all_vals:
        # Empirical CDFs
        cdf_a = sum(1 for x in a if x <= v) / n_a
        cdf_b = sum(1 for x in b if x <= v) / n_b
        d = abs(cdf_a - cdf_b)
        if d > max_diff:
            max_diff = d
    return max_diff


# ---------------------------------------------------------------------------
# Guardrails (G1, G2, G3)
# ---------------------------------------------------------------------------


def guardrails_g1_snapshot_completeness(pre_date: str, post_date: str) -> Tuple[bool, List[str]]:
    """G1: rankings.csv + institutional_summary_delta.json present in both,
    inst_delta_z has sd > floor."""
    failures = []
    for label, snap in [("pre", pre_date), ("post", post_date)]:
        rows = _load_rankings(snap)
        if not rows:
            failures.append(f"{label} ({snap}): rankings.csv missing")
            continue
        delta = _load_delta_json(snap)
        if delta is None:
            failures.append(f"{label} ({snap}): institutional_summary_delta.json missing")
        sd = _column_sd(rows, "inst_delta_z")
        if sd is None or sd <= INST_DELTA_SD_FLOOR:
            failures.append(
                f"{label} ({snap}): inst_delta_z sd={sd} ≤ {INST_DELTA_SD_FLOOR} (incomplete-run fallback?)"
            )
    return (len(failures) == 0, failures)


def guardrails_g2_producer_freshness(pre_date: str, post_date: str) -> Tuple[bool, List[str]]:
    """G2: institutional_summary mtime advanced AND prior_date in delta
    advanced between pre and post."""
    failures = []
    pre_delta = _load_delta_json(pre_date)
    post_delta = _load_delta_json(post_date)
    if pre_delta and post_delta:
        pre_prior = pre_delta.get("prior_date", "")
        post_prior = post_delta.get("prior_date", "")
        if post_prior <= pre_prior:
            failures.append(
                f"prior_date in delta JSON did not advance: pre={pre_prior} post={post_prior} "
                f"(refresh hasn't landed yet)"
            )
    summary = _load_summary()
    if summary:
        cache_date = summary.get("cache_as_of_date", "")
        # Pre-refresh cache_as_of_date should be < post_date
        if cache_date <= pre_date:
            failures.append(f"institutional_summary.cache_as_of_date={cache_date} not advanced past pre={pre_date}")
    return (len(failures) == 0, failures)


def guardrails_g3_attribution_check(pre_date: str, post_date: str) -> Tuple[bool, List[str]]:
    """G3: best-effort check whether manager-level change occurred. If the
    registry didn't change AND no new managers in the post-summary, then any
    change is window-level (prior_date roll) only — flag for interpretation."""
    notes = []
    # This guardrail is informational, not blocking. It just notes whether the
    # caller should expect manager-driven vs window-driven changes.
    summary = _load_summary()
    if summary:
        notes.append(f"institutional_summary.elite_managers_total = {summary.get('elite_managers_total')}")
        notes.append(f"institutional_summary.cache_as_of_date = {summary.get('cache_as_of_date')}")
    return (True, notes)


# ---------------------------------------------------------------------------
# Diff sections (A through E)
# ---------------------------------------------------------------------------


def diff_section_b_coverage(pre_delta: dict, post_delta: dict) -> dict:
    """B. Coverage diff from delta JSONs."""
    return {
        "tickers_in_current_pre": pre_delta.get("tickers_in_current"),
        "tickers_in_current_post": post_delta.get("tickers_in_current"),
        "tickers_common_pre": pre_delta.get("tickers_common"),
        "tickers_common_post": post_delta.get("tickers_common"),
        "prior_date_pre": pre_delta.get("prior_date"),
        "prior_date_post": post_delta.get("prior_date"),
    }


def diff_section_c_scores(pre_rows: List[dict], post_rows: List[dict]) -> dict:
    """C. Per-ticker score diff for coinvest_score_z and inst_delta_z."""
    pre_by_t = {r["ticker"]: r for r in pre_rows}
    post_by_t = {r["ticker"]: r for r in post_rows}
    common = sorted(set(pre_by_t.keys()) & set(post_by_t.keys()))

    def stats(field: str) -> dict:
        deltas = []
        pre_vals = []
        post_vals = []
        for t in common:
            pv = _safe_float(pre_by_t[t].get(field))
            ev = _safe_float(post_by_t[t].get(field))
            if pv is not None:
                pre_vals.append(pv)
            if ev is not None:
                post_vals.append(ev)
            if pv is not None and ev is not None:
                deltas.append((t, ev - pv))
        if not deltas:
            return {"n": 0}
        sorted_by_abs = sorted(deltas, key=lambda x: -abs(x[1]))
        return {
            "n": len(deltas),
            "mean_abs_delta": round(sum(abs(d[1]) for d in deltas) / len(deltas), 4),
            "max_abs_delta": round(abs(sorted_by_abs[0][1]), 4),
            "n_large_change": sum(1 for d in deltas if abs(d[1]) > 1.0),
            "ks_stat_vs_pre": (
                round(_ks_stat(pre_vals, post_vals), 4) if _ks_stat(pre_vals, post_vals) is not None else None
            ),
            "top_3_movers": [(t, round(d, 4)) for t, d in sorted_by_abs[:3]],
        }

    return {
        "coinvest_score_z": stats("coinvest_score_z"),
        "inst_delta_z": stats("inst_delta_z"),
        "common_tickers": len(common),
    }


def diff_section_d_top30(pre_rows: List[dict], post_rows: List[dict]) -> dict:
    """D. Top-30 churn. Top-30 = actionable_rank ≤ 30."""

    def top30_set(rows):
        ranked = []
        for r in rows:
            ar = _safe_float(r.get("actionable_rank"))
            if ar is not None and ar <= 30:
                ranked.append((r["ticker"], int(ar)))
        return {t for t, _ in ranked}

    pre_set = top30_set(pre_rows)
    post_set = top30_set(post_rows)
    intersection = pre_set & post_set
    union = pre_set | post_set
    jaccard = len(intersection) / len(union) if union else 0.0
    entered = sorted(post_set - pre_set)
    left = sorted(pre_set - post_set)
    return {
        "n_pre_top30": len(pre_set),
        "n_post_top30": len(post_set),
        "n_intersect": len(intersection),
        "jaccard": round(jaccard, 3),
        "entered": entered,
        "left": left,
    }


def diff_section_e_skew(pre_rows: List[dict], post_rows: List[dict]) -> dict:
    """E. Sector / market_cap / stage_bucket skew of top-30."""
    from collections import Counter

    def top30_buckets(rows: List[dict], field: str) -> dict:
        counts = Counter()
        for r in rows:
            ar = _safe_float(r.get("actionable_rank"))
            if ar is not None and ar <= 30:
                counts[r.get(field, "")] += 1
        return dict(counts)

    return {
        "industry_group_pre": top30_buckets(pre_rows, "industry_group"),
        "industry_group_post": top30_buckets(post_rows, "industry_group"),
        "market_cap_bucket_pre": top30_buckets(pre_rows, "market_cap_bucket"),
        "market_cap_bucket_post": top30_buckets(post_rows, "market_cap_bucket"),
        "stage_bucket_pre": top30_buckets(pre_rows, "stage_bucket"),
        "stage_bucket_post": top30_buckets(post_rows, "stage_bucket"),
    }


# ---------------------------------------------------------------------------
# Quarantine decision
# ---------------------------------------------------------------------------


def quarantine_decision(diff: dict) -> Tuple[str, List[str]]:
    """Apply §3 thresholds to the diff and return (verdict, reasons)."""
    reasons = []
    verdict = "NO_QUARANTINE"
    j = diff["top30"]["jaccard"]
    if j < TOP30_JACCARD_QUARANTINE:
        reasons.append(f"Top-30 Jaccard {j:.2f} < {TOP30_JACCARD_QUARANTINE}: cohort-contaminated 10 trading days")
        verdict = "QUARANTINE"
    elif j < TOP30_JACCARD_REVIEW:
        reasons.append(f"Top-30 Jaccard {j:.2f} in [0.70, 0.85): standard cohort window, attribution-only")
        verdict = "STANDARD_COHORT_WINDOW"
    cov_pre = diff["coverage"].get("tickers_common_pre") or 0
    cov_post = diff["coverage"].get("tickers_common_post") or 0
    if cov_pre and cov_post:
        drop_pp = (cov_pre - cov_post) / cov_pre * 100
        if drop_pp >= COVERAGE_DROP_PP_THRESHOLD:
            reasons.append(f"Coverage dropped {drop_pp:.1f}pp ≥ {COVERAGE_DROP_PP_THRESHOLD}: producer audit required")
            verdict = "PRODUCER_AUDIT_REQUIRED"
    ks_inst = diff["scores"]["inst_delta_z"].get("ks_stat_vs_pre")
    if ks_inst is not None and ks_inst >= INST_DELTA_KS_THRESHOLD:
        reasons.append(f"inst_delta_z KS={ks_inst:.2f} ≥ {INST_DELTA_KS_THRESHOLD}: refresh confirmed (expected)")
    ks_cv = diff["scores"]["coinvest_score_z"].get("ks_stat_vs_pre")
    if ks_cv is not None and ks_cv >= COINVEST_KS_THRESHOLD:
        reasons.append(
            f"coinvest_score_z KS={ks_cv:.2f} ≥ {COINVEST_KS_THRESHOLD}: registry change suspected, manual review"
        )
    return (verdict, reasons)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(pre_date: str, post_date: str, output: Optional[Path] = None) -> int:
    log.info(f"13F cohort-quarantine diff: pre={pre_date} post={post_date}")

    # Guardrails
    g1_pass, g1_msgs = guardrails_g1_snapshot_completeness(pre_date, post_date)
    if not g1_pass:
        log.error("G1 (snapshot completeness) FAILED:")
        for m in g1_msgs:
            log.error(f"  - {m}")
        log.error("Verdict: INCOMPLETE_RUN_FALLBACK — refusing to compute diff.")
        return 2
    log.info("G1 (snapshot completeness): PASS")

    g2_pass, g2_msgs = guardrails_g2_producer_freshness(pre_date, post_date)
    if not g2_pass:
        log.error("G2 (producer freshness) FAILED:")
        for m in g2_msgs:
            log.error(f"  - {m}")
        log.error("Verdict: REFRESH_NOT_LANDED — wait for next snapshot.")
        return 2
    log.info("G2 (producer freshness): PASS")

    g3_pass, g3_notes = guardrails_g3_attribution_check(pre_date, post_date)
    log.info("G3 (attribution context):")
    for n in g3_notes:
        log.info(f"  - {n}")

    # Load data
    pre_rows = _load_rankings(pre_date)
    post_rows = _load_rankings(post_date)
    pre_delta = _load_delta_json(pre_date)
    post_delta = _load_delta_json(post_date)

    # Diff sections
    diff = {
        "pre_date": pre_date,
        "post_date": post_date,
        "coverage": diff_section_b_coverage(pre_delta, post_delta),
        "scores": diff_section_c_scores(pre_rows, post_rows),
        "top30": diff_section_d_top30(pre_rows, post_rows),
        "skew": diff_section_e_skew(pre_rows, post_rows),
    }

    verdict, reasons = quarantine_decision(diff)
    diff["verdict"] = verdict
    diff["verdict_reasons"] = reasons

    # Markdown output
    md_lines = [
        f"# 13F Cohort-Quarantine Diff — {pre_date} → {post_date}",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "**Reasons:**",
        *[f"- {r}" for r in reasons or ["(no triggers fired)"]],
        "",
        "## Top-30 churn",
        "",
        f"- Jaccard: **{diff['top30']['jaccard']}**",
        f"- Pre top-30: {diff['top30']['n_pre_top30']}; Post top-30: {diff['top30']['n_post_top30']}",
        f"- Names entering: {diff['top30']['entered']}",
        f"- Names leaving: {diff['top30']['left']}",
        "",
        "## Per-ticker score deltas",
        "",
        f"- coinvest_score_z: {diff['scores']['coinvest_score_z']}",
        f"- inst_delta_z: {diff['scores']['inst_delta_z']}",
        "",
        "## Coverage",
        "",
        f"- {diff['coverage']}",
        "",
        "## Top-30 skew (industry / market_cap / stage)",
        "",
        f"- industry_group: pre={diff['skew']['industry_group_pre']}, post={diff['skew']['industry_group_post']}",
        f"- market_cap_bucket: pre={diff['skew']['market_cap_bucket_pre']}, post={diff['skew']['market_cap_bucket_post']}",
        f"- stage_bucket: pre={diff['skew']['stage_bucket_pre']}, post={diff['skew']['stage_bucket_post']}",
    ]
    md = "\n".join(md_lines) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md)
        log.info(f"Wrote {output}")
    else:
        print(md)

    return 1 if verdict in ("QUARANTINE", "PRODUCER_AUDIT_REQUIRED") else 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pre-date", required=True, help="Last clean pre-refresh snapshot YYYY-MM-DD")
    p.add_argument("--post-date", required=True, help="First post-refresh snapshot YYYY-MM-DD")
    p.add_argument("--output", type=Path, default=None, help="Output Markdown path (default stdout)")
    args = p.parse_args(argv)
    return run(args.pre_date, args.post_date, args.output)


if __name__ == "__main__":
    sys.exit(main())
