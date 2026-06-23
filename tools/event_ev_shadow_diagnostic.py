"""Event EV shadow diagnostic — market-implied expectation vs base rate.

LABEL: EVENT_EV_SHADOW_DIAGNOSTIC_ONLY_NO_ALPHA_PROMOTION

EES_STATUS: CALIBRATION_DIAGNOSTIC_BUILT
PREDICTIVE_STATUS: UNPROVEN
PROMOTION_STATUS: DO_NOT_PROMOTE

This tool is calibration analysis, not predictive validation. It tests
whether the options-implied move (priced_move_pct) is consistent with
historical base rates by cohort. It does NOT test EES scores against
future returns, realized catalyst outcomes, or IC. EES has not been
shown to predict future returns.

What would prove EES predictive (not done here):
  1. Cross-sectional IC: EES vs future 5d/20d/60d excess return
  2. Event outcome test: EES vs realized catalyst move / hit-miss
  3. Calibration test: underpriced bucket outperforms in-range/overpriced
  4. Stability: works across clinical, regulatory, phase2, phase3 cohorts
  5. Walk-forward test: no same-period tuning or look-ahead

Outputs
-------
  <snapshot_dir>/event_ev_shadow_diagnostic.json
  <snapshot_dir>/event_ev_shadow_diagnostic.md

Governance constraints
----------------------
- Read-only: reads rankings.csv, writes diagnostic artifacts only
- No scoring integration: outputs do NOT feed ranker/selector/final_score
- No alpha promotion: flags are diagnostic labels, not trade recommendations
- Freeze-safe: does not import or modify any frozen module
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Base rate table (mirrors event_ev/expectation_error_model.py) ─────────────
# p50 = median realised abs move (%); iqr = p75 - p25 (%).
_BASE_RATE: dict[str, dict[str, float]] = {
    "CLINICAL|phase3": {"p50": 35.0, "iqr": 37.0},
    "CLINICAL|phase2": {"p50": 35.0, "iqr": 40.0},
    "CLINICAL|early": {"p50": 25.0, "iqr": 38.0},
    "REGULATORY|phase3": {"p50": 19.0, "iqr": 27.0},
    "REGULATORY|phase2": {"p50": 23.5, "iqr": 30.0},
    "REGULATORY|early": {"p50": 20.0, "iqr": 30.0},
    "SAFETY|any": {"p50": 20.0, "iqr": 35.0},
}

# Conditional expected move (probability-weighted)
_COND_MOVE: dict[str, float] = {
    "CLINICAL|phase3": 29.2,
    "CLINICAL|phase2": 32.7,
    "CLINICAL|early": 23.4,
    "REGULATORY|phase3": 15.1,
    "REGULATORY|phase2": 24.0,
    "REGULATORY|early": 17.0,
}

# Miscalibration thresholds
_OVER_THRESH = 0.40  # implied > base_p50 * (1 + thresh) → overpriced
_UNDER_THRESH = 0.25  # implied < base_p50 * (1 - thresh) → underpriced


def _sf(v: object) -> Optional[float]:
    try:
        return float(v) if v and str(v).strip() not in ("", "None", "nan") else None
    except (ValueError, TypeError):
        return None


def _phase_bucket(phase_str: object) -> str:
    try:
        p = float(phase_str)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return "early"
    if p >= 3:
        return "phase3"
    if p >= 2:
        return "phase2"
    return "early"


def _cohort_key(family: str, phase_bucket: str) -> str:
    key = f"{family}|{phase_bucket}"
    if key in _BASE_RATE:
        return key
    fallback = f"{family}|any"
    if fallback in _BASE_RATE:
        return fallback
    return ""


def _miscal_label(priced: float, base_p50: float) -> str:
    ratio = priced / (base_p50 + 1e-6)
    if ratio > 1 + _OVER_THRESH:
        return "OVERPRICED"
    if ratio < 1 - _UNDER_THRESH:
        return "UNDERPRICED"
    return "IN_RANGE"


def run(snapshot_dir: Path) -> None:
    rankings_path = snapshot_dir / "rankings.csv"
    if not rankings_path.exists():
        print(f"ERROR: {rankings_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(rankings_path) as f:
        rows = list(csv.DictReader(f))

    as_of = (snapshot_dir / "metadata.json").exists() and json.loads((snapshot_dir / "metadata.json").read_text()).get(
        "as_of_date", snapshot_dir.name
    )

    # ── Per-name analysis ───────────────────────────────────────────────────
    names: list[dict] = []
    cohort_data: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        family = row.get("catalyst_family", "")
        if family in ("", "NO_CATALYST"):
            continue

        priced = _sf(row.get("priced_move_pct"))
        if priced is None:
            continue

        phase = _phase_bucket(row.get("lead_program_phase", ""))
        cohort = _cohort_key(family, phase)
        if not cohort:
            continue

        base = _BASE_RATE[cohort]
        cond_move = _COND_MOVE.get(cohort)
        label = _miscal_label(priced, base["p50"])
        gap_pp = round(priced - base["p50"], 2)
        ratio = round(priced / (base["p50"] + 1e-6), 3)

        entry: dict = {
            "ticker": row.get("ticker", ""),
            "cohort": cohort,
            "catalyst_date_precision": row.get("catalyst_date_precision", ""),
            "priced_move_pct": priced,
            "base_rate_p50": base["p50"],
            "base_rate_iqr": base["iqr"],
            "conditional_expected_move": cond_move,
            "gap_pp": gap_pp,
            "ratio": ratio,
            "miscal_label": label,
            "base_rate_gap_score": _sf(row.get("base_rate_gap_score")),
            "ees_v2_score": _sf(row.get("ees_v2_score")),
            "ees_v3_score": _sf(row.get("ees_v3_score")),
        }
        names.append(entry)
        cohort_data[cohort].append(priced)

    # ── Cohort summaries ────────────────────────────────────────────────────
    cohort_summaries: list[dict] = []
    for cohort, prices in sorted(cohort_data.items()):
        base = _BASE_RATE[cohort]
        cond = _COND_MOVE.get(cohort)
        n = len(prices)
        med = statistics.median(prices)
        over = sum(1 for p in prices if _miscal_label(p, base["p50"]) == "OVERPRICED")
        under = sum(1 for p in prices if _miscal_label(p, base["p50"]) == "UNDERPRICED")
        in_range = n - over - under
        cohort_summaries.append(
            {
                "cohort": cohort,
                "n": n,
                "implied_median_pct": round(med, 1),
                "base_rate_p50": base["p50"],
                "conditional_expected_move": cond,
                "median_gap_pp": round(med - base["p50"], 1),
                "pct_overpriced": round(100 * over / n, 1) if n else 0,
                "pct_underpriced": round(100 * under / n, 1) if n else 0,
                "pct_in_range": round(100 * in_range / n, 1) if n else 0,
                "systematic_flag": (
                    "SYSTEMATIC_OVER" if over / n > 0.6 else "SYSTEMATIC_UNDER" if under / n > 0.4 else "MIXED"
                ),
            }
        )

    # ── Miscalibration flags ────────────────────────────────────────────────
    overpriced = sorted(
        [n for n in names if n["miscal_label"] == "OVERPRICED"],
        key=lambda x: -x["gap_pp"],
    )[:20]
    underpriced = sorted(
        [n for n in names if n["miscal_label"] == "UNDERPRICED"],
        key=lambda x: x["gap_pp"],
    )[:20]

    # ── Universe summary ────────────────────────────────────────────────────
    total_with_priced = len(names)
    n_over = sum(1 for n in names if n["miscal_label"] == "OVERPRICED")
    n_under = sum(1 for n in names if n["miscal_label"] == "UNDERPRICED")
    n_in = total_with_priced - n_over - n_under
    all_gaps = [n["gap_pp"] for n in names]
    universe_summary = {
        "n_with_priced_move": total_with_priced,
        "n_overpriced": n_over,
        "n_underpriced": n_under,
        "n_in_range": n_in,
        "pct_overpriced": round(100 * n_over / total_with_priced, 1) if total_with_priced else 0,
        "pct_underpriced": round(100 * n_under / total_with_priced, 1) if total_with_priced else 0,
        "gap_pp_mean": round(statistics.mean(all_gaps), 1) if all_gaps else None,
        "gap_pp_median": round(statistics.median(all_gaps), 1) if all_gaps else None,
        "gap_pp_stdev": round(statistics.stdev(all_gaps), 1) if len(all_gaps) > 1 else None,
        "thresholds_used": {
            "overpriced_if_ratio_above": 1 + _OVER_THRESH,
            "underpriced_if_ratio_below": 1 - _UNDER_THRESH,
        },
    }

    # ── JSON output ─────────────────────────────────────────────────────────
    output = {
        "label": "EVENT_EV_SHADOW_DIAGNOSTIC_ONLY_NO_ALPHA_PROMOTION",
        "as_of_date": as_of or snapshot_dir.name,
        "governance": {
            "read_only": True,
            "no_scoring_integration": True,
            "no_alpha_promotion": True,
            "freeze_safe": True,
        },
        "universe_summary": universe_summary,
        "cohort_summaries": cohort_summaries,
        "overpriced_names": overpriced,
        "underpriced_names": underpriced,
    }

    json_path = snapshot_dir / "event_ev_shadow_diagnostic.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {json_path}")

    # ── Markdown output ─────────────────────────────────────────────────────
    lines: list[str] = [
        "# Event EV Shadow Diagnostic",
        "",
        f"**Date:** {as_of or snapshot_dir.name}  ",
        "**Label:** `EVENT_EV_SHADOW_DIAGNOSTIC_ONLY_NO_ALPHA_PROMOTION`  ",
        "**Governance:** Read-only diagnostic — outputs do NOT feed ranker/selector/scoring",
        "",
        "---",
        "",
        "## Universe Summary",
        "",
        f"- Names with priced move: **{total_with_priced}**",
        f"- Overpriced (implied > base×{1+_OVER_THRESH:.0%}): **{n_over}** ({universe_summary['pct_overpriced']}%)",
        f"- Underpriced (implied < base×{1-_UNDER_THRESH:.0%}): **{n_under}** ({universe_summary['pct_underpriced']}%)",
        f"- In range: **{n_in}**",
        f"- Mean gap (implied − base p50): **{universe_summary['gap_pp_mean']} pp**",
        f"- Median gap: **{universe_summary['gap_pp_median']} pp**",
        "",
        "---",
        "",
        "## Cohort Breakdown",
        "",
        "| Cohort | N | Implied Median | Base p50 | Cond. EV | Gap (pp) | % Over | % Under | Flag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cs in cohort_summaries:
        cond = f"{cs['conditional_expected_move']:.1f}%" if cs["conditional_expected_move"] else "—"
        lines.append(
            f"| {cs['cohort']} | {cs['n']} | {cs['implied_median_pct']}% "
            f"| {cs['base_rate_p50']}% | {cond} | {cs['median_gap_pp']:+.1f} "
            f"| {cs['pct_overpriced']}% | {cs['pct_underpriced']}% | {cs['systematic_flag']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Most Overpriced Names (implied > base by >{:.0%})".format(_OVER_THRESH),
        "",
        "| Ticker | Cohort | Implied | Base p50 | Gap | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for n in overpriced[:15]:
        lines.append(
            f"| {n['ticker']} | {n['cohort']} | {n['priced_move_pct']:.1f}% "
            f"| {n['base_rate_p50']:.0f}% | +{n['gap_pp']:.1f} pp "
            f"| {n['catalyst_date_precision']} |"
        )

    if underpriced:
        lines += [
            "",
            "## Most Underpriced Names (implied < base by >{:.0%})".format(_UNDER_THRESH),
            "",
            "| Ticker | Cohort | Implied | Base p50 | Gap | Precision |",
            "|---|---|---|---|---|---|",
        ]
        for n in underpriced[:15]:
            lines.append(
                f"| {n['ticker']} | {n['cohort']} | {n['priced_move_pct']:.1f}% "
                f"| {n['base_rate_p50']:.0f}% | {n['gap_pp']:.1f} pp "
                f"| {n['catalyst_date_precision']} |"
            )
    else:
        lines += ["", "## Underpriced Names", "", "None above threshold."]

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **Overpriced**: implied move exceeds historical base rate — flags cohort as a calibration review candidate for trap risk, NOT a trade signal",
        "- **Underpriced**: implied move is below historical base rate — highest-priority calibration review candidate, NOT an alpha signal",
        "- EES scores (`ees_v2_score`, `ees_v3_score`) appear in per-name output for reference; they have NOT been validated against future returns or catalyst outcomes",
        "- `priced_move_pct` is the straddle-implied move from the options surface (already in rankings.csv)",
        "- Base rates are static historical medians from `event_ev/expectation_error_model.py:_BASE_RATE_TABLE`",
        "- Conditional EV is the probability-weighted expected abs move from `_CONDITIONAL_MOVE_TABLE`",
        "- EES_STATUS: CALIBRATION_DIAGNOSTIC_BUILT | PREDICTIVE_STATUS: UNPROVEN | PROMOTION_STATUS: DO_NOT_PROMOTE",
    ]

    md_path = snapshot_dir / "event_ev_shadow_diagnostic.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")

    # ── Print summary to stdout ─────────────────────────────────────────────
    print(f"\nSnapshot: {snapshot_dir.name}")
    print(f"Names with priced move: {total_with_priced}")
    print(f"  Overpriced:  {n_over} ({universe_summary['pct_overpriced']}%)")
    print(f"  In range:    {n_in}")
    print(f"  Underpriced: {n_under} ({universe_summary['pct_underpriced']}%)")
    print(f"Mean gap: {universe_summary['gap_pp_mean']} pp  Median: {universe_summary['gap_pp_median']} pp")
    print("\nCohort breakdown:")
    for cs in cohort_summaries:
        print(
            f"  {cs['cohort']:30s}  n={cs['n']:3d}  "
            f"implied_med={cs['implied_median_pct']:5.1f}%  "
            f"base_p50={cs['base_rate_p50']:4.0f}%  "
            f"gap={cs['median_gap_pp']:+5.1f}pp  {cs['systematic_flag']}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        snap = Path(sys.argv[1])
    else:
        # Default to most recent snapshot
        snaps_dir = Path(__file__).parent.parent / "data" / "snapshots"
        candidates = sorted(
            (p for p in snaps_dir.iterdir() if p.is_dir() and len(p.name) == 10 and p.name.count("-") == 2),
            key=lambda p: p.name,
            reverse=True,
        )
        if not candidates:
            print("No dated snapshot dirs found", file=sys.stderr)
            sys.exit(1)
        snap = candidates[0]

    print(f"Running Event EV shadow diagnostic on {snap}")
    run(snap)
