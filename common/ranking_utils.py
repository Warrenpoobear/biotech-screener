"""Ranking CSV backfill and safe-float utilities.

Shared between eval_forward_returns.py, rerank_snapshots.py, and any
other module that needs to re-sort historical snapshot rankings through
a DecisionRuleset.

Extracted from scripts/research/rerank_snapshots.py to avoid eval
depending on a "research" module.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional


def safe_float(v: Any) -> Optional[float]:
    """Parse a string/value to float, returning None on empty/NaN/invalid."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (ValueError, TypeError):
        return None


def _z_score_by_group(
    rows: List[Dict[str, str]],
    value_col: str,
    group_col: str,
    out_col: str,
) -> None:
    """Compute per-group z-scores (ddof=0) and write to out_col."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        g = r.get(group_col, "") or "unknown"
        groups.setdefault(g, []).append(i)

    for g, indices in groups.items():
        vals = []
        for i in indices:
            v = safe_float(rows[i].get(value_col))
            if v is not None:
                vals.append((i, v))

        if len(vals) < 2:
            for i in indices:
                rows[i][out_col] = ""
            continue

        raw = [v for _, v in vals]
        mu = statistics.mean(raw)
        # Population std (ddof=0)
        var = sum((x - mu) ** 2 for x in raw) / len(raw)
        std = var ** 0.5

        for i in indices:
            v = safe_float(rows[i].get(value_col))
            if v is not None and std > 0:
                rows[i][out_col] = str(round((v - mu) / std, 6))
            else:
                rows[i][out_col] = ""


def backfill_columns(rows: List[Dict[str, str]]) -> None:
    """Backfill missing sort-signal columns with safe defaults.

    Historical snapshots may lack columns added in later rulesets
    (clinical_score_z_tier, coinvest_score_z, etc.).  This function
    fills them so that ``compute_actionable_sort_key()`` can operate
    on any vintage of rankings.csv.
    """
    if not rows:
        return

    sample = rows[0]

    # alpha_cohort_pct: use score_rank_pct as proxy if missing
    if "alpha_cohort_pct" not in sample or not sample.get("alpha_cohort_pct"):
        for r in rows:
            r["alpha_cohort_pct"] = r.get("score_rank_pct", "") or r.get("clinical_optionality_pct_dev", "") or ""

    # clinical_score_z: z-score of clinical_score by archetype cohort
    if "clinical_score_z" not in sample:
        def _cohort(r: Dict[str, str]) -> str:
            a = r.get("archetype", "")
            if a.startswith("drug_developer"):
                return "drug_developer"
            elif a.startswith("commercial_"):
                return a
            return "other"

        for r in rows:
            r["_cohort"] = _cohort(r)
        _z_score_by_group(rows, "clinical_score", "_cohort", "clinical_score_z")
        for r in rows:
            r.pop("_cohort", None)

    # clinical_score_z_tier: z-score by tier within archetype
    if "clinical_score_z_tier" not in sample:
        for r in rows:
            arch = r.get("archetype", "")
            if arch.startswith("drug_developer"):
                tier = r.get("tier_dev", "")
            elif arch.startswith("commercial_"):
                tier = r.get("tier_commercial", "") or r.get("tier_dev", "")
            else:
                tier = ""
            r["_tier_group"] = f"{arch}_{tier}"
        _z_score_by_group(rows, "clinical_score", "_tier_group", "clinical_score_z_tier")
        for r in rows:
            r.pop("_tier_group", None)

    # coinvest_score_z: default 0 (no penalty, no boost)
    if "coinvest_score_z" not in sample:
        for r in rows:
            r["coinvest_score_z"] = "0"

    # inst_delta_z: default 0
    if "inst_delta_z" not in sample:
        for r in rows:
            r["inst_delta_z"] = "0"
            r["inst_delta_net"] = "0"
            r["inst_delta_new"] = "0"
            r["inst_delta_exit"] = "0"

    # missingness_penalty: default 0
    if "missingness_penalty" not in sample:
        for r in rows:
            r["missingness_penalty"] = "0"

    # missing_components: default empty
    if "missing_components" not in sample:
        for r in rows:
            r["missing_components"] = ""

    # commercial_quality_pct: default empty
    if "commercial_quality_pct" not in sample:
        for r in rows:
            r["commercial_quality_pct"] = ""

    # stage_bucket: infer from archetype if missing
    if "stage_bucket" not in sample:
        for r in rows:
            r["stage_bucket"] = ""

    # sponsor_tier1_count: default 0
    if "sponsor_tier1_count" not in sample:
        for r in rows:
            r["sponsor_tier1_count"] = r.get("de_tier1_count", "0") or "0"

    # catalyst_mode: use de_catalyst_mode if available
    if "catalyst_mode" not in sample:
        for r in rows:
            r["catalyst_mode"] = r.get("de_catalyst_mode", "") or ""

    # Clinical Calendar Alpha v2 backfill defaults
    if "clinical_score_v2" not in sample:
        for r in rows:
            r["clinical_score_v2"] = r.get("clinical_score", "")
    if "competitive_intensity_z" not in sample:
        for r in rows:
            r["competitive_intensity_z"] = "0"
    if "sizing_multiplier_clinical" not in sample:
        for r in rows:
            r["sizing_multiplier_clinical"] = "1.0"
    if "clinical_score_v2_z" not in sample:
        for r in rows:
            r["clinical_score_v2_z"] = "0"
    if "de_vol_60d" not in sample:
        for r in rows:
            r["de_vol_60d"] = ""
