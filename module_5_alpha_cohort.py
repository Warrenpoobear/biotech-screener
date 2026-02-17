"""Alpha cohort scoring — table-driven alternative ranking signal.

Assigns each ticker to a cell in a (stage_bucket × catalyst_horizon_band ×
clinical_z_sign) grid, looks up historical mean excess return, and applies
James–Stein-style shrinkage + clipping to produce ``alpha_cohort_raw``.
Percentile ranks are then assigned deterministically.

Activated via ``sort_anchor="alpha_cohort"`` in the DecisionRuleset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Horizon band boundaries (upper-exclusive except last)
# ---------------------------------------------------------------------------
_HORIZON_BANDS = [
    (0, 31, "near_0_30"),
    (31, 91, "near_31_90"),
    (91, 181, "near_91_180"),
    (181, 271, "near_181_270"),
    (271, 541, "far_271_540"),
]

_VALID_STAGES = {"early", "mid", "late"}
_VALID_SIGNS = {"pos", "nonpos"}
_VALID_HORIZONS = {b[2] for b in _HORIZON_BANDS} | {"none"}


def load_alpha_cohort_table(path: Path) -> dict:
    """Load and validate an alpha cohort lookup table from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("cells"), dict):
        raise ValueError(f"alpha_cohort_table missing 'cells' dict: {path}")
    return data


def _classify_horizon(catalyst_mode: str, catalyst_days: Any) -> str:
    """Map catalyst_mode + catalyst_days to a horizon band string."""
    if catalyst_mode == "far_window":
        return "far_271_540"
    if catalyst_mode not in ("specific_days", "blended_window"):
        return "none"
    # Parse days
    if catalyst_days is None or catalyst_days == "":
        return "none"
    try:
        days = int(catalyst_days)
    except (ValueError, TypeError):
        return "none"
    for lo, hi, band in _HORIZON_BANDS:
        if lo <= days < hi:
            return band
    # days >= 541 → far bucket
    if days >= 271:
        return "far_271_540"
    return "none"


def compute_alpha_cohort_key(row: Dict[str, Any]) -> str:
    """Derive the cohort key from a row's fields.

    Returns a pipe-delimited string: ``stage|horizon|sign``.
    """
    stage = str(row.get("stage_bucket", "") or "").strip()
    if stage not in _VALID_STAGES:
        stage = "early"

    catalyst_mode = str(row.get("catalyst_mode", "") or "")
    catalyst_days = row.get("catalyst_days", "")
    horizon = _classify_horizon(catalyst_mode, catalyst_days)

    # clinical_score_z_tier > 0 → "pos", else "nonpos"
    cz_raw = row.get("clinical_score_z_tier")
    if cz_raw is None or cz_raw == "":
        sign = "nonpos"
    else:
        try:
            sign = "pos" if float(cz_raw) > 0 else "nonpos"
        except (ValueError, TypeError):
            sign = "nonpos"

    return f"{stage}|{horizon}|{sign}"


def compute_alpha_raw(
    key: str,
    table: dict,
    shrink_k: float,
    clip_min: float,
    clip_max: float,
) -> float:
    """Lookup cell mean, apply shrinkage and clipping.

    Shrinkage: ``w = n / (n + shrink_k)``, ``alpha = mean * w``.
    Missing key or n=0 → alpha=0.0.
    """
    cells = table.get("cells", {})
    cell = cells.get(key)
    if cell is None:
        return 0.0
    n = cell.get("n", 0)
    if n <= 0:
        return 0.0
    mean = cell.get("mean_excess_ret_6m", 0.0)
    w = n / (n + shrink_k)
    alpha = mean * w
    return max(clip_min, min(clip_max, alpha))


def attach_alpha_scores(
    csv_rows: List[Dict[str, Any]],
    table: dict,
    shrink_k: float,
    clip_min: float,
    clip_max: float,
) -> None:
    """Compute and attach alpha cohort scores to each row in-place.

    Sets: ``alpha_cohort_key``, ``alpha_cohort_raw``, ``alpha_cohort_pct``.
    Percentile: sort by (alpha_cohort_raw DESC, ticker ASC), assign
    ``pct = (rank - 0.5) / N`` rounded to 6dp. Rank 1 = highest alpha.
    """
    if not csv_rows:
        return

    # Step 1: compute key + raw for each row
    for row in csv_rows:
        key = compute_alpha_cohort_key(row)
        raw = compute_alpha_raw(key, table, shrink_k, clip_min, clip_max)
        row["alpha_cohort_key"] = key
        row["alpha_cohort_raw"] = raw

    # Step 2: deterministic percentile assignment
    n = len(csv_rows)
    # Sort: highest alpha first, then alphabetical ticker for ties
    ranked = sorted(
        csv_rows,
        key=lambda r: (-r["alpha_cohort_raw"], r.get("ticker", "")),
    )
    for i, row in enumerate(ranked, start=1):
        row["alpha_cohort_pct"] = round((i - 0.5) / n, 6)
