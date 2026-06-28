"""Options feature expansion — Stage 1: shadow scores and expectation gap.

Classification: OPTIONS_SHADOW_EXPANSION / EXPECTATION_LAYER_PLUMBING
Constraints (enforced by construction):
  NO_MODEL_CHANGE, NO_RANKER_CHANGE, NO_SELECTOR_CHANGE,
  NO_SIZING_CHANGE, NO_TRADING_ACTION, SHADOW_ONLY

Adds the following shadow fields to each rankings row:
  options_quality_score     — float [0, 1]: overall chain usability
  options_quality_status    — verdict: OPTIONS_USABLE|THIN|STALE|
                              VENDOR_DISAGREE|NO_EVENT_EXPIRY|UNUSABLE|MISSING
  event_premium_iv_pp       — float: (front_iv - back_iv) * 100 in pp
  event_premium_ratio       — float: front_iv / back_iv (NaN if back=0)
  expectation_gap_score     — float: model_opportunity_z - priced_move_z
  options_shadow_verdict    — verdict: OPTIONS_CONFIRMED|CROWDING_WARN|
                              EVENT_ALREADY_PRICED|DATA_WARN|NO_OPTIONS_DATA|NEUTRAL

None of these fields affects actionable_rank, composite_score, or any
selector / ranker / sizing gate.

Forward-validation cohort labels (written to sidecar, not rankings.csv):
  core_top30               — actionable_rank <= 30
  options_confirmed_top30  — core_top30 AND options_shadow_verdict == OPTIONS_CONFIRMED
  event_already_priced     — options_shadow_verdict == EVENT_ALREADY_PRICED
  options_unusable         — quality status in {UNUSABLE, MISSING, STALE}
  positive_expectation_gap — expectation_gap_score > 0 AND status == OPTIONS_USABLE
  negative_expectation_gap — expectation_gap_score < 0 AND status == OPTIONS_USABLE
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "options_features"

# ─── Quality status constants ─────────────────────────────────────────────────

OPTIONS_MISSING = "OPTIONS_MISSING"
OPTIONS_STALE = "OPTIONS_STALE"
OPTIONS_UNUSABLE = "OPTIONS_UNUSABLE"
OPTIONS_VENDOR_DISAGREE = "OPTIONS_VENDOR_DISAGREE"
OPTIONS_THIN = "OPTIONS_THIN"
OPTIONS_NO_EVENT_EXPIRY = "OPTIONS_NO_EVENT_EXPIRY"
OPTIONS_USABLE = "OPTIONS_USABLE"

# ─── Shadow verdict constants ─────────────────────────────────────────────────

VERDICT_NO_DATA = "NO_OPTIONS_DATA"
VERDICT_DATA_WARN = "DATA_WARN"
VERDICT_CROWDING_WARN = "CROWDING_WARN"
VERDICT_EVENT_ALREADY_PRICED = "EVENT_ALREADY_PRICED"
VERDICT_OPTIONS_CONFIRMED = "OPTIONS_CONFIRMED"
VERDICT_NEUTRAL = "NEUTRAL"

# Fields injected into csv_rows
OPTIONS_FEATURE_COLUMNS = [
    "options_quality_score",
    "options_quality_status",
    "event_premium_iv_pp",
    "event_premium_ratio",
    "expectation_gap_score",
    "options_shadow_verdict",
]

# Thresholds
_CROWDING_RATIO_THRESH = 1.75
_PRICED_MOVE_HIGH_PCTILE = 75.0  # top quartile of priced_move_pct
_SCORE_TOP_QUARTILE = 75.0  # score_rank_pct threshold for "top quartile"


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ─── Per-ticker quality assessment ───────────────────────────────────────────


def compute_options_quality(row: Dict[str, Any]) -> Tuple[float, str]:
    """Compute (options_quality_score [0-1], options_quality_status) from a row.

    Priority-ordered verdicts:
      1. no chain data            → OPTIONS_MISSING   (score=0.0)
      2. chain present but stale  → OPTIONS_STALE     (score=0.1)
      3. liquidity absent         → OPTIONS_UNUSABLE  (score=0.2)
      4. EXTREME iv regime        → OPTIONS_VENDOR_DISAGREE (score varies)
      5. thin liquidity           → OPTIONS_THIN      (score < 0.7)
      6. no event premium         → OPTIONS_NO_EVENT_EXPIRY (score < 0.9)
      7. all gates pass           → OPTIONS_USABLE    (score >= 0.6)
    """
    has_data = str(row.get("opt_has_data", "0")) == "1"
    if not has_data:
        return 0.0, OPTIONS_MISSING

    use_for_judgment = str(row.get("opt_use_for_judgment", "")) == "YES"
    if not use_for_judgment:
        return 0.1, OPTIONS_STALE

    liquidity_state = str(row.get("opt_liquidity_state", "")).strip()
    iv_regime = str(row.get("opt_iv_regime", "")).strip()
    event_premium = str(row.get("opt_event_premium", "")).strip()

    if liquidity_state == "absent":
        return 0.2, OPTIONS_UNUSABLE

    # Build score from components
    score = 0.0
    score += 0.40  # base: use_for_judgment gate passed
    score += 0.20 if liquidity_state == "liquid" else 0.08  # thin gets partial credit
    score += 0.20 if event_premium == "YES" else 0.0
    if iv_regime == "EXTREME":
        score -= 0.15
    elif iv_regime in ("NORMAL", "ELEVATED"):
        score += 0.05
    score = round(max(0.0, min(1.0, score)), 4)

    if iv_regime == "EXTREME":
        return score, OPTIONS_VENDOR_DISAGREE
    if liquidity_state == "thin":
        return score, OPTIONS_THIN
    if event_premium != "YES":
        return score, OPTIONS_NO_EVENT_EXPIRY
    return score, OPTIONS_USABLE


# ─── Event-pricing features ───────────────────────────────────────────────────


def compute_event_premium_features(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Compute (event_premium_iv_pp, event_premium_ratio) from front/back IV.

    opt_front_iv and opt_back_iv are in decimal form (0.83 = 83% IV).
    event_premium_iv_pp is in percentage points: (front - back) * 100.
    event_premium_ratio is dimensionless: front / back.

    Returns (None, None) if either IV is unavailable.
    """
    front = _safe_float(row.get("opt_front_iv"))
    back = _safe_float(row.get("opt_back_iv"))
    if front is None or back is None:
        return None, None
    iv_pp = round((front - back) * 100.0, 4)
    ratio = round(front / back, 4) if back > 0 else None
    return iv_pp, ratio


# ─── Cross-sectional expectation gap ─────────────────────────────────────────


def _zscore_series(values: List[Optional[float]]) -> List[Optional[float]]:
    """Z-score a list, returning None for missing values. Stable with std=0."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0 if v is not None else None for v in values]
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    std = math.sqrt(var)
    if std < 1e-9:
        return [0.0 if v is not None else None for v in values]
    return [round((v - mean) / std, 4) if v is not None else None for v in values]


def compute_expectation_gap_scores(rows: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Cross-sectional expectation gap: model_opportunity_z - priced_move_z.

    model_event_opportunity is proxied by score_rank_pct (already 0-100).
    priced_move_pct is the options-implied expected move in percentage points.

    Positive gap: model sees more opportunity than options already price.
    Negative gap: market has already priced the expected move or more.
    """
    opportunity_vals: List[Optional[float]] = []
    priced_move_vals: List[Optional[float]] = []

    for row in rows:
        opportunity_vals.append(_safe_float(row.get("score_rank_pct")))
        priced_move_vals.append(_safe_float(row.get("priced_move_pct")))

    opportunity_z = _zscore_series(opportunity_vals)
    priced_move_z = _zscore_series(priced_move_vals)

    gaps: List[Optional[float]] = []
    for op_z, pm_z in zip(opportunity_z, priced_move_z):
        if op_z is not None and pm_z is not None:
            gaps.append(round(op_z - pm_z, 4))
        else:
            gaps.append(None)
    return gaps


# ─── Shadow verdict ───────────────────────────────────────────────────────────


def compute_options_shadow_verdict(
    row: Dict[str, Any],
    quality_status: str,
    event_premium_ratio: Optional[float],
    expectation_gap_score: Optional[float],
    priced_move_top_quartile_threshold: float,
) -> str:
    """Assign a shadow verdict (no model impact).

    Verdict priority:
      1. No data                → NO_OPTIONS_DATA
      2. Stale / unusable data  → DATA_WARN
      3. Crowded setup          → CROWDING_WARN
      4. Already priced         → EVENT_ALREADY_PRICED
      5. Confirmed good chain   → OPTIONS_CONFIRMED (status=USABLE + priced_move available)
      6. Default                → NEUTRAL
    """
    if quality_status == OPTIONS_MISSING:
        return VERDICT_NO_DATA

    if quality_status in (OPTIONS_STALE, OPTIONS_UNUSABLE):
        return VERDICT_DATA_WARN

    priced_move = _safe_float(row.get("priced_move_pct"))
    score_rank = _safe_float(row.get("score_rank_pct"))

    # CROWDING_WARN: high event premium + high priced move but NOT top-quartile model score
    if (
        event_premium_ratio is not None
        and event_premium_ratio > _CROWDING_RATIO_THRESH
        and priced_move is not None
        and priced_move >= priced_move_top_quartile_threshold
        and (score_rank is None or score_rank < _SCORE_TOP_QUARTILE)
    ):
        return VERDICT_CROWDING_WARN

    # EVENT_ALREADY_PRICED: very high implied move and negative expectation gap
    if (
        priced_move is not None
        and priced_move >= priced_move_top_quartile_threshold
        and expectation_gap_score is not None
        and expectation_gap_score <= 0
    ):
        return VERDICT_EVENT_ALREADY_PRICED

    # OPTIONS_CONFIRMED: usable chain with priced move available
    if quality_status == OPTIONS_USABLE and priced_move is not None:
        return VERDICT_OPTIONS_CONFIRMED

    return VERDICT_NEUTRAL


# ─── Cohort builder ───────────────────────────────────────────────────────────


def build_forward_validation_cohorts(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build forward-validation cohort ticker lists.

    These cohorts are used for retrospective autopsy after 5d and 20d windows:
      Did CROWDING_WARN names underperform?
      Did OPTIONS_CONFIRMED names produce cleaner returns?
      Did positive_expectation_gap predict forward excess?
    """
    cohorts: Dict[str, List[str]] = {
        "core_top30": [],
        "options_confirmed_top30": [],
        "event_already_priced": [],
        "options_unusable": [],
        "positive_expectation_gap": [],
        "negative_expectation_gap": [],
    }

    for row in rows:
        ticker = row.get("ticker", "")
        if not ticker:
            continue

        ar = _safe_float(row.get("actionable_rank"))
        status = row.get("options_quality_status", "")
        verdict = row.get("options_shadow_verdict", "")
        gap = _safe_float(row.get("expectation_gap_score"))

        if ar is not None and ar <= 30:
            cohorts["core_top30"].append(ticker)
            if verdict == VERDICT_OPTIONS_CONFIRMED:
                cohorts["options_confirmed_top30"].append(ticker)

        if verdict == VERDICT_EVENT_ALREADY_PRICED:
            cohorts["event_already_priced"].append(ticker)

        if status in (OPTIONS_UNUSABLE, OPTIONS_MISSING, OPTIONS_STALE):
            cohorts["options_unusable"].append(ticker)

        if status == OPTIONS_USABLE and gap is not None:
            if gap > 0:
                cohorts["positive_expectation_gap"].append(ticker)
            elif gap < 0:
                cohorts["negative_expectation_gap"].append(ticker)

    return cohorts


# ─── Sidecar writers ─────────────────────────────────────────────────────────

_SIDECAR_COLUMNS = [
    "as_of_date",
    "ticker",
    "opt_has_data",
    "opt_use_for_judgment",
    "opt_liquidity_state",
    "opt_iv_regime",
    "opt_event_premium",
    "opt_front_iv",
    "opt_back_iv",
    "opt_atm_iv",
    "opt_term_slope",
    "priced_move_pct",
    "score_rank_pct",
    "options_quality_score",
    "options_quality_status",
    "event_premium_iv_pp",
    "event_premium_ratio",
    "expectation_gap_score",
    "options_shadow_verdict",
]


def write_options_features_sidecar(
    rows: List[Dict[str, Any]],
    as_of_date: str,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Path:
    """Write per-ticker options feature sidecar to artifacts/options_features/."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{as_of_date}_options_features.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SIDECAR_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**{"as_of_date": as_of_date}, **row})
    logger.info("[options_features] Wrote %d rows → %s", len(rows), out_path)
    return out_path


def write_options_cohorts_sidecar(
    cohorts: Dict[str, List[str]],
    as_of_date: str,
    n_universe: int,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Path:
    """Write forward-validation cohorts to artifacts/options_features/."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{as_of_date}_options_cohorts.json"
    payload = {
        "schema": "options_cohorts.v1",
        "as_of_date": as_of_date,
        "n_universe": n_universe,
        "cohorts": cohorts,
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("[options_features] Cohorts → %s (core_top30=%d)", out_path, len(cohorts["core_top30"]))
    return out_path


# ─── Primary entry point ──────────────────────────────────────────────────────


def enrich_csv_rows(
    csv_rows: List[Dict[str, Any]],
    as_of_date: str,
    write_sidecars: bool = True,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Dict[str, Any]:
    """Compute Stage 1 options shadow features and inject into rows in-place.

    Returns a summary dict (for logging / sidecar metadata). Does NOT affect
    actionable_rank, composite_score, or any selector / ranker field.

    Args:
        csv_rows:       The live rankings rows list (mutated in-place).
        as_of_date:     ISO date string (YYYY-MM-DD) for artifact naming.
        write_sidecars: If True, write options_features CSV + cohorts JSON.
        artifact_dir:   Override artifact directory (used in tests).

    Returns:
        Dict with n_usable, n_thin, n_stale, n_missing, n_confirmed, n_gap_positive, etc.
    """
    if not csv_rows:
        logger.warning("[options_features] Empty csv_rows — nothing to enrich")
        return {}

    # Step 1: per-ticker quality + event-premium features
    for row in csv_rows:
        score, status = compute_options_quality(row)
        row["options_quality_score"] = score
        row["options_quality_status"] = status

        iv_pp, ratio = compute_event_premium_features(row)
        row["event_premium_iv_pp"] = iv_pp if iv_pp is not None else ""
        row["event_premium_ratio"] = ratio if ratio is not None else ""

    # Step 2: cross-sectional expectation gap (needs full universe)
    gaps = compute_expectation_gap_scores(csv_rows)
    for row, gap in zip(csv_rows, gaps):
        row["expectation_gap_score"] = gap if gap is not None else ""

    # Step 3: priced_move top-quartile threshold (for verdict logic)
    pm_values = [_safe_float(r.get("priced_move_pct")) for r in csv_rows]
    pm_present = sorted(v for v in pm_values if v is not None)
    if pm_present:
        idx = int(len(pm_present) * 0.75)
        pm_top_q = pm_present[min(idx, len(pm_present) - 1)]
    else:
        pm_top_q = float("inf")

    # Step 4: shadow verdict per ticker
    for row in csv_rows:
        ratio = _safe_float(row.get("event_premium_ratio"))
        gap = _safe_float(row.get("expectation_gap_score"))
        verdict = compute_options_shadow_verdict(
            row,
            quality_status=str(row.get("options_quality_status", "")),
            event_premium_ratio=ratio,
            expectation_gap_score=gap,
            priced_move_top_quartile_threshold=pm_top_q,
        )
        row["options_shadow_verdict"] = verdict

    # Step 5: summary counts
    summary = {
        "n_universe": len(csv_rows),
        "n_usable": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_USABLE),
        "n_thin": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_THIN),
        "n_stale": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_STALE),
        "n_unusable": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_UNUSABLE),
        "n_missing": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_MISSING),
        "n_vendor_disagree": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_VENDOR_DISAGREE),
        "n_no_event_expiry": sum(1 for r in csv_rows if r.get("options_quality_status") == OPTIONS_NO_EVENT_EXPIRY),
        "n_confirmed": sum(1 for r in csv_rows if r.get("options_shadow_verdict") == VERDICT_OPTIONS_CONFIRMED),
        "n_crowding_warn": sum(1 for r in csv_rows if r.get("options_shadow_verdict") == VERDICT_CROWDING_WARN),
        "n_already_priced": sum(1 for r in csv_rows if r.get("options_shadow_verdict") == VERDICT_EVENT_ALREADY_PRICED),
        "n_gap_positive": sum(1 for r in csv_rows if _safe_float(r.get("expectation_gap_score"), 0) > 0),
        "n_gap_negative": sum(1 for r in csv_rows if _safe_float(r.get("expectation_gap_score"), 0) < 0),
    }

    logger.info(
        "[options_features] usable=%d thin=%d stale=%d unusable=%d missing=%d "
        "confirmed=%d crowding_warn=%d already_priced=%d gap+=%d gap-=%d",
        summary["n_usable"],
        summary["n_thin"],
        summary["n_stale"],
        summary["n_unusable"],
        summary["n_missing"],
        summary["n_confirmed"],
        summary["n_crowding_warn"],
        summary["n_already_priced"],
        summary["n_gap_positive"],
        summary["n_gap_negative"],
    )

    if write_sidecars:
        try:
            write_options_features_sidecar(csv_rows, as_of_date, artifact_dir=artifact_dir)
        except Exception as exc:
            logger.warning("[options_features] Sidecar write failed: %s", exc)
        try:
            cohorts = build_forward_validation_cohorts(csv_rows)
            write_options_cohorts_sidecar(cohorts, as_of_date, n_universe=len(csv_rows), artifact_dir=artifact_dir)
        except Exception as exc:
            logger.warning("[options_features] Cohorts write failed: %s", exc)

    return summary
