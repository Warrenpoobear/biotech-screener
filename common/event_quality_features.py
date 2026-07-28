"""Event Quality Features — per-family quality decomposition for binary catalysts.

Produces family-specific quality columns that complement the composite
binary_quality_score.  These are **passive** (informational only, no ranking
change) and intended for downstream return-attribution analysis.

Columns added:
  regulatory_quality  — 0–1 score for REGULATORY events (ADCOM, source, timing)
  clinical_quality    — 0–1 score for CLINICAL events (phase, design, program depth)
  has_adcom           — 1 if any ADCOM event contributes to the catalyst, else 0
  single_asset_risk   — 1 if program_count == 1 (binary concentrated), else 0

Clinical 91-180 quality columns (CLINICAL family, informational + optional sort):
  clinical_days_precision     — DAY|WEEK|MONTH|QUARTER|HALF_YEAR|YEAR|UNKNOWN
  clinical_date_confidence    — 0–1 confidence proxy from precision + source
  clinical_design_quality     — 0–1 from trial metadata (randomized/controlled/phase/endpoint)
  clinical_program_depth      — 0–1 inverse of single-asset risk + program count
  clinical_quality_composite  — 0–1 weighted combination of the above 4

Usage:
    from common.event_quality_features import compute_event_quality_features
    features = compute_event_quality_features(row)
    row.update(features)
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Regulatory quality components
# ---------------------------------------------------------------------------

_REG_EVENT_SCORE: Dict[str, float] = {
    "PDUFA": 1.0,
    "FDA_PDUFA_DATE": 1.0,
    "FDA_ADCOM": 0.95,
    "FDA_DECISION": 0.90,
    "FDA_APPROVAL": 0.85,
    "EMA_AGENDA": 0.80,
    "EMA_OUTCOME": 0.80,
    "EMA_COMMITTEE_AGENDA": 0.75,
    "EMA_COMMITTEE_OUTCOME": 0.75,
    "FDA_SUBMISSION": 0.50,
    "FDA_DESIGNATION": 0.40,
    "FDA_CRL": 0.30,
    "FDA_RTF": 0.25,
}

_REG_SOURCE_SCORE: Dict[str, float] = {
    "SEC_8K_FILING": 1.0,
    "PDUFA_MANUAL": 0.95,
    "FDA_FEDREG": 0.90,
    "FDA_ADCOM": 0.90,
    "CTGOV_CALENDAR": 0.50,
}

_ADCOM_EVENT_TYPES = frozenset({"FDA_ADCOM"})


def _regulatory_quality(event_type: str, source: str) -> float:
    """Score regulatory catalyst quality [0, 1].

    Components:
      0.60 * event_type score (PDUFA > ADCOM > EMA > submission)
      0.40 * source reliability (SEC 8-K > manual > Fed Register > CTgov)
    """
    evt = _REG_EVENT_SCORE.get(event_type, 0.3)
    src = _REG_SOURCE_SCORE.get(source, 0.4)
    return round(0.60 * evt + 0.40 * src, 4)


# ---------------------------------------------------------------------------
# Clinical quality components
# ---------------------------------------------------------------------------

_CLIN_EVENT_SCORE: Dict[str, float] = {
    "DATA_READOUT": 0.90,
    "CT_DATE_CONFIRMED_ACTUAL": 0.85,
    "DATA_PRESENTATION": 0.80,
    "CT_PRIMARY_COMPLETION": 0.65,
    "CT_STUDY_COMPLETION": 0.50,
    "CT_RESULTS_POSTED": 0.45,
    "DATA_PUBLICATION": 0.40,
    "CT_TIMELINE_PULLIN": 0.35,
    "CT_STATUS_UPGRADE": 0.30,
}

_PHASE_SCORE: Dict[float, float] = {
    3.0: 1.0,
    2.5: 0.80,
    2.0: 0.55,
    1.5: 0.35,
    1.0: 0.15,
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _clinical_quality(
    event_type: str,
    phase_str: str,
    design_quality: str,
    program_count_str: str,
) -> float:
    """Score clinical catalyst quality [0, 1].

    Components:
      0.30 * event_type (DATA_READOUT > PCD > completion)
      0.30 * phase (phase 3 > 2 > 1)
      0.20 * design_quality (reused from calendar alpha v2)
      0.20 * program depth (multi-program > single)
    """
    evt = _CLIN_EVENT_SCORE.get(event_type, 0.3)

    phase_val = _safe_float(phase_str, 0.0)
    phase = _PHASE_SCORE.get(phase_val, 0.3)

    design = _safe_float(design_quality, 0.3)
    design = max(0.0, min(1.0, design))

    prog = _safe_float(program_count_str, 1.0)
    # multi-program: more diversified, lower single-name risk
    depth = min(1.0, prog / 5.0) if prog > 0 else 0.0

    return round(0.30 * evt + 0.30 * phase + 0.20 * design + 0.20 * depth, 4)


# ---------------------------------------------------------------------------
# Clinical 91-180 quality features
# ---------------------------------------------------------------------------

PRECISION_LEVELS = ("DAY", "WEEK", "MONTH", "QUARTER", "HALF_YEAR", "YEAR", "UNKNOWN")

# catalyst_mode → precision mapping
_MODE_PRECISION: Dict[str, str] = {
    "specific_days": "DAY",
    "blended_window": "MONTH",
    "far_window": "QUARTER",
    "no_upcoming": "UNKNOWN",
    "missing": "UNKNOWN",
}

# source → precision override (higher-confidence sources → finer precision)
_SOURCE_PRECISION: Dict[str, str] = {
    "SEC_8K_FILING": "DAY",
    "SEC_MULTI_FORM": "DAY",
    "PDUFA_MANUAL": "DAY",
    "FDA_CALENDAR": "DAY",
    "FDA_ADCOM_CALENDAR": "DAY",
    "FEDERAL_REGISTER": "DAY",
    "CTGOV_CALENDAR": "MONTH",
    "CTGOV_PCD_FAR": "QUARTER",
    "CORPORATE_CALENDAR": "WEEK",
}

_PRECISION_CONFIDENCE: Dict[str, float] = {
    "DAY": 0.95,
    "WEEK": 0.80,
    "MONTH": 0.60,
    "QUARTER": 0.40,
    "HALF_YEAR": 0.25,
    "YEAR": 0.15,
    "UNKNOWN": 0.10,
}

# Coarseness ranking — higher is weaker/less precise. Used to pick the more
# conservative of two precision claims.
PRECISION_WEAKNESS: Dict[str, int] = {
    "DAY": 0,
    "WEEK": 1,
    "MONTH": 2,
    "QUARTER": 3,
    "HALF_YEAR": 4,
    "YEAR": 5,
    "UNKNOWN": 6,
}


def source_precision_floor(source: str) -> str:
    """Weakest precision a source can legitimately support.

    Reads ``_SOURCE_PRECISION`` so that table stays the single authority on
    per-source precision. An unrecognised source returns ``UNKNOWN`` — never
    ``DAY``, since defaulting to the strongest claim is the defect this guards
    against (see specs/changes/spec_114_catalyst_date_precision_provenance_2026_07_28.md).

    Diagnostic/shadow use only — no routing or scoring path consumes this.
    """
    return _SOURCE_PRECISION.get(source, "UNKNOWN")


def weakest_precision(*precisions: str) -> str:
    """Most conservative (coarsest) of the given precision labels.

    Unrankable labels are ignored; if nothing is rankable the result is
    ``UNKNOWN``.
    """
    known = [p for p in precisions if p in PRECISION_WEAKNESS]
    if not known:
        return "UNKNOWN"
    return max(known, key=lambda p: PRECISION_WEAKNESS[p])


# source reliability bonus (stacks with precision)
_SOURCE_CONFIDENCE_BONUS: Dict[str, float] = {
    "SEC_8K_FILING": 0.05,
    "SEC_MULTI_FORM": 0.05,
    "PDUFA_MANUAL": 0.05,
    "FDA_CALENDAR": 0.05,
}


def compute_clinical_days_precision(
    catalyst_mode: str,
    catalyst_source: str,
    corroborated: bool = True,
) -> str:
    """Derive date precision from catalyst_mode and source.

    When ``corroborated`` is False and source is a noisy clinical source,
    precision is capped at MONTH regardless of the source's native precision.
    This prevents false exact-date trust from empirically unreliable sources.
    """
    from common.clinical_corroboration import DOWNGRADED_PRECISION, should_downgrade_precision

    # Source override takes priority (e.g., SEC_8K always DAY)
    src_prec = _SOURCE_PRECISION.get(catalyst_source)
    mode_prec = _MODE_PRECISION.get(catalyst_mode, "UNKNOWN")
    if src_prec:
        # Pick the finer of source vs mode precision
        src_idx = PRECISION_LEVELS.index(src_prec) if src_prec in PRECISION_LEVELS else 6
        mode_idx = PRECISION_LEVELS.index(mode_prec) if mode_prec in PRECISION_LEVELS else 6
        precision = PRECISION_LEVELS[min(src_idx, mode_idx)]
    else:
        precision = mode_prec

    # Corroboration gate: noisy clinical sources get capped at MONTH
    if should_downgrade_precision(catalyst_source, "CLINICAL", corroborated):
        prec_idx = PRECISION_LEVELS.index(precision) if precision in PRECISION_LEVELS else 6
        down_idx = PRECISION_LEVELS.index(DOWNGRADED_PRECISION) if DOWNGRADED_PRECISION in PRECISION_LEVELS else 6
        # Only downgrade if current precision is finer than the cap
        if prec_idx < down_idx:
            precision = DOWNGRADED_PRECISION

    return precision


def compute_clinical_date_confidence(
    precision: str,
    catalyst_source: str,
    pcd_overdue: bool = False,
    catalyst_event_type: str = "",
) -> float:
    """Confidence proxy [0, 1] from precision + source reliability.

    Applies a penalty when the primary completion date (PCD) has passed
    without results, indicating a serially-slipping trial. Also penalizes
    soft catalyst types (CT_STUDY_COMPLETION) which are completion-date
    derived and historically unreliable.
    """
    base = _PRECISION_CONFIDENCE.get(precision, 0.10)
    bonus = _SOURCE_CONFIDENCE_BONUS.get(catalyst_source, 0.0)
    conf = min(1.0, base + bonus)

    # Penalty for overdue PCD: trial date has passed without resolution.
    # This catches serially-slipping trials where the system default (0.95)
    # overstates reliability.
    if pcd_overdue:
        conf *= 0.60  # 40% haircut

    # Penalty for soft catalyst types derived from completion dates
    # (less reliable than hard catalysts like PDUFA dates)
    if catalyst_event_type in ("CT_STUDY_COMPLETION", "CT_PRIMARY_COMPLETION"):
        conf *= 0.85  # 15% haircut

    return round(max(0.05, conf), 4)


def compute_clinical_design_quality(row: Dict[str, Any]) -> float:
    """Design quality [0, 1] from trial metadata already on the row.

    Components (reuses existing fields, no new data needed):
      0.35 * design_quality_score (randomized, blinded, controlled)
      0.30 * phase score (phase 3 > 2 > 1)
      0.20 * endpoint_strength_score (TA-based proxy)
      0.15 * (1 if phase >= 2.5 else 0.5)  — confirmatory vs exploratory proxy
    """
    design = _safe_float(row.get("design_quality_score"), 0.3)
    design = max(0.0, min(1.0, design))

    phase_val = _safe_float(row.get("lead_program_phase"), 0.0)
    phase = _PHASE_SCORE.get(phase_val, 0.3)

    endpoint = _safe_float(row.get("endpoint_strength_score"), 0.5)
    endpoint = max(0.0, min(1.0, endpoint))

    confirmatory = 1.0 if phase_val >= 2.5 else 0.5

    return round(0.35 * design + 0.30 * phase + 0.20 * endpoint + 0.15 * confirmatory, 4)


def compute_clinical_program_depth(row: Dict[str, Any]) -> float:
    """Program depth [0, 1] — inverse of single-asset risk + program count."""
    prog = _safe_float(row.get("program_count"), 1.0)
    single = _safe_float(row.get("single_asset_risk"), 0.0)

    # Multi-program diversity: more programs → higher depth
    count_score = min(1.0, prog / 4.0) if prog > 0 else 0.0
    # Penalty for single-asset risk
    risk_penalty = 0.3 if single == 1 else 0.0

    return round(max(0.0, count_score - risk_penalty), 4)


def compute_clinical_quality_composite(
    date_confidence: float,
    design_quality: float,
    program_depth: float,
) -> float:
    """Weighted composite [0, 1] of the 3 clinical quality dimensions.

    Weights: 0.40 date_confidence + 0.40 design_quality + 0.20 program_depth
    Date confidence gets equal weight to design because precision is the #1
    predictor of whether the catalyst is real and tradeable.
    """
    return round(0.40 * date_confidence + 0.40 * design_quality + 0.20 * program_depth, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Columns produced by compute_event_quality_features
EVENT_QUALITY_COLUMNS = [
    "regulatory_quality",
    "clinical_quality",
    "has_adcom",
    "single_asset_risk",
]

# Additional clinical 91-180 quality columns
CLINICAL_91_180_QUALITY_COLUMNS = [
    "clinical_days_precision",
    "clinical_date_confidence",
    "clinical_design_quality",
    "clinical_program_depth",
    "clinical_quality_composite",
]


def compute_event_quality_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-family quality features from a rankings row.

    Returns a dict with EVENT_QUALITY_COLUMNS keys.
    """
    family = str(row.get("catalyst_family", "") or "")
    event_type = str(row.get("catalyst_event_type", "") or "")
    source = str(row.get("catalyst_source", "") or "")
    phase_str = str(row.get("lead_program_phase", "") or "")
    design_str = str(row.get("design_quality_score", "") or "")
    prog_str = str(row.get("program_count", "") or "")

    # Regulatory quality (only meaningful for REGULATORY family)
    reg_q = _regulatory_quality(event_type, source) if family == "REGULATORY" else 0.0

    # Clinical quality (only meaningful for CLINICAL family)
    clin_q = _clinical_quality(event_type, phase_str, design_str, prog_str) if family == "CLINICAL" else 0.0

    # ADCOM flag: is the nearest event an ADCOM?
    has_adcom = 1 if event_type in _ADCOM_EVENT_TYPES else 0

    # Single-asset risk: company has only 1 clinical program
    prog_count = _safe_float(prog_str, 0.0)
    single_asset = 1 if prog_count == 1 else 0

    return {
        "regulatory_quality": reg_q,
        "clinical_quality": clin_q,
        "has_adcom": has_adcom,
        "single_asset_risk": single_asset,
    }


def compute_clinical_91_180_quality(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute clinical 91-180 quality features from a rankings row.

    Only meaningful for CLINICAL family rows. Returns neutral defaults
    for non-CLINICAL rows.
    """
    family = str(row.get("catalyst_family", "") or "")
    catalyst_mode = str(row.get("catalyst_mode", "") or "")
    catalyst_source = str(row.get("catalyst_source", "") or "")

    if family != "CLINICAL":
        return {
            "clinical_days_precision": "",
            "clinical_date_confidence": "",
            "clinical_design_quality": "",
            "clinical_program_depth": "",
            "clinical_quality_composite": "",
        }

    corroborated = row.get("catalyst_corroborated", "") != "0"
    precision = compute_clinical_days_precision(catalyst_mode, catalyst_source, corroborated)

    # Detect overdue PCD: clinical_readout_days == 0 means PCD has passed
    # but event is still listed (trial hasn't posted results).
    readout_days = _safe_float(row.get("clinical_readout_days"), -1)
    pcd_overdue = readout_days == 0 or (
        readout_days < 0 and _safe_float(row.get("lead_program_readout_days"), -1) > 180
    )
    catalyst_event_type = str(row.get("catalyst_event_type", "") or "")

    date_conf = compute_clinical_date_confidence(
        precision,
        catalyst_source,
        pcd_overdue=pcd_overdue,
        catalyst_event_type=catalyst_event_type,
    )
    design_q = compute_clinical_design_quality(row)
    prog_depth = compute_clinical_program_depth(row)
    composite = compute_clinical_quality_composite(date_conf, design_q, prog_depth)

    return {
        "clinical_days_precision": precision,
        "clinical_date_confidence": round(date_conf, 4),
        "clinical_design_quality": round(design_q, 4),
        "clinical_program_depth": round(prog_depth, 4),
        "clinical_quality_composite": round(composite, 4),
    }


# ---------------------------------------------------------------------------
# Options quality composite (tastytrade diagnostics → [0, 1])
# ---------------------------------------------------------------------------

OPTIONS_QUALITY_COLUMNS = [
    "options_quality_composite",
]


def compute_options_quality_composite(row: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded [0, 1] composite from tastytrade diagnostics.

    Components (additive, capped at 1.0):
      - use_for_judgment gate:  0.0 if opt_use_for_judgment != "YES" (hard zero)
      - event_premium credit:   +0.40 if opt_event_premium == "YES"
      - liquidity credit:       +0.20 if opt_liquidity_ok == "1"
      - iv_regime penalty:      -0.20 if opt_iv_regime == "EXTREME", else 0
      - term_slope bonus:       +0.20 * min(1.0, abs(slope) / 0.30) if slope < 0
      - skew context:           +0.20 * clamp(skew / 0.10, 0, 1) if skew > 0
    """
    # Hard gate: no usable chain → empty (DE defaults to 0.0)
    if str(row.get("opt_use_for_judgment", "")) != "YES":
        return {"options_quality_composite": ""}

    score = 0.0

    # Event premium (+0.40): market sees a binary event
    if str(row.get("opt_event_premium", "")) == "YES":
        score += 0.40

    # Liquidity (+0.20): baseline chain quality
    if str(row.get("opt_liquidity_state", "")) == "liquid":
        score += 0.20

    # IV regime penalty (-0.20): EXTREME chains are noise
    if str(row.get("opt_iv_regime", "")) == "EXTREME":
        score -= 0.20

    # Term slope bonus (+0.20): graduated backwardation strength
    slope_raw = row.get("opt_term_slope", "")
    if slope_raw != "" and slope_raw is not None:
        slope = _safe_float(slope_raw, default=0.0)
        if slope < 0:
            score += 0.20 * min(1.0, abs(slope) / 0.30)

    # Skew context (+0.20): put demand proxy
    skew_raw = row.get("opt_put_call_skew", "")
    if skew_raw != "" and skew_raw is not None:
        skew = _safe_float(skew_raw, default=0.0)
        if skew > 0:
            score += 0.20 * min(1.0, skew / 0.10)

    return {"options_quality_composite": round(max(0.0, min(1.0, score)), 4)}
