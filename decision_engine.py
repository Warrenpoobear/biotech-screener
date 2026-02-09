"""
Decision Engine v1 — Post-processing layer for structured investment decisions.

Consumes existing Module 5 pipeline outputs and emits per-ticker decision
fields: eligibility gates, overlay signals, sizing bands, and dev-tier labels.

This is a pure function layer with no side effects. Called from
run_screen.py → save_validation_snapshot() after building csv_rows.

Data sources on each rec:
  - rec["smart_money_signal"]  → tier1_holders, holders_increasing/decreasing,
                                  overlap_count, tier_breakdown
  - rec["coinvest"]            → tier1_count
  - rec["catalyst_decay"]      → days_to_catalyst, in_optimal_window (top-level)
  - rec["defensive_features"]  → vol_60d, beta_xbi_60d, drawdown, rsi_14d
  - rec["score_breakdown"]["enhancements"]["momentum"] → alpha_60d
  - rec["momentum_signal"]     → alpha_60d (fallback)
  - rec["fundamental_red_flag"] / rec["severity"] / rec["confidence_overall"]
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# VERSIONING
# =============================================================================

# Bump VERSION when thresholds, gates, or tier logic change.
# Bump RULESET_ID (or recompute) when any parameter changes.
VERSION = "v1.0.0"

# Deterministic ruleset hash: computed from the threshold constants below.
# Recomputed at import time so any parameter change auto-updates the ID.
def _compute_ruleset_id() -> str:
    """Short hash of all tunable parameters for change detection."""
    params = (
        f"DRAWDOWN_GATE={-0.40}|"
        f"VOL_HIGH={1.20}|BETA_HIGH={1.80}|DD_FLAG={-0.35}|RSI_OB={70}|"
        f"CONF_LOW={0.30}|"
        f"ALPHA_TW={0.05}|ALPHA_HW={-0.05}|"
        f"TIER_A={0.60}|TIER_B={0.30}|"
        f"SPONSOR_CONFIRM=2"
    )
    return hashlib.sha256(params.encode()).hexdigest()[:8]

RULESET_ID = _compute_ruleset_id()


# =============================================================================
# THRESHOLDS
# =============================================================================

# Layer 0 — Eligibility gates (hard gates only)
DRAWDOWN_GATE = -0.40            # Worse than -40% drawdown → ineligible

# Layer 2 — Risk flag thresholds
VOL_HIGH_THRESHOLD = 1.20        # Annualized vol > 120%
BETA_HIGH_THRESHOLD = 1.80       # Beta to XBI > 1.8
DRAWDOWN_FLAG_THRESHOLD = -0.35  # Drawdown worse than -35%
RSI_OVERBOUGHT = 70              # RSI > 70
CONFIDENCE_LOW_THRESHOLD = 0.30  # Confidence < 30% → risk flag (not hard gate)

# Layer 2 — Momentum classification
ALPHA_TAILWIND = 0.05            # alpha_60d > +5%
ALPHA_HEADWIND = -0.05           # alpha_60d < -5%

# Layer 4 — Tier cutoffs
TIER_A_OPTIONALITY_FLOOR = 0.60  # Top 40% → optionality_pct >= 0.60
TIER_B_OPTIONALITY_FLOOR = 0.30  # Top 70% → optionality_pct >= 0.30


# =============================================================================
# HELPERS
# =============================================================================

def _safe_float(val, default=None):
    """Convert a value to float, handling None/str/Decimal."""
    if val is None:
        return default
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return default


# Sentinel for "data not present" vs "present and zero"
_MISSING = object()


# =============================================================================
# LAYER 0 — ELIGIBILITY
# =============================================================================

def _compute_eligibility(rec: Dict) -> Tuple[bool, List[str]]:
    """Check eligibility gates.

    Hard gates only — survivability, liquidity, drawdown.
    Confidence is deliberately NOT a hard gate (sparse dev-stage coverage
    would exclude the exact optionality names we want to keep).

    Returns (eligible, list_of_reasons).
    """
    reasons: List[str] = []

    # Gate: fundamental red flag (runway < 6m, survivability critical, etc.)
    if rec.get("fundamental_red_flag"):
        reasons.append("fundamental_red_flag")

    # Gate: severity SEV3 (excluded by pipeline already, but double-check)
    if str(rec.get("severity", "")).upper() == "SEV3":
        reasons.append("sev3")

    # Gate: deep drawdown
    df = rec.get("defensive_features") or {}
    dd = _safe_float(df.get("drawdown"))
    if dd is not None and dd < DRAWDOWN_GATE:
        reasons.append("deep_drawdown")

    # Gate: check attn_flags for liquidity/ADV issues
    attn = rec.get("attn_flags") or []
    flags = rec.get("flags") or []
    all_flags = attn + flags
    for f in all_flags:
        fl = str(f).lower()
        if "adv_fail" in fl or "liquidity_fail" in fl:
            reasons.append("adv_fail")
            break

    eligible = len(reasons) == 0
    return eligible, reasons


# =============================================================================
# LAYER 2 — OVERLAYS
# =============================================================================

def _compute_overlays(rec: Dict) -> Dict[str, Any]:
    """Extract overlay signals from rec.

    Sponsorship:  rec["smart_money_signal"] + rec["coinvest"]
    Catalyst:     rec["catalyst_decay"] (top-level)
    Momentum:     score_breakdown.enhancements.momentum → fallback to rec["momentum_signal"]
    Risk:         rec["defensive_features"]

    Uses blank-when-missing semantics: blank means "no data available",
    0 means "data present and the value is zero".
    """
    out: Dict[str, Any] = {}

    # --- Sponsorship (from top-level smart_money_signal + coinvest) ---
    sms = rec.get("smart_money_signal") or {}
    coinvest = rec.get("coinvest") or {}

    # tier1_count: prefer coinvest.tier1_count (int), blank if missing
    t1_raw = coinvest.get("tier1_count")
    if t1_raw is not None:
        out["sponsor_tier1_count"] = int(t1_raw)
    else:
        out["sponsor_tier1_count"] = ""

    # overlap_count: prefer smart_money_signal, blank if missing
    oc_raw = sms.get("overlap_count")
    if oc_raw is not None:
        out["sponsor_overlap_count"] = int(oc_raw)
    else:
        out["sponsor_overlap_count"] = ""

    # net buying: count holders_increasing vs holders_decreasing (lists)
    holders_inc = sms.get("holders_increasing")
    holders_dec = sms.get("holders_decreasing")
    if holders_inc is not None or holders_dec is not None:
        n_inc = len(holders_inc) if isinstance(holders_inc, list) else 0
        n_dec = len(holders_dec) if isinstance(holders_dec, list) else 0
        if n_inc > n_dec:
            out["sponsor_net_buying"] = "buying"
        elif n_dec > n_inc:
            out["sponsor_net_buying"] = "selling"
        else:
            out["sponsor_net_buying"] = "neutral"
    else:
        out["sponsor_net_buying"] = ""

    # --- Catalyst (from top-level rec["catalyst_decay"]) ---
    cd = rec.get("catalyst_decay") or {}
    days = cd.get("days_to_catalyst")
    in_win = cd.get("in_optimal_window")
    # days_to_catalyst: 0 means "no upcoming catalyst", positive int means real distance
    # None/absent means "no data"
    if days is not None:
        out["catalyst_days"] = int(days)
    else:
        out["catalyst_days"] = ""
    if in_win is not None:
        out["catalyst_in_window"] = "1" if in_win else "0"
    else:
        out["catalyst_in_window"] = ""

    # Catalyst mode: explicit label for how catalyst proximity was determined
    if isinstance(days, (int, float)) and days > 0:
        out["catalyst_mode"] = "specific_days"
    elif days == 0 and in_win is True:
        out["catalyst_mode"] = "blended_window"
    elif days is not None or in_win is not None:
        out["catalyst_mode"] = "no_upcoming"
    else:
        out["catalyst_mode"] = "missing"

    # --- Runway bucket (from severity / fundamental_red_flag_reasons) ---
    sev = str(rec.get("severity", "")).upper()
    red_reasons = rec.get("fundamental_red_flag_reasons") or []
    red_set = {str(r).lower() for r in red_reasons}
    if "cash_runway_lt_6m" in red_set or sev == "SEV3":
        out["runway_bucket"] = "critical"
    elif sev == "SEV2":
        out["runway_bucket"] = "short"
    elif sev in ("SEV1", "NONE", ""):
        out["runway_bucket"] = "adequate"
    else:
        out["runway_bucket"] = ""

    # --- Momentum state ---
    mom_enh = (rec.get("score_breakdown") or {}).get("enhancements", {}).get("momentum") or {}
    alpha = _safe_float(mom_enh.get("alpha_60d"))
    if alpha is None:
        # Fallback: try top-level momentum_signal
        mom_top = rec.get("momentum_signal") or {}
        alpha = _safe_float(mom_top.get("alpha_60d"))
    if alpha is not None:
        if alpha > ALPHA_TAILWIND:
            out["mom_state"] = "tailwind"
        elif alpha < ALPHA_HEADWIND:
            out["mom_state"] = "headwind"
        else:
            out["mom_state"] = "neutral"
    else:
        out["mom_state"] = "neutral"

    # --- Risk flags ---
    df = rec.get("defensive_features") or {}
    risk_flags: List[str] = []
    vol = _safe_float(df.get("vol_60d"))
    if vol is not None and vol > VOL_HIGH_THRESHOLD:
        risk_flags.append("high_vol")
    beta = _safe_float(df.get("beta_xbi_60d"))
    if beta is not None and beta > BETA_HIGH_THRESHOLD:
        risk_flags.append("high_beta")
    dd = _safe_float(df.get("drawdown"))
    if dd is not None and dd < DRAWDOWN_FLAG_THRESHOLD:
        risk_flags.append("deep_drawdown")
    rsi = _safe_float(df.get("rsi_14d"))
    if rsi is not None and rsi > RSI_OVERBOUGHT:
        risk_flags.append("overbought_rsi")
    conf = _safe_float(rec.get("confidence_overall"))
    if conf is not None and conf < CONFIDENCE_LOW_THRESHOLD:
        risk_flags.append("low_confidence")
    out["risk_flags"] = "|".join(risk_flags) if risk_flags else ""

    return out


# =============================================================================
# LAYER 3 — SIZING
# =============================================================================

_BAND_ORDER = ["XS", "S", "M", "L"]


def _compute_size_band(
    eligible: bool,
    tier_dev: str,
    optionality: Optional[float],
    overlays: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Rule-based sizing band. Returns (band, list_of_reasons)."""
    if not eligible:
        return "XS", ["ineligible"]

    idx = 2  # Start at M (index 2)
    reasons: List[str] = []

    # Tier A dev with high optionality → push toward L
    if tier_dev == "A" and optionality is not None and optionality >= TIER_A_OPTIONALITY_FLOOR:
        idx += 1
        reasons.append("tier_a_dev")

    # Sponsorship confirmation (only when data present)
    t1 = overlays.get("sponsor_tier1_count")
    if isinstance(t1, (int, float)) and t1 >= 2:
        idx += 1
        reasons.append("sponsor_confirmed")

    # Momentum
    mom = overlays.get("mom_state", "neutral")
    if mom == "tailwind":
        idx += 1
        reasons.append("momentum_tailwind")
    elif mom == "headwind":
        idx -= 1
        reasons.append("momentum_headwind")

    # Runway
    runway = overlays.get("runway_bucket", "adequate")
    if runway in ("short", "critical"):
        idx -= 1
        reasons.append(f"runway_{runway}")

    # Risk flags
    risk = overlays.get("risk_flags", "")
    if "high_vol" in risk or "high_beta" in risk:
        idx -= 1
        reasons.append("high_risk")

    # Clamp
    idx = max(0, min(len(_BAND_ORDER) - 1, idx))
    return _BAND_ORDER[idx], reasons


# =============================================================================
# LAYER 4 — DEV TIER
# =============================================================================

def _compute_tier_dev(
    archetype: str,
    eligible: bool,
    optionality: Optional[float],
    catalyst_in_window: str,
    catalyst_days: Any,
) -> Tuple[str, str]:
    """Compute dev tier (A/B/C/D) and tier_reason.

    Only for drug_developer archetype.
    Degrades gracefully when catalyst data is missing: tiers on optionality
    alone and marks the reason.

    Returns (tier_letter, tier_reason).
    """
    if archetype != "drug_developer":
        return "", ""

    if not eligible:
        return "D", "ineligible"

    if optionality is None:
        return "C", "no_optionality_data"

    # Determine catalyst proximity
    # days_to_catalyst > 0 means a specific upcoming catalyst with known distance.
    # days_to_catalyst == 0 + in_optimal_window == "1" means "blended proximity mode"
    #   (pipeline couldn't pin exact days but proximity scoring is active).
    # days_to_catalyst absent + in_optimal_window absent/False = no catalyst data.
    has_specific_days = isinstance(catalyst_days, (int, float)) and catalyst_days > 0
    has_blended_window = catalyst_in_window == "1"
    has_catalyst_data = has_specific_days or has_blended_window
    has_catalyst_near = (has_specific_days and catalyst_days <= 90) or has_blended_window

    # Catalyst suffix for tier_reason
    cat_tag = "catalyst_near" if has_specific_days and catalyst_days <= 90 else (
        "catalyst_window" if has_blended_window and not has_specific_days else (
        "catalyst_far" if has_catalyst_data and not has_catalyst_near else ""))

    if has_catalyst_data:
        if optionality >= TIER_A_OPTIONALITY_FLOOR and has_catalyst_near:
            return "A", f"high_opt+{cat_tag}"
        elif optionality >= TIER_A_OPTIONALITY_FLOOR:
            return "B", f"high_opt+{cat_tag}" if cat_tag else "high_opt+catalyst_far"
        elif optionality >= TIER_B_OPTIONALITY_FLOOR and has_catalyst_near:
            return "B", f"mod_opt+{cat_tag}"
        else:
            return "C", "low_opt"
    else:
        # No catalyst data: tier on optionality only
        if optionality >= TIER_A_OPTIONALITY_FLOOR:
            return "B", "high_opt+no_catalyst_data"
        elif optionality >= TIER_B_OPTIONALITY_FLOOR:
            return "C", "mod_opt+no_catalyst_data"
        else:
            return "C", "low_opt+no_catalyst_data"


# =============================================================================
# PUBLIC API
# =============================================================================

# Column names emitted by the decision engine (for SNAPSHOT_COLUMNS)
DECISION_COLUMNS = [
    "decision_engine_version", "decision_engine_ruleset_id",
    "eligible", "ineligible_reasons",
    "sponsor_tier1_count", "sponsor_overlap_count", "sponsor_net_buying",
    "catalyst_days", "catalyst_in_window", "catalyst_mode",
    "runway_bucket", "mom_state", "risk_flags",
    "size_band", "size_reasons",
    "tier_dev", "tier_reason",
]


def compute_decision_fields(
    rec: Dict[str, Any],
    archetype: str,
    optionality_pct_dev: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute all decision engine fields for a single ticker.

    Args:
        rec: ranked_securities record from Module 5
        archetype: company archetype string
        optionality_pct_dev: pre-computed optionality percentile (float or None)

    Returns:
        dict with all decision engine column values
    """
    # Layer 0 — Eligibility
    eligible, reasons = _compute_eligibility(rec)

    # Layer 2 — Overlays
    overlays = _compute_overlays(rec)

    # Layer 4 — Dev Tier (before sizing, since tier affects size)
    tier_dev, tier_reason = _compute_tier_dev(
        archetype=archetype,
        eligible=eligible,
        optionality=optionality_pct_dev,
        catalyst_in_window=overlays.get("catalyst_in_window", ""),
        catalyst_days=overlays.get("catalyst_days", ""),
    )

    # Layer 3 — Sizing
    size_band, size_reasons = _compute_size_band(
        eligible=eligible,
        tier_dev=tier_dev,
        optionality=optionality_pct_dev,
        overlays=overlays,
    )

    # Assemble output
    fields: Dict[str, Any] = {
        "decision_engine_version": VERSION,
        "decision_engine_ruleset_id": RULESET_ID,
        "eligible": "1" if eligible else "0",
        "ineligible_reasons": "|".join(reasons) if reasons else "",
        **overlays,
        "size_band": size_band,
        "size_reasons": "|".join(size_reasons) if size_reasons else "",
        "tier_dev": tier_dev,
        "tier_reason": tier_reason,
    }
    return fields
