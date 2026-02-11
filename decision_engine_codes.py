"""
Canonical reason-code registry for the Decision Engine.

Single source of truth for all codes emitted by decision_engine.py:
    - eligibility gates (ineligible_reasons)
    - risk flags (risk_flags)
    - sizing modifiers (size_reasons)
    - tier reasons (tier_reason)
    - overlay enums (catalyst_mode, catalyst_strength, mom_state, etc.)

Also provides:
    - build_explanation(fields) — structured explanation from decision output
    - describe_code(code) — human-readable description lookup
    - registry_fingerprint() — SHA256[:12] of canonical code set for drift detection
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional


# =============================================================================
# REASON CODE DATACLASS
# =============================================================================

@dataclass(frozen=True)
class ReasonCode:
    """A single reason code emitted by the decision engine."""
    code: str
    domain: str       # "eligibility" | "tier" | "sizing" | "risk" | "overlay"
    description: str  # Short human-readable explanation


# =============================================================================
# ELIGIBILITY GATES (ineligible_reasons pipe-separated codes)
# =============================================================================

_ELIGIBILITY_CODE_LIST: List[ReasonCode] = [
    ReasonCode("fundamental_red_flag", "eligibility",
               "Fundamental red flag (survivability / runway critical)"),
    ReasonCode("sev3", "eligibility",
               "Severity SEV3 hard exclusion"),
    ReasonCode("deep_drawdown", "eligibility",
               "Drawdown below hard gate threshold"),
    ReasonCode("adv_fail", "eligibility",
               "Liquidity / ADV below minimum"),
]

ELIGIBILITY_CODES: FrozenSet[str] = frozenset(rc.code for rc in _ELIGIBILITY_CODE_LIST)


# =============================================================================
# RISK FLAGS (risk_flags pipe-separated codes)
# =============================================================================

_RISK_FLAG_CODE_LIST: List[ReasonCode] = [
    ReasonCode("high_vol", "risk",
               "60d volatility above threshold"),
    ReasonCode("high_beta", "risk",
               "60d beta vs XBI above threshold"),
    ReasonCode("deep_drawdown", "risk",
               "Drawdown below flag threshold (less severe than gate)"),
    ReasonCode("overbought_rsi", "risk",
               "14d RSI above overbought threshold"),
    ReasonCode("low_confidence", "risk",
               "Overall confidence below threshold"),
    ReasonCode("deep_drawdown_rel_xbi", "risk",
               "Relative drawdown vs XBI below threshold"),
    ReasonCode("drawdown_data_missing", "risk",
               "No drawdown data available for assessment"),
]

RISK_FLAG_CODES: FrozenSet[str] = frozenset(rc.code for rc in _RISK_FLAG_CODE_LIST)


# =============================================================================
# SIZING MODIFIERS (size_reasons pipe-separated codes)
# =============================================================================

_SIZING_CODE_LIST: List[ReasonCode] = [
    ReasonCode("ineligible", "sizing",
               "Ineligible ticker — forced to XS band"),
    ReasonCode("tier_a_dev", "sizing",
               "A-tier dev with high optionality — band upgrade"),
    ReasonCode("catalyst_far_lift", "sizing",
               "FAR catalyst data present — minor band upgrade vs missing"),
    ReasonCode("sponsor_confirmed", "sizing",
               "Tier-1 sponsor count above threshold — band upgrade"),
    ReasonCode("momentum_tailwind", "sizing",
               "Positive 60d alpha momentum — band upgrade"),
    ReasonCode("momentum_headwind", "sizing",
               "Negative 60d alpha momentum — band downgrade"),
    ReasonCode("runway_short", "sizing",
               "SEV2 runway concern — band downgrade"),
    ReasonCode("runway_critical", "sizing",
               "Critical runway (SEV3 / <6m cash) — band downgrade"),
    ReasonCode("high_risk", "sizing",
               "High volatility or beta — band downgrade"),
    ReasonCode("drawdown_penalty", "sizing",
               "Soft drawdown mode breach — band downgrade"),
]

SIZING_CODES: FrozenSet[str] = frozenset(rc.code for rc in _SIZING_CODE_LIST)

# Band index deltas for each sizing modifier
_SIZING_DELTA: Dict[str, int] = {
    "ineligible": 0,           # forces XS, no delta
    "tier_a_dev": +1,
    "catalyst_far_lift": +1,
    "sponsor_confirmed": +1,
    "momentum_tailwind": +1,
    "momentum_headwind": -1,
    "runway_short": -1,
    "runway_critical": -1,
    "high_risk": -1,
    "drawdown_penalty": -1,    # ruleset.drawdown_size_penalty (typically -1)
}


# =============================================================================
# OVERLAY ENUMS (valid values for categorical overlay fields)
# =============================================================================

VALID_CATALYST_MODES: FrozenSet[str] = frozenset({
    "specific_days", "blended_window", "no_upcoming", "missing",
})

VALID_CATALYST_STRENGTHS: FrozenSet[str] = frozenset({
    "near", "mid", "far", "missing",
})

VALID_MOM_STATES: FrozenSet[str] = frozenset({
    "tailwind", "neutral", "headwind",
})

VALID_RUNWAY_BUCKETS: FrozenSet[str] = frozenset({
    "critical", "short", "adequate", "",
})

VALID_TIER_DEVS: FrozenSet[str] = frozenset({
    "A", "B", "C", "D", "",
})

VALID_SIZE_BANDS: FrozenSet[str] = frozenset({
    "L", "M", "S", "XS",
})


# =============================================================================
# TIER REASONS — exhaustive set from _compute_tier_dev() code paths
# =============================================================================

VALID_TIER_REASONS: FrozenSet[str] = frozenset({
    "",                           # non-drug-developer
    "ineligible",                 # D — not eligible
    "no_optionality_data",        # C — optionality is None
    "high_opt+catalyst_near",     # A — opt >= a_floor, actionable, specific_days near
    "high_opt+catalyst_mid",      # A — opt >= a_floor, actionable, specific_days mid
    "high_opt+catalyst_window",   # A — opt >= a_floor, actionable, blended_window
    "high_opt+catalyst_far",      # B — opt >= a_floor, NOT actionable (far catalyst)
    "mod_opt+catalyst_near",      # B — b_floor <= opt < a_floor, actionable, near
    "mod_opt+catalyst_mid",       # B — b_floor <= opt < a_floor, actionable, mid
    "mod_opt+catalyst_window",    # B — b_floor <= opt < a_floor, actionable, blended
    "high_opt+no_catalyst_data",  # B — opt >= a_floor, no catalyst data
    "mod_opt+no_catalyst_data",   # C — b_floor <= opt < a_floor, no catalyst data
    "low_opt+no_catalyst_data",   # C — opt < b_floor, no catalyst data
    "low_opt",                    # C — opt < b_floor with catalyst, or mod_opt+far
})


# =============================================================================
# TIER REASON DESCRIPTIONS (human-readable)
# =============================================================================

_TIER_REASON_DESCRIPTIONS: Dict[str, str] = {
    "": "Non-drug-developer archetype (tier not applicable)",
    "ineligible": "Ineligible — failed one or more eligibility gates",
    "no_optionality_data": "No optionality data available",
    "high_opt+catalyst_near": "High optionality with near-term catalyst",
    "high_opt+catalyst_mid": "High optionality with mid-term catalyst",
    "high_opt+catalyst_window": "High optionality in blended catalyst window",
    "high_opt+catalyst_far": "High optionality but catalyst too distant for A-tier",
    "mod_opt+catalyst_near": "Moderate optionality with near-term catalyst",
    "mod_opt+catalyst_mid": "Moderate optionality with mid-term catalyst",
    "mod_opt+catalyst_window": "Moderate optionality in blended catalyst window",
    "high_opt+no_catalyst_data": "High optionality but no catalyst data available",
    "mod_opt+no_catalyst_data": "Moderate optionality but no catalyst data available",
    "low_opt+no_catalyst_data": "Low optionality and no catalyst data available",
    "low_opt": "Low optionality (or moderate with distant catalyst)",
}

# Component descriptions used in decomposed explanations
_COMPONENT_DESCRIPTIONS: Dict[str, str] = {
    "high_opt": "Optionality above A-tier floor",
    "mod_opt": "Optionality between B-tier and A-tier floors",
    "low_opt": "Optionality below B-tier floor",
    "catalyst_near": "Catalyst within near-term window",
    "catalyst_mid": "Catalyst within mid-term window",
    "catalyst_far": "Catalyst beyond mid-term window",
    "catalyst_window": "In blended catalyst proximity window",
    "no_catalyst_data": "No catalyst data available",
}


# =============================================================================
# MASTER REGISTRY (all codes, all domains)
# =============================================================================

_ALL_CODES: Dict[str, ReasonCode] = {}
for _rc_list in (_ELIGIBILITY_CODE_LIST, _RISK_FLAG_CODE_LIST, _SIZING_CODE_LIST):
    for _rc in _rc_list:
        _ALL_CODES[_rc.code] = _rc


# =============================================================================
# TIER REASON PARSING
# =============================================================================

def _parse_tier_reason(tier_reason: str) -> Dict[str, str]:
    """Decompose a compound tier_reason string into components.

    Examples:
        "high_opt+catalyst_near" → {"optionality_band": "high_opt",
                                     "catalyst_tag": "catalyst_near"}
        "ineligible"             → {"special": "ineligible"}
        "low_opt"                → {"optionality_band": "low_opt"}
        ""                       → {}
    """
    if not tier_reason:
        return {}

    # Special single-token reasons
    if tier_reason in ("ineligible", "no_optionality_data"):
        return {"special": tier_reason}

    parts = tier_reason.split("+", 1)
    result: Dict[str, str] = {"optionality_band": parts[0]}
    if len(parts) > 1:
        result["catalyst_tag"] = parts[1]
    return result


# =============================================================================
# EXPLANATION BUILDER
# =============================================================================

def build_explanation(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured explanation from decision engine output fields.

    Args:
        fields: dict with decision engine column values (from compute_decision_fields
                or from a rankings CSV row). Expected keys match DECISION_COLUMNS.

    Returns:
        Structured explanation dict with domains: eligibility, tier, sizing, risk, overlays.
    """
    # --- Eligibility ---
    eligible_raw = str(fields.get("eligible", "0")).strip()
    is_eligible = eligible_raw == "1"

    inelig_raw = str(fields.get("ineligible_reasons", "")).strip()
    failed_gates = [g for g in inelig_raw.split("|") if g] if inelig_raw else []

    gate_details: Dict[str, str] = {}
    for g in failed_gates:
        rc = _ALL_CODES.get(g)
        gate_details[g] = rc.description if rc else g

    # Missing inputs: if no defensive data, drawdown gate can't fire
    elig_missing: List[str] = []

    eligibility_block = {
        "passed": is_eligible,
        "failed_gates": failed_gates,
        "gate_details": gate_details,
        "missing_inputs": elig_missing,
    }

    # --- Tier ---
    tier_dev = str(fields.get("tier_dev", "")).strip()
    tier_reason_raw = str(fields.get("tier_reason", "")).strip()
    parsed = _parse_tier_reason(tier_reason_raw)

    primary_reasons: List[str] = []
    if "optionality_band" in parsed:
        primary_reasons.append(parsed["optionality_band"])
    if "catalyst_tag" in parsed:
        primary_reasons.append(parsed["catalyst_tag"])
    if "special" in parsed:
        primary_reasons.append(parsed["special"])

    reason_descs: Dict[str, str] = {}
    for r in primary_reasons:
        reason_descs[r] = _COMPONENT_DESCRIPTIONS.get(
            r, _TIER_REASON_DESCRIPTIONS.get(r, r)
        )

    tier_block = {
        "tier": tier_dev,
        "tier_reason_raw": tier_reason_raw,
        "primary_reasons": primary_reasons,
        "reason_descriptions": reason_descs,
        "optionality_band": parsed.get("optionality_band", ""),
        "catalyst_proximity": parsed.get("catalyst_tag", ""),
    }

    # --- Sizing ---
    size_band = str(fields.get("size_band", "")).strip()
    size_reasons_raw = str(fields.get("size_reasons", "")).strip()
    size_codes = [s for s in size_reasons_raw.split("|") if s] if size_reasons_raw else []

    modifiers: List[Dict[str, Any]] = []
    net_modifier = 0
    for code in size_codes:
        delta = _SIZING_DELTA.get(code, 0)
        rc = _ALL_CODES.get(code)
        modifiers.append({
            "code": code,
            "delta": delta,
            "description": rc.description if rc else code,
        })
        net_modifier += delta

    sizing_block = {
        "band": size_band,
        "modifiers": modifiers,
        "net_modifier": net_modifier,
    }

    # --- Risk ---
    risk_raw = str(fields.get("risk_flags", "")).strip()
    risk_codes = [r for r in risk_raw.split("|") if r] if risk_raw else []

    flag_details: Dict[str, str] = {}
    for r in risk_codes:
        rc = _ALL_CODES.get(r)
        flag_details[r] = rc.description if rc else r

    risk_missing: List[str] = []

    risk_block = {
        "flags": risk_codes,
        "flag_details": flag_details,
        "missing_inputs": risk_missing,
    }

    # --- Overlays ---
    # Parse catalyst_days safely
    cat_days_raw = fields.get("catalyst_days", "")
    cat_days: Optional[int] = None
    if cat_days_raw != "" and cat_days_raw is not None:
        try:
            cat_days = int(float(str(cat_days_raw)))
        except (ValueError, TypeError):
            pass

    # sponsor_confirmed: check if sponsor_confirmed is in size_reasons
    sponsor_confirmed = "sponsor_confirmed" in size_codes

    overlays_block = {
        "catalyst_mode": str(fields.get("catalyst_mode", "")).strip(),
        "catalyst_strength": str(fields.get("catalyst_strength", "")).strip(),
        "catalyst_days": cat_days,
        "mom_state": str(fields.get("mom_state", "")).strip(),
        "runway_bucket": str(fields.get("runway_bucket", "")).strip(),
        "sponsor_confirmed": sponsor_confirmed,
    }

    return {
        "eligibility": eligibility_block,
        "tier": tier_block,
        "sizing": sizing_block,
        "risk": risk_block,
        "overlays": overlays_block,
    }


# =============================================================================
# PUBLIC UTILITIES
# =============================================================================

def describe_code(code: str) -> str:
    """Look up the human-readable description for a reason code.

    Falls back to the code itself if not found in the registry.
    """
    rc = _ALL_CODES.get(code)
    if rc:
        return rc.description
    # Check tier reason descriptions
    desc = _TIER_REASON_DESCRIPTIONS.get(code)
    if desc:
        return desc
    # Check component descriptions
    desc = _COMPONENT_DESCRIPTIONS.get(code)
    if desc:
        return desc
    return code


def registry_fingerprint() -> str:
    """Compute a SHA256[:12] fingerprint of the canonical code registry.

    Changes whenever codes are added, removed, or renamed. Used for
    drift detection between engine versions.
    """
    hasher = hashlib.sha256()

    # Hash all canonical code sets in deterministic order
    sections = [
        ("eligibility", sorted(ELIGIBILITY_CODES)),
        ("risk", sorted(RISK_FLAG_CODES)),
        ("sizing", sorted(SIZING_CODES)),
        ("tier_reasons", sorted(VALID_TIER_REASONS)),
        ("catalyst_modes", sorted(VALID_CATALYST_MODES)),
        ("catalyst_strengths", sorted(VALID_CATALYST_STRENGTHS)),
        ("mom_states", sorted(VALID_MOM_STATES)),
        ("runway_buckets", sorted(VALID_RUNWAY_BUCKETS)),
        ("tier_devs", sorted(VALID_TIER_DEVS)),
        ("size_bands", sorted(VALID_SIZE_BANDS)),
    ]
    for section_name, codes in sections:
        hasher.update(f"{section_name}:{','.join(codes)}".encode())

    return hasher.hexdigest()[:12]
