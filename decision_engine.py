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
import json
from dataclasses import dataclass, fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# RULESET CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class DecisionRuleset:
    """Immutable configuration for all tunable decision engine thresholds.

    Frozen so instances are hashable and can produce a deterministic ruleset_id.
    All defaults match the original v1.0.0 hardcoded values.
    """
    # Layer 0 — Eligibility
    drawdown_gate: float = -0.40
    drawdown_gate_mode: str = "hard"      # "hard" (v1 behavior) or "soft" (sizing penalty)
    drawdown_size_penalty: int = -1       # band steps to subtract when soft + breached
    drawdown_hard_floor: float = -0.75    # hard-exclude even in soft mode if below this

    # Layer 2 — Risk flag thresholds
    vol_high_threshold: float = 1.20
    beta_high_threshold: float = 1.80
    drawdown_flag_threshold: float = -0.35
    rsi_overbought: float = 70.0
    confidence_low_threshold: float = 0.30

    # Layer 2 — Momentum classification
    alpha_tailwind: float = 0.05
    alpha_headwind: float = -0.05

    # Layer 4 — Tier cutoffs
    tier_a_optionality_floor: float = 0.60
    tier_b_optionality_floor: float = 0.30

    # Previously hardcoded inline values
    catalyst_near_days: int = 90
    sponsor_confirm_threshold: int = 2

    # Sizing weights (tuple-of-tuples for frozen hashability)
    sizing_weights: tuple = (("L", 1.0), ("M", 0.6), ("S", 0.3), ("XS", 0.15))

    @property
    def sizing_weights_dict(self) -> Dict[str, float]:
        """Convert sizing_weights to dict for runtime use."""
        return dict(self.sizing_weights)

    def _canonical_json(self) -> str:
        """Deterministic JSON representation for hashing."""
        d = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if f.name == "sizing_weights":
                d[f.name] = dict(val)
            else:
                d[f.name] = val
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def ruleset_id(self) -> str:
        """8-char hex hash of all parameters for change detection."""
        return hashlib.sha256(self._canonical_json().encode()).hexdigest()[:8]

    def to_json(self, path: str) -> None:
        """Write ruleset to a JSON file."""
        d = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if f.name == "sizing_weights":
                d[f.name] = dict(val)
            else:
                d[f.name] = val
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def from_json(cls, path: str) -> DecisionRuleset:
        """Load ruleset from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        # Convert sizing_weights dict back to tuple-of-tuples
        if "sizing_weights" in d and isinstance(d["sizing_weights"], dict):
            d["sizing_weights"] = tuple(
                (k, v) for k, v in sorted(d["sizing_weights"].items())
            )
        return cls(**d)


# =============================================================================
# VERSIONING
# =============================================================================

VERSION = "v1.2.0"

DEFAULT_RULESET = DecisionRuleset()

# Backward-compat exports (same value as DEFAULT_RULESET for default params)
RULESET_ID = DEFAULT_RULESET.ruleset_id
SIZING_WEIGHTS = DEFAULT_RULESET.sizing_weights_dict


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

def _compute_eligibility(rec: Dict, ruleset: DecisionRuleset) -> Tuple[bool, List[str]]:
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
    if dd is not None and dd < ruleset.drawdown_gate:
        if ruleset.drawdown_gate_mode == "hard":
            reasons.append("deep_drawdown")
        elif dd < ruleset.drawdown_hard_floor:
            reasons.append("deep_drawdown")
        # soft mode above hard_floor: don't fail eligibility (handled in sizing)

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

def _compute_overlays(rec: Dict, ruleset: DecisionRuleset) -> Dict[str, Any]:
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
        if alpha > ruleset.alpha_tailwind:
            out["mom_state"] = "tailwind"
        elif alpha < ruleset.alpha_headwind:
            out["mom_state"] = "headwind"
        else:
            out["mom_state"] = "neutral"
    else:
        out["mom_state"] = "neutral"

    # --- Risk flags ---
    df = rec.get("defensive_features") or {}
    risk_flags: List[str] = []
    vol = _safe_float(df.get("vol_60d"))
    if vol is not None and vol > ruleset.vol_high_threshold:
        risk_flags.append("high_vol")
    beta = _safe_float(df.get("beta_xbi_60d"))
    if beta is not None and beta > ruleset.beta_high_threshold:
        risk_flags.append("high_beta")
    dd = _safe_float(df.get("drawdown"))
    if dd is None:
        risk_flags.append("drawdown_data_missing")
    elif dd < ruleset.drawdown_flag_threshold:
        risk_flags.append("deep_drawdown")
    rsi = _safe_float(df.get("rsi_14d"))
    if rsi is not None and rsi > ruleset.rsi_overbought:
        risk_flags.append("overbought_rsi")
    conf = _safe_float(rec.get("confidence_overall"))
    if conf is not None and conf < ruleset.confidence_low_threshold:
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
    ruleset: DecisionRuleset,
    rec: Optional[Dict] = None,
) -> Tuple[str, List[str]]:
    """Rule-based sizing band. Returns (band, list_of_reasons)."""
    if not eligible:
        return "XS", ["ineligible"]

    idx = 2  # Start at M (index 2)
    reasons: List[str] = []

    # Tier A dev with high optionality → push toward L
    if tier_dev == "A" and optionality is not None and optionality >= ruleset.tier_a_optionality_floor:
        idx += 1
        reasons.append("tier_a_dev")

    # Sponsorship confirmation (only when data present)
    t1 = overlays.get("sponsor_tier1_count")
    if isinstance(t1, (int, float)) and t1 >= ruleset.sponsor_confirm_threshold:
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

    # Soft drawdown penalty
    if rec is not None and ruleset.drawdown_gate_mode == "soft":
        _df = rec.get("defensive_features") or {}
        _dd = _safe_float(_df.get("drawdown"))
        if _dd is not None and _dd < ruleset.drawdown_gate:
            idx += ruleset.drawdown_size_penalty  # negative → band downgrade
            reasons.append("drawdown_penalty")

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
    ruleset: DecisionRuleset,
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
    has_catalyst_near = (has_specific_days and catalyst_days <= ruleset.catalyst_near_days) or has_blended_window

    # Catalyst suffix for tier_reason
    cat_tag = "catalyst_near" if has_specific_days and catalyst_days <= ruleset.catalyst_near_days else (
        "catalyst_window" if has_blended_window and not has_specific_days else (
        "catalyst_far" if has_catalyst_data and not has_catalyst_near else ""))

    if has_catalyst_data:
        if optionality >= ruleset.tier_a_optionality_floor and has_catalyst_near:
            return "A", f"high_opt+{cat_tag}"
        elif optionality >= ruleset.tier_a_optionality_floor:
            return "B", f"high_opt+{cat_tag}" if cat_tag else "high_opt+catalyst_far"
        elif optionality >= ruleset.tier_b_optionality_floor and has_catalyst_near:
            return "B", f"mod_opt+{cat_tag}"
        else:
            return "C", "low_opt"
    else:
        # No catalyst data: tier on optionality only
        if optionality >= ruleset.tier_a_optionality_floor:
            return "B", "high_opt+no_catalyst_data"
        elif optionality >= ruleset.tier_b_optionality_floor:
            return "C", "mod_opt+no_catalyst_data"
        else:
            return "C", "low_opt+no_catalyst_data"


# =============================================================================
# ACTIONABLE ORDERING
# =============================================================================

# Priority mappings for sort key construction
_TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "": 4}
_CATALYST_MODE_ORDER = {"specific_days": 0, "blended_window": 1, "no_upcoming": 2, "missing": 3}
_MOM_STATE_ORDER = {"tailwind": 0, "neutral": 1, "headwind": 2}


def compute_actionable_sort_key(
    decision_fields: Dict[str, Any],
    archetype: str,
    optionality: Optional[float],
    composite_rank: Optional[int],
    ticker: str,
) -> Tuple:
    """Return a sort-key tuple for deterministic actionable ordering.

    Lower tuple values sort first. Eligible dev-stage tickers with the
    best tier, nearest catalyst, and highest optionality rank first.
    """
    eligible_val = decision_fields.get("eligible", "0")
    is_eligible = 0 if eligible_val == "1" else 1

    is_dev = 0 if archetype == "drug_developer" else 1

    tier = decision_fields.get("tier_dev", "")
    tier_ord = _TIER_ORDER.get(str(tier), 4)

    cat_mode = decision_fields.get("catalyst_mode", "missing")
    cat_mode_ord = _CATALYST_MODE_ORDER.get(str(cat_mode), 3)

    cat_days_raw = decision_fields.get("catalyst_days", "")
    cat_days = int(cat_days_raw) if cat_days_raw != "" and cat_days_raw is not None else 9999

    opt_neg = -(float(optionality)) if optionality is not None else 0.0

    sponsor_raw = decision_fields.get("sponsor_tier1_count", "")
    sponsor_neg = -(int(sponsor_raw)) if sponsor_raw != "" and sponsor_raw is not None else 0

    mom = decision_fields.get("mom_state", "neutral")
    mom_ord = _MOM_STATE_ORDER.get(str(mom), 1)

    comp_rank = int(composite_rank) if composite_rank is not None else 9999

    return (
        is_eligible,    # 0: eligible first
        is_dev,         # 1: dev first
        tier_ord,       # 2: A < B < C < D < blank
        cat_mode_ord,   # 3: specific < blended < no_upcoming < missing
        cat_days,       # 4: ascending days (missing=9999)
        opt_neg,        # 5: descending optionality (negated)
        sponsor_neg,    # 6: descending sponsor count (negated)
        mom_ord,        # 7: tailwind < neutral < headwind
        comp_rank,      # 8: ascending composite rank
        ticker,         # 9: alphabetic tiebreak
    )


# =============================================================================
# TARGET WEIGHT SIZING
# =============================================================================

def compute_target_weights(
    rows: List[Dict[str, Any]],
    ruleset: Optional[DecisionRuleset] = None,
) -> List[Dict[str, Any]]:
    """Assign target_weight_pct to eligible rows based on size_band.

    Takes already-sorted, eligible-only rows. Normalizes raw weights
    so they sum to 100%. Rounds to 2 decimal places.

    Returns the same rows with 'target_weight_pct' added.
    """
    rs = ruleset or DEFAULT_RULESET
    weights_map = rs.sizing_weights_dict

    raw_weights = []
    for row in rows:
        band = row.get("size_band", "XS")
        raw_weights.append(weights_map.get(str(band), 0.15))

    total = sum(raw_weights)
    if total <= 0:
        for row in rows:
            row["target_weight_pct"] = ""
        return rows

    for row, rw in zip(rows, raw_weights):
        row["target_weight_pct"] = round(rw / total * 100, 2)

    return rows


# =============================================================================
# PUBLIC API
# =============================================================================

# Columns added by the actionable ordering layer
ACTIONABLE_COLUMNS = ["actionable_rank", "target_weight_pct"]

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
    ruleset: Optional[DecisionRuleset] = None,
) -> Dict[str, Any]:
    """Compute all decision engine fields for a single ticker.

    Args:
        rec: ranked_securities record from Module 5
        archetype: company archetype string
        optionality_pct_dev: pre-computed optionality percentile (float or None)
        ruleset: DecisionRuleset config (defaults to DEFAULT_RULESET)

    Returns:
        dict with all decision engine column values
    """
    rs = ruleset or DEFAULT_RULESET

    # Layer 0 — Eligibility
    eligible, reasons = _compute_eligibility(rec, rs)

    # Layer 2 — Overlays
    overlays = _compute_overlays(rec, rs)

    # Layer 4 — Dev Tier (before sizing, since tier affects size)
    tier_dev, tier_reason = _compute_tier_dev(
        archetype=archetype,
        eligible=eligible,
        optionality=optionality_pct_dev,
        catalyst_in_window=overlays.get("catalyst_in_window", ""),
        catalyst_days=overlays.get("catalyst_days", ""),
        ruleset=rs,
    )

    # Layer 3 — Sizing
    size_band, size_reasons = _compute_size_band(
        eligible=eligible,
        tier_dev=tier_dev,
        optionality=optionality_pct_dev,
        overlays=overlays,
        ruleset=rs,
        rec=rec,
    )

    # Assemble output
    fields: Dict[str, Any] = {
        "decision_engine_version": VERSION,
        "decision_engine_ruleset_id": rs.ruleset_id,
        "eligible": "1" if eligible else "0",
        "ineligible_reasons": "|".join(reasons) if reasons else "",
        **overlays,
        "size_band": size_band,
        "size_reasons": "|".join(size_reasons) if size_reasons else "",
        "tier_dev": tier_dev,
        "tier_reason": tier_reason,
    }
    return fields
