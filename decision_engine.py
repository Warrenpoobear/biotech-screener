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
    catalyst_mid_days: int = 180
    catalyst_time_decay_mode: str = "hard"           # "hard" | "logistic"
    sparse_signal_mode: str = "legacy"               # "legacy" | "exclude_missing"
    catalyst_logistic_midpoint_days: int = 150        # sigmoid center
    catalyst_logistic_scale_days: float = 30.0        # sigmoid steepness
    sponsor_confirm_threshold: int = 2

    # Regime-aware drawdown gate
    drawdown_rel_xbi_gate: float = -0.20
    drawdown_gate_require_both: bool = True

    # Near-rel-gate rescue: override drawdown failure when relative margin
    # is close to the gate (sector-driven drawdown, not idiosyncratic blow-up)
    enable_dd_rel_margin_rescue: bool = False
    dd_rel_margin_rescue_threshold: float = -0.05

    # Sizing weights (tuple-of-tuples for frozen hashability)
    sizing_weights: tuple = (("L", 1.0), ("M", 0.6), ("S", 0.3), ("XS", 0.15))

    # Layer 3 — Cost-aware sizing (opt-in, no behavior change when disabled)
    enable_cost_haircut: bool = False
    cost_haircut_buckets: tuple = ((400, 1.0), (1000, 0.85), (2000, 0.70))
    cost_haircut_floor_mult: float = 0.55
    cost_impact_cap_bps: float = 200.0

    # Layer 3 — Catalyst strength sizing tilt (opt-in, multiplicative on weight)
    enable_catalyst_tilt: bool = False
    catalyst_tilt_mults: tuple = (
        ("NEAR", 1.10), ("MID", 1.05), ("FAR", 0.95), ("MISSING", 0.90)
    )

    # Layer 3 — Momentum state sizing tilt (opt-in, multiplicative on weight)
    enable_mom_state_tilt: bool = False
    mom_state_tilt_mults: tuple = (
        ("tailwind", 1.0), ("neutral", 1.0), ("headwind", 1.0),
    )

    # Missingness penalties (opt-in, default off — tracking always on)
    enable_missingness_sort_penalty: bool = False
    enable_missingness_size_penalty: bool = False

    # Clinical signal sort tilt (opt-in, default OFF)
    # When enabled, blends clinical_score_z_tier into sort anchor (comp_rank
    # or opt_neg), gated by stage_bucket.  Positive-only mode (default) only
    # boosts high-clinical names without penalising low ones.
    enable_clinical_sort_signal: bool = False
    clinical_sort_weight: float = 1.0
    clinical_positive_only: bool = True
    clinical_stage_mults: tuple = (("early", 0.0), ("mid", 1.0), ("late", 1.5))

    # Coinvest sort tilt (opt-in, default OFF)
    # When enabled, blends coinvest conviction score into sort anchor,
    # following the clinical_sort_signal pattern.  Positive-only: only
    # boosts high-conviction names, never penalises zero-coinvest.
    enable_coinvest_sort_signal: bool = False
    coinvest_sort_weight: float = 0.5          # scale factor (lower than clinical — advisory signal)
    coinvest_positive_only: bool = True         # only boost, never penalise
    coinvest_score_mode: str = "tier1_count"   # "tier1_count" (others deferred)

    # Institutional delta sort tilt (opt-in, default OFF)
    # When enabled, blends net_elite_holders_delta z-score into sort anchor,
    # following the coinvest pattern.  Positive-only: only boosts names with
    # increasing institutional ownership, never penalises decreasing.
    enable_institutional_sort_signal: bool = False
    institutional_sort_weight: float = 0.3
    institutional_positive_only: bool = True

    # Far-horizon catalyst (opt-in, off by default; set far_window_days > 0 to enable)
    far_window_days: int = 0                          # max PCD horizon for far_window mode (0 = off)
    far_window_decay_mult: float = 0.15               # catalyst_decay_w for far_window tickers

    # Sort anchor (default composite_rank = current behavior)
    sort_anchor: str = "composite_rank"               # "composite_rank" | "optionality_pct" | "alpha_cohort"

    # Alpha cohort scoring (opt-in via sort_anchor="alpha_cohort")
    alpha_cohort_table_path: str = "production_data/alpha_cohort_tables/v1.json"
    alpha_cohort_shrink_k: float = 50.0
    alpha_cohort_clip_min: float = -0.10
    alpha_cohort_clip_max: float = 0.10

    # Alpha cohort training configuration (adaptive alpha)
    alpha_train_mode: str = "trailing-6"      # "expanding" | "trailing-N" | "decay-H"
    alpha_train_min_train_dates: int = 6      # minimum training dates (>= 2)
    alpha_train_horizon: int = 84             # forward horizon in trading days (63|84|126)
    alpha_table_rebuild_policy: str = "never" # "never" | "if_missing" | "daily"

    # Composite engine (default legacy = Module 5 composite scoring)
    composite_engine: str = "legacy"   # "legacy" | "alpha_cohort"

    # Commercial tiering (opt-in, default dev_first = current behavior)
    tiering_priority_mode: str = "dev_first"          # "dev_first" | "tier_first"
    tier_a_commercial_floor: float = 0.85             # quality_pct cutoff for commercial A
    tier_b_commercial_floor: float = 0.60             # quality_pct cutoff for commercial B

    # Actionable ordering — catalyst source/type priority (opt-in)
    enable_catalyst_priority: bool = False
    catalyst_priority_map: tuple = (
        # (event_type, source) → priority int (lower = better)
        # First-match-wins: specific FDA types before wildcards.
        ("FDA_PDUFA_DATE", "*", 1),
        ("FDA_ADCOM", "*", 1),
        ("FDA_CRL", "*", 1),
        ("FDA_RTF", "*", 1),
        ("FDA_WARNING_LETTER", "*", 1),
        ("FDA_DECISION", "*", 1),
        ("FDA_APPROVAL", "*", 1),
        ("FDA_SUBMISSION", "*", 1),
        ("DATA_READOUT", "CTGOV_CALENDAR", 2),
        ("DATA_READOUT", "*", 2),
        ("CT_PRIMARY_COMPLETION", "*", 3),
        ("CT_STUDY_COMPLETION", "*", 3),
        ("TRIAL_ONGOING", "*", 3),
        ("*", "CORPORATE_CALENDAR", 3),
        ("*", "SEC_8K_FILING", 2),
        ("*", "SEC_10Q_FILING", 2),
        ("*", "SEC_10K_FILING", 2),
        ("*", "SEC_6K_FILING", 2),
        ("*", "FEDERAL_REGISTER", 3),
    )
    catalyst_priority_default: int = 9
    catalyst_priority_unknown: int = 99

    # Alpha modifier (nudges ordering using OOS alpha signal, does NOT replace composite)
    alpha_modifier_mode: str = "off"           # "off" | "tiebreak" | "within_tier"
    alpha_modifier_weight: float = 0.00        # weight for within_tier mode; capped [0, 0.25]

    # Actionable ordering — catalyst priority mode (supersedes enable_catalyst_priority)
    catalyst_priority_mode: str = "off"       # "off" | "tiebreaker" | "blended"
    catalyst_priority_rank_bonuses: tuple = (  # blended mode: priority → rank bonus
        (1, 5.0),     # FDA events: effective_rank -= 5
        (2, 2.0),     # CTGOV/SEC readout: effective_rank -= 2
        (3, 0.0),     # corporate/ongoing
        (9, 0.0),     # no catalyst
        (99, 0.0),    # unknown
    )

    def __post_init__(self):
        # Validate cost_haircut_buckets: ascending thresholds
        buckets = self.cost_haircut_buckets
        if buckets:
            for i in range(1, len(buckets)):
                if buckets[i][0] <= buckets[i - 1][0]:
                    raise ValueError(
                        f"cost_haircut_buckets thresholds must be strictly ascending: {buckets}"
                    )
        # Validate cost_haircut_floor_mult in (0, 1]
        if not (0 < self.cost_haircut_floor_mult <= 1.0):
            raise ValueError(
                f"cost_haircut_floor_mult must be in (0, 1], got {self.cost_haircut_floor_mult}"
            )
        # Validate cost_impact_cap_bps > 0
        if self.cost_impact_cap_bps <= 0:
            raise ValueError(
                f"cost_impact_cap_bps must be > 0, got {self.cost_impact_cap_bps}"
            )
        # Validate catalyst_tilt_mults: keys must be valid bands, mults > 0
        valid_bands = {"NEAR", "MID", "FAR", "MISSING"}
        for band, mult in self.catalyst_tilt_mults:
            if band not in valid_bands:
                raise ValueError(
                    f"catalyst_tilt_mults: unknown band '{band}', "
                    f"expected one of {sorted(valid_bands)}"
                )
            if mult <= 0:
                raise ValueError(
                    f"catalyst_tilt_mults: mult must be > 0, got {mult} for '{band}'"
                )
        # Validate mom_state_tilt_mults: keys must be valid states, mults > 0
        valid_mom_states = {"tailwind", "neutral", "headwind"}
        for state, mult in self.mom_state_tilt_mults:
            if state not in valid_mom_states:
                raise ValueError(
                    f"mom_state_tilt_mults: unknown state '{state}', "
                    f"expected one of {sorted(valid_mom_states)}"
                )
            if mult <= 0:
                raise ValueError(
                    f"mom_state_tilt_mults: mult must be > 0, got {mult} for '{state}'"
                )
        # Validate catalyst_time_decay_mode
        if self.catalyst_time_decay_mode not in ("hard", "logistic"):
            raise ValueError(
                f"catalyst_time_decay_mode must be 'hard' or 'logistic', "
                f"got '{self.catalyst_time_decay_mode}'"
            )
        if self.catalyst_logistic_scale_days <= 0:
            raise ValueError(
                f"catalyst_logistic_scale_days must be > 0, "
                f"got {self.catalyst_logistic_scale_days}"
            )
        # Validate sparse_signal_mode
        if self.sparse_signal_mode not in ("legacy", "exclude_missing"):
            raise ValueError(
                f"sparse_signal_mode must be 'legacy' or 'exclude_missing', "
                f"got '{self.sparse_signal_mode}'"
            )
        # Validate catalyst_priority_mode
        if self.catalyst_priority_mode not in ("off", "tiebreaker", "blended"):
            raise ValueError(
                f"catalyst_priority_mode must be 'off', 'tiebreaker', or 'blended', "
                f"got '{self.catalyst_priority_mode}'"
            )
        # Validate sort_anchor
        if self.sort_anchor not in ("composite_rank", "optionality_pct", "alpha_cohort"):
            raise ValueError(
                f"sort_anchor must be 'composite_rank', 'optionality_pct', or 'alpha_cohort', "
                f"got '{self.sort_anchor}'"
            )
        # Validate composite_engine
        if self.composite_engine not in ("legacy", "alpha_cohort"):
            raise ValueError(
                f"composite_engine must be 'legacy' or 'alpha_cohort', "
                f"got '{self.composite_engine}'"
            )
        # Validate alpha_train_mode: expanding | trailing-N | decay-H
        _atm = self.alpha_train_mode.strip().lower()
        _atm_valid = False
        if _atm == "expanding":
            _atm_valid = True
        elif _atm.startswith("trailing-") or _atm.startswith("decay-"):
            try:
                _param = int(_atm.split("-", 1)[1])
                _atm_valid = _param >= 1
            except (ValueError, IndexError):
                pass
        if not _atm_valid:
            raise ValueError(
                f"alpha_train_mode must be 'expanding', 'trailing-N', or 'decay-H', "
                f"got '{self.alpha_train_mode}'"
            )
        # Validate alpha_train_min_train_dates
        if self.alpha_train_min_train_dates < 2:
            raise ValueError(
                f"alpha_train_min_train_dates must be >= 2, "
                f"got {self.alpha_train_min_train_dates}"
            )
        # Validate alpha_train_horizon
        if self.alpha_train_horizon not in (63, 84, 126):
            raise ValueError(
                f"alpha_train_horizon must be 63, 84, or 126, "
                f"got {self.alpha_train_horizon}"
            )
        # Validate alpha_table_rebuild_policy
        if self.alpha_table_rebuild_policy not in ("never", "if_missing", "daily"):
            raise ValueError(
                f"alpha_table_rebuild_policy must be 'never', 'if_missing', or 'daily', "
                f"got '{self.alpha_table_rebuild_policy}'"
            )
        # Validate alpha_modifier_mode
        if self.alpha_modifier_mode not in ("off", "tiebreak", "within_tier"):
            raise ValueError(
                f"alpha_modifier_mode must be 'off', 'tiebreak', or 'within_tier', "
                f"got '{self.alpha_modifier_mode}'"
            )
        # Validate alpha_modifier_weight
        if not (0.0 <= self.alpha_modifier_weight <= 0.25):
            raise ValueError(
                f"alpha_modifier_weight must be in [0.0, 0.25], "
                f"got {self.alpha_modifier_weight}"
            )
        # Validate tiering_priority_mode
        if self.tiering_priority_mode not in ("dev_first", "tier_first"):
            raise ValueError(
                f"tiering_priority_mode must be 'dev_first' or 'tier_first', "
                f"got '{self.tiering_priority_mode}'"
            )
        # Validate coinvest_score_mode (only tier1_count implemented)
        if self.coinvest_score_mode not in ("tier1_count",):
            raise ValueError(
                f"coinvest_score_mode must be 'tier1_count', "
                f"got '{self.coinvest_score_mode}'"
            )
        # Validate drawdown_gate_mode
        if self.drawdown_gate_mode not in ("hard", "soft"):
            raise ValueError(
                f"drawdown_gate_mode must be 'hard' or 'soft', "
                f"got '{self.drawdown_gate_mode}'"
            )
        # Validate dd_rel_margin_rescue_threshold: must be <= 0
        if self.dd_rel_margin_rescue_threshold > 0:
            raise ValueError(
                f"dd_rel_margin_rescue_threshold must be <= 0, "
                f"got {self.dd_rel_margin_rescue_threshold}"
            )
        # Validate far_window_days: must be >= 0
        if self.far_window_days < 0:
            raise ValueError(
                f"far_window_days must be >= 0, got {self.far_window_days}"
            )
        # Validate far_window_decay_mult: must be in (0, 1]
        if not (0.0 < self.far_window_decay_mult <= 1.0):
            raise ValueError(
                f"far_window_decay_mult must be in (0, 1], "
                f"got {self.far_window_decay_mult}"
            )

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
        """8-char hex hash for change detection.

        When loaded via ``from_json``, hashes the *raw file content* so that
        adding new optional fields to the schema does not change historical IDs.
        When constructed programmatically, hashes all dataclass fields.
        """
        # Prefer file-content hash set by from_json (schema-stable)
        file_hash = getattr(self, "_file_content_hash", None)
        if file_hash:
            return file_hash
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
        """Load ruleset from a JSON file.

        Computes a stable ``ruleset_id`` from the raw file content *before*
        type normalization.  This ensures that adding new optional fields to
        the dataclass schema does not change the hash of existing files.
        """
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        # Compute stable hash from raw dict BEFORE type conversions.
        # Uses the same compact-separator format as _canonical_json.
        file_hash = hashlib.sha256(
            json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]
        # Convert sizing_weights dict back to tuple-of-tuples
        if "sizing_weights" in d and isinstance(d["sizing_weights"], dict):
            d["sizing_weights"] = tuple(
                (k, v) for k, v in sorted(d["sizing_weights"].items())
            )
        # Convert cost_haircut_buckets list-of-lists back to tuple-of-tuples
        if "cost_haircut_buckets" in d and isinstance(d["cost_haircut_buckets"], list):
            d["cost_haircut_buckets"] = tuple(
                tuple(pair) for pair in d["cost_haircut_buckets"]
            )
        # Convert catalyst_tilt_mults list-of-lists back to tuple-of-tuples
        if "catalyst_tilt_mults" in d and isinstance(d["catalyst_tilt_mults"], list):
            d["catalyst_tilt_mults"] = tuple(
                tuple(pair) for pair in d["catalyst_tilt_mults"]
            )
        # Convert mom_state_tilt_mults list-of-lists back to tuple-of-tuples
        if "mom_state_tilt_mults" in d and isinstance(d["mom_state_tilt_mults"], list):
            d["mom_state_tilt_mults"] = tuple(
                tuple(pair) for pair in d["mom_state_tilt_mults"]
            )
        # Convert catalyst_priority_map list-of-lists back to tuple-of-tuples
        if "catalyst_priority_map" in d and isinstance(d["catalyst_priority_map"], list):
            d["catalyst_priority_map"] = tuple(
                tuple(rule) for rule in d["catalyst_priority_map"]
            )
        # Convert catalyst_priority_rank_bonuses list-of-lists back to tuple-of-tuples
        if "catalyst_priority_rank_bonuses" in d and isinstance(d["catalyst_priority_rank_bonuses"], list):
            d["catalyst_priority_rank_bonuses"] = tuple(
                tuple(pair) for pair in d["catalyst_priority_rank_bonuses"]
            )
        # Convert clinical_stage_mults list-of-lists back to tuple-of-tuples
        if "clinical_stage_mults" in d and isinstance(d["clinical_stage_mults"], list):
            d["clinical_stage_mults"] = tuple(
                tuple(pair) for pair in d["clinical_stage_mults"]
            )
        # Migration: enable_catalyst_priority=True → catalyst_priority_mode="tiebreaker"
        if d.get("enable_catalyst_priority") and "catalyst_priority_mode" not in d:
            d["catalyst_priority_mode"] = "tiebreaker"
        # Filter to known fields for forward compatibility (ignore unknown keys
        # that may have been added by a newer schema version).
        import dataclasses as _dc
        _known = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in d.items() if k in _known}
        instance = cls(**d)
        # Store file-content hash on the frozen instance (bypasses __setattr__)
        object.__setattr__(instance, "_file_content_hash", file_hash)
        return instance


# =============================================================================
# VERSIONING
# =============================================================================

VERSION = "v1.3.0"

DEFAULT_RULESET = DecisionRuleset()

# Backward-compat exports (same value as DEFAULT_RULESET for default params)
RULESET_ID = DEFAULT_RULESET.ruleset_id
SIZING_WEIGHTS = DEFAULT_RULESET.sizing_weights_dict


# =============================================================================
# HELPERS
# =============================================================================

def _safe_float(val, default=None):
    """Convert a value to float, handling None/str/Decimal/NaN."""
    if val is None:
        return default
    try:
        if isinstance(val, str) and val.strip() == "":
            return default
        v = float(val)
        # NaN should behave like missing (pandas frequently produces NaN)
        if v != v:
            return default
        return v
    except (ValueError, TypeError):
        return default


def _logistic_decay(days: float | None, midpoint: float, scale: float) -> float:
    """Continuous proximity weight: 1.0 (imminent) -> 0.0 (distant). Midpoint -> 0.5."""
    if days is None or days < 0:
        return 0.0
    from math import exp
    return 1.0 / (1.0 + exp((days - midpoint) / max(scale, 1e-6)))


def _has_flag(rf, flag: str) -> bool:
    """Check if a risk-flag string contains a specific flag."""
    if not rf:
        return False
    return flag in str(rf).replace("|", ",").split(",")


def _compute_missing_components(decision_fields: dict) -> list:
    """Identify missing critical data components from decision output fields."""
    missing = []
    if decision_fields.get("catalyst_mode") == "missing":
        missing.append("catalyst")
    if decision_fields.get("sponsor_tier1_count") in ("", None):
        missing.append("sponsor")
    if _has_flag(decision_fields.get("risk_flags"), "drawdown_data_missing"):
        missing.append("drawdown")
    return missing


# =============================================================================
# LAYER 0 — ELIGIBILITY
# =============================================================================

def _compute_eligibility(
    rec: Dict, ruleset: DecisionRuleset
) -> Tuple[bool, List[str], bool]:
    """Check eligibility gates.

    Hard gates only — survivability, liquidity, drawdown.
    Confidence is deliberately NOT a hard gate (sparse dev-stage coverage
    would exclude the exact optionality names we want to keep).

    Returns (eligible, list_of_reasons, dd_rel_margin_rescued).
    dd_rel_margin_rescued is True when a drawdown failure was overridden
    because the relative margin was close to the gate (sector-driven drawdown).
    """
    reasons: List[str] = []
    dd_rel_margin_rescued = False

    # Gate 0: financials missing — if survivability data is absent,
    # don't trust red flag checks (they'd produce false positives).
    surv_coverage = (rec.get("survivability_signal") or {}).get("coverage") or []
    if "missing_cash" in surv_coverage and "missing_burn_data" in surv_coverage:
        reasons.append("financials_missing")
    else:
        # Gate 1: fundamental red flag (runway < 6m, survivability critical, etc.)
        # If the boolean is present (live run), use it; if absent (archive run),
        # compute it directly via the red-flag detection rules.
        has_red_flag = rec.get("fundamental_red_flag")
        if has_red_flag is None:
            from defensive_overlay_adapter import detect_fundamental_red_flags
            has_red_flag, _ = detect_fundamental_red_flags(rec)
        if has_red_flag:
            reasons.append("fundamental_red_flag")

    # Gate: severity SEV3 (excluded by pipeline already, but double-check)
    if str(rec.get("severity", "")).upper() == "SEV3":
        reasons.append("sev3")

    # Gate: deep drawdown (regime-aware: absolute + relative to XBI)
    df = rec.get("defensive_features") or {}
    dd = _safe_float(df.get("drawdown"))
    dd_rel = _safe_float(df.get("drawdown_rel_xbi"))

    if ruleset.drawdown_gate_mode == "hard":
        abs_breach = dd is not None and dd < ruleset.drawdown_gate
        rel_breach = dd_rel is not None and dd_rel < ruleset.drawdown_rel_xbi_gate

        if dd_rel is None:
            # No relative data — fall back to absolute-only
            if abs_breach:
                reasons.append("deep_drawdown")
        elif ruleset.drawdown_gate_require_both:
            # AND: both absolute AND relative must breach
            if abs_breach and rel_breach:
                # Near-rel-gate rescue: if relative margin is close to the gate,
                # override the failure (sector-driven drawdown, not idiosyncratic).
                if (ruleset.enable_dd_rel_margin_rescue
                        and dd_rel is not None):
                    rel_margin = dd_rel - ruleset.drawdown_rel_xbi_gate
                    if rel_margin > ruleset.dd_rel_margin_rescue_threshold:
                        dd_rel_margin_rescued = True
                    else:
                        reasons.append("deep_drawdown")
                else:
                    reasons.append("deep_drawdown")
        else:
            # OR: either breaches
            if abs_breach or rel_breach:
                reasons.append("deep_drawdown")
    else:
        # Soft mode: only hard floor exclusion
        if dd is not None and dd < ruleset.drawdown_hard_floor:
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
    return eligible, reasons, dd_rel_margin_rescued


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
    t1_val = _safe_float(t1_raw, default=None)
    if t1_val is not None:
        out["sponsor_tier1_count"] = int(t1_val)
    else:
        out["sponsor_tier1_count"] = ""

    # overlap_count: prefer smart_money_signal, blank if missing
    oc_raw = sms.get("overlap_count")
    oc_val = _safe_float(oc_raw, default=None)
    if oc_val is not None:
        out["sponsor_overlap_count"] = int(oc_val)
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

    # Coinvest conviction fields (computed by _convert_holdings_to_coinvest)
    out["coinvest_conviction"] = _safe_float(coinvest.get("conviction_overlap"), default=0.0)
    out["coinvest_tier1_conviction"] = _safe_float(coinvest.get("tier1_conviction_overlap"), default=0.0)
    out["coinvest_max_position_pct"] = _safe_float(coinvest.get("max_tier1_position_pct"), default=0.0)
    filing_age_raw = coinvest.get("days_since_latest_filing")
    # Guard: treat non-numeric as absent.  int() raises ValueError on NaN
    # and TypeError on non-numeric types, so no separate NaN check needed.
    if filing_age_raw is not None:
        try:
            filing_age = int(filing_age_raw)
        except (ValueError, TypeError):
            filing_age = None
    else:
        filing_age = None
    out["coinvest_filing_age_days"] = filing_age if filing_age is not None else ""
    if filing_age is None:
        out["coinvest_recency_state"] = ""
    elif filing_age <= 90:
        out["coinvest_recency_state"] = "fresh"
    else:
        out["coinvest_recency_state"] = "stale"

    # --- Catalyst (from top-level rec["catalyst_decay"]) ---
    cd = rec.get("catalyst_decay") or {}
    days = cd.get("days_to_catalyst")
    in_win = cd.get("in_optimal_window")
    # days_to_catalyst: 0 means "no upcoming catalyst", positive int means real distance
    # None/absent means "no data"
    days_val = _safe_float(days, default=None)
    if days_val is not None:
        out["catalyst_days"] = max(0, int(days_val))
    else:
        out["catalyst_days"] = ""
    if in_win is not None:
        out["catalyst_in_window"] = "1" if in_win else "0"
    else:
        out["catalyst_in_window"] = ""

    # Catalyst mode: explicit label for how catalyst proximity was determined
    if days_val is not None and days_val > 0:
        out["catalyst_mode"] = "specific_days"
    elif days_val == 0 and bool(in_win):
        out["catalyst_mode"] = "blended_window"
    elif days is not None or in_win is not None:
        out["catalyst_mode"] = "no_upcoming"
    else:
        out["catalyst_mode"] = "missing"

    # Catalyst strength: near/mid/far/missing (using catalyst_mid_days boundary)
    if out["catalyst_mode"] == "blended_window":
        out["catalyst_strength"] = "near"
    elif out["catalyst_mode"] == "specific_days" and days_val is not None and days_val > 0:
        if days_val <= ruleset.catalyst_near_days:
            out["catalyst_strength"] = "near"
        elif days_val <= ruleset.catalyst_mid_days:
            out["catalyst_strength"] = "mid"
        else:
            out["catalyst_strength"] = "far"
    else:
        out["catalyst_strength"] = "missing"

    # Catalyst decay weight: continuous proximity score
    if ruleset.catalyst_time_decay_mode == "logistic":
        if out["catalyst_mode"] == "blended_window":
            out["catalyst_decay_w"] = 1.0
        elif out["catalyst_mode"] == "specific_days" and days_val is not None and days_val >= 0:
            out["catalyst_decay_w"] = round(
                _logistic_decay(days_val, ruleset.catalyst_logistic_midpoint_days,
                                ruleset.catalyst_logistic_scale_days), 4
            )
        else:
            out["catalyst_decay_w"] = 0.0
        # Override catalyst_strength to match logistic decay thresholds
        if out["catalyst_mode"] in ("specific_days", "blended_window"):
            w = out["catalyst_decay_w"]
            if w >= 0.5:
                out["catalyst_strength"] = "near"
            elif w >= 0.2:
                out["catalyst_strength"] = "mid"
            else:
                out["catalyst_strength"] = "far"
    else:
        # Hard mode: derive from bucket for backward compat
        _hard_w = {"near": 1.0, "mid": 0.7, "far": 0.3, "missing": 0.0}
        out["catalyst_decay_w"] = _hard_w.get(out["catalyst_strength"], 0.0)

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
    dd_rel = _safe_float(df.get("drawdown_rel_xbi"))
    if dd_rel is not None and dd_rel < ruleset.drawdown_rel_xbi_gate:
        risk_flags.append("deep_drawdown_rel_xbi")
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


def _compute_cost_mult(
    est_cost_bps: Optional[float],
    ruleset: DecisionRuleset,
) -> Tuple[float, str]:
    """Compute cost multiplier and bucket label from estimated trading cost.

    Returns (multiplier, bucket_label).  When cost haircut is disabled
    or cost data is unavailable, returns (1.0, "").
    """
    cost_val = _safe_float(est_cost_bps)
    if not ruleset.enable_cost_haircut or cost_val is None:
        return 1.0, ""
    for upper, mult in ruleset.cost_haircut_buckets:
        if cost_val <= upper:
            return mult, f"<={int(upper)}bps"
    last_upper = ruleset.cost_haircut_buckets[-1][0] if ruleset.cost_haircut_buckets else 0
    return ruleset.cost_haircut_floor_mult, f">{int(last_upper)}bps"


def _compute_size_band(
    eligible: bool,
    effective_tier: str,
    optionality: Optional[float],
    overlays: Dict[str, Any],
    ruleset: DecisionRuleset,
    rec: Optional[Dict] = None,
    cost_mult: float = 1.0,
    cost_bucket: str = "",
    commercial_quality_pct: Optional[float] = None,
) -> Tuple[str, List[str]]:
    """Rule-based sizing band. Returns (band, list_of_reasons)."""
    if not eligible:
        return "XS", ["ineligible"]

    idx = 2  # Start at M (index 2)
    reasons: List[str] = []

    # Tier A with high optionality/quality → push toward L
    if effective_tier == "A" and optionality is not None and optionality >= ruleset.tier_a_optionality_floor:
        idx += 1
        reasons.append("tier_a_dev")
    elif effective_tier == "A" and optionality is None and commercial_quality_pct is not None and commercial_quality_pct >= ruleset.tier_a_commercial_floor:
        idx += 1
        reasons.append("tier_a_commercial")

    # Catalyst FAR lift (far data is better than missing — at least we have a date)
    cat_strength = overlays.get("catalyst_strength", "missing")
    if cat_strength == "far":
        idx += 1
        reasons.append("catalyst_far_lift")

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

    # Missingness size penalty (opt-in)
    if ruleset.enable_missingness_size_penalty:
        if _has_flag(overlays.get("risk_flags"), "drawdown_data_missing"):
            idx -= 1
            reasons.append("missing_drawdown")
        if effective_tier in ("A", "B") and overlays.get("sponsor_tier1_count") in ("", None):
            idx -= 1
            reasons.append("missing_sponsor")
        if effective_tier == "A" and overlays.get("catalyst_mode") == "missing":
            idx -= 1
            reasons.append("missing_catalyst")

    # Soft drawdown penalty
    if rec is not None and ruleset.drawdown_gate_mode == "soft":
        _df = rec.get("defensive_features") or {}
        _dd = _safe_float(_df.get("drawdown"))
        if _dd is not None and _dd < ruleset.drawdown_gate:
            idx += ruleset.drawdown_size_penalty  # negative → band downgrade
            reasons.append("drawdown_penalty")

    # Cost-aware band step-down (heavy haircut → drop one band)
    if cost_mult <= 0.70 and cost_bucket:
        idx -= 1
        reasons.append(f"cost_haircut_{cost_bucket}")

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
    catalyst_decay_w: float = 0.0,
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

    # Determine catalyst proximity and strength band.
    # days_to_catalyst > 0 means a specific upcoming catalyst with known distance.
    # days_to_catalyst == 0 + in_optimal_window == "1" means "blended proximity mode"
    #   (pipeline couldn't pin exact days but proximity scoring is active).
    # days_to_catalyst absent + in_optimal_window absent/False = no catalyst data.
    c_days_float = _safe_float(catalyst_days, default=0.0)
    has_specific_days = c_days_float > 0
    has_blended_window = catalyst_in_window == "1"
    has_catalyst_data = has_specific_days or has_blended_window

    # Catalyst strength: near/mid/far/missing
    if ruleset.catalyst_time_decay_mode == "logistic" and has_catalyst_data:
        # Smooth: map continuous decay_w back to discrete strength labels
        decay_w = _safe_float(catalyst_decay_w, default=0.0)
        if decay_w >= 0.5:
            strength = "near"
        elif decay_w >= 0.2:
            strength = "mid"
        else:
            strength = "far"
        is_actionable = decay_w >= 0.2
    elif has_blended_window:
        strength = "near"
        is_actionable = True
    elif has_specific_days:
        if c_days_float <= ruleset.catalyst_near_days:
            strength = "near"
        elif c_days_float <= ruleset.catalyst_mid_days:
            strength = "mid"
        else:
            strength = "far"
        is_actionable = strength in ("near", "mid")
    else:
        strength = "missing"
        is_actionable = False

    # Catalyst tag for tier_reason
    if has_blended_window and not has_specific_days:
        cat_tag = "catalyst_window"
    elif strength == "near":
        cat_tag = "catalyst_near"
    elif strength == "mid":
        cat_tag = "catalyst_mid"
    elif strength == "far":
        cat_tag = "catalyst_far"
    else:
        cat_tag = ""

    if has_catalyst_data:
        if optionality >= ruleset.tier_a_optionality_floor and is_actionable:
            return "A", f"high_opt+{cat_tag}"
        elif optionality >= ruleset.tier_a_optionality_floor:
            return "B", f"high_opt+{cat_tag}" if cat_tag else "high_opt+catalyst_far"
        elif optionality >= ruleset.tier_b_optionality_floor and is_actionable:
            return "B", f"mod_opt+{cat_tag}"
        elif optionality >= ruleset.tier_b_optionality_floor:
            return "C", f"mod_opt+{cat_tag}" if cat_tag else "mod_opt+catalyst_far"
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
# LAYER 4b — COMMERCIAL TIER
# =============================================================================

def _compute_tier_commercial(
    archetype: str,
    eligible: bool,
    quality_pct: Optional[float],
    catalyst_in_window: str,
    catalyst_days: Any,
    ruleset: DecisionRuleset,
    catalyst_decay_w: float = 0.0,
) -> Tuple[str, str]:
    """Compute commercial tier (A/B/C/D) and tier_reason.

    For non-drug-developer archetypes (commercial_* and platform_*).
    Mirrors _compute_tier_dev structure but uses commercial_quality_pct
    instead of optionality_pct_dev.

    Returns (tier_letter, tier_reason).
    """
    arch = str(archetype)
    if arch == "drug_developer" or not arch:
        return "", ""

    if not eligible:
        return "D", "ineligible"

    if quality_pct is None:
        return "C", "no_quality_data"

    # Determine catalyst proximity (same logic as _compute_tier_dev)
    c_days_float = _safe_float(catalyst_days, default=0.0)
    has_specific_days = c_days_float > 0
    has_blended_window = catalyst_in_window == "1"
    has_catalyst_data = has_specific_days or has_blended_window

    if ruleset.catalyst_time_decay_mode == "logistic" and has_catalyst_data:
        decay_w = _safe_float(catalyst_decay_w, default=0.0)
        if decay_w >= 0.5:
            strength = "near"
        elif decay_w >= 0.2:
            strength = "mid"
        else:
            strength = "far"
        is_actionable = decay_w >= 0.2
    elif has_blended_window:
        strength = "near"
        is_actionable = True
    elif has_specific_days:
        if c_days_float <= ruleset.catalyst_near_days:
            strength = "near"
        elif c_days_float <= ruleset.catalyst_mid_days:
            strength = "mid"
        else:
            strength = "far"
        is_actionable = strength in ("near", "mid")
    else:
        strength = "missing"
        is_actionable = False

    # Catalyst tag for tier_reason
    if has_blended_window and not has_specific_days:
        cat_tag = "catalyst_window"
    elif strength == "near":
        cat_tag = "catalyst_near"
    elif strength == "mid":
        cat_tag = "catalyst_mid"
    elif strength == "far":
        cat_tag = "catalyst_far"
    else:
        cat_tag = ""

    if has_catalyst_data:
        if quality_pct >= ruleset.tier_a_commercial_floor and is_actionable:
            return "A", f"high_quality+{cat_tag}"
        elif quality_pct >= ruleset.tier_a_commercial_floor:
            return "B", f"high_quality+{cat_tag}" if cat_tag else "high_quality+catalyst_far"
        elif quality_pct >= ruleset.tier_b_commercial_floor and is_actionable:
            return "B", f"mod_quality+{cat_tag}"
        elif quality_pct >= ruleset.tier_b_commercial_floor:
            return "C", f"mod_quality+{cat_tag}" if cat_tag else "mod_quality+catalyst_far"
        else:
            return "C", "low_quality"
    else:
        if quality_pct >= ruleset.tier_a_commercial_floor:
            return "B", "high_quality+no_catalyst_data"
        elif quality_pct >= ruleset.tier_b_commercial_floor:
            return "C", "mod_quality+no_catalyst_data"
        else:
            return "C", "low_quality+no_catalyst_data"


# =============================================================================
# ACTIONABLE ORDERING
# =============================================================================

# Priority mappings for sort key construction
_TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "": 4}
_CATALYST_MODE_ORDER = {"specific_days": 0, "blended_window": 1, "far_window": 2, "no_upcoming": 3, "missing": 4}
_MOM_STATE_ORDER = {"tailwind": 0, "neutral": 1, "headwind": 2}


def resolve_catalyst_priority(
    event_type: str,
    source: str,
    ruleset: "DecisionRuleset",
) -> int:
    """Map (event_type, source) → priority int via ruleset.catalyst_priority_map.

    Match rules: exact match on event_type and source, with ``"*"`` acting as
    wildcard.  First matching rule wins (rules are ordered).

    Returns ``catalyst_priority_default`` (9) when no dated catalyst, or
    ``catalyst_priority_unknown`` (99) when values are "unknown".
    """
    et = str(event_type).strip().upper() if event_type else ""
    src = str(source).strip().upper() if source else ""

    # No catalyst data → default (9)
    if et in ("", "NONE") and src in ("", "NONE"):
        return ruleset.catalyst_priority_default

    # Broken data → unknown (99)
    if et == "UNKNOWN" or src == "UNKNOWN":
        return ruleset.catalyst_priority_unknown

    for rule in ruleset.catalyst_priority_map:
        rule_et, rule_src, priority = rule
        et_match = (rule_et == "*" or rule_et.upper() == et)
        src_match = (rule_src == "*" or rule_src.upper() == src)
        if et_match and src_match:
            return int(priority)

    # No rule matched — return default
    return ruleset.catalyst_priority_default


# Fields injected by run_screen.py (not computed by the DE itself).
# compute_actionable_sort_key() reads these from decision_fields and defaults
# to 0.0 when absent, so the DE remains self-contained for unit testing.
_EXTERNAL_SORT_FIELDS: frozenset = frozenset({
    "clinical_score_z_tier",
    "coinvest_score_z",
    "inst_delta_z",
    "stage_bucket",
})


def compute_actionable_sort_key(
    decision_fields: Dict[str, Any],
    archetype: str,
    optionality: Optional[float],
    composite_rank: Optional[int],
    ticker: str,
    *,
    catalyst_event_type: str = "",
    catalyst_source: str = "",
    ruleset: Optional["DecisionRuleset"] = None,
    tiebreaker_pct: Optional[float] = None,
    alpha_raw: Optional[float] = None,
) -> Tuple:
    """Return a sort-key tuple for deterministic actionable ordering.

    Lower tuple values sort first. Eligible dev-stage tickers with the
    best tier, nearest catalyst, and highest optionality rank first.

    Sort behavior is controlled by ``ruleset.catalyst_priority_mode``:

    - **off**: priority neutral — catalyst mode/days dominate after tier.
    - **tiebreaker**: composite_rank dominates; priority only breaks ties
      among tickers with the same composite rank.
    - **blended**: small rank bonus applied per priority level — FDA tickers
      get ``effective_rank = comp_rank - bonus``, a gentle tilt that can
      leapfrog nearby ranks but not distant ones.

    External fields (see ``_EXTERNAL_SORT_FIELDS``): ``clinical_score_z_tier``,
    ``coinvest_score_z``, ``inst_delta_z``, and ``stage_bucket`` are injected
    by ``run_screen.py`` after the DE loop and default to 0.0 when absent.
    """
    eligible_val = decision_fields.get("eligible", "0")
    is_eligible = 0 if eligible_val == "1" else 1

    is_dev = 0 if archetype == "drug_developer" else 1

    rs = ruleset or DEFAULT_RULESET

    # Select tier based on tiering priority mode
    if rs.tiering_priority_mode == "tier_first":
        tier = decision_fields.get("tier_any", "") or decision_fields.get("tier_dev", "")
    else:
        tier = decision_fields.get("tier_dev", "")
    tier_ord = _TIER_ORDER.get(str(tier), 4)

    # Build prefix: tier_first puts tier before archetype
    if rs.tiering_priority_mode == "tier_first":
        prefix = (is_eligible, tier_ord, is_dev)
    else:
        prefix = (is_eligible, is_dev, tier_ord)

    mode = rs.catalyst_priority_mode

    # Resolve catalyst priority (needed for tiebreaker + blended modes)
    if mode != "off":
        cat_priority = resolve_catalyst_priority(
            catalyst_event_type, catalyst_source, rs,
        )
    else:
        cat_priority = 0  # neutral — no effect on ordering

    cat_mode = decision_fields.get("catalyst_mode", "missing")
    cat_mode_ord = _CATALYST_MODE_ORDER.get(str(cat_mode), 4)

    cat_days_raw = decision_fields.get("catalyst_days", "")
    cat_days = int(_safe_float(cat_days_raw, default=9999))

    opt_neg = -(_safe_float(optionality, default=0.0))

    sponsor_val = _safe_float(decision_fields.get("sponsor_tier1_count"), default=0.0)
    sponsor_neg = -int(sponsor_val)

    mom = decision_fields.get("mom_state", "neutral")
    mom_ord = _MOM_STATE_ORDER.get(str(mom), 1)

    _cr = _safe_float(composite_rank, default=None)
    comp_rank = int(_cr) if _cr is not None else 9999

    # Resolve anchor value based on sort_anchor mode
    if rs.sort_anchor in ("optionality_pct", "alpha_cohort"):
        anchor = -(tiebreaker_pct if tiebreaker_pct is not None else 0.0)  # higher pct → more negative → sorts first
    else:
        anchor = float(comp_rank)           # existing behavior

    # Missingness sort penalty: higher count sorts later (0 when disabled)
    missing_count = 0
    if rs.enable_missingness_sort_penalty:
        missing_count = int(_safe_float(decision_fields.get("missingness_penalty", 0), default=0))

    # Clinical sort signal: blended into anchor term (0 when disabled).
    # Uses tier-local z-score, gated by stage_bucket (early=0, mid/late active).
    # Positive-only mode: only boosts high-clinical, never penalises low.
    # Clamped to ±2.0 to prevent spiky z in small tier cohorts.
    clin_adj = 0.0
    if rs.enable_clinical_sort_signal:
        cz_tier = _safe_float(decision_fields.get("clinical_score_z_tier"), default=0.0)
        stage = str(decision_fields.get("stage_bucket", ""))
        stage_mult = dict(rs.clinical_stage_mults).get(stage, 0.0)
        if rs.clinical_positive_only:
            cz_eff = min(2.0, max(0.0, cz_tier))
        else:
            cz_eff = max(-2.0, min(2.0, cz_tier))
        clin_adj = rs.clinical_sort_weight * cz_eff * stage_mult

    # Coinvest sort tilt: blended into anchor (same pattern as clinical).
    # Uses cross-sectional z-score of tier1_count, positive-only default.
    # Clamped to ±2.0 to prevent spiky z in small cohorts.
    coinvest_adj = 0.0
    if rs.enable_coinvest_sort_signal:
        cz = _safe_float(decision_fields.get("coinvest_score_z"), default=0.0)
        if rs.coinvest_positive_only:
            cz_eff = min(2.0, max(0.0, cz))
        else:
            cz_eff = max(-2.0, min(2.0, cz))
        coinvest_adj = rs.coinvest_sort_weight * cz_eff

    # Institutional delta sort tilt: blended into anchor (same pattern).
    # Uses cross-sectional z-score of net_elite_holders_delta.
    # Clamped to ±2.0; positive-only default avoids penalising decreases.
    inst_adj = 0.0
    if rs.enable_institutional_sort_signal:
        iz = _safe_float(decision_fields.get("inst_delta_z"), default=0.0)
        if rs.institutional_positive_only:
            iz_eff = min(2.0, max(0.0, iz))
        else:
            iz_eff = max(-2.0, min(2.0, iz))
        inst_adj = rs.institutional_sort_weight * iz_eff

    # Alpha modifier signal (0.0 when mode=off or missing)
    _alpha_sig = _safe_float(alpha_raw, default=0.0)

    # --- Mode dispatch ---
    # prefix is (is_eligible, is_dev, tier_ord) in dev_first mode
    # or (is_eligible, tier_ord, is_dev) in tier_first mode
    # Alpha modifier adjustments (0.0 when mode=off)
    _alpha_within = rs.alpha_modifier_weight * _alpha_sig if rs.alpha_modifier_mode == "within_tier" else 0.0
    _alpha_tie = -_alpha_sig if rs.alpha_modifier_mode == "tiebreak" else 0.0

    if mode == "tiebreaker":
        # anchor dominates after tier; priority only breaks ties
        effective_comp_rank = anchor - clin_adj - coinvest_adj - inst_adj - _alpha_within
        return prefix + (
            effective_comp_rank,  # anchor with clinical tilt + alpha within_tier
            missing_count,  # fewer missing components first
            cat_priority,   # priority breaks comp_rank ties
            cat_days,       # ascending days
            opt_neg,        # descending optionality
            sponsor_neg,    # descending sponsor count
            mom_ord,        # tailwind < neutral < headwind
            cat_mode_ord,   # specific < blended < no_upcoming < missing
            _alpha_tie,     # alpha tiebreak (0 unless mode=tiebreak)
            ticker,         # alphabetic tiebreak
        )

    if mode == "blended":
        # Build bonus map from ruleset tuples
        bonus_map = dict(rs.catalyst_priority_rank_bonuses)
        bonus = bonus_map.get(cat_priority, 0.0)
        effective_comp_rank = anchor - bonus - clin_adj - coinvest_adj - inst_adj - _alpha_within
        return prefix + (
            effective_comp_rank,  # anchor with bonus + clinical tilt + alpha within_tier
            missing_count,        # fewer missing components first
            cat_mode_ord,         # specific < blended < no_upcoming < missing
            cat_days,             # ascending days
            opt_neg,              # descending optionality
            sponsor_neg,          # descending sponsor count
            mom_ord,              # tailwind < neutral < headwind
            cat_priority,         # priority as tiebreaker
            _alpha_tie,           # alpha tiebreak (0 unless mode=tiebreak)
            ticker,               # alphabetic tiebreak
        )

    # mode == "off" (default)
    effective_opt_neg = opt_neg - clin_adj - coinvest_adj - inst_adj - _alpha_within  # higher z → more negative → sorts earlier
    return prefix + (
        cat_priority,       # 0 (neutral) — no effect on ordering
        cat_mode_ord,       # specific < blended < no_upcoming < missing
        cat_days,           # ascending days (missing=9999)
        missing_count,      # fewer missing components first
        effective_opt_neg,  # optionality with clinical tilt (negated)
        sponsor_neg,        # descending sponsor count (negated)
        mom_ord,            # tailwind < neutral < headwind
        anchor,             # ascending composite rank (or negated pct)
        _alpha_tie,         # alpha tiebreak (0 unless mode=tiebreak)
        ticker,             # alphabetic tiebreak
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
        cm = _safe_float(row.get("cost_mult"), default=1.0)
        tm = _safe_float(row.get("catalyst_tilt_mult"), default=1.0)
        mm = _safe_float(row.get("mom_state_tilt_mult"), default=1.0)
        raw_weights.append(weights_map.get(str(band), 0.15) * cm * tm * mm)

    # Floor prevents explosive weights from near-zero floating-point totals.
    # Smallest real weight in production is ~0.15, so 1e-9 is safely below.
    _WEIGHT_TOTAL_FLOOR = 1e-9
    total = sum(raw_weights)
    if total <= _WEIGHT_TOTAL_FLOOR:
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
    "coinvest_conviction", "coinvest_tier1_conviction",
    "coinvest_max_position_pct", "coinvest_filing_age_days",
    "catalyst_days", "catalyst_in_window", "catalyst_mode", "catalyst_strength", "catalyst_decay_w",
    "runway_bucket", "mom_state", "risk_flags",
    "size_band", "size_reasons",
    "cost_mult", "cost_bucket", "cost_haircut_applied",
    "catalyst_tilt_mult", "catalyst_tilt_applied",
    "mom_state_tilt_mult", "mom_state_tilt_applied",
    "dd_rel_margin_rescued",
    "tier_dev", "tier_reason",
    "tier_commercial", "tier_any", "tier_any_reason",
    "missing_components", "missingness_penalty",
]


def compute_decision_fields(
    rec: Dict[str, Any],
    archetype: str,
    optionality_pct_dev: Optional[float] = None,
    ruleset: Optional[DecisionRuleset] = None,
    est_cost_bps: Optional[float] = None,
    commercial_quality_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute all decision engine fields for a single ticker.

    Args:
        rec: ranked_securities record from Module 5
        archetype: company archetype string
        optionality_pct_dev: pre-computed optionality percentile (float or None)
        ruleset: DecisionRuleset config (defaults to DEFAULT_RULESET)
        est_cost_bps: estimated round-trip trading cost in basis points (float or None)
        commercial_quality_pct: pre-computed quality percentile for commercial archetypes (float or None)

    Returns:
        dict with all decision engine column values
    """
    rs = ruleset or DEFAULT_RULESET

    # Layer 0 — Eligibility
    eligible, reasons, dd_rel_margin_rescued = _compute_eligibility(rec, rs)

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
        catalyst_decay_w=overlays.get("catalyst_decay_w", 0.0),
    )

    # Layer 4b — Commercial Tier
    tier_commercial, tier_commercial_reason = _compute_tier_commercial(
        archetype=archetype,
        eligible=eligible,
        quality_pct=commercial_quality_pct,
        catalyst_in_window=overlays.get("catalyst_in_window", ""),
        catalyst_days=overlays.get("catalyst_days", ""),
        ruleset=rs,
        catalyst_decay_w=overlays.get("catalyst_decay_w", 0.0),
    )
    tier_any = tier_dev or tier_commercial  # first non-empty
    tier_any_reason = tier_reason if tier_dev else tier_commercial_reason

    # Cost multiplier (opt-in, affects sizing band + weight)
    cost_mult, cost_bucket = _compute_cost_mult(est_cost_bps, rs)

    # Catalyst tilt multiplier (opt-in, affects weight only)
    catalyst_tilt_mult = 1.0
    if rs.enable_catalyst_tilt:
        if rs.catalyst_time_decay_mode == "logistic":
            # Continuous tilt: w=1.0→1.10, w=0.5→1.00, w=0.0→0.90
            decay_w = overlays.get("catalyst_decay_w", 0.0)
            catalyst_tilt_mult = 0.90 + 0.20 * decay_w
        else:
            tilt_map = dict(rs.catalyst_tilt_mults)
            cat_strength = overlays.get("catalyst_strength", "missing")
            catalyst_tilt_mult = tilt_map.get(cat_strength.upper(), 1.0)

    # Momentum state tilt multiplier (opt-in, affects weight only)
    mom_state_tilt_mult = 1.0
    if rs.enable_mom_state_tilt:
        mom_tilt_map = dict(rs.mom_state_tilt_mults)
        mom = overlays.get("mom_state", "neutral")
        mom_state_tilt_mult = mom_tilt_map.get(mom, 1.0)

    # Layer 3 — Sizing (effective tier depends on tiering priority mode)
    # For non-drug-developer archetypes, always use tier_any (tier_dev is empty).
    if archetype != "drug_developer":
        effective_tier = tier_any
    elif rs.tiering_priority_mode == "tier_first":
        effective_tier = tier_any
    else:
        effective_tier = tier_dev
    size_band, size_reasons = _compute_size_band(
        eligible=eligible,
        effective_tier=effective_tier,
        optionality=optionality_pct_dev,
        overlays=overlays,
        ruleset=rs,
        rec=rec,
        cost_mult=cost_mult,
        cost_bucket=cost_bucket,
        commercial_quality_pct=commercial_quality_pct,
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
        "tier_commercial": tier_commercial,
        "tier_any": tier_any,
        "tier_any_reason": tier_any_reason,
        "cost_mult": cost_mult,
        "cost_bucket": cost_bucket,
        "cost_haircut_applied": "1" if cost_mult < 1.0 else "0",
        "catalyst_tilt_mult": catalyst_tilt_mult,
        "catalyst_tilt_applied": "1" if catalyst_tilt_mult != 1.0 else "0",
        "mom_state_tilt_mult": mom_state_tilt_mult,
        "mom_state_tilt_applied": "1" if mom_state_tilt_mult != 1.0 else "0",
        "dd_rel_margin_rescued": "1" if dd_rel_margin_rescued else "0",
    }

    # Missingness tracking (always computed, penalties gated by ruleset toggles)
    missing = _compute_missing_components(fields)
    fields["missing_components"] = "|".join(missing) if missing else ""
    fields["missingness_penalty"] = len(missing)

    return fields


# =============================================================================
# GATE MARGIN DIAGNOSTICS (panel-only — NOT in DECISION_COLUMNS)
# =============================================================================

def compute_gate_margins(rec: Dict, ruleset: Optional[DecisionRuleset] = None) -> Dict[str, Any]:
    """Compute per-gate margin diagnostics for a single ticker.

    Mirrors _compute_eligibility() gate logic exactly but returns how far
    each gate is from its threshold (margin > 0 = safe, margin < 0 = breached)
    plus rescued_by_rel and counterfactual fields.

    Panel-only analytics — not stored in DECISION_COLUMNS or rankings.csv.
    """
    rs = ruleset or DEFAULT_RULESET
    df = rec.get("defensive_features") or {}
    dd = _safe_float(df.get("drawdown"))
    dd_rel = _safe_float(df.get("drawdown_rel_xbi"))

    # --- Per-gate results ---
    gates: List[Dict[str, Any]] = []

    # Gate 1: fundamental_red_flag (boolean)
    # Mirror the fallback from _compute_eligibility: if not pre-computed,
    # detect it from the rec fields directly.
    has_red_flag = rec.get("fundamental_red_flag")
    if has_red_flag is None:
        from defensive_overlay_adapter import detect_fundamental_red_flags
        has_red_flag, _ = detect_fundamental_red_flags(rec)
    has_red_flag = bool(has_red_flag)
    gates.append({
        "gate": "fundamental_red_flag",
        "passed": not has_red_flag,
        "margin": None,
        "actual": has_red_flag,
        "threshold": True,
    })

    # Gate 2: sev3 (boolean)
    severity = str(rec.get("severity", "")).upper()
    sev3_hit = severity == "SEV3"
    gates.append({
        "gate": "sev3",
        "passed": not sev3_hit,
        "margin": None,
        "actual": severity,
        "threshold": "SEV3",
    })

    # Gate 3: deep_drawdown_abs
    dd_abs_margin: Optional[float] = None
    if dd is not None:
        dd_abs_margin = round(dd - rs.drawdown_gate, 6)
    abs_passed = dd is None or dd >= rs.drawdown_gate
    gates.append({
        "gate": "deep_drawdown_abs",
        "passed": abs_passed,
        "margin": dd_abs_margin,
        "actual": dd,
        "threshold": rs.drawdown_gate,
    })

    # Gate 4: deep_drawdown_rel
    dd_rel_margin: Optional[float] = None
    if dd_rel is not None:
        dd_rel_margin = round(dd_rel - rs.drawdown_rel_xbi_gate, 6)
    rel_passed = dd_rel is None or dd_rel >= rs.drawdown_rel_xbi_gate
    gates.append({
        "gate": "deep_drawdown_rel",
        "passed": rel_passed,
        "margin": dd_rel_margin,
        "actual": dd_rel,
        "threshold": rs.drawdown_rel_xbi_gate,
    })

    # Gate 5: deep_drawdown (combined) — mirrors _compute_eligibility logic
    abs_breach = dd is not None and dd < rs.drawdown_gate
    rel_breach = dd_rel is not None and dd_rel < rs.drawdown_rel_xbi_gate

    if rs.drawdown_gate_mode == "hard":
        if dd_rel is None:
            dd_combined_passed = not abs_breach
            mode = "abs_only_fallback"
        elif rs.drawdown_gate_require_both:
            if abs_breach and rel_breach:
                # Mirror the rescue path from _compute_eligibility
                if (rs.enable_dd_rel_margin_rescue and dd_rel is not None):
                    rel_margin = dd_rel - rs.drawdown_rel_xbi_gate
                    dd_combined_passed = rel_margin > rs.dd_rel_margin_rescue_threshold
                else:
                    dd_combined_passed = False
            else:
                dd_combined_passed = True
            mode = "require_both"
        else:
            dd_combined_passed = not (abs_breach or rel_breach)
            mode = "require_either"
    else:
        hard_floor_breach = dd is not None and dd < rs.drawdown_hard_floor
        dd_combined_passed = not hard_floor_breach
        mode = "soft"

    gates.append({
        "gate": "deep_drawdown",
        "passed": dd_combined_passed,
        "mode": mode,
    })

    # Gate 6: adv_fail (boolean from flag scan)
    attn = rec.get("attn_flags") or []
    flags = rec.get("flags") or []
    all_flags = attn + flags
    adv_fail = False
    for f in all_flags:
        fl = str(f).lower()
        if "adv_fail" in fl or "liquidity_fail" in fl:
            adv_fail = True
            break
    gates.append({
        "gate": "adv_fail",
        "passed": not adv_fail,
        "margin": None,
        "actual": adv_fail,
        "threshold": True,
    })

    # --- Rescued by relative gate ---
    rescued_by_rel = (
        rs.drawdown_gate_mode == "hard"
        and dd_abs_margin is not None and dd_abs_margin < 0
        and dd_rel_margin is not None and dd_rel_margin >= 0
        and rs.drawdown_gate_require_both
    )

    # --- Effective gate order (matches _compute_eligibility output) ---
    # The raw per-gate abs/rel results stay in gates[] for diagnostics,
    # but eligibility is determined by the *effective* gate sequence:
    #   red_flag → sev3 → deep_drawdown (combined) → adv_fail
    _EFFECTIVE_GATES = ("fundamental_red_flag", "sev3", "deep_drawdown", "adv_fail")
    effective_pass = {g["gate"]: g["passed"] for g in gates if g["gate"] in _EFFECTIVE_GATES}

    first_failed_effective = ""
    for gname in _EFFECTIVE_GATES:
        if not effective_pass.get(gname, True):
            if not first_failed_effective:
                first_failed_effective = gname

    eligible = all(effective_pass.get(g, True) for g in _EFFECTIVE_GATES)

    # --- Counterfactual (only when ineligible) ---
    # Minimal: fewest field changes to flip eligibility.
    counterfactual: Dict[str, Any] = {}
    if not eligible:
        for gname in _EFFECTIVE_GATES:
            if effective_pass.get(gname, True):
                continue
            if gname == "fundamental_red_flag":
                counterfactual["fundamental_red_flag"] = False
            elif gname == "sev3":
                counterfactual["severity"] = "SEV2"
            elif gname == "deep_drawdown":
                if mode == "abs_only_fallback":
                    counterfactual["drawdown"] = rs.drawdown_gate
                elif mode == "require_both":
                    # AND: both must breach to fail → flip ONE to rescue.
                    # Pick the side requiring the smaller move.
                    abs_delta = abs(dd_abs_margin) if dd_abs_margin is not None else float("inf")
                    rel_delta = abs(dd_rel_margin) if dd_rel_margin is not None else float("inf")
                    if abs_delta <= rel_delta:
                        counterfactual["drawdown"] = rs.drawdown_gate
                    else:
                        counterfactual["drawdown_rel_xbi"] = rs.drawdown_rel_xbi_gate
                elif mode == "require_either":
                    # OR: either breach fails → flip every side that breached.
                    if abs_breach:
                        counterfactual["drawdown"] = rs.drawdown_gate
                    if rel_breach:
                        counterfactual["drawdown_rel_xbi"] = rs.drawdown_rel_xbi_gate
                elif mode == "soft":
                    counterfactual["drawdown"] = rs.drawdown_hard_floor
            elif gname == "adv_fail":
                counterfactual["flags"] = "remove_adv_fail"

    return {
        "dd_abs_margin": dd_abs_margin,
        "dd_rel_margin": dd_rel_margin,
        "rescued_by_rel": rescued_by_rel,
        "first_failed_gate": first_failed_effective,
        "first_failed_effective_gate": first_failed_effective,
        "gates": gates,
        "counterfactual": counterfactual,
    }


def compute_tier_margins(
    optionality: Optional[float],
    catalyst_strength: str,
    ruleset: Optional[DecisionRuleset] = None,
) -> Dict[str, Any]:
    """Compute tier-threshold margin diagnostics.

    Simple arithmetic: how far is optionality from A/B floors?
    Panel-only analytics — not stored in DECISION_COLUMNS or rankings.csv.
    """
    rs = ruleset or DEFAULT_RULESET

    opt_margin_a: Optional[float] = None
    opt_margin_b: Optional[float] = None
    if optionality is not None:
        opt_margin_a = round(optionality - rs.tier_a_optionality_floor, 6)
        opt_margin_b = round(optionality - rs.tier_b_optionality_floor, 6)

    actionable = str(catalyst_strength).strip() in ("near", "mid")

    return {
        "optionality_margin_a": opt_margin_a,
        "optionality_margin_b": opt_margin_b,
        "actionable_catalyst": actionable,
    }
