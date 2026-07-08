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
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from decision_engine_codes import canonicalize_reasons

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
    drawdown_gate_mode: str = "hard"  # "hard" (v1 behavior) or "soft" (sizing penalty)
    drawdown_size_penalty: int = -1  # band steps to subtract when soft + breached
    drawdown_hard_floor: float = -0.75  # hard-exclude even in soft mode if below this
    financials_missing_bypass_market_cap: float = 0.0  # USD raw; 0 = disabled

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
    catalyst_time_decay_mode: str = "hard"  # "hard" | "logistic"
    sparse_signal_mode: str = "legacy"  # "legacy" | "exclude_missing"
    catalyst_logistic_midpoint_days: int = 150  # sigmoid center
    catalyst_logistic_scale_days: float = 30.0  # sigmoid steepness
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
    catalyst_tilt_mults: tuple = (("NEAR", 1.10), ("MID", 1.05), ("FAR", 0.95), ("MISSING", 0.90))

    # Layer 3 — Catalyst type quality tilt (opt-in, multiplicative on weight)
    # Scales weight by event type quality: confirmed regulatory > pivotal data >
    # calendar milestones > softer signals.  See Spec 030.
    enable_catalyst_type_tilt: bool = False
    catalyst_type_mults: tuple = (
        ("T1", 1.00),  # Confirmed regulatory (PDUFA, AdCom, CRL, etc.)
        ("T2", 0.90),  # Pivotal data (DATA_READOUT, DATA_PRESENTATION)
        ("T3", 0.60),  # ClinTrials calendar (CT_PRIMARY_COMPLETION, CT_STUDY_COMPLETION)
        ("T4", 0.40),  # Softer signals (CT_RESULTS_POSTED, CT_STATUS_UPGRADE, etc.)
        ("T5", 0.50),  # Unknown / missing event type
    )

    # Layer 3 — Momentum state sizing tilt (opt-in, multiplicative on weight)
    # NOTE: default multipliers are all 1.0 (identity) — enabling this flag
    # without setting non-trivial multipliers has zero behavioral effect.
    enable_mom_state_tilt: bool = False
    mom_state_tilt_mults: tuple = (
        ("tailwind", 1.0),
        ("neutral", 1.0),
        ("headwind", 1.0),
    )

    # Oncology crowding penalty (opt-in, default OFF — Spec 033)
    # Penalizes oncology names in crowded indications. Only fires when
    # therapeutic_area == "oncology"; non-oncology names are unaffected.
    enable_oncology_crowding_penalty: bool = False
    oncology_crowding_weight: float = 0.3

    # Options verdict agreement tilt (opt-in, default OFF — Spec 038)
    # Uses fused 4-lens verdict composite as within-bucket tiebreaker.
    # Only fires in less_binary/build_window buckets. Research-phase.
    enable_options_verdict_tilt: bool = False
    options_verdict_weight: float = 0.3

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
    coinvest_sort_weight: float = 0.5  # scale factor (lower than clinical — advisory signal)
    coinvest_positive_only: bool = True  # only boost, never penalise
    coinvest_score_mode: str = "tier1_count"  # "tier1_count" (others deferred)

    # Institutional delta sort tilt (opt-in, default OFF)
    # When enabled, blends net_elite_holders_delta z-score into sort anchor,
    # following the coinvest pattern.  Positive-only: only boosts names with
    # increasing institutional ownership, never penalises decreasing.
    enable_institutional_sort_signal: bool = False
    institutional_sort_weight: float = 0.3
    institutional_positive_only: bool = True
    institutional_sort_min_nonzero_pct: float = 10.0  # coverage guard: zero z if <X% nonzero

    # Clinical Calendar Alpha v2 sort tilt (opt-in, default OFF)
    # When enabled, blends clinical_score_v2_z into sort anchor,
    # following the institutional/coinvest pattern. Positive-only default.
    enable_calendar_alpha_sort: bool = False
    calendar_alpha_sort_weight: float = 0.5
    # Clinical Calendar Alpha v2 sizing (opt-in, default OFF)
    enable_clinical_sizing: bool = False

    # Far-horizon catalyst (opt-in, off by default; set far_window_days > 0 to enable)
    far_window_days: int = 0  # max PCD horizon for far_window mode (0 = off)
    far_window_decay_mult: float = 0.15  # catalyst_decay_w for far_window tickers

    # Sort anchor (default composite_rank = current behavior)
    sort_anchor: str = (
        "composite_rank"  # "composite_rank" | "optionality_pct" | "alpha_cohort" | "options_anchor_hybrid" | "selector_score"
    )

    # Options anchor hybrid (opt-in via sort_anchor="options_anchor_hybrid", Spec 039)
    # Uses options_quality_composite when present, falls back to optionality_pct.
    options_anchor_regulatory_boost: float = 1.10
    options_anchor_near_catalyst_boost: float = 1.05

    # Alpha cohort scoring (opt-in via sort_anchor="alpha_cohort")
    alpha_cohort_table_path: str = "production_data/alpha_cohort_tables/v1.json"
    alpha_cohort_shrink_k: float = 50.0
    alpha_cohort_clip_min: float = -0.10
    alpha_cohort_clip_max: float = 0.10

    # Alpha cohort training configuration (adaptive alpha)
    alpha_train_mode: str = "trailing-6"  # "expanding" | "trailing-N" | "decay-H"
    alpha_train_min_train_dates: int = 6  # minimum training dates (>= 2)
    alpha_train_horizon: int = 84  # forward horizon in trading days (63|84|126)
    alpha_table_rebuild_policy: str = "never"  # "never" | "if_missing" | "daily"

    # Composite engine (default legacy = Module 5 composite scoring)
    composite_engine: str = "legacy"  # "legacy" | "alpha_cohort"

    # Commercial tiering (opt-in, default dev_first = current behavior)
    tiering_priority_mode: str = "dev_first"  # "dev_first" | "tier_first"
    tier_a_commercial_floor: float = 0.85  # quality_pct cutoff for commercial A
    tier_b_commercial_floor: float = 0.60  # quality_pct cutoff for commercial B

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

    # Alpha cohort percentile tiebreak — uses alpha_cohort_pct (unique per ticker, no
    # ceiling ties unlike alpha_raw).  Blended as a small secondary sort contribution:
    #   delta = alpha_cohort_tiebreak_weight * alpha_cohort_pct
    # 0 = disabled (default).  Ablation: w=0.02/0.05/0.10 on 2025 IS.
    alpha_cohort_tiebreak_weight: float = 0.0  # 0.0 = off; range [0.0, 0.50]

    # Binary sleeve risk cap (L3 enforcement, applied after normalization)
    # Defaults of 100.0 mean no behavioral change without explicit opt-in.
    binary_sleeve_max_weight_pct: float = 100.0  # aggregate cap (100 = disabled)
    binary_sleeve_per_name_max_pct: float = 100.0  # per-name cap (100 = disabled)
    binary_sleeve_days_threshold: int = 30  # catalyst_days cutoff for "binary"

    # Less-binary (core) sleeve construction mode.
    # Controls how names without dated catalysts are treated in sizing:
    #   "include"      — standard top-K selection + sizing (default, status quo)
    #   "equal_weight" — equal weight among eligible core names (beta sleeve)
    #   "bottom_k"     — contrarian: select worst-ranked K (research only)
    #   "exclude"      — zero weight for core names (alpha-only portfolio)
    less_binary_construction: str = "include"

    # Bucket hysteresis: prevent churn at 30/90/180 day boundaries.
    # When enabled, a ticker stays in its previous bucket until catalyst_days
    # crosses the exit threshold (boundary + 7 days).  Default OFF.
    enable_bucket_hysteresis: bool = False

    # Binary 91-180 tier flattening (opt-in, default OFF)
    # When enabled, removes tier discrimination from the sort key for names
    # in the binary_91_180 bucket (catalyst_bucket == "less_binary").
    # Tier labels are preserved for eligibility and labeling — only the
    # sort ordering is affected. Lets the anchor (optionality_pct) and
    # sort contributions dominate directly without tier gating.
    binary_91_180_flatten_tier_sort: bool = False

    # Binary 91-180 within-bucket re-ranking (opt-in, default "baseline")
    # Controls how names inside the less_binary bucket are re-ordered AFTER
    # the global sort key is computed.  Only affects intra-bucket ordering.
    #   "baseline":                   no change (current behavior)
    #   "quality_primary":            binary_quality_score dominates within bucket
    #   "quality_plus_institutional": binary_quality_score + institutional delta z
    #   "clinical_quality":           clinical_quality_composite for CLINICAL family
    #   "options_quality":            options_quality_composite for REGULATORY family
    #   "clinical_plus_options":      both clinical_quality (CLINICAL) + options_quality (REGULATORY)
    binary_91_180_sort_mode: str = (
        "baseline"  # "baseline" | "quality_primary" | "quality_plus_institutional" | "clinical_quality" | "options_quality" | "clinical_plus_options"
    )
    binary_91_180_quality_weight: float = 1.0  # scale for binary_quality_score contribution
    binary_91_180_institutional_weight: float = 0.3  # scale for inst_delta_z (quality_plus_institutional only)
    binary_91_180_clinical_quality_weight: float = 0.0  # scale for clinical_quality_composite (clinical_quality mode)
    binary_91_180_options_quality_weight: float = 0.0  # scale for options_quality_composite (options_quality mode)

    # Binary 0-90 within-bucket re-ranking (shadow, default "baseline")
    # Same pattern as binary_91_180 but for binary_now + build_window buckets.
    # Only affects intra-bucket ordering within the near-term binary names.
    #   "baseline":                   no change (current behavior)
    #   "quality_primary":            binary_quality_score dominates within bucket
    #   "quality_plus_institutional": binary_quality_score + institutional delta z
    binary_now_sort_mode: str = "baseline"
    binary_now_quality_weight: float = 1.0
    binary_now_institutional_weight: float = 0.3

    # Build-window clinical sort tilt (shadow, default "baseline")
    # Bucket-gated version of clinical sort: only applies inside build_window.
    # Uses clinical_score_z_tier (PIT-safe), same clamp/stage logic as global
    # clinical sort but restricted to one sleeve.
    #   "baseline":     no change
    #   "clinical_z":   clinical_score_z_tier sort tilt in build_window only
    build_window_clinical_mode: str = "baseline"
    build_window_clinical_weight: float = 0.5

    # Put/call ratio penalty (shadow, default OFF)
    # Applies a negative sort penalty for high pre-event put/call ratio,
    # gated to build_window bucket + CLINICAL family only.
    #   "off":       no penalty (default)
    #   "tiebreak":  small last-key penalty (safest live read)
    pcr_penalty_mode: str = "off"
    pcr_penalty_weight: float = 0.25

    # Event EV Promotion Ladder (Spec 061, all opt-in, default OFF)
    # Staged promotion of Event EV into production decisions:
    #   "off":            no Event EV influence (diagnostic only, Spec 060)
    #   "tiebreaker":     EV breaks ties among names with identical sort keys
    #   "rank_overlay":   bounded sort contribution (like coinvest/institutional)
    #   "sizing_overlay": multiplicative sizing tilt on target_weight_pct
    #   "composite":      full composite replacement (NOT IMPLEMENTED, placeholder)
    event_ev_stage: str = "off"

    # Stage 2 — Rank overlay parameters
    event_ev_rank_overlay_weight: float = 0.0  # [0.0, 1.0]; 0.0 = no overlay
    event_ev_rank_overlay_cap: float = 0.15  # max |delta| on anchor scale
    event_ev_min_analog_confidence: str = "ok"  # "ok" | "low" | "insufficient"

    # Stage 3 — Sizing tilt parameters (multipliers on target_weight_pct)
    event_ev_sizing_tilt_mults: tuple = (
        ("high_ev", 1.0),
        ("mid_ev", 1.0),
        ("low_ev", 1.0),
        ("no_ev", 1.0),
    )
    event_ev_sizing_high_threshold: float = 3.0  # ds_adj_ev % above → high_ev
    event_ev_sizing_low_threshold: float = -1.0  # ds_adj_ev % below → low_ev

    # Portfolio mechanics — rebalance buffer for top-K evaluation.
    # Existing holdings stay unless they fall below rank K + buffer.
    # Reduces turnover from small rank oscillations.  0 = disabled.
    rebalance_buffer_ranks: int = 0  # 0 = off; range [0, 50]

    # Construction risk controls (Spec 050, all opt-in / disabled by default)
    # Per-name weight cap as % of portfolio. 100.0 = disabled.
    construction_name_cap_pct: float = 100.0
    # Weight cap for names with liquidity risk flags. 100.0 = disabled.
    construction_liquidity_cap_pct: float = 100.0
    # Exclude names with est_cost_bps above this threshold. 0.0 = disabled.
    construction_cost_gate_bps: float = 0.0

    # Actionable ordering — catalyst priority mode (supersedes enable_catalyst_priority)
    catalyst_priority_mode: str = "off"  # "off" | "tiebreaker" | "blended"
    catalyst_priority_rank_bonuses: tuple = (  # blended mode: priority → rank bonus
        (1, 5.0),  # FDA events: effective_rank -= 5
        (2, 2.0),  # CTGOV/SEC readout: effective_rank -= 2
        (3, 0.0),  # corporate/ongoing
        (9, 0.0),  # no catalyst
        (99, 0.0),  # unknown
    )

    def __post_init__(self):
        # Validate cost_haircut_buckets: ascending thresholds
        buckets = self.cost_haircut_buckets
        if buckets:
            for i in range(1, len(buckets)):
                if buckets[i][0] <= buckets[i - 1][0]:
                    raise ValueError(f"cost_haircut_buckets thresholds must be strictly ascending: {buckets}")
        # Validate cost_haircut_floor_mult in (0, 1]
        if not (0 < self.cost_haircut_floor_mult <= 1.0):
            raise ValueError(f"cost_haircut_floor_mult must be in (0, 1], got {self.cost_haircut_floor_mult}")
        # Validate cost_impact_cap_bps > 0
        if self.cost_impact_cap_bps <= 0:
            raise ValueError(f"cost_impact_cap_bps must be > 0, got {self.cost_impact_cap_bps}")
        # Validate catalyst_tilt_mults: keys must be valid bands, mults > 0
        valid_bands = {"NEAR", "MID", "FAR", "MISSING"}
        for band, mult in self.catalyst_tilt_mults:
            if band not in valid_bands:
                raise ValueError(
                    f"catalyst_tilt_mults: unknown band '{band}', " f"expected one of {sorted(valid_bands)}"
                )
            if mult <= 0:
                raise ValueError(f"catalyst_tilt_mults: mult must be > 0, got {mult} for '{band}'")
        # Validate catalyst_type_mults: keys must be valid tiers, mults > 0
        valid_type_tiers = {"T1", "T2", "T3", "T4", "T5"}
        for tier, mult in self.catalyst_type_mults:
            if tier not in valid_type_tiers:
                raise ValueError(
                    f"catalyst_type_mults: unknown tier '{tier}', " f"expected one of {sorted(valid_type_tiers)}"
                )
            if mult < 0:
                raise ValueError(f"catalyst_type_mults: mult must be >= 0, got {mult} for '{tier}'")
        # Validate mom_state_tilt_mults: keys must be valid states, mults > 0
        valid_mom_states = {"tailwind", "neutral", "headwind"}
        for state, mult in self.mom_state_tilt_mults:
            if state not in valid_mom_states:
                raise ValueError(
                    f"mom_state_tilt_mults: unknown state '{state}', " f"expected one of {sorted(valid_mom_states)}"
                )
            if mult <= 0:
                raise ValueError(f"mom_state_tilt_mults: mult must be > 0, got {mult} for '{state}'")
        # Validate catalyst_time_decay_mode
        if self.catalyst_time_decay_mode not in ("hard", "logistic"):
            raise ValueError(
                f"catalyst_time_decay_mode must be 'hard' or 'logistic', " f"got '{self.catalyst_time_decay_mode}'"
            )
        if self.catalyst_logistic_scale_days <= 0:
            raise ValueError(f"catalyst_logistic_scale_days must be > 0, " f"got {self.catalyst_logistic_scale_days}")
        # Validate sparse_signal_mode
        if self.sparse_signal_mode not in ("legacy", "exclude_missing"):
            raise ValueError(
                f"sparse_signal_mode must be 'legacy' or 'exclude_missing', " f"got '{self.sparse_signal_mode}'"
            )
        # Validate catalyst_priority_mode
        if self.catalyst_priority_mode not in ("off", "tiebreaker", "blended"):
            raise ValueError(
                f"catalyst_priority_mode must be 'off', 'tiebreaker', or 'blended', "
                f"got '{self.catalyst_priority_mode}'"
            )
        # Validate sort_anchor
        _VALID_ANCHORS = (
            "composite_rank",
            "optionality_pct",
            "alpha_cohort",
            "options_anchor_hybrid",
            "selector_score",
        )
        if self.sort_anchor not in _VALID_ANCHORS:
            raise ValueError(f"sort_anchor must be one of {_VALID_ANCHORS}, " f"got '{self.sort_anchor}'")
        # Validate composite_engine
        if self.composite_engine not in ("legacy", "alpha_cohort"):
            raise ValueError(f"composite_engine must be 'legacy' or 'alpha_cohort', " f"got '{self.composite_engine}'")
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
                f"alpha_train_mode must be 'expanding', 'trailing-N', or 'decay-H', " f"got '{self.alpha_train_mode}'"
            )
        # Validate alpha_train_min_train_dates
        if self.alpha_train_min_train_dates < 2:
            raise ValueError(f"alpha_train_min_train_dates must be >= 2, " f"got {self.alpha_train_min_train_dates}")
        # Validate alpha_train_horizon
        if self.alpha_train_horizon not in (63, 84, 126):
            raise ValueError(f"alpha_train_horizon must be 63, 84, or 126, " f"got {self.alpha_train_horizon}")
        # Validate alpha_table_rebuild_policy
        if self.alpha_table_rebuild_policy not in ("never", "if_missing", "daily"):
            raise ValueError(
                f"alpha_table_rebuild_policy must be 'never', 'if_missing', or 'daily', "
                f"got '{self.alpha_table_rebuild_policy}'"
            )
        # Validate alpha_cohort_tiebreak_weight
        if not (0.0 <= self.alpha_cohort_tiebreak_weight <= 0.50):
            raise ValueError(
                f"alpha_cohort_tiebreak_weight must be in [0.0, 0.50], " f"got {self.alpha_cohort_tiebreak_weight}"
            )
        # Validate rebalance_buffer_ranks
        if not (0 <= self.rebalance_buffer_ranks <= 50):
            raise ValueError(f"rebalance_buffer_ranks must be in [0, 50], " f"got {self.rebalance_buffer_ranks}")
        # Validate tiering_priority_mode
        if self.tiering_priority_mode not in ("dev_first", "tier_first"):
            raise ValueError(
                f"tiering_priority_mode must be 'dev_first' or 'tier_first', " f"got '{self.tiering_priority_mode}'"
            )
        # Validate coinvest_score_mode (only tier1_count implemented)
        if self.coinvest_score_mode not in ("tier1_count",):
            raise ValueError(f"coinvest_score_mode must be 'tier1_count', " f"got '{self.coinvest_score_mode}'")
        # Validate drawdown_gate_mode
        if self.drawdown_gate_mode not in ("hard", "soft"):
            raise ValueError(f"drawdown_gate_mode must be 'hard' or 'soft', " f"got '{self.drawdown_gate_mode}'")
        # Validate dd_rel_margin_rescue_threshold: must be <= 0
        if self.dd_rel_margin_rescue_threshold > 0:
            raise ValueError(
                f"dd_rel_margin_rescue_threshold must be <= 0, " f"got {self.dd_rel_margin_rescue_threshold}"
            )
        # Validate far_window_days: must be >= 0
        if self.far_window_days < 0:
            raise ValueError(f"far_window_days must be >= 0, got {self.far_window_days}")
        # Validate far_window_decay_mult: must be in (0, 1]
        if not (0.0 < self.far_window_decay_mult <= 1.0):
            raise ValueError(f"far_window_decay_mult must be in (0, 1], " f"got {self.far_window_decay_mult}")
        # Validate binary sleeve caps: must be in (0, 100]
        if not (0.0 < self.binary_sleeve_max_weight_pct <= 100.0):
            raise ValueError(
                f"binary_sleeve_max_weight_pct must be in (0, 100], " f"got {self.binary_sleeve_max_weight_pct}"
            )
        if not (0.0 < self.binary_sleeve_per_name_max_pct <= 100.0):
            raise ValueError(
                f"binary_sleeve_per_name_max_pct must be in (0, 100], " f"got {self.binary_sleeve_per_name_max_pct}"
            )
        if self.binary_sleeve_days_threshold < 0:
            raise ValueError(f"binary_sleeve_days_threshold must be >= 0, " f"got {self.binary_sleeve_days_threshold}")
        _valid_lb_modes = {"include", "equal_weight", "bottom_k", "exclude"}
        if self.less_binary_construction not in _valid_lb_modes:
            raise ValueError(
                f"less_binary_construction must be one of {_valid_lb_modes}, " f"got {self.less_binary_construction!r}"
            )
        # Validate event_ev_stage (Spec 061)
        _valid_ev_stages = ("off", "tiebreaker", "rank_overlay", "sizing_overlay", "composite")
        if self.event_ev_stage not in _valid_ev_stages:
            raise ValueError(f"event_ev_stage must be one of {_valid_ev_stages}, " f"got '{self.event_ev_stage}'")
        if not (0.0 <= self.event_ev_rank_overlay_weight <= 1.0):
            raise ValueError(
                f"event_ev_rank_overlay_weight must be in [0.0, 1.0], " f"got {self.event_ev_rank_overlay_weight}"
            )
        if not (0.0 <= self.event_ev_rank_overlay_cap <= 1.0):
            raise ValueError(
                f"event_ev_rank_overlay_cap must be in [0.0, 1.0], " f"got {self.event_ev_rank_overlay_cap}"
            )
        _valid_analog_conf = ("ok", "low", "insufficient")
        if self.event_ev_min_analog_confidence not in _valid_analog_conf:
            raise ValueError(
                f"event_ev_min_analog_confidence must be one of {_valid_analog_conf}, "
                f"got '{self.event_ev_min_analog_confidence}'"
            )
        # Validate event_ev_sizing_tilt_mults
        _valid_ev_buckets = {"high_ev", "mid_ev", "low_ev", "no_ev"}
        for bucket, mult in self.event_ev_sizing_tilt_mults:
            if bucket not in _valid_ev_buckets:
                raise ValueError(
                    f"event_ev_sizing_tilt_mults: unknown bucket '{bucket}', "
                    f"expected one of {sorted(_valid_ev_buckets)}"
                )
            if mult <= 0:
                raise ValueError(f"event_ev_sizing_tilt_mults: mult must be > 0, got {mult} for '{bucket}'")

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
        # Schema version guard: log if the file has a version we don't recognize
        _sv = d.get("schema_version", "")
        if _sv and not _sv.startswith("v1."):
            import logging as _lg

            _lg.getLogger(__name__).warning(
                "Ruleset %s has schema_version=%s (expected v1.x) — "
                "unknown fields will be ignored but behavior may differ",
                path,
                _sv,
            )
        # Compute stable hash from raw dict BEFORE type conversions.
        # Uses the same compact-separator format as _canonical_json.
        file_hash = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
        # Convert sizing_weights dict back to tuple-of-tuples
        if "sizing_weights" in d and isinstance(d["sizing_weights"], dict):
            d["sizing_weights"] = tuple((k, v) for k, v in sorted(d["sizing_weights"].items()))
        # Convert cost_haircut_buckets list-of-lists back to tuple-of-tuples
        if "cost_haircut_buckets" in d and isinstance(d["cost_haircut_buckets"], list):
            d["cost_haircut_buckets"] = tuple(tuple(pair) for pair in d["cost_haircut_buckets"])
        # Convert catalyst_tilt_mults list-of-lists back to tuple-of-tuples
        if "catalyst_tilt_mults" in d and isinstance(d["catalyst_tilt_mults"], list):
            d["catalyst_tilt_mults"] = tuple(tuple(pair) for pair in d["catalyst_tilt_mults"])
        # Convert catalyst_type_mults list-of-lists back to tuple-of-tuples
        if "catalyst_type_mults" in d and isinstance(d["catalyst_type_mults"], list):
            d["catalyst_type_mults"] = tuple(tuple(pair) for pair in d["catalyst_type_mults"])
        # Convert mom_state_tilt_mults list-of-lists back to tuple-of-tuples
        if "mom_state_tilt_mults" in d and isinstance(d["mom_state_tilt_mults"], list):
            d["mom_state_tilt_mults"] = tuple(tuple(pair) for pair in d["mom_state_tilt_mults"])
        # Convert catalyst_priority_map list-of-lists back to tuple-of-tuples
        if "catalyst_priority_map" in d and isinstance(d["catalyst_priority_map"], list):
            d["catalyst_priority_map"] = tuple(tuple(rule) for rule in d["catalyst_priority_map"])
        # Convert catalyst_priority_rank_bonuses list-of-lists back to tuple-of-tuples
        if "catalyst_priority_rank_bonuses" in d and isinstance(d["catalyst_priority_rank_bonuses"], list):
            d["catalyst_priority_rank_bonuses"] = tuple(tuple(pair) for pair in d["catalyst_priority_rank_bonuses"])
        # Convert clinical_stage_mults list-of-lists back to tuple-of-tuples
        if "clinical_stage_mults" in d and isinstance(d["clinical_stage_mults"], list):
            d["clinical_stage_mults"] = tuple(tuple(pair) for pair in d["clinical_stage_mults"])
        # Convert event_ev_sizing_tilt_mults list-of-lists back to tuple-of-tuples (Spec 061)
        if "event_ev_sizing_tilt_mults" in d and isinstance(d["event_ev_sizing_tilt_mults"], list):
            d["event_ev_sizing_tilt_mults"] = tuple(tuple(pair) for pair in d["event_ev_sizing_tilt_mults"])
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

VERSION = "v1.4.0"  # Spec 050: A4 selector + ranker promoted

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


def _safe_decimal(val, default=Decimal("0")) -> Decimal:
    """Convert a value to Decimal for scoring paths (CCFT determinism).

    Uses str() intermediary to avoid float representation artifacts.
    NaN and missing values return the default.
    """
    if val is None:
        return default
    if isinstance(val, Decimal):
        return val
    try:
        if isinstance(val, str) and val.strip() == "":
            return default
        # Convert via str to avoid float→Decimal imprecision
        v = float(val)
        if v != v:  # NaN check
            return default
        return Decimal(str(v))
    except (ValueError, TypeError, ArithmeticError):
        return default


_D = Decimal  # shorthand for literals in scoring paths
_D0 = Decimal("0")
_D1 = Decimal("1")
_D2 = Decimal("2")
_D_NEG2 = Decimal("-2")
_D_HALF = Decimal("0.5")


def _clamp_d(val: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal to [lo, hi]."""
    return max(lo, min(hi, val))


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
    rec: Dict,
    ruleset: DecisionRuleset,
    market_cap: Optional[float] = None,
) -> Tuple[bool, List[str], bool, bool]:
    """Check eligibility gates.

    Hard gates only — survivability, liquidity, drawdown.
    Confidence is deliberately NOT a hard gate (sparse dev-stage coverage
    would exclude the exact optionality names we want to keep).

    Returns (eligible, list_of_reasons, dd_rel_margin_rescued, financials_bypassed).
    dd_rel_margin_rescued is True when a drawdown failure was overridden
    because the relative margin was close to the gate (sector-driven drawdown).
    financials_bypassed is True when the financials_missing gate was skipped
    because market_cap exceeds the bypass threshold.
    """
    reasons: List[str] = []
    dd_rel_margin_rescued = False
    financials_bypassed = False

    # Gate 0: financials missing — if survivability data is truly absent,
    # don't trust red flag checks (they'd produce false positives).
    # The coverage flags can be misleading for profitable companies where
    # cash arrives via MarketableSecurities (not cash_and_equivalents) and
    # positive OCF means zero burn (not missing burn data).  Use cash_total
    # from the metrics as the ground-truth indicator of data presence.
    surv = rec.get("survivability_signal") or {}
    surv_coverage = surv.get("coverage") or []
    surv_metrics = surv.get("metrics") or {}
    cash_total = surv_metrics.get("cash_total") or 0
    if "missing_cash" in surv_coverage and "missing_burn_data" in surv_coverage and cash_total <= 0:
        # Mega-cap bypass: skip financials_missing for very large names
        _bypass = (
            ruleset.financials_missing_bypass_market_cap > 0
            and market_cap is not None
            and market_cap >= ruleset.financials_missing_bypass_market_cap
        )
        if _bypass:
            financials_bypassed = True
        else:
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
                if ruleset.enable_dd_rel_margin_rescue and dd_rel is not None:
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
    return eligible, reasons, dd_rel_margin_rescued, financials_bypassed


# =============================================================================
# CATALYST BUCKET
# =============================================================================

# Bucket thresholds (days)
BUCKET_BINARY_NOW_MAX = 30
BUCKET_BUILD_WINDOW_MAX = 90
BUCKET_LESS_BINARY_MAX = 180

# Hysteresis buffer: once in a bucket, don't exit until crossing
# threshold + buffer.  Prevents calendar-driven churn at boundaries.
# Only active when enable_bucket_hysteresis=True in ruleset.
BUCKET_HYSTERESIS_BUFFER = 7  # days

_BUCKET_CORE_MODES = frozenset({"no_upcoming", "missing"})


def assign_catalyst_bucket(catalyst_days: Optional[float], catalyst_mode: str) -> str:
    """Assign a canonical catalyst horizon bucket.

    Returns one of: "binary_now", "build_window", "less_binary", "core".
    Mirrors action_packet.assign_bucket() but lives in DE for snapshot output.
    """
    if catalyst_mode in _BUCKET_CORE_MODES:
        return "core"
    if catalyst_days is None:
        return "core"
    if catalyst_days <= BUCKET_BINARY_NOW_MAX:
        return "binary_now"
    if catalyst_days <= BUCKET_BUILD_WINDOW_MAX:
        return "build_window"
    if catalyst_days <= BUCKET_LESS_BINARY_MAX:
        return "less_binary"
    return "core"


def assign_catalyst_bucket_with_hysteresis(
    catalyst_days: Optional[float],
    catalyst_mode: str,
    prev_bucket: Optional[str] = None,
    buffer: int = BUCKET_HYSTERESIS_BUFFER,
) -> str:
    """Assign bucket with hysteresis to prevent boundary churn.

    When ``prev_bucket`` is provided, the ticker only exits its current
    bucket once catalyst_days crosses the boundary + buffer.  This keeps
    the action lists stable when catalyst_days drifts by ±1-2 days.

    Entry thresholds (same as assign_catalyst_bucket):
        binary_now:    days <= 30
        build_window:  days <= 90
        less_binary:   days <= 180

    Exit thresholds (entry + buffer):
        binary_now  → build_window:  days >= 30 + buffer (37)
        build_window → less_binary:  days >= 90 + buffer (97)
        less_binary → core:          days >= 180 + buffer (187)

    Returns one of: "binary_now", "build_window", "less_binary", "core".
    """
    # Without previous state, fall through to standard assignment
    if prev_bucket is None:
        return assign_catalyst_bucket(catalyst_days, catalyst_mode)

    if catalyst_mode in _BUCKET_CORE_MODES:
        return "core"
    if catalyst_days is None:
        return "core"

    # Apply hysteresis: stay in prev bucket unless past exit threshold
    if prev_bucket == "binary_now":
        if catalyst_days <= BUCKET_BINARY_NOW_MAX + buffer:
            return "binary_now"
    elif prev_bucket == "build_window":
        # Can re-enter binary_now (days decreased) or exit to less_binary
        if catalyst_days <= BUCKET_BINARY_NOW_MAX:
            return "binary_now"
        if catalyst_days <= BUCKET_BUILD_WINDOW_MAX + buffer:
            return "build_window"
    elif prev_bucket == "less_binary":
        if catalyst_days <= BUCKET_BINARY_NOW_MAX:
            return "binary_now"
        if catalyst_days <= BUCKET_BUILD_WINDOW_MAX:
            return "build_window"
        if catalyst_days <= BUCKET_LESS_BINARY_MAX + buffer:
            return "less_binary"

    # Standard assignment for new entries or when past exit threshold
    return assign_catalyst_bucket(catalyst_days, catalyst_mode)


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
    # Far-out relabel: specific_days > 540 → no_upcoming (prevents "dated catalysts"
    # 3+ years out from contaminating near-term action lists)
    FAR_OUT_DAYS_THRESHOLD = 540
    if days_val is not None and days_val > FAR_OUT_DAYS_THRESHOLD:
        out["catalyst_mode"] = "no_upcoming"
    elif days_val is not None and days_val > 0:
        out["catalyst_mode"] = "specific_days"
    elif days_val == 0 and bool(in_win):
        out["catalyst_mode"] = "blended_window"
    elif days is not None or in_win is not None:
        out["catalyst_mode"] = "no_upcoming"
    else:
        out["catalyst_mode"] = "missing"

    # Catalyst bucket: canonical horizon classification for action lists
    out["catalyst_bucket"] = assign_catalyst_bucket(days_val, out["catalyst_mode"])

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
                _logistic_decay(
                    days_val, ruleset.catalyst_logistic_midpoint_days, ruleset.catalyst_logistic_scale_days
                ),
                4,
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
    elif (
        effective_tier == "A"
        and optionality is None
        and commercial_quality_pct is not None
        and commercial_quality_pct >= ruleset.tier_a_commercial_floor
    ):
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
# CATALYST TYPE QUALITY TIER (Spec 030)
# =============================================================================

# Event types → quality tier mapping.  First match wins (case-insensitive).
_CATALYST_TYPE_TIER_MAP: dict = {
    # T1: Confirmed regulatory — date-certain, binary outcome
    "FDA_PDUFA_DATE": "T1",
    "FDA_ADCOM": "T1",
    "FDA_APPROVAL": "T1",
    "FDA_CRL": "T1",
    "FDA_RTF": "T1",
    "FDA_DECISION": "T1",
    "FDA_SUBMISSION": "T1",
    "FDA_DESIGNATION": "T1",
    "EMA_AGENDA": "T1",
    "EMA_OUTCOME": "T1",
    "EMA_COMMITTEE_AGENDA": "T1",
    "EMA_COMMITTEE_OUTCOME": "T1",
    # T2: Pivotal data — high conviction, date less certain
    "DATA_READOUT": "T2",
    "DATA_PRESENTATION": "T2",
    "DATA_PUBLICATION": "T2",
    # T3: ClinTrials calendar — milestones, often noisy/delayed
    "CT_PRIMARY_COMPLETION": "T3",
    "CT_STUDY_COMPLETION": "T3",
    "CLINICAL_PCD": "T3",
    "CLINICAL_CD": "T3",
    # T4: Softer signals — activity indicators, not event dates
    "CT_RESULTS_POSTED": "T4",
    "CT_DATE_CONFIRMED_ACTUAL": "T4",
    "CT_TIMELINE_PULLIN": "T4",
    "CT_STATUS_UPGRADE": "T4",
    "CLINICAL_HOLD": "T4",
    "SAFETY_SIGNAL": "T4",
    "FDA_WARNING_LETTER": "T4",
    "CT_TRIAL_TERMINATED": "T4",
    "CT_TRIAL_WITHDRAWN": "T4",
}


def classify_catalyst_type_tier(event_type: str) -> str:
    """Map catalyst_event_type to a quality tier (T1–T5).

    T1 = confirmed regulatory, T2 = pivotal data, T3 = calendar milestone,
    T4 = softer signal, T5 = unknown/missing.  See Spec 030.
    """
    if not event_type or not event_type.strip():
        return "T5"
    return _CATALYST_TYPE_TIER_MAP.get(event_type.strip(), "T5")


# =============================================================================
# CATALYST STRENGTH RESOLUTION (shared by dev + commercial tier)
# =============================================================================


def _resolve_catalyst_strength(
    catalyst_in_window: str,
    catalyst_days: Any,
    ruleset: DecisionRuleset,
    catalyst_decay_w: float = 0.0,
) -> Tuple[str, bool, bool, str]:
    """Resolve catalyst proximity into strength band, actionability, and reason tag.

    Determines catalyst proximity and strength band from raw inputs:
      - days_to_catalyst > 0 → specific upcoming catalyst with known distance
      - days_to_catalyst == 0 + in_optimal_window == "1" → blended proximity mode
      - both absent → no catalyst data

    Returns (strength, is_actionable, has_catalyst_data, cat_tag) where:
      - strength: "near" | "mid" | "far" | "missing"
      - is_actionable: True when strength is near or mid
      - has_catalyst_data: True when any catalyst proximity info exists
      - cat_tag: reason fragment for tier_reason (e.g. "catalyst_near")
    """
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

    return strength, is_actionable, has_catalyst_data, cat_tag


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

    strength, is_actionable, has_catalyst_data, cat_tag = _resolve_catalyst_strength(
        catalyst_in_window,
        catalyst_days,
        ruleset,
        catalyst_decay_w,
    )

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
    Uses commercial_quality_pct instead of optionality_pct_dev.

    Returns (tier_letter, tier_reason).
    """
    arch = str(archetype)
    if arch == "drug_developer" or not arch:
        return "", ""

    if not eligible:
        return "D", "ineligible"

    if quality_pct is None:
        return "C", "no_quality_data"

    strength, is_actionable, has_catalyst_data, cat_tag = _resolve_catalyst_strength(
        catalyst_in_window,
        catalyst_days,
        ruleset,
        catalyst_decay_w,
    )

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
        et_match = rule_et == "*" or rule_et.upper() == et
        src_match = rule_src == "*" or rule_src.upper() == src
        if et_match and src_match:
            return int(priority)

    # No rule matched — return default
    return ruleset.catalyst_priority_default


# =============================================================================
# EVENT EV PROMOTION LADDER (Spec 061)
# =============================================================================

# Ordered stages — higher index = more aggressive integration.
# _ev_stage_at_least() compares current stage against a minimum.
_EV_STAGE_ORDER = {
    "off": 0,
    "tiebreaker": 1,
    "rank_overlay": 2,
    "sizing_overlay": 3,
    "composite": 4,
}

# Analog confidence ordering for gate comparison.
_ANALOG_CONF_ORDER = {"insufficient": 0, "low": 1, "ok": 2}


def _ev_stage_at_least(ruleset: "DecisionRuleset", minimum: str) -> bool:
    """Return True if ruleset.event_ev_stage >= minimum in the ladder."""
    return _EV_STAGE_ORDER.get(ruleset.event_ev_stage, 0) >= _EV_STAGE_ORDER.get(minimum, 0)


def _analog_confidence_meets(actual: str, minimum: str) -> bool:
    """Return True if actual analog confidence >= minimum threshold."""
    return _ANALOG_CONF_ORDER.get(actual, 0) >= _ANALOG_CONF_ORDER.get(minimum, 0)


# =============================================================================
# SORT CONTRIBUTION HELPERS
# =============================================================================

# Authoritative ordered list of contribution keys.  Order matters for
# deterministic CSV column emission.  Every name that _build_sort_contributions
# can emit must appear here.
SORT_CONTRIB_KEYS: Tuple[str, ...] = (
    "clinical",
    "coinvest",
    "institutional",
    "calendar_alpha",
    "alpha_cohort_tb",
    "catalyst_bonus",
    "binary_quality",
    "binary_institutional",
    "clinical_quality_91_180",
    "options_quality_91_180",
    "binary_quality_now",
    "binary_institutional_now",
    "clinical_build_window",
    "pcr_penalty_bw",
    "oncology_crowding",
    "options_verdict",
    "event_ev",
)


class SortContribution(NamedTuple):
    """One signal's adjustment to the sort anchor."""

    name: str  # signal identifier ("clinical", "coinvest", etc.)
    raw: Decimal  # input z-score before clamp
    weight: Decimal  # effective weight applied
    delta: Decimal  # final adjustment (subtracted from anchor)


def _build_sort_contributions(
    decision_fields: Dict[str, Any],
    ruleset: "DecisionRuleset",
    *,
    alpha_raw: float,
    catalyst_bonus: float,
    catalyst_type_mult: float = 1.0,
) -> List[SortContribution]:
    """Compute all sort-anchor adjustments as structured contributions.

    Each contribution's ``delta`` is subtracted from the anchor value by the
    caller.  Uses Decimal arithmetic for CCFT determinism.

    ``catalyst_bonus`` is non-zero only in blended mode (caller resolves from
    the priority map).  ``alpha_raw`` feeds the within_tier modifier only;
    the tiebreak mode lives in a separate tuple position and is NOT included.
    """
    contribs: List[SortContribution] = []
    _ctm = _safe_decimal(catalyst_type_mult, _D1)

    # Helper: convert ruleset float weights to Decimal at point of use
    def _w(val) -> Decimal:
        return _safe_decimal(val, _D0)

    # 1. Clinical sort signal
    if ruleset.enable_clinical_sort_signal:
        cz_tier = _safe_decimal(decision_fields.get("clinical_score_z_tier"))
        stage = str(decision_fields.get("stage_bucket", ""))
        stage_mult = _w(dict(ruleset.clinical_stage_mults).get(stage, 0.0))
        if ruleset.clinical_positive_only:
            cz_eff = _clamp_d(cz_tier, _D0, _D2)
        else:
            cz_eff = _clamp_d(cz_tier, _D_NEG2, _D2)
        w = _w(ruleset.clinical_sort_weight)
        delta = w * cz_eff * stage_mult
        contribs.append(SortContribution("clinical", cz_tier, w, delta))

    # 2. Coinvest sort tilt — RESEARCHED & REJECTED
    if ruleset.enable_coinvest_sort_signal:
        cz = _safe_decimal(decision_fields.get("coinvest_score_z"))
        if ruleset.coinvest_positive_only:
            cz_eff = _clamp_d(cz, _D0, _D2)
        else:
            cz_eff = _clamp_d(cz, _D_NEG2, _D2)
        w = _w(ruleset.coinvest_sort_weight)
        delta = w * cz_eff
        contribs.append(SortContribution("coinvest", cz, w, delta))

    # 3. Institutional delta sort tilt
    if ruleset.enable_institutional_sort_signal:
        iz = _safe_decimal(decision_fields.get("inst_delta_z"))
        if ruleset.institutional_positive_only:
            iz_eff = _clamp_d(iz, _D0, _D2)
        else:
            iz_eff = _clamp_d(iz, _D_NEG2, _D2)
        w = _w(ruleset.institutional_sort_weight)
        delta = w * iz_eff
        contribs.append(SortContribution("institutional", iz, w, delta))

    # 4. Calendar Alpha v2 sort tilt
    if ruleset.enable_calendar_alpha_sort:
        cv2_z = _safe_decimal(decision_fields.get("clinical_score_v2_z"))
        cv2_eff = _clamp_d(cv2_z, _D0, _D2)  # positive-only
        w = _w(ruleset.calendar_alpha_sort_weight)
        delta = w * cv2_eff
        contribs.append(SortContribution("calendar_alpha", cv2_z, w, delta))

    # 5. Alpha cohort percentile tiebreak
    if ruleset.alpha_cohort_tiebreak_weight > 0.0:
        ac_pct = _safe_decimal(decision_fields.get("alpha_cohort_pct"))
        w = _w(ruleset.alpha_cohort_tiebreak_weight)
        delta = w * ac_pct
        contribs.append(SortContribution("alpha_cohort_tb", ac_pct, w, delta))

    # 6. Catalyst bonus (non-zero only in blended mode)
    if catalyst_bonus != 0.0:
        _cb = _safe_decimal(catalyst_bonus)
        scaled_bonus = _cb * _ctm
        contribs.append(SortContribution("catalyst_bonus", _cb, _ctm, scaled_bonus))

    # 7. Binary 91-180 within-bucket quality re-ranking
    bucket = str(decision_fields.get("catalyst_bucket", ""))
    b91_mode = ruleset.binary_91_180_sort_mode
    if bucket == "less_binary" and b91_mode != "baseline":
        bqs = _safe_decimal(decision_fields.get("binary_quality_score"))
        bqs_w = _w(ruleset.binary_91_180_quality_weight)
        delta = bqs_w * bqs * _ctm
        contribs.append(SortContribution("binary_quality", bqs, bqs_w, delta))

        # 8. Binary 91-180 institutional tilt
        if b91_mode == "quality_plus_institutional":
            iz = _safe_decimal(decision_fields.get("inst_delta_z"))
            iz_eff = _clamp_d(iz, _D0, _D2)
            bi_w = _w(ruleset.binary_91_180_institutional_weight)
            delta_i = bi_w * iz_eff
            contribs.append(SortContribution("binary_institutional", iz, bi_w, delta_i))

    # 9. Binary 91-180 clinical quality tilt
    if bucket == "less_binary" and b91_mode in ("clinical_quality", "clinical_plus_options"):
        family = str(decision_fields.get("catalyst_family", ""))
        if family == "CLINICAL":
            cqc = _safe_decimal(decision_fields.get("clinical_quality_composite"))
            cq_w = _w(ruleset.binary_91_180_clinical_quality_weight)
            delta_cq = cq_w * cqc * _ctm
            contribs.append(SortContribution("clinical_quality_91_180", cqc, cq_w, delta_cq))

    # 10. Binary 91-180 options quality tilt
    if b91_mode in ("options_quality", "clinical_plus_options"):
        _has_reg = str(decision_fields.get("has_regulatory_upcoming_180d", "")) == "1"
        _reg_days = _safe_float(decision_fields.get("regulatory_days"), default=None)
        _in_less_binary = (
            _reg_days is not None and _reg_days > BUCKET_BUILD_WINDOW_MAX and _reg_days <= BUCKET_LESS_BINARY_MAX
        )
        if _has_reg and _in_less_binary:
            oqc = _safe_decimal(decision_fields.get("options_quality_composite"))
            oq_w = _w(ruleset.binary_91_180_options_quality_weight)
            delta_oq = oq_w * oqc
            contribs.append(SortContribution("options_quality_91_180", oqc, oq_w, delta_oq))

    # 11. Binary 0-90 within-bucket quality re-ranking
    bn_mode = ruleset.binary_now_sort_mode
    if bucket in ("binary_now", "build_window") and bn_mode != "baseline":
        bqs = _safe_decimal(decision_fields.get("binary_quality_score"))
        bqs_w = _w(ruleset.binary_now_quality_weight)
        delta_bn = bqs_w * bqs
        contribs.append(SortContribution("binary_quality_now", bqs, bqs_w, delta_bn))

        if bn_mode == "quality_plus_institutional":
            iz = _safe_decimal(decision_fields.get("inst_delta_z"))
            iz_eff = _clamp_d(iz, _D0, _D2)
            bi_w = _w(ruleset.binary_now_institutional_weight)
            delta_i = bi_w * iz_eff
            contribs.append(SortContribution("binary_institutional_now", iz, bi_w, delta_i))

    # 12. Build-window clinical sort tilt
    bw_mode = ruleset.build_window_clinical_mode
    if bucket == "build_window" and bw_mode == "clinical_z":
        cz_tier = _safe_decimal(decision_fields.get("clinical_score_z_tier"))
        stage = str(decision_fields.get("stage_bucket", ""))
        stage_mult = _w(dict(ruleset.clinical_stage_mults).get(stage, 0.0))
        cz_eff = _clamp_d(cz_tier, _D0, _D2)
        bw_w = _w(ruleset.build_window_clinical_weight)
        delta_bw = bw_w * cz_eff * stage_mult
        contribs.append(SortContribution("clinical_build_window", cz_tier, bw_w, delta_bw))

    # 13. Put/call ratio penalty
    pcr_mode = ruleset.pcr_penalty_mode
    if pcr_mode == "tiebreak" and bucket == "build_window":
        cat_fam = str(decision_fields.get("catalyst_family", ""))
        if cat_fam == "CLINICAL":
            pcr_raw = _safe_decimal(decision_fields.get("pre_event_put_call_ratio"))
            pcr_excess = max(_D0, pcr_raw - _D_HALF) * _D("2")
            pcr_w = _w(ruleset.pcr_penalty_weight)
            delta_pcr = -(pcr_w * pcr_excess)
            contribs.append(SortContribution("pcr_penalty_bw", pcr_raw, pcr_w, delta_pcr))

    # 14. Oncology crowding penalty
    if ruleset.enable_oncology_crowding_penalty:
        ta = str(decision_fields.get("therapeutic_area", ""))
        if ta == "oncology":
            ci_z = _safe_decimal(decision_fields.get("competitive_intensity_z"))
            oc_w = _w(ruleset.oncology_crowding_weight)
            delta_oc = -(oc_w * ci_z)
            contribs.append(SortContribution("oncology_crowding", ci_z, oc_w, delta_oc))

    # 15. Options verdict agreement tilt (Spec 038, research-phase)
    # Uses fused 4-lens verdict composite [0, 1] as within-bucket tiebreaker.
    # Only activates in less_binary/build_window buckets where options data
    # is most likely to carry alpha near secondary regulatory events.
    if ruleset.enable_options_verdict_tilt:
        if bucket in ("less_binary", "build_window"):
            ovf = _safe_decimal(decision_fields.get("ovf_composite"))
            ovf_w = _w(ruleset.options_verdict_weight)
            delta_ovf = ovf_w * ovf
            contribs.append(SortContribution("options_verdict", ovf, ovf_w, delta_ovf))

    # 16. Event EV rank overlay (Spec 061, Stage 2)
    # Adds a bounded sort contribution based on downside_adjusted_ev z-score.
    # Gated on event_ev_stage >= "rank_overlay" AND analog confidence threshold.
    if _ev_stage_at_least(ruleset, "rank_overlay") and ruleset.event_ev_rank_overlay_weight > 0:
        ev_z = _safe_decimal(decision_fields.get("event_ev_score_z"))
        ev_conf = str(decision_fields.get("event_ev_analog_confidence", ""))
        if _analog_confidence_meets(ev_conf, ruleset.event_ev_min_analog_confidence):
            ev_cap = _safe_decimal(ruleset.event_ev_rank_overlay_cap)
            ev_eff = _clamp_d(ev_z, -ev_cap, ev_cap)
            ev_w = _w(ruleset.event_ev_rank_overlay_weight)
            delta_ev = ev_w * ev_eff
            contribs.append(SortContribution("event_ev", ev_z, ev_w, delta_ev))

    return contribs


# Fields injected by run_screen.py (not computed by the DE itself).
# compute_actionable_sort_key() reads these from decision_fields and defaults
# to 0.0 when absent, so the DE remains self-contained for unit testing.
_EXTERNAL_SORT_FIELDS: frozenset = frozenset(
    {
        "clinical_score_z_tier",
        # "coinvest_score_z" — removed: never injected by run_screen.py; coinvest sort
        # signal researched & rejected (look-ahead contaminated, PIT-correct harmful).
        "inst_delta_z",
        "stage_bucket",
        "clinical_score_v2_z",
        "alpha_cohort_pct",  # used by alpha_cohort_tiebreak_weight contribution
        "binary_quality_score",  # used by binary_91_180_sort_mode contribution
        "catalyst_bucket",  # used by binary_91_180_sort_mode bucket gate
        "clinical_quality_composite",  # used by clinical_quality sort mode
        "options_quality_composite",  # used by options_quality sort mode
        "catalyst_family",  # used by clinical_quality sort mode family gate
        "has_regulatory_upcoming_180d",  # used by options_quality sort mode secondary reg path
        "regulatory_days",  # used by options_quality sort mode secondary reg path
        "pre_event_put_call_ratio",  # used by pcr_penalty_mode penalty
        "event_ev_score",  # used by Spec 061 Stage 1 tiebreaker
        "event_ev_score_z",  # used by Spec 061 Stage 2 rank overlay
        "event_ev_analog_confidence",  # used by Spec 061 rank overlay confidence gate
        "event_ev_bucket",  # used by Spec 061 Stage 3 sizing overlay
    }
)


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
    ``inst_delta_z``, and ``stage_bucket`` are injected by ``run_screen.py``
    after the DE loop and default to 0.0 when absent.
    """
    eligible_val = decision_fields.get("eligible", "0")
    is_eligible = 0 if eligible_val == "1" else 1

    is_dev = 0 if archetype == "drug_developer" else 1

    rs = ruleset or DEFAULT_RULESET

    # Spec 061 Stage 4: composite mode is recognized but not implemented.
    if rs.event_ev_stage == "composite":
        raise NotImplementedError(
            "event_ev_stage='composite' is reserved for a future promotion spec. "
            "Use 'tiebreaker', 'rank_overlay', or 'sizing_overlay' for now."
        )

    # Select tier based on tiering priority mode
    if rs.tiering_priority_mode == "tier_first":
        tier = decision_fields.get("tier_any", "") or decision_fields.get("tier_dev", "")
    else:
        tier = decision_fields.get("tier_dev", "")
    tier_ord = _TIER_ORDER.get(str(tier), 4)

    # Flatten tier within binary_91_180 bucket when flag is enabled.
    # This removes tier as a sort discriminator for names in the 91-180
    # catalyst window, letting the anchor and sort contributions dominate.
    if rs.binary_91_180_flatten_tier_sort:
        bucket = decision_fields.get("catalyst_bucket", "")
        if bucket == "less_binary" and is_eligible == 0:
            tier_ord = 1  # treat as tier A for sort purposes only

    # Build prefix: tier_first puts tier before archetype.
    # When sort_anchor=selector_score (Spec 050), flatten tier so the selector
    # score is the primary ordering — tier is preserved as metadata but does
    # not gate rank order.
    if rs.sort_anchor == "selector_score":
        prefix = (is_eligible, 0, 0)  # tier and archetype flattened
    elif rs.tiering_priority_mode == "tier_first":
        prefix = (is_eligible, tier_ord, is_dev)
    else:
        prefix = (is_eligible, is_dev, tier_ord)

    mode = rs.catalyst_priority_mode

    # Resolve catalyst priority (needed for tiebreaker + blended modes)
    if mode != "off":
        cat_priority = resolve_catalyst_priority(
            catalyst_event_type,
            catalyst_source,
            rs,
        )
    else:
        cat_priority = 0  # neutral — no effect on ordering

    cat_mode = decision_fields.get("catalyst_mode", "missing")
    cat_mode_ord = _CATALYST_MODE_ORDER.get(str(cat_mode), 4)

    cat_days_raw = decision_fields.get("catalyst_days", "")
    cat_days = int(_safe_float(cat_days_raw, default=9999))

    opt_neg = -_safe_decimal(optionality)

    sponsor_val = _safe_float(decision_fields.get("sponsor_tier1_count"), default=0.0)
    sponsor_neg = -int(sponsor_val)

    mom = decision_fields.get("mom_state", "neutral")
    mom_ord = _MOM_STATE_ORDER.get(str(mom), 1)

    _cr = _safe_float(composite_rank, default=None)
    comp_rank = int(_cr) if _cr is not None else 9999

    # Resolve anchor value based on sort_anchor mode (Decimal for CCFT)
    if rs.sort_anchor == "options_anchor_hybrid":
        # Spec 039: use options_quality_composite when present, fallback to optionality_pct
        _oqc = _safe_decimal(decision_fields.get("options_quality_composite"), default=_D0)
        if _oqc > _D0:
            anchor = -_oqc  # higher composite → more negative → sorts first
            # Bounded modifiers for regulatory proximity and catalyst nearness
            _has_reg = str(decision_fields.get("has_regulatory_upcoming_180d", "")) == "1"
            _reg_days = _safe_float(decision_fields.get("regulatory_days"), default=None)
            if _has_reg and _reg_days is not None and _reg_days > 90 and _reg_days <= 180:
                anchor = anchor * _safe_decimal(rs.options_anchor_regulatory_boost, _D1)
            _cat_days_raw = _safe_float(decision_fields.get("catalyst_days"), default=None)
            if _cat_days_raw is not None and _cat_days_raw <= 30:
                anchor = anchor * _safe_decimal(rs.options_anchor_near_catalyst_boost, _D1)
        else:
            # Fallback: use optionality_pct or comp_rank
            anchor = -_safe_decimal(tiebreaker_pct) if tiebreaker_pct is not None else _D(str(comp_rank))
    elif rs.sort_anchor == "selector_score":
        # Spec 050: use final_score (selector + ranker) as anchor.
        # Falls back to selector_score if ranker inactive.
        _final = _safe_decimal(decision_fields.get("final_score"), default=_D0)
        if _final > _D0:
            anchor = -_final
        else:
            _sel = _safe_decimal(decision_fields.get("selector_score"), default=_D0)
            anchor = -_sel  # higher score → more negative → sorts first
    elif rs.sort_anchor in ("optionality_pct", "alpha_cohort"):
        anchor = -_safe_decimal(tiebreaker_pct)  # higher pct → more negative → sorts first
    else:
        anchor = _D(str(comp_rank))  # existing behavior

    # Missingness sort penalty: higher count sorts later (0 when disabled)
    missing_count = 0
    if rs.enable_missingness_sort_penalty:
        missing_count = int(_safe_float(decision_fields.get("missingness_penalty", 0), default=0))

    # --- Build structured contributions and compute total adjustment ---
    # Catalyst bonus is only non-zero in blended mode.
    catalyst_bonus_f = 0.0
    if mode == "blended":
        bonus_map = dict(rs.catalyst_priority_rank_bonuses)
        catalyst_bonus_f = bonus_map.get(cat_priority, 0.0)

    # Resolve catalyst type multiplier for sort-key scaling (Spec 030)
    _ctm_f = 1.0
    if rs.enable_catalyst_type_tilt:
        _ct_tier = classify_catalyst_type_tier(catalyst_event_type)
        _ctm_f = dict(rs.catalyst_type_mults).get(_ct_tier, 1.0)

    contribs = _build_sort_contributions(
        decision_fields,
        rs,
        alpha_raw=_safe_float(alpha_raw, default=0.0),
        catalyst_bonus=catalyst_bonus_f,
        catalyst_type_mult=_ctm_f,
    )
    total_adj = sum((c.delta for c in contribs), _D0)

    # --- Mode dispatch ---
    # prefix is (is_eligible, is_dev, tier_ord) in dev_first mode
    # or (is_eligible, tier_ord, is_dev) in tier_first mode

    # Spec 061: Event EV tiebreaker (Stage 1+).
    # When event_ev_stage >= "tiebreaker", insert -ev_score before the
    # alphabetic ticker tiebreak. Higher EV → more negative → sorts first.
    _ev_tiebreak = _D0
    if _ev_stage_at_least(rs, "tiebreaker"):
        _ev_tiebreak = -_safe_decimal(decision_fields.get("event_ev_score"))

    # Spec 050: when sort_anchor=selector_score, the score IS the primary
    # ordering signal regardless of catalyst_priority_mode. All other sort
    # elements are tiebreakers only.
    if rs.sort_anchor == "selector_score":
        return prefix + (
            anchor,  # -final_score: primary ordering
            missing_count,
            _ev_tiebreak,  # Spec 061: EV tiebreaker (0 when stage=off)
            ticker,
        )

    if mode == "tiebreaker":
        # anchor dominates after tier; priority only breaks ties
        effective_comp_rank = anchor - total_adj
        return prefix + (
            effective_comp_rank,  # anchor with signal tilts
            missing_count,  # fewer missing components first
            cat_priority,  # priority breaks comp_rank ties
            cat_days,  # ascending days
            opt_neg,  # descending optionality
            sponsor_neg,  # descending sponsor count
            mom_ord,  # tailwind < neutral < headwind
            cat_mode_ord,  # specific < blended < no_upcoming < missing
            _ev_tiebreak,  # Spec 061: EV tiebreaker (0 when stage=off)
            ticker,  # alphabetic tiebreak
        )

    if mode == "blended":
        # catalyst_bonus already included in contribs
        effective_comp_rank = anchor - total_adj
        return prefix + (
            effective_comp_rank,  # anchor with bonus + signal tilts
            missing_count,  # fewer missing components first
            cat_mode_ord,  # specific < blended < no_upcoming < missing
            cat_days,  # ascending days
            opt_neg,  # descending optionality
            sponsor_neg,  # descending sponsor count
            mom_ord,  # tailwind < neutral < headwind
            cat_priority,  # priority as tiebreaker
            _ev_tiebreak,  # Spec 061: EV tiebreaker (0 when stage=off)
            ticker,  # alphabetic tiebreak
        )

    # mode == "off" (default)
    effective_opt_neg = opt_neg - total_adj  # Decimal: higher z → more negative → sorts earlier

    return prefix + (
        cat_priority,  # 0 (neutral) — no effect on ordering
        cat_mode_ord,  # specific < blended < no_upcoming < missing
        cat_days,  # ascending days (missing=9999)
        missing_count,  # fewer missing components first
        effective_opt_neg,  # optionality with signal tilts (negated)
        sponsor_neg,  # descending sponsor count (negated)
        mom_ord,  # tailwind < neutral < headwind
        anchor,  # ascending composite rank (or negated pct)
        _ev_tiebreak,  # Spec 061: EV tiebreaker (0 when stage=off)
        ticker,  # alphabetic tiebreak
    )


def compute_sort_contribs(
    decision_fields: Dict[str, Any],
    archetype: str,
    *,
    ruleset: Optional["DecisionRuleset"] = None,
    tiebreaker_pct: Optional[float] = None,
    alpha_raw: Optional[float] = None,
    catalyst_event_type: str = "",
    catalyst_source: str = "",
) -> Tuple[Decimal, Dict[str, Decimal]]:
    """Return sort contribution deltas without recomputing the full sort key.

    Returns ``(total_adj, contrib_map)`` where *contrib_map* has an entry
    for **every** key in :data:`SORT_CONTRIB_KEYS` (0.0 when inactive).
    Callers can rely on ``total_adj == sum(contrib_map.values())`` within
    floating-point tolerance.
    """
    rs = ruleset or DEFAULT_RULESET

    mode = rs.catalyst_priority_mode
    catalyst_bonus = 0.0
    if mode == "blended":
        bonus_map = dict(rs.catalyst_priority_rank_bonuses)
        cat_priority = resolve_catalyst_priority(
            catalyst_event_type,
            catalyst_source,
            rs,
        )
        catalyst_bonus = bonus_map.get(cat_priority, 0.0)

    # Resolve catalyst type multiplier for sort-key scaling (Spec 030)
    _ctm = 1.0
    if rs.enable_catalyst_type_tilt:
        _ct_tier = classify_catalyst_type_tier(catalyst_event_type)
        _ctm = dict(rs.catalyst_type_mults).get(_ct_tier, 1.0)

    contribs = _build_sort_contributions(
        decision_fields,
        rs,
        alpha_raw=_safe_float(alpha_raw, default=0.0),
        catalyst_bonus=catalyst_bonus,
        catalyst_type_mult=_ctm,
    )

    # Build the output map — every key present, Decimal 0 when inactive.
    contrib_map: Dict[str, Decimal] = {k: _D0 for k in SORT_CONTRIB_KEYS}
    for c in contribs:
        contrib_map[c.name] = c.delta

    total_adj = sum(contrib_map.values(), _D0)
    return total_adj, contrib_map


# =============================================================================
# TARGET WEIGHT SIZING
# =============================================================================


_BINARY_CATALYST_MODES = {"specific_days", "blended_window"}


def _is_binary_name(row: Dict[str, Any], days_threshold: int) -> bool:
    """Return True if row represents a binary-event name (imminent catalyst)."""
    mode = str(row.get("catalyst_mode", "")).strip().lower()
    if mode not in _BINARY_CATALYST_MODES:
        return False
    days = _safe_float(row.get("catalyst_days"), default=float("inf"))
    return days <= days_threshold


def apply_binary_sleeve_caps(
    rows: List[Dict[str, Any]],
    ruleset: DecisionRuleset,
) -> List[str]:
    """Enforce per-name and aggregate weight caps on binary-event names.

    Operates in-place on rows that already have ``target_weight_pct``.
    Excess weight is redistributed proportionally to non-binary names.
    After capping, weights are re-normalized to sum to 100%.

    No-op when both caps are 100% (the default).

    Returns list of warning strings (empty when no issues detected).
    """
    warnings: List[str] = []
    per_cap = ruleset.binary_sleeve_per_name_max_pct
    agg_cap = ruleset.binary_sleeve_max_weight_pct
    threshold = ruleset.binary_sleeve_days_threshold

    # Fast path: defaults → nothing to do
    if per_cap >= 100.0 and agg_cap >= 100.0:
        return warnings

    # Classify rows
    binary_idx = []
    non_binary_idx = []
    for i, row in enumerate(rows):
        w = _safe_float(row.get("target_weight_pct"), default=0.0)
        if w <= 0:
            continue
        if _is_binary_name(row, threshold):
            binary_idx.append(i)
        else:
            non_binary_idx.append(i)

    if not binary_idx:
        return warnings

    # --- Per-name cap ---
    excess = 0.0
    for i in binary_idx:
        w = _safe_float(rows[i]["target_weight_pct"], default=0.0)
        if w > per_cap:
            excess += w - per_cap
            rows[i]["target_weight_pct"] = round(per_cap, 2)

    # Redistribute per-name excess to non-binary proportionally
    if excess > 0:
        if not non_binary_idx:
            warnings.append(
                f"BINARY_SLEEVE: {excess:.2f}pp per-name excess cannot be "
                f"redistributed (no non-binary names); excess held in binary "
                f"names via re-normalize"
            )
        else:
            nb_total = sum(_safe_float(rows[j]["target_weight_pct"], default=0.0) for j in non_binary_idx)
            if nb_total > 0:
                for j in non_binary_idx:
                    w = _safe_float(rows[j]["target_weight_pct"], default=0.0)
                    rows[j]["target_weight_pct"] = round(w + excess * (w / nb_total), 2)
            else:
                warnings.append(
                    f"BINARY_SLEEVE: {excess:.2f}pp per-name excess cannot be "
                    f"redistributed (non-binary weights sum to 0)"
                )

    # --- Aggregate cap ---
    binary_total = sum(_safe_float(rows[i]["target_weight_pct"], default=0.0) for i in binary_idx)
    if binary_total > agg_cap:
        scale = agg_cap / binary_total
        agg_excess = binary_total - agg_cap
        for i in binary_idx:
            w = _safe_float(rows[i]["target_weight_pct"], default=0.0)
            rows[i]["target_weight_pct"] = round(w * scale, 2)
        # Redistribute aggregate excess to non-binary
        if not non_binary_idx:
            warnings.append(
                f"BINARY_SLEEVE: {agg_excess:.2f}pp aggregate excess cannot "
                f"be redistributed (no non-binary names); excess held in "
                f"binary names via re-normalize"
            )
        else:
            nb_total = sum(_safe_float(rows[j]["target_weight_pct"], default=0.0) for j in non_binary_idx)
            if nb_total > 0:
                for j in non_binary_idx:
                    w = _safe_float(rows[j]["target_weight_pct"], default=0.0)
                    rows[j]["target_weight_pct"] = round(w + agg_excess * (w / nb_total), 2)
            else:
                warnings.append(
                    f"BINARY_SLEEVE: {agg_excess:.2f}pp aggregate excess "
                    f"cannot be redistributed (non-binary weights sum to 0)"
                )

    # --- Re-normalize to 100% ---
    # Only scale up non-binary names when binary names are capped.
    # Scaling binary names back up would undo the cap.
    all_idx = binary_idx + non_binary_idx
    total = sum(_safe_float(rows[i]["target_weight_pct"], default=0.0) for i in all_idx)
    if total > 0 and abs(total - 100.0) > 0.01:
        if non_binary_idx:
            # Absorb the gap into non-binary names only
            nb_total = sum(_safe_float(rows[j]["target_weight_pct"], default=0.0) for j in non_binary_idx)
            gap = 100.0 - total
            if nb_total > 0:
                for j in non_binary_idx:
                    w = _safe_float(rows[j]["target_weight_pct"], default=0.0)
                    rows[j]["target_weight_pct"] = round(w + gap * (w / nb_total), 2)
            else:
                # Fallback: scale all (caps effectively unenforceable)
                for i in all_idx:
                    w = _safe_float(rows[i]["target_weight_pct"], default=0.0)
                    rows[i]["target_weight_pct"] = round(w / total * 100, 2)
        else:
            # All-binary universe: scale all (caps effectively unenforceable)
            for i in all_idx:
                w = _safe_float(rows[i]["target_weight_pct"], default=0.0)
                rows[i]["target_weight_pct"] = round(w / total * 100, 2)
            warnings.append(
                "BINARY_SLEEVE: re-normalize scaled binary names above caps "
                "(all-binary universe, caps unenforceable)"
            )

    return warnings


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

    # Spec 061 Stage 3: Event EV sizing tilt (multiplicative, default all 1.0)
    _ev_sizing_active = _ev_stage_at_least(rs, "sizing_overlay")
    _ev_tilt_map = dict(rs.event_ev_sizing_tilt_mults) if _ev_sizing_active else {}

    raw_weights = []
    for row in rows:
        band = row.get("size_band", "XS")
        cm = _safe_float(row.get("cost_mult"), default=1.0)
        tm = _safe_float(row.get("catalyst_tilt_mult"), default=1.0)
        mm = _safe_float(row.get("mom_state_tilt_mult"), default=1.0)
        csm = _safe_float(row.get("sizing_multiplier_clinical"), default=1.0)
        ctm = _safe_float(row.get("catalyst_type_mult"), default=1.0)
        evm = _ev_tilt_map.get(str(row.get("event_ev_bucket", "no_ev")), 1.0) if _ev_sizing_active else 1.0
        raw_weights.append(weights_map.get(str(band), 0.15) * cm * tm * mm * csm * ctm * evm)

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

    # Apply binary sleeve caps (no-op when defaults are 100%)
    sleeve_warnings = apply_binary_sleeve_caps(rows, rs)
    if sleeve_warnings and rows:
        # Attach warnings to first row for downstream visibility
        existing = rows[0].get("_sizing_warnings", [])
        rows[0]["_sizing_warnings"] = existing + sleeve_warnings

    # Less-binary construction mode
    _apply_less_binary_construction(rows, rs)

    # Spec 050: construction risk controls (all no-op when defaults)
    _apply_construction_risk_controls(rows, rs)

    return rows


def _apply_construction_risk_controls(
    rows: List[Dict[str, Any]],
    ruleset: DecisionRuleset,
) -> None:
    """Apply Spec 050 construction risk controls in-place.

    Controls:
      - construction_name_cap_pct: per-name weight cap
      - construction_liquidity_cap_pct: cap for names with liquidity flags
      - construction_cost_gate_bps: exclude names above cost threshold

    Excess weight is redistributed proportionally to uncapped names.
    All controls are no-op at default values (100.0 / 100.0 / 0.0).
    """
    name_cap = ruleset.construction_name_cap_pct
    liq_cap = ruleset.construction_liquidity_cap_pct
    cost_gate = ruleset.construction_cost_gate_bps

    # Early exit: all controls disabled
    if name_cap >= 100.0 and liq_cap >= 100.0 and cost_gate <= 0.0:
        return

    # --- Cost gate: zero-out names above cost threshold ---
    if cost_gate > 0.0:
        for row in rows:
            cost_bps = _safe_float(row.get("est_cost_bps"), default=0.0)
            if cost_bps > cost_gate:
                row["target_weight_pct"] = 0.0

    # --- Per-name cap ---
    if name_cap < 100.0:
        _apply_weight_cap(rows, name_cap, lambda _row: True)

    # --- Liquidity cap: tighter cap for names with liquidity risk ---
    if liq_cap < 100.0:

        def _has_liq_risk(row):
            flags = str(row.get("risk_flags", ""))
            return "low_adv" in flags or "liquidity_fail" in flags

        _apply_weight_cap(rows, liq_cap, _has_liq_risk)


def _apply_weight_cap(
    rows: List[Dict[str, Any]],
    cap_pct: float,
    predicate,
) -> None:
    """Cap target_weight_pct for rows matching predicate, redistribute excess."""
    excess = 0.0
    uncapped_total = 0.0
    for row in rows:
        w = _safe_float(row.get("target_weight_pct"), default=0.0)
        if w <= 0:
            continue
        if predicate(row) and w > cap_pct:
            excess += w - cap_pct
            row["target_weight_pct"] = round(cap_pct, 2)
        else:
            uncapped_total += w

    # Redistribute excess proportionally to uncapped names
    if excess > 0 and uncapped_total > 0:
        for row in rows:
            w = _safe_float(row.get("target_weight_pct"), default=0.0)
            if w > 0 and w <= cap_pct and not (predicate(row) and w == cap_pct):
                row["target_weight_pct"] = round(w + excess * (w / uncapped_total), 2)


def _apply_less_binary_construction(
    rows: List[Dict[str, Any]],
    ruleset: DecisionRuleset,
) -> None:
    """Apply less-binary (core) sleeve construction policy in-place.

    Modes:
        include:      no change (default)
        exclude:      zero weight for core names, renormalize binaries to 100%
        equal_weight: set core names to equal weight, renormalize total to 100%
    """
    mode = ruleset.less_binary_construction
    if mode == "include":
        return

    threshold = ruleset.binary_sleeve_days_threshold
    core_idx = []
    binary_idx = []
    for i, row in enumerate(rows):
        w = _safe_float(row.get("target_weight_pct"), default=0.0)
        if w <= 0:
            continue
        if _is_binary_name(row, threshold):
            binary_idx.append(i)
        else:
            core_idx.append(i)

    if not core_idx:
        return

    if mode == "exclude":
        # Zero out core names
        for i in core_idx:
            rows[i]["target_weight_pct"] = 0.0
        # Renormalize binaries to 100%
        if binary_idx:
            total = sum(_safe_float(rows[i]["target_weight_pct"], default=0.0) for i in binary_idx)
            if total > 0:
                for i in binary_idx:
                    w = _safe_float(rows[i]["target_weight_pct"], default=0.0)
                    rows[i]["target_weight_pct"] = round(w / total * 100, 2)

    elif mode == "equal_weight":
        # Set core names to equal weight, preserving their total allocation
        core_total = sum(_safe_float(rows[i]["target_weight_pct"], default=0.0) for i in core_idx)
        ew = round(core_total / len(core_idx), 2) if core_idx else 0.0
        for i in core_idx:
            rows[i]["target_weight_pct"] = ew
        # Renormalize total to 100%
        all_idx = binary_idx + core_idx
        total = sum(_safe_float(rows[i]["target_weight_pct"], default=0.0) for i in all_idx)
        if total > 0 and abs(total - 100.0) > 0.01:
            for i in all_idx:
                w = _safe_float(rows[i]["target_weight_pct"], default=0.0)
                rows[i]["target_weight_pct"] = round(w / total * 100, 2)


# =============================================================================
# PUBLIC API
# =============================================================================

# Columns added by the actionable ordering layer
ACTIONABLE_COLUMNS = ["actionable_rank", "target_weight_pct"]

# Column names emitted by the decision engine (for SNAPSHOT_COLUMNS)
DECISION_COLUMNS = [
    "decision_engine_version",
    "decision_engine_ruleset_id",
    "eligible",
    "ineligible_reasons",
    "sponsor_tier1_count",
    "sponsor_overlap_count",
    "sponsor_net_buying",
    "coinvest_conviction",
    "coinvest_tier1_conviction",
    "coinvest_max_position_pct",
    "coinvest_filing_age_days",
    "catalyst_days",
    "catalyst_in_window",
    "catalyst_mode",
    "catalyst_bucket",
    "catalyst_strength",
    "catalyst_decay_w",
    "runway_bucket",
    "mom_state",
    "risk_flags",
    "size_band",
    "size_reasons",
    "cost_mult",
    "cost_bucket",
    "cost_haircut_applied",
    "catalyst_tilt_mult",
    "catalyst_tilt_applied",
    "mom_state_tilt_mult",
    "mom_state_tilt_applied",
    "dd_rel_margin_rescued",
    "tier_dev",
    "tier_reason",
    "tier_commercial",
    "tier_any",
    "tier_any_reason",
    "missing_components",
    "missingness_penalty",
    # --- Spec 050: Selector/Ranker columns (populated by run_screen.py) ---
    "selector_score",
    "selector_rank_bucket",
    "selector_clinical_block",
    "selector_catalyst_block",
    "selector_survivability_block",
    "selector_institutional_block",
    "selector_market_block",
    "ranker_active",
    "ranker_adjustment",
    "final_score",
    "ranker_options_block",
    "ranker_inst_block",
    "ranker_aact_block",
    # --- Spec 061: Event EV columns (populated by run_screen.py) ---
    "event_ev_score",
    "event_ev_score_z",
    "event_ev_bucket",
    "event_ev_analog_confidence",
    # --- Spec 057: Clinical Quality Score (monitor-only, populated by run_screen.py) ---
    "clinical_quality_score",
    "clinical_quality_confidence",
    "endpoint_strength_tier",
    "design_rigor_tier",
    "prior_evidence_tier",
    "mechanism_maturity_tier",
    "clinical_quality_notes",
]


def compute_decision_fields(
    rec: Dict[str, Any],
    archetype: str,
    optionality_pct_dev: Optional[float] = None,
    ruleset: Optional[DecisionRuleset] = None,
    est_cost_bps: Optional[float] = None,
    commercial_quality_pct: Optional[float] = None,
    market_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute all decision engine fields for a single ticker.

    Args:
        rec: ranked_securities record from Module 5
        archetype: company archetype string
        optionality_pct_dev: pre-computed optionality percentile (float or None)
        ruleset: DecisionRuleset config (defaults to DEFAULT_RULESET)
        est_cost_bps: estimated round-trip trading cost in basis points (float or None)
        commercial_quality_pct: pre-computed quality percentile for commercial archetypes (float or None)
        market_cap: market capitalisation in USD (float or None)

    Returns:
        dict with all decision engine column values
    """
    rs = ruleset or DEFAULT_RULESET

    # Layer 0 — Eligibility
    eligible, reasons, dd_rel_margin_rescued, financials_bypassed = _compute_eligibility(rec, rs, market_cap=market_cap)

    # Layer 2 — Overlays
    overlays = _compute_overlays(rec, rs)

    # Tag financials_partial when mega-cap bypass was used
    if financials_bypassed:
        rf = overlays.get("risk_flags", "")
        overlays["risk_flags"] = (rf + "|financials_partial") if rf else "financials_partial"

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

    # Catalyst type quality tilt multiplier (opt-in, affects weight only — Spec 030)
    catalyst_type_tier = classify_catalyst_type_tier(overlays.get("catalyst_event_type", ""))
    catalyst_type_mult = 1.0
    if rs.enable_catalyst_type_tilt:
        type_tilt_map = dict(rs.catalyst_type_mults)
        catalyst_type_mult = type_tilt_map.get(catalyst_type_tier, 1.0)

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
        "ineligible_reasons": "|".join(canonicalize_reasons(reasons)),
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
        "catalyst_type_tier": catalyst_type_tier,
        "catalyst_type_mult": catalyst_type_mult,
        "catalyst_type_tilt_applied": "1" if catalyst_type_mult != 1.0 else "0",
        "mom_state_tilt_mult": mom_state_tilt_mult,
        "mom_state_tilt_applied": "1" if mom_state_tilt_mult != 1.0 else "0",
        "dd_rel_margin_rescued": "1" if dd_rel_margin_rescued else "0",
    }

    # Missingness tracking (always computed, penalties gated by ruleset toggles)
    missing = _compute_missing_components(fields)
    fields["missing_components"] = "|".join(missing) if missing else ""
    fields["missingness_penalty"] = len(missing)

    # Spec 050: selector/ranker placeholders (populated later by run_screen.py)
    fields["selector_score"] = ""
    fields["selector_rank_bucket"] = ""
    fields["selector_clinical_block"] = ""
    fields["selector_catalyst_block"] = ""
    fields["selector_survivability_block"] = ""
    fields["selector_institutional_block"] = ""
    fields["selector_market_block"] = ""
    fields["ranker_active"] = ""
    fields["ranker_adjustment"] = ""
    fields["final_score"] = ""
    fields["ranker_options_block"] = ""
    fields["ranker_inst_block"] = ""
    fields["ranker_aact_block"] = ""

    # Spec 061: Event EV placeholders (populated later by run_screen.py)
    fields["event_ev_score"] = ""
    fields["event_ev_score_z"] = ""
    fields["event_ev_bucket"] = ""
    fields["event_ev_analog_confidence"] = ""

    # Spec 057: Clinical Quality Score placeholders (populated later by run_screen.py)
    fields["clinical_quality_score"] = ""
    fields["clinical_quality_confidence"] = ""
    fields["endpoint_strength_tier"] = ""
    fields["design_rigor_tier"] = ""
    fields["prior_evidence_tier"] = ""
    fields["mechanism_maturity_tier"] = ""
    fields["clinical_quality_notes"] = ""

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
    gates.append(
        {
            "gate": "fundamental_red_flag",
            "passed": not has_red_flag,
            "margin": None,
            "actual": has_red_flag,
            "threshold": True,
        }
    )

    # Gate 2: sev3 (boolean)
    severity = str(rec.get("severity", "")).upper()
    sev3_hit = severity == "SEV3"
    gates.append(
        {
            "gate": "sev3",
            "passed": not sev3_hit,
            "margin": None,
            "actual": severity,
            "threshold": "SEV3",
        }
    )

    # Gate 3: deep_drawdown_abs
    dd_abs_margin: Optional[float] = None
    if dd is not None:
        dd_abs_margin = round(dd - rs.drawdown_gate, 6)
    abs_passed = dd is None or dd >= rs.drawdown_gate
    gates.append(
        {
            "gate": "deep_drawdown_abs",
            "passed": abs_passed,
            "margin": dd_abs_margin,
            "actual": dd,
            "threshold": rs.drawdown_gate,
        }
    )

    # Gate 4: deep_drawdown_rel
    dd_rel_margin: Optional[float] = None
    if dd_rel is not None:
        dd_rel_margin = round(dd_rel - rs.drawdown_rel_xbi_gate, 6)
    rel_passed = dd_rel is None or dd_rel >= rs.drawdown_rel_xbi_gate
    gates.append(
        {
            "gate": "deep_drawdown_rel",
            "passed": rel_passed,
            "margin": dd_rel_margin,
            "actual": dd_rel,
            "threshold": rs.drawdown_rel_xbi_gate,
        }
    )

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
                if rs.enable_dd_rel_margin_rescue and dd_rel is not None:
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

    gates.append(
        {
            "gate": "deep_drawdown",
            "passed": dd_combined_passed,
            "mode": mode,
        }
    )

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
    gates.append(
        {
            "gate": "adv_fail",
            "passed": not adv_fail,
            "margin": None,
            "actual": adv_fail,
            "threshold": True,
        }
    )

    # --- Rescued by relative gate ---
    rescued_by_rel = (
        rs.drawdown_gate_mode == "hard"
        and dd_abs_margin is not None
        and dd_abs_margin < 0
        and dd_rel_margin is not None
        and dd_rel_margin >= 0
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
