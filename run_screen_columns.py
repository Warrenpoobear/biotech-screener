"""
Column definitions and operational constants for the screening pipeline.

Extracted from run_screen.py to reduce its size. These are pure data
constants with no logic or dependencies.
"""

from __future__ import annotations

from pathlib import Path

from common.adcom_vote_features import ADCOM_VOTE_COLUMNS
from common.event_quality_features import OPTIONS_QUALITY_COLUMNS
from common.options_diagnostics import OPTIONS_DIAGNOSTIC_COLUMNS
from common.options_features import OPTIONS_FEATURE_COLUMNS
from decision_engine import SORT_CONTRIB_KEYS

# =============================================================================
# VALIDATION SNAPSHOT (for forward-looking backtest)
# =============================================================================

# Columns saved in the validation snapshot CSV.
# These are the minimum fields needed for forward IC / decile-lift analysis.
SNAPSHOT_COLUMNS = (
    [
        # --- Identity ---
        "ticker",
        "company_name",
        # --- What drives the ranking (read left-to-right) ---
        "actionable_rank",
        "target_weight_pct",
        "tier_any",
        "tier_any_reason",
        "tier_dev",
        "tier_reason",
        "tier_commercial",
        "alpha_cohort_pct",
        "commercial_quality_pct",
        "has_commercial_quality",
        "clinical_optionality_pct_dev",
        "has_clinical_optionality_dev",
        "clinical_rank_pct_dev",
        "catalyst_days",
        "catalyst_in_window",
        "catalyst_mode",
        "catalyst_bucket",
        "cat_priority",
        "mom_state",
        "risk_flags",
        "size_band",
        "size_reasons",
        # --- Explanation fields ---
        "top_3_drivers",
        "catalyst_reason_detail",
        # --- Engine / run metadata ---
        "decision_engine_version",
        "decision_engine_ruleset_id",
        "eligible",
        "ineligible_reasons",
        # --- Red flag audit trail ---
        "fundamental_red_flag",
        "fundamental_red_flag_reasons",
        "fundamental_red_flag_inputs",
        # --- Diagnostics / supporting DE signals ---
        "alpha_cohort_key",
        "alpha_cohort_raw",
        "commercial_quality",
        "clinical_alpha_z",
        "clinical_readout_days",
        "clinical_coverage_flag",
        "clinical_score_z",
        "clinical_score_z_tier",
        "sponsor_tier1_count",
        "sponsor_overlap_count",
        "sponsor_net_buying",
        "coinvest_score_z",
        "coinvest_tag",
        "coinvest_conviction",
        "coinvest_tier1_conviction",
        "coinvest_max_position_pct",
        "coinvest_filing_age_days",
        "coinvest_recency_state",
        "inst_delta_z",  # z of net_elite_holders_delta (cross-sectional, ddof=0)
        "inst_delta_net",  # raw net_elite_holders_delta
        "inst_delta_new",  # elite_new_count
        "inst_delta_exit",  # elite_exit_count
        "inst_delta_nonzero_pct",  # % of tickers with nonzero net delta (coverage guard telemetry)
        "has_coinvest_signal",  # True when sponsor_tier1_count is real data
        "has_inst_delta",  # True when institutional delta is available
        "catalyst_strength",
        "catalyst_decay_w",
        "runway_bucket",
        "cost_bucket",
        "est_cost_bps",
        "cost_mult",
        "cost_haircut_applied",
        "dd_rel_margin_rescued",
        "catalyst_tilt_mult",
        "catalyst_tilt_applied",
        "catalyst_type_tier",
        "catalyst_type_mult",
        "catalyst_type_tilt_applied",
        "mom_state_tilt_mult",
        "mom_state_tilt_applied",
        "de_catalyst_days",
        "de_catalyst_in_window",
        "de_catalyst_mode",
        "de_alpha_60d",
        "de_alpha_60d_source",
        "de_alpha_60d_missing_reason",
        "de_tier1_count",
        "de_beta_xbi_60d",
        "de_beta_xbi_60d_source",
        "de_beta_xbi_60d_missing_reason",
        "de_drawdown",
        "de_drawdown_missing_reason",
        "de_rsi_14d",
        "de_vol_60d",
        "de_drawdown_xbi",
        "de_drawdown_rel_xbi",
        # --- Earnings calendar ---
        "next_earnings_date",
        # --- AACT execution score ---
        "aact_execution_score",
        # --- Context / provenance ---
        "stage_bucket",
        # Display-only development stage (preclinical / phase_1 / phase_1_2 /
        # phase_2 / phase_2_3 / phase_3 / nda_bla / approved / commercial /
        # unknown). Derived from archetype + tier_commercial + Module 4
        # lead_phase. NOT a scoring/selector input — see _derive_development_stage.
        "development_stage",
        "development_stage_source",
        "lead_program_phase_raw",
        # Display-only ranker_v2 cohort stability (per audit 2026-04-26).
        # cohort_membership = "in" | "out" depending on whether ranker_v2_score
        # is populated. cohort_membership_streak counts consecutive prior
        # snapshots with the same membership state (capped at 30 days
        # walkback). NOT a scoring/selector input.
        "cohort_membership",
        "cohort_membership_streak",
        "market_cap_bucket",
        "severity",
        "archetype",
        "industry_group",
        "returns_source",
        "catalyst_source",
        "catalyst_event_type",
        "is_hard_catalyst",
        "catalyst_family",
        # --- Catalyst calendar v2 fields (Spec 062+ sharpening) ---
        "next_catalyst_date",
        "catalyst_date_lower",
        "catalyst_date_upper",
        "catalyst_date_precision",
        "calendar_confidence",
        "has_catalyst_signal",
        "has_tradeable_calendar",
        "catalyst_quality",  # Spec 078: binary_alpha / registry_only / low_confidence / corporate_update
        # --- Institutional flow diagnostics ---
        "inst_delta_regime",
        "inst_flow_abs_positive",
        "inst_flow_abs_negative",
        "inst_relative_underperformance",
        "inst_relative_outperformance",
        "inst_flow_diagnostic",
        "binary_quality_score",
        "regulatory_quality",
        "clinical_quality",
        "has_adcom",
        "single_asset_risk",
        # --- Clinical 91-180 quality features ---
        "clinical_days_precision",
        "clinical_date_confidence",
        "clinical_design_quality",
        "clinical_program_depth",
        "clinical_quality_composite",
        # --- FDA AdCom voting-pattern pilot (passive, informational) ---
        *ADCOM_VOTE_COLUMNS,
        # --- Options-implied vol/skew diagnostics (passive, tastytrade) ---
        *OPTIONS_DIAGNOSTIC_COLUMNS,
        # --- Options quality composite (derived from diagnostics) ---
        *OPTIONS_QUALITY_COLUMNS,
        # --- Options Stage 1 shadow features (quality score, expectation gap) ---
        *OPTIONS_FEATURE_COLUMNS,
        # --- Market-model disagreement (shadow diagnostic, not ranking) ---
        "implied_event_move",
        "pos_divergence",
        "market_model_disagreement",
        # --- IV crush stress test (from Massive chain analytics) ---
        "iv_crush_breakeven_pct",
        "crush_adjusted_implied_move",
        # --- Market data pass-through for Event EV expectation model ---
        "short_interest_pct",
        "close_price",
        "market_cap_mm",
        "priced_move_pct",
        # insider_net_buy_value_90d is a diagnostic pass-through only — the
        # scoring lane was closed 2026-04-05 and is NOT reopened by this
        # column appearing in rankings.csv. Wired via common.insider_enrichment.
        "insider_net_buy_value_90d",
        # --- Expectation Error Model (Jane Street 6-mistake framework, overlay-only) ---
        "base_rate_gap_score",
        "conditional_misprice_score",
        "slippage_penalty_score",
        "divergence_score",
        "crowding_bias_score",
        "timing_decay_risk_score",
        "expectation_error_score",
        "expectation_confidence",
        "expectation_notes",
        "quality_overlay_score",
        "trap_overlay_score",
        "ees_v2_score",
        "ees_quality_gate",
        "ees_trap_gate",
        "ees_eligible",
        # --- EES v3 (conditional misprice + expected move, diagnostic overlay) ---
        "ees_v3_score",
        "ees_v3_gate",
        "ees_v3_pctile",
        "ees_v3_misprice_available",
        "conditional_misprice_z",
        "conditional_expected_move_z",
        # --- Conditional Model (Tier 1 alpha candidate, diagnostic only) ---
        "conditional_bucket",
        "conditional_base_rate",
        "conditional_expected_move",
        "conditional_gap_score",
        "conditional_confidence",
        "conditional_notes",
        # --- Execution Capacity Layer (Tier 1 sizing/construction, not alpha) ---
        "dollar_volume",
        "adv_20d",
        "adv_60d",
        "median_dollar_volume_20d",
        "execution_capacity_score",
        "max_position_dollars",
        "max_position_weight",
        "execution_bucket",
        "execution_notes",
        # --- Runway Severity (financing-truth cross-layer, diagnostic overlay) ---
        "runway_severity_score",
        "ev_severity_score",
        "runway_buffer_months",
        "financing_truth_gate",
        "dilution_haircut",
        "size_multiplier",
        "severity_bucket",
        "severity_notes",
        # --- Runway Transition Model (Markov Chain Phase 1, shadow-only diagnostic) ---
        "transition_runway_state",
        "transition_p_runway_worse_60d",
        "transition_p_financing_90d",
        "transition_p_distress_90d",
        # --- Straddle mispricing (from event_move_table + chain/IV) ---
        "cheap_vol_score",
        "vol_classification",
        "straddle_price",
        # --- Pre-event put/call ratio (from Massive day aggs, shadow) ---
        "pre_event_put_call_ratio",
        # --- Term structure validation flags (Agent 0 staleness / blind spot) ---
        "ts_flag",
        "ts_flag_type",
        "ts_flag_reason",
        # --- Secondary regulatory catalyst (independent of nearest) ---
        "regulatory_days",
        "regulatory_event_type",
        "regulatory_confidence",
        "has_regulatory_upcoming_180d",
        "missing_components",
        "missingness_penalty",
        "confidence_overall",
        # --- Source reliability (empirical slip-based) ---
        "source_reliability_action",
        "source_reliability_penalty",
        # --- Underlying module scores (informational) ---
        "momentum_score",
        "catalyst_score",
        "smart_money_score",
        "valuation_score",
        "clinical_score",
        "financial_score",
        # --- Clinical Calendar Alpha v2 (informational, sort/sizing off by default) ---
        "clinical_score_v2",
        "clinical_score_v2_z",
        "lead_program_phase",
        "lead_program_readout_days",
        "program_count",
        "program_diversification",
        "readout_curve_score",
        "readout_density_90",
        "late_stage_readouts_180",
        "execution_momentum",
        "design_quality_score",
        "endpoint_strength_score",
        "therapeutic_area",
        "competitive_intensity_z",
        "crowding_level",
        "sizing_multiplier_clinical",
        # --- Morningstar research diagnostics (not in composite) ---
        "ms_volatility_3yr",
        "ms_volatility_5yr",
        "ms_star_rating",
        "ms_return_ytd",
        "ms_return_annualized_3yr",
        "ms_return_annualized_5yr",
        # --- Surface signal fields (Spec 020, were computed but not persisted) ---
        "atm_iv_change_5d",
        "actual_implied_move_pctile",
        "surface_move_extreme",
        "iv_ramp_flag",
        "post_event_drift_risk",
        "rr_25d_trend_7d",
        "rr_trend_flag",
        "surface_signal_quality",
        "surface_validation_basis",
        # --- Options verdict research features (Spec 038) ---
        "ovf_agreement_count",
        "ovf_severity_score",
        "ovf_near_catalyst",
        "ovf_has_event_premium",
        "ovf_has_iv_ramp",
        "ovf_has_quiet_before",
        "ovf_surface_confirmed",
        "ovf_composite",
        # --- Options Monitor v1.1 research features (Spec 040) ---
        "ovf11_ep",
        "ovf11_sr",
        "ovf11_sk",
        "ovf11_dv",
        "ovf11_quality",
        "ovf11_confidence",
        "ovf11_score",
        "ovf11_primary_factor",
        "ovf11_monitor_verdict",
        "ovf11_trade_bias",
        "ovf11_event_window_flag",
        "ovf11_catalyst_class",
        # --- Legacy Module 5 composite fields (far right) ---
        "composite_rank",
        "composite_score",
        "score_rank_pct",
        "score_z",
        "composite_score_attn",
        "score_rank_pct_attn",
        "score_z_attn",
        # --- Sort contribution diagnostics (populated at sort time) ---
        "de_sort_total_adj",
    ]
    + [f"de_sort_contrib_{k}" for k in SORT_CONTRIB_KEYS]
    + [
        # --- Spec 050: Selector/Ranker columns ---
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
        "regime_label",
        "ranker_v2_score",
        "ranker_v2_rank",
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
)

# Phase-2 decision portfolio output columns
PHASE2_PORTFOLIO_COLUMNS = [
    "ticker",
    "company_name",
    # DE output first
    "actionable_rank",
    # Tiers + reasons
    "tier_any",
    "tier_any_reason",
    "tier_dev",
    "tier_reason",
    "tier_commercial",
    # Primary DE drivers
    "alpha_cohort_pct",
    "clinical_optionality_pct_dev",
    "clinical_alpha_z",
    "clinical_score_z",
    "clinical_score_z_tier",
    "commercial_quality_pct",
    "catalyst_days",
    "catalyst_mode",
    "cat_priority",
    "mom_state",
    "risk_flags",
    "size_band",
    "size_reasons",
    # Earnings
    "next_earnings_date",
    # Display-only development stage (matches rankings.csv column).
    "development_stage",
    "development_stage_source",
    # Metadata + missingness
    "decision_engine_version",
    "decision_engine_ruleset_id",
    "alpha_cohort_key",
    "alpha_cohort_raw",
    "missing_components",
    "archetype",
    # Legacy composite (far right)
    "composite_rank",
    "composite_score",
]

# Phase-2 portfolio positions output columns (weighted top-K subset)
PORTFOLIO_POSITIONS_COLUMNS = [
    "ticker",
    "company_name",
    "actionable_rank",
    "target_weight_pct",
    # Tiers
    "tier_any",
    "tier_any_reason",
    "tier_dev",
    "tier_reason",
    "tier_commercial",
    # Primary DE drivers
    "alpha_cohort_pct",
    "clinical_optionality_pct_dev",
    "catalyst_days",
    "catalyst_mode",
    "mom_state",
    "risk_flags",
    "size_band",
    "size_reasons",
    # Earnings
    "next_earnings_date",
    # Metadata
    "archetype",
    "eligible",
    # Legacy composite (far right)
    "composite_rank",
    "composite_score",
]

# Phase-2 operational defaults
PHASE2_DEFAULT_RULESET_PATH = (
    Path(__file__).resolve().parent / "production_data" / "decision_rulesets" / "v1.14.0_coinvest_only_selector.json"
)
PHASE2_DEFAULT_TIER_FILTER = ["A", "B"]
PHASE2_DEFAULT_TOP_K = 20
# Spec 050: EW Top-30 positions (all eligible, no tier filter)
POSITIONS_TOP_K = 30
PHASE2_PINNED_RULESET_ID = "8887576e"  # v1.14.0 coinvest-only selector (2026-05-04; was 2a3e79eb)
PHASE2_DEFAULT_HEALTH_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent / "production_data" / "phase2_health_thresholds" / "v1.json"
)
PHASE2_PINNED_THRESHOLDS_ID = "70636854"
