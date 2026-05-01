"""Active production ranker/selector contract.

Single source of truth for which scoring fields drive production output.
Documentation-as-code: this module computes nothing, declares nothing
runtime-loadable, and has no side effects. It exists so:

  1. Tests assert the live model artifact + selector config match this
     registry, catching drift on every CI run.
  2. Future readers can grep ACTIVE_RANKER_FEATURES / ACTIVE_SELECTOR_BLOCKS
     before adding new fields to scoring without going through promotion.
  3. Attribution traces can render the contract programmatically and
     distinguish active scoring fields from diagnostic / sidecar / inert
     fields — so "is X driving the rank?" has a deterministic answer.

Update only when:
  - Promoting a feature (Checklist v2 cleared) — bump ACTIVE_RANKER_FEATURES
    or block weights, retrain, redeploy ranker_v2_model.json.
  - Closing a lane — move field from ACTIVE_* to DEPRECATED_INERT.
  - Adding a new sidecar — register under SIDECAR_RESEARCH or
    DIAGNOSTIC_PASS_THROUGH.

Do NOT update without a corresponding code change. The test suite asserts
this file matches `production_data/ranker_v2_model.json` and the
`A4_SELECTOR_CONFIG` constant in `run_screen.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class FieldStatus(str, Enum):
    """Production-status taxonomy for scoring fields."""

    ACTIVE_RANKER = "active_ranker"
    """Consumed by production_data/ranker_v2_model.json with non-zero weight."""

    ACTIVE_SELECTOR = "active_selector"
    """Consumed by A4_SELECTOR_CONFIG with non-zero block-weight × signal-weight."""

    DIAGNOSTIC_PASS_THROUGH = "diagnostic_pass_through"
    """Written to rankings.csv for inspection; not consumed by ranker or selector.
    May power audit/attribution tooling. Examples: insider_net_buy_value_90d
    (Form 4 scoring lane closed 2026-04-05; field is data-integrity gate only)."""

    SIDECAR_RESEARCH = "sidecar_research"
    """Written to a separate sidecar artifact (e.g. ees_v3_overlay.json) for
    research/forward-monitoring. Behind a flag or explicit shadow path; not
    consumed by ranker or selector. Examples: EES v2/v3, expression overlay."""

    DEPRECATED_INERT = "deprecated_inert"
    """Was active in a prior model; now inert by block-weight=0 or removed
    from feature_set. Computation kept (e.g. for research lanes / shadow
    diagnostics) but does NOT reach final_score. Examples: clinical_score_v2_z
    (Spec 055 ablation -0.35pp, removed from ranker minimal_v2)."""


@dataclass(frozen=True)
class FieldContract:
    name: str
    status: FieldStatus
    note: str = ""


# ---------------------------------------------------------------------------
# Active ranker contract
# ---------------------------------------------------------------------------
# Must equal `model.feature_names` in `production_data/ranker_v2_model.json`.
# Asserted by tests/test_ranker_active_contract.py.
#
# Production model: pairwise logistic, feature_set="minimal_v2".
# Deployed weights are the capped Family C live-pilot vector; the artifact's
# `provenance` block is authoritative. Trained vector ≠ deployed vector.
# Frozen 2026-04-04 (alpha stack freeze policy); next change requires
# Checklist v2 + retrain + redeploy.
ACTIVE_RANKER_FEATURES: Tuple[str, ...] = (
    "coinvest_score_z",
    "financial_score",
)


# ---------------------------------------------------------------------------
# Active selector contract
# ---------------------------------------------------------------------------
# Must equal `A4_SELECTOR_CONFIG.block_weights` in run_screen.py.
# Asserted by tests/test_ranker_active_contract.py.
#
# A4 production selector (Spec 050 v1.1, validated true-PIT backtest +2.34pp/mo
# net, t=2.60). Clinical block zeroed Spec 055 (confirmed destructive).
ACTIVE_SELECTOR_BLOCKS: Tuple[Tuple[str, float], ...] = (
    ("clinical", 0.00),
    ("catalyst", 0.15),
    ("survivability", 0.10),
    ("institutional", 0.65),
    ("market_structure", 0.10),
)

# Institutional block signal weights override (defined in run_screen.py:A4_SELECTOR_CONFIG.
# institutional_signals). Other block signal weights inherit SelectorConfig defaults.
ACTIVE_INSTITUTIONAL_SIGNALS: Tuple[Tuple[str, float], ...] = (
    ("coinvest_score_z", 0.65),
    ("inst_delta_z", 0.35),
    ("coinvest_recency_state", 0.00),
)


# ---------------------------------------------------------------------------
# Field-by-field registry
# ---------------------------------------------------------------------------
# Not exhaustive over every column in rankings.csv — focuses on fields that
# have caused (or could cause) attribution confusion. Add to this list when:
#   - Closing a scoring lane (mark DEPRECATED_INERT).
#   - Promoting a sidecar (move SIDECAR_RESEARCH → ACTIVE_*).
#   - A new pass-through field is mistaken for active.
#
# Tests use this as the lookup table for known-status assertions.
SCORING_FIELDS: Tuple[FieldContract, ...] = (
    # --- Active ranker features ---
    FieldContract(
        "coinvest_score_z",
        FieldStatus.ACTIVE_RANKER,
        "Ranker v2 weight +0.02 (capped from trained +0.0613). Also active in "
        "selector institutional block at signal-weight 0.65 (block weight 0.65). "
        "Effective selector contribution = 0.4225.",
    ),
    FieldContract(
        "financial_score",
        FieldStatus.ACTIVE_RANKER,
        "Ranker v2 weight -0.0533 (penalizes safe names). Also active in "
        "selector survivability block at signal-weight 0.35 (block weight 0.10). "
        "Module 5 rank-normalized, NOT raw Module 2 financial output.",
    ),
    # --- Active selector signals (non-ranker) ---
    FieldContract(
        "inst_delta_z",
        FieldStatus.ACTIVE_SELECTOR,
        "Selector institutional block, signal-weight 0.35 (block weight 0.65). "
        "Effective selector contribution = 0.2275. NOT in ranker_v2 (removed "
        "in minimal_v2 promotion 2026-04-05; ranker audit flagged as noise).",
    ),
    # --- Sidecar research overlays (not in active path) ---
    FieldContract(
        "ees_v2_score",
        FieldStatus.SIDECAR_RESEARCH,
        "EES v2 expectation-error overlay. Sidecar: expectation_error_overlay.json. "
        "Not consumed by ranker or selector.",
    ),
    FieldContract(
        "ees_v3_score",
        FieldStatus.SIDECAR_RESEARCH,
        "EES v3 (0.70×misprice_z + 0.30×expected_move_z). Sidecar: ees_v3_overlay.json. "
        "Awaiting Checklist v2 5/5 (currently 4/5; WS4 fail t_adj=1.53 vs 1.65 threshold) "
        "before promotion. Diagnostic overlay only.",
    ),
    FieldContract(
        "expectation_error_score",
        FieldStatus.SIDECAR_RESEARCH,
        "Jane Street 6-mistake framework overlay. Sidecar (rankings.csv pass-through "
        "and expectation_error_overlay.json). Not consumed by ranker or selector.",
    ),
    FieldContract(
        "conditional_misprice_score",
        FieldStatus.SIDECAR_RESEARCH,
        "EES v3 component (true alpha source per WS audit, IC +0.089, NW t=2.07). "
        "Sidecar via EES v3 overlay; not in ranker pending Checklist v2 promotion.",
    ),
    FieldContract(
        "base_rate_gap_score",
        FieldStatus.SIDECAR_RESEARCH,
        "EES v3 component, NEGATIVE CONTROL ONLY (IC -0.090, anti-predictive — "
        "the trap is dead, market is right). Kept for attribution; never promote.",
    ),
    # --- Diagnostic pass-throughs (Form 4, etc.) ---
    FieldContract(
        "insider_net_buy_value_90d",
        FieldStatus.DIAGNOSTIC_PASS_THROUGH,
        "Form 4 wired into rankings.csv as pass-through (Pass B, 2026-04-24). "
        "Scoring lane CLOSED 2026-04-05 and not reopened. `required=True` in "
        "production_qa_check.py is data-integrity gate only — does NOT promote "
        "to selector/ranker/sizing. See Spec 065.",
    ),
    # --- Deprecated / inert ---
    FieldContract(
        "clinical_score_v2_z",
        FieldStatus.DEPRECATED_INERT,
        "Removed from ranker feature_set in minimal_v2 promotion (2026-04-05). "
        "Ablation study confirmed -0.35pp destructive (Spec 055). Computation "
        "retained for clinical research module + selector clinical block, but "
        "selector clinical block weight is 0.00 in A4 — field does not reach "
        "selector_score. All clinical lanes CLOSED for promotion.",
    ),
    FieldContract(
        "clinical_quality_score",
        FieldStatus.DEPRECATED_INERT,
        "Spec 057 monitor-only. Sizing tilt TESTED and REJECTED. Conditional IC "
        "within top coinvest +0.103 (t=3.53), but standalone borderline; sizing "
        "backtest showed B6^1.5 alone wins. Use for attribution / cohort analysis "
        "only. NOT for rank/sizing/selection.",
    ),
    FieldContract(
        "clinical_alpha_z",
        FieldStatus.DEPRECATED_INERT,
        "Cross-sectional clinical z within dev cohort. Selector clinical block "
        "zeroed Spec 055. Computation kept; does not reach selector_score.",
    ),
    FieldContract(
        "coinvest_recency_state",
        FieldStatus.DEPRECATED_INERT,
        "Selector institutional categorical. Default SelectorConfig weights it "
        "0.20; A4_SELECTOR_CONFIG overrides to 0.00. Computation kept; does not "
        "reach selector_score under A4.",
    ),
)


# ---------------------------------------------------------------------------
# Lookup helpers (zero side effects)
# ---------------------------------------------------------------------------


def field_status(name: str) -> FieldStatus:
    """Return the contract status for `name`, or raise KeyError if unregistered.

    Unregistered ≠ inactive. Most rankings.csv columns are not in the registry;
    this helper is for fields where status is known and asserted.
    """
    for fc in SCORING_FIELDS:
        if fc.name == name:
            return fc.status
    raise KeyError(name)


def fields_with_status(status: FieldStatus) -> Tuple[str, ...]:
    """Return all registered field names matching `status`."""
    return tuple(fc.name for fc in SCORING_FIELDS if fc.status == status)
