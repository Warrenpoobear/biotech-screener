"""Tests asserting the active ranker/selector contract in
`common.ranker_active_contract` matches production reality:

  - production_data/ranker_v2_model.json (`model.feature_names`)
  - run_screen.A4_SELECTOR_CONFIG (block weights + institutional signals)

These tests are the gate that prevents "we said feature X is inactive but it's
actually weighted in production" or vice versa. If the registry diverges from
ground truth, fail loudly and force the author to update both.

Also includes per-field guards for known-dead lanes flagged in alpha
attribution audits — so a future reader cannot accidentally re-classify
clinical_score_v2_z, EES, or insider_net_buy_value_90d as ACTIVE without
this test failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.ranker_active_contract import (
    ACTIVE_INSTITUTIONAL_SIGNALS,
    ACTIVE_RANKER_FEATURES,
    ACTIVE_SELECTOR_BLOCKS,
    SCORING_FIELDS,
    FieldStatus,
    field_status,
    fields_with_status,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Drift gates — registry vs. ground truth
# ---------------------------------------------------------------------------


class TestActiveRankerDrift:
    """Active ranker registry must equal `production_data/ranker_v2_model.json`."""

    def test_active_ranker_features_match_model_artifact(self):
        artifact_path = REPO_ROOT / "production_data" / "ranker_v2_model.json"
        assert artifact_path.exists(), f"Model artifact missing: {artifact_path}"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        assert tuple(artifact["model"]["feature_names"]) == ACTIVE_RANKER_FEATURES, (
            "Drift: ranker_v2_model.json feature_names != ACTIVE_RANKER_FEATURES. "
            "Update common/ranker_active_contract.py:ACTIVE_RANKER_FEATURES OR "
            "the model artifact, but never silently."
        )

    def test_ranker_artifact_is_minimal_v2(self):
        """Sanity: deployed config_id should match `feature_set='minimal_v2'`."""
        artifact_path = REPO_ROOT / "production_data" / "ranker_v2_model.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["config"]["feature_set"] == "minimal_v2"
        assert artifact["model"]["n_features"] == 2

    def test_inactive_fields_not_in_ranker_features(self):
        """Inactive (sidecar / deprecated / pass-through) fields must NEVER
        appear in the deployed ranker_v2_model.json feature list."""
        artifact_path = REPO_ROOT / "production_data" / "ranker_v2_model.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        feature_names = set(artifact["model"]["feature_names"])

        inactive_statuses = {
            FieldStatus.DIAGNOSTIC_PASS_THROUGH,
            FieldStatus.SIDECAR_RESEARCH,
            FieldStatus.DEPRECATED_INERT,
        }
        for fc in SCORING_FIELDS:
            if fc.status in inactive_statuses:
                assert fc.name not in feature_names, (
                    f"Field {fc.name!r} is registered as {fc.status.value} but "
                    f"appears in production ranker_v2_model.json — registry says "
                    f"it should not drive scoring."
                )


class TestActiveSelectorDrift:
    """Active selector registry must equal `run_screen.A4_SELECTOR_CONFIG`."""

    def test_active_selector_blocks_match_run_screen_config(self):
        from run_screen import A4_SELECTOR_CONFIG

        actual = tuple((bw.name, bw.weight) for bw in A4_SELECTOR_CONFIG.block_weights)
        assert actual == ACTIVE_SELECTOR_BLOCKS, (
            "Drift: run_screen.A4_SELECTOR_CONFIG.block_weights != "
            "ACTIVE_SELECTOR_BLOCKS. Update common/ranker_active_contract.py:"
            "ACTIVE_SELECTOR_BLOCKS OR run_screen.py, but never silently."
        )

    def test_active_institutional_signals_match_run_screen_config(self):
        from run_screen import A4_SELECTOR_CONFIG

        actual = tuple((s.name, s.weight) for s in A4_SELECTOR_CONFIG.institutional_signals)
        assert actual == ACTIVE_INSTITUTIONAL_SIGNALS, (
            "Drift: run_screen.A4_SELECTOR_CONFIG.institutional_signals != " "ACTIVE_INSTITUTIONAL_SIGNALS."
        )

    def test_clinical_block_is_zero(self):
        """Clinical block weight must remain 0.00 — Spec 055 confirmed
        destructive. Any non-zero clinical block requires Checklist v2."""
        from run_screen import A4_SELECTOR_CONFIG

        clinical = next(bw for bw in A4_SELECTOR_CONFIG.block_weights if bw.name == "clinical")
        assert clinical.weight == 0.00, (
            f"Clinical block weight = {clinical.weight}, expected 0.00. "
            "Spec 055 confirmed clinical block is destructive in production. "
            "Re-enabling requires Checklist v2 (FM + bootstrap + FDR + LOSO + year stability)."
        )

    def test_block_weights_sum_to_one(self):
        from run_screen import A4_SELECTOR_CONFIG

        total = sum(bw.weight for bw in A4_SELECTOR_CONFIG.block_weights)
        assert abs(total - 1.0) < 1e-9, f"Block weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# Per-field guards — known-dead lanes
# ---------------------------------------------------------------------------


class TestKnownDeadLanes:
    """Specific guards: lanes that have been explicitly closed must stay
    classified as DEPRECATED_INERT / SIDECAR_RESEARCH / DIAGNOSTIC_PASS_THROUGH.
    These guards prevent a future reader from accidentally re-tagging them as
    ACTIVE without also redeploying a model that consumes them."""

    @pytest.mark.parametrize(
        "field_name,expected_status",
        [
            # Clinical lanes — Spec 055 confirmed destructive
            ("clinical_score_v2_z", FieldStatus.DEPRECATED_INERT),
            ("clinical_quality_score", FieldStatus.DEPRECATED_INERT),
            ("clinical_alpha_z", FieldStatus.DEPRECATED_INERT),
            # Coinvest recency — A4 override zeroed it
            ("coinvest_recency_state", FieldStatus.DEPRECATED_INERT),
            # EES — sidecar pending Checklist v2
            ("ees_v2_score", FieldStatus.SIDECAR_RESEARCH),
            ("ees_v3_score", FieldStatus.SIDECAR_RESEARCH),
            ("expectation_error_score", FieldStatus.SIDECAR_RESEARCH),
            ("conditional_misprice_score", FieldStatus.SIDECAR_RESEARCH),
            ("base_rate_gap_score", FieldStatus.SIDECAR_RESEARCH),
            # Form 4 — scoring lane closed 2026-04-05
            ("insider_net_buy_value_90d", FieldStatus.DIAGNOSTIC_PASS_THROUGH),
        ],
    )
    def test_field_status(self, field_name, expected_status):
        actual = field_status(field_name)
        assert actual == expected_status, (
            f"{field_name}: expected {expected_status.value}, got {actual.value}. "
            "If this field's status genuinely changed, a model retrain or lane "
            "re-opening is required — not just a registry edit."
        )

    def test_active_ranker_features_are_registered_active(self):
        """Every entry in ACTIVE_RANKER_FEATURES must also be in the
        SCORING_FIELDS registry tagged ACTIVE_RANKER, so the per-field guards
        and the top-level constant agree."""
        registered_active = set(fields_with_status(FieldStatus.ACTIVE_RANKER))
        for feat in ACTIVE_RANKER_FEATURES:
            assert feat in registered_active, (
                f"{feat} is in ACTIVE_RANKER_FEATURES but not registered as " f"ACTIVE_RANKER in SCORING_FIELDS."
            )


# ---------------------------------------------------------------------------
# Registry hygiene
# ---------------------------------------------------------------------------


class TestRegistryHygiene:
    def test_no_duplicate_field_names(self):
        names = [fc.name for fc in SCORING_FIELDS]
        assert len(names) == len(set(names)), "Duplicate field name in SCORING_FIELDS"

    def test_every_entry_has_a_note(self):
        """Every contract entry must include a note explaining the
        classification — the note is the audit trail."""
        for fc in SCORING_FIELDS:
            assert fc.note.strip(), f"{fc.name}: empty note"

    def test_field_status_unknown_raises(self):
        with pytest.raises(KeyError):
            field_status("definitely_not_a_real_field_xyz")
