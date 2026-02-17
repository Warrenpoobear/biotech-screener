"""
Decision Engine Explanation Contract Tests (CI-enforced).

Validates that:
    1. All codes emitted by the decision engine belong to the canonical registry
    2. Overlay enum values are exhaustive
    3. Mutual exclusivity invariants hold
    4. build_explanation() produces correct structured output
    5. Registry fingerprint is pinned (forces explicit update on code changes)

Uses the same golden fixtures and _rec() builder from test_decision_engine_contract.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DECISION_COLUMNS,
    compute_decision_fields,
)
from decision_engine_codes import (
    ELIGIBILITY_CODES,
    RISK_FLAG_CODES,
    SIZING_CODES,
    VALID_CATALYST_MODES,
    VALID_CATALYST_STRENGTHS,
    VALID_MOM_STATES,
    VALID_RUNWAY_BUCKETS,
    VALID_TIER_REASONS,
    build_explanation,
    describe_code,
    registry_fingerprint,
)
from tests.test_decision_engine_contract import (
    ALL_GOLDEN,
    _compute_golden,
    _rec,
)


# =============================================================================
# HELPERS
# =============================================================================

def _all_golden_fields() -> List[Dict[str, Any]]:
    """Compute decision fields for all 10 golden fixtures."""
    return [_compute_golden(g) for g in ALL_GOLDEN]


def _parse_pipe(val: str) -> List[str]:
    """Split pipe-separated string, filtering empties."""
    if not val:
        return []
    return [s for s in val.split("|") if s]


# =============================================================================
# 1. REGISTRY COMPLETENESS — every emitted code is in the registry
# =============================================================================

class TestRegistryCompleteness:
    """Every code emitted by golden fixtures must be in the canonical registry."""

    @pytest.fixture(scope="class")
    def all_fields(self) -> List[Dict[str, Any]]:
        return _all_golden_fields()

    def test_all_ineligible_reasons_in_registry(self, all_fields):
        """Every pipe-parsed code from ineligible_reasons is in ELIGIBILITY_CODES."""
        for fields in all_fields:
            codes = _parse_pipe(fields.get("ineligible_reasons", ""))
            for code in codes:
                assert code in ELIGIBILITY_CODES, (
                    f"Ineligible reason '{code}' not in ELIGIBILITY_CODES. "
                    f"Update decision_engine_codes.py to add it."
                )

    def test_all_risk_flags_in_registry(self, all_fields):
        """Every pipe-parsed code from risk_flags is in RISK_FLAG_CODES."""
        for fields in all_fields:
            codes = _parse_pipe(fields.get("risk_flags", ""))
            for code in codes:
                assert code in RISK_FLAG_CODES, (
                    f"Risk flag '{code}' not in RISK_FLAG_CODES. "
                    f"Update decision_engine_codes.py to add it."
                )

    def test_all_size_reasons_in_registry(self, all_fields):
        """Every pipe-parsed code from size_reasons is in SIZING_CODES."""
        for fields in all_fields:
            codes = _parse_pipe(fields.get("size_reasons", ""))
            for code in codes:
                assert code in SIZING_CODES, (
                    f"Sizing reason '{code}' not in SIZING_CODES. "
                    f"Update decision_engine_codes.py to add it."
                )

    def test_all_tier_reasons_in_registry(self, all_fields):
        """Every tier_reason string is in VALID_TIER_REASONS."""
        for fields in all_fields:
            tr = fields.get("tier_reason", "")
            assert tr in VALID_TIER_REASONS, (
                f"Tier reason '{tr}' not in VALID_TIER_REASONS. "
                f"Update decision_engine_codes.py to add it."
            )


# =============================================================================
# 2. ENUM EXHAUSTIVENESS — overlay values are in valid sets
# =============================================================================

class TestEnumExhaustiveness:
    """All golden fixture overlay values belong to canonical enum sets."""

    @pytest.fixture(scope="class")
    def all_fields(self) -> List[Dict[str, Any]]:
        return _all_golden_fields()

    def test_catalyst_mode_in_valid_set(self, all_fields):
        """All golden catalyst_mode values in VALID_CATALYST_MODES."""
        for fields in all_fields:
            val = fields.get("catalyst_mode", "")
            assert val in VALID_CATALYST_MODES, (
                f"catalyst_mode '{val}' not in VALID_CATALYST_MODES"
            )

    def test_catalyst_strength_in_valid_set(self, all_fields):
        """All golden catalyst_strength values in VALID_CATALYST_STRENGTHS."""
        for fields in all_fields:
            val = fields.get("catalyst_strength", "")
            assert val in VALID_CATALYST_STRENGTHS, (
                f"catalyst_strength '{val}' not in VALID_CATALYST_STRENGTHS"
            )

    def test_mom_state_in_valid_set(self, all_fields):
        """All golden mom_state values in VALID_MOM_STATES."""
        for fields in all_fields:
            val = fields.get("mom_state", "")
            assert val in VALID_MOM_STATES, (
                f"mom_state '{val}' not in VALID_MOM_STATES"
            )

    def test_runway_bucket_in_valid_set(self, all_fields):
        """All golden runway_bucket values in VALID_RUNWAY_BUCKETS."""
        for fields in all_fields:
            val = fields.get("runway_bucket", "")
            assert val in VALID_RUNWAY_BUCKETS, (
                f"runway_bucket '{val}' not in VALID_RUNWAY_BUCKETS"
            )


# =============================================================================
# 3. MUTUAL EXCLUSIVITY — certain codes never co-occur
# =============================================================================

class TestMutualExclusivity:
    """Structural invariants: certain codes are mutually exclusive."""

    def test_dd_missing_and_deep_dd_exclusive_in_risk(self):
        """drawdown_data_missing and deep_drawdown never co-occur in risk_flags."""
        # Test with missing drawdown
        rec_missing = _rec("ME_MISS", drawdown=None)
        f1 = compute_decision_fields(rec_missing, "drug_developer", 0.65)
        flags1 = _parse_pipe(f1.get("risk_flags", ""))
        assert not ("drawdown_data_missing" in flags1 and "deep_drawdown" in flags1), (
            "drawdown_data_missing and deep_drawdown must be mutually exclusive"
        )

        # Test with deep drawdown
        rec_deep = _rec("ME_DEEP", drawdown=-0.55)
        f2 = compute_decision_fields(rec_deep, "drug_developer", 0.65)
        flags2 = _parse_pipe(f2.get("risk_flags", ""))
        assert not ("drawdown_data_missing" in flags2 and "deep_drawdown" in flags2), (
            "drawdown_data_missing and deep_drawdown must be mutually exclusive"
        )

    def test_ineligible_and_tier_a_dev_exclusive_in_sizing(self):
        """ineligible and tier_a_dev never co-occur in size_reasons."""
        for fields in _all_golden_fields():
            codes = _parse_pipe(fields.get("size_reasons", ""))
            assert not ("ineligible" in codes and "tier_a_dev" in codes), (
                "ineligible and tier_a_dev must be mutually exclusive in sizing"
            )

    def test_eligible_zero_implies_tier_d_or_blank(self):
        """eligible='0' with drug_developer archetype → tier_dev='D'."""
        for g in ALL_GOLDEN:
            if g["archetype"] != "drug_developer":
                continue
            fields = _compute_golden(g)
            if fields["eligible"] == "0":
                assert fields["tier_dev"] == "D", (
                    f"[{g['ticker']}] eligible=0 + drug_developer should give tier_dev=D, "
                    f"got '{fields['tier_dev']}'"
                )

    def test_tier_a_requires_actionable_catalyst(self):
        """tier_dev='A' → catalyst_strength in ('near', 'mid')."""
        for g in ALL_GOLDEN:
            fields = _compute_golden(g)
            if fields.get("tier_dev") == "A":
                strength = fields.get("catalyst_strength", "")
                assert strength in ("near", "mid"), (
                    f"[{g['ticker']}] A-tier requires actionable catalyst "
                    f"(near or mid), got '{strength}'"
                )


# =============================================================================
# 4. BUILD EXPLANATION — structured output correctness
# =============================================================================

class TestBuildExplanation:
    """build_explanation() produces correct structured output."""

    def test_explanation_has_all_domains(self):
        """Output has eligibility, tier, sizing, risk, overlays domains."""
        for fields in _all_golden_fields():
            explanation = build_explanation(fields)
            expected_domains = {"eligibility", "tier", "sizing", "risk", "overlays"}
            assert set(explanation.keys()) == expected_domains, (
                f"Explanation missing domains: "
                f"{expected_domains - set(explanation.keys())}"
            )

            # Check sub-structure
            assert "passed" in explanation["eligibility"]
            assert "failed_gates" in explanation["eligibility"]
            assert "gate_details" in explanation["eligibility"]

            assert "tier" in explanation["tier"]
            assert "tier_reason_raw" in explanation["tier"]
            assert "primary_reasons" in explanation["tier"]
            assert "reason_descriptions" in explanation["tier"]

            assert "band" in explanation["sizing"]
            assert "modifiers" in explanation["sizing"]
            assert "net_modifier" in explanation["sizing"]

            assert "flags" in explanation["risk"]
            assert "flag_details" in explanation["risk"]

            assert "catalyst_mode" in explanation["overlays"]
            assert "catalyst_strength" in explanation["overlays"]
            assert "mom_state" in explanation["overlays"]

    def test_tier_a_decomposition(self):
        """A-tier golden: primary_reasons contains 'high_opt' and a catalyst tag."""
        # GOLDEN_1 is A-tier with high_opt+catalyst_near
        fields = _compute_golden(ALL_GOLDEN[0])
        assert fields["tier_dev"] == "A", "Fixture 1 should be A-tier"

        explanation = build_explanation(fields)
        tier_block = explanation["tier"]

        assert tier_block["tier"] == "A"
        assert "high_opt" in tier_block["primary_reasons"]
        # Should have a catalyst component
        catalyst_reasons = [r for r in tier_block["primary_reasons"]
                           if r.startswith("catalyst_")]
        assert len(catalyst_reasons) == 1, (
            f"Expected exactly one catalyst tag in A-tier reasons, "
            f"got {catalyst_reasons}"
        )
        assert tier_block["optionality_band"] == "high_opt"
        assert tier_block["catalyst_proximity"] != ""


# =============================================================================
# 5. REGISTRY FINGERPRINT — pinned value for drift detection
# =============================================================================

class TestRegistryFingerprint:
    """Registry fingerprint must match pinned value."""

    EXPECTED_FINGERPRINT = "3877c5c0b774"

    def test_registry_fingerprint_pinned(self):
        """registry_fingerprint() matches pinned value.

        If this test fails:
          1. Verify the registry change is intentional
          2. Update EXPECTED_FINGERPRINT below
          3. Add a note to RULESET_CHANGELOG.md
        """
        actual = registry_fingerprint()
        assert actual == self.EXPECTED_FINGERPRINT, (
            f"Registry fingerprint changed: {self.EXPECTED_FINGERPRINT} → {actual}. "
            f"A code was added, removed, or renamed in decision_engine_codes.py. "
            f"If intentional: update EXPECTED_FINGERPRINT."
        )

    def test_registry_change_requires_changelog_entry(self):
        """When registry fingerprint changes, RULESET_CHANGELOG.md must reference it.

        Skip-when-no-change: green when fingerprint matches pinned value.
        """
        actual = registry_fingerprint()
        if actual == self.EXPECTED_FINGERPRINT:
            pytest.skip("Fingerprint unchanged — no changelog entry required")

        changelog = PROJECT_ROOT / "RULESET_CHANGELOG.md"
        assert changelog.exists(), "RULESET_CHANGELOG.md not found at project root"
        text = changelog.read_text(encoding="utf-8")
        assert actual in text, (
            f"Registry fingerprint changed to {actual} but "
            f"RULESET_CHANGELOG.md does not reference it. "
            f"Add a changelog entry that includes the new fingerprint."
        )

    def test_registry_change_requires_non_draft_changelog(self):
        """Changelog entry containing the new fingerprint must not be [DRAFT].

        Skip-when-no-change: green when fingerprint matches pinned value.
        """
        actual = registry_fingerprint()
        if actual == self.EXPECTED_FINGERPRINT:
            pytest.skip("Fingerprint unchanged — no draft check required")

        changelog = PROJECT_ROOT / "RULESET_CHANGELOG.md"
        assert changelog.exists(), "RULESET_CHANGELOG.md not found at project root"
        text = changelog.read_text(encoding="utf-8")

        # Find the section header (## [...]) that contains the fingerprint
        for line in text.splitlines():
            if line.startswith("## ") and actual in line:
                assert "[DRAFT]" not in line, (
                    f"Changelog entry for registry fingerprint {actual} is still "
                    f"marked [DRAFT]. Finalize the entry before merging."
                )
                return
        # If we reach here, the fingerprint wasn't in any section header —
        # the first test already catches this case.
        pytest.fail(
            f"Registry fingerprint {actual} not found in any ## section header "
            f"of RULESET_CHANGELOG.md."
        )
