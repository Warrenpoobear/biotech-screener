"""
test_pos_model_v2.py - Comprehensive test suite for PoS Model v2

Tests cover:
1. Base rate lookups (exact, partial, default fallback)
2. Modifier calculations (FDA designations, trial characteristics)
3. Boundary conditions (floor, ceiling, modifier caps)
4. Decimal precision and rounding
5. Audit trail completeness
6. Integration helpers
"""

import json
from decimal import Decimal
from datetime import date

from pos_model_v2 import (
    PosModelV2,
    ClinicalStage,
    TherapeuticArea,
    MechanismClass,
    FDADesignation,
    TrialCharacteristic,
    BaseRateEntry,
    POS_FLOOR,
    POS_CEILING,
    MAX_MODIFIER_PRODUCT,
    MIN_MODIFIER_PRODUCT,
    enhanced_stage_score,
    pos_to_catalyst_ev_weight,
    validate_model_consistency,
)


class TestResults:
    """Accumulator for test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append((name, message))
            print(f"  ✗ {name}: {message}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Tests: {total} | Passed: {self.passed} | Failed: {self.failed}")
        if self.errors:
            print("\nFailed tests:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_enum_parsing():
    """Test enum parsing from string inputs."""
    results = TestResults()
    print("\n--- Enum Parsing Tests ---")
    
    # Stage parsing
    results.record(
        "Parse 'phase_3' to PHASE_3",
        ClinicalStage.from_string("phase_3") == ClinicalStage.PHASE_3
    )
    results.record(
        "Parse 'Phase 3' to PHASE_3",
        ClinicalStage.from_string("Phase 3") == ClinicalStage.PHASE_3
    )
    results.record(
        "Parse 'P3' to PHASE_3",
        ClinicalStage.from_string("P3") == ClinicalStage.PHASE_3
    )
    results.record(
        "Parse 'NDA' to FILED",
        ClinicalStage.from_string("NDA") == ClinicalStage.FILED
    )
    
    # Therapeutic area parsing
    results.record(
        "Parse 'oncology' to ONCOLOGY",
        TherapeuticArea.from_string("oncology") == TherapeuticArea.ONCOLOGY
    )
    results.record(
        "Parse 'rare disease' to RARE_DISEASE",
        TherapeuticArea.from_string("rare disease") == TherapeuticArea.RARE_DISEASE
    )
    results.record(
        "Parse 'neuro' to CNS",
        TherapeuticArea.from_string("neuro") == TherapeuticArea.CNS
    )
    results.record(
        "Parse unknown to OTHER",
        TherapeuticArea.from_string("xyz_unknown") == TherapeuticArea.OTHER
    )
    
    # Mechanism parsing
    results.record(
        "Parse 'mab' to MONOCLONAL_ANTIBODY",
        MechanismClass.from_string("mab") == MechanismClass.MONOCLONAL_ANTIBODY
    )
    results.record(
        "Parse 'gene therapy' to GENE_THERAPY",
        MechanismClass.from_string("gene therapy") == MechanismClass.GENE_THERAPY
    )
    results.record(
        "Parse 'car_t' to CAR_T",
        MechanismClass.from_string("car_t") == MechanismClass.CAR_T
    )
    
    return results


def test_base_rate_lookup():
    """Test base rate lookup with fallback hierarchy."""
    results = TestResults()
    print("\n--- Base Rate Lookup Tests ---")
    
    model = PosModelV2()
    
    # Test exact match
    result = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
    )
    results.record(
        "Exact match lookup type",
        result.base_pos_lookup == "exact",
        f"Got: {result.base_pos_lookup}"
    )
    results.record(
        "Exact match base_pos = 0.42",
        result.base_pos == Decimal("0.42"),
        f"Got: {result.base_pos}"
    )
    
    # Test partial match (indication)
    result2 = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.BISPECIFIC,  # Not in matrix
    )
    results.record(
        "Partial match (indication) lookup type",
        result2.base_pos_lookup in ("partial_indication", "partial_mechanism"),
        f"Got: {result2.base_pos_lookup}"
    )
    
    # Test default fallback
    result3 = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.DERMATOLOGY,  # Limited coverage
        mechanism_class=MechanismClass.OTHER,
    )
    # Should either find a partial match or use default
    results.record(
        "Fallback lookup type valid",
        result3.base_pos_lookup in ("partial_indication", "partial_mechanism", "default"),
        f"Got: {result3.base_pos_lookup}"
    )
    
    return results


def test_modifier_calculations():
    """Test FDA and trial characteristic modifier calculations."""
    results = TestResults()
    print("\n--- Modifier Calculation Tests ---")
    
    model = PosModelV2()
    
    # Test single FDA modifier
    result = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
        fda_designations=[FDADesignation.BREAKTHROUGH_THERAPY],
    )
    results.record(
        "BTD modifier = 1.25",
        result.fda_modifier_product == Decimal("1.25"),
        f"Got: {result.fda_modifier_product}"
    )
    
    # Test multiple FDA modifiers
    result2 = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.RARE_DISEASE,
        mechanism_class=MechanismClass.GENE_THERAPY,
        fda_designations=[
            FDADesignation.ORPHAN_DRUG,
            FDADesignation.FAST_TRACK,
        ],
    )
    expected = Decimal("1.15") * Decimal("1.10")  # 1.265
    results.record(
        "Multiple FDA modifiers multiply correctly",
        result2.fda_modifier_product == expected,
        f"Expected: {expected}, Got: {result2.fda_modifier_product}"
    )
    
    # Test trial characteristic modifiers
    result3 = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
        trial_characteristics=[
            TrialCharacteristic.BIOMARKER_SELECTED,
            TrialCharacteristic.FIRST_IN_CLASS,
        ],
    )
    expected_trial = Decimal("1.20") * Decimal("1.30")  # 1.56
    results.record(
        "Trial modifiers multiply correctly",
        result3.trial_modifier_product == expected_trial,
        f"Expected: {expected_trial}, Got: {result3.trial_modifier_product}"
    )
    
    # Test negative modifiers
    result4 = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
        trial_characteristics=[
            TrialCharacteristic.SURROGATE_ENDPOINT,
            TrialCharacteristic.ACTIVE_COMPARATOR,
        ],
    )
    expected_neg = Decimal("0.75") * Decimal("0.85")  # 0.6375
    results.record(
        "Negative modifiers multiply correctly",
        result4.trial_modifier_product == expected_neg,
        f"Expected: {expected_neg}, Got: {result4.trial_modifier_product}"
    )
    
    return results


def test_boundary_conditions():
    """Test floor, ceiling, and modifier cap enforcement."""
    results = TestResults()
    print("\n--- Boundary Condition Tests ---")
    
    model = PosModelV2()
    
    # Test ceiling enforcement (many positive modifiers)
    result = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.RARE_DISEASE,
        mechanism_class=MechanismClass.GENE_THERAPY,  # 0.58 base
        fda_designations=[
            FDADesignation.BREAKTHROUGH_THERAPY,
            FDADesignation.ORPHAN_DRUG,
            FDADesignation.FAST_TRACK,
            FDADesignation.RMAT,
        ],
        trial_characteristics=[
            TrialCharacteristic.BIOMARKER_SELECTED,
            TrialCharacteristic.FIRST_IN_CLASS,
        ],
    )
    results.record(
        f"PoS ceiling enforced (<= {POS_CEILING})",
        result.adjusted_pos <= POS_CEILING,
        f"Got: {result.adjusted_pos}"
    )
    results.record(
        f"Modifier cap enforced (<= {MAX_MODIFIER_PRODUCT})",
        result.total_modifier_product <= MAX_MODIFIER_PRODUCT,
        f"Got: {result.total_modifier_product}"
    )
    
    # Test floor enforcement (many negative modifiers on low base)
    result2 = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_2,
        therapeutic_area=TherapeuticArea.CNS,
        mechanism_class=MechanismClass.SMALL_MOLECULE,  # 0.22 base
        trial_characteristics=[
            TrialCharacteristic.SURROGATE_ENDPOINT,
            TrialCharacteristic.ACTIVE_COMPARATOR,
        ],
    )
    results.record(
        f"PoS floor enforced (>= {POS_FLOOR})",
        result2.adjusted_pos >= POS_FLOOR,
        f"Got: {result2.adjusted_pos}"
    )
    
    # Test modifier floor enforcement
    results.record(
        f"Modifier floor enforced (>= {MIN_MODIFIER_PRODUCT})",
        result2.total_modifier_product >= MIN_MODIFIER_PRODUCT,
        f"Got: {result2.total_modifier_product}"
    )
    
    return results


def test_decimal_precision():
    """Test Decimal arithmetic precision and rounding."""
    results = TestResults()
    print("\n--- Decimal Precision Tests ---")
    
    model = PosModelV2()
    
    result = model.calculate(
        ticker="TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
        fda_designations=[FDADesignation.BREAKTHROUGH_THERAPY],
    )
    
    # Check that result is Decimal
    results.record(
        "adjusted_pos is Decimal type",
        isinstance(result.adjusted_pos, Decimal),
        f"Got: {type(result.adjusted_pos)}"
    )
    
    # Check precision (4 decimal places)
    str_result = str(result.adjusted_pos)
    if '.' in str_result:
        decimal_places = len(str_result.split('.')[1])
        results.record(
            "Rounded to 4 decimal places",
            decimal_places <= 4,
            f"Got: {decimal_places} places"
        )
    else:
        results.record("Rounded to 4 decimal places", True)
    
    # Verify calculation: 0.42 * 1.25 = 0.525
    expected = Decimal("0.5250")
    results.record(
        "Calculation precision: 0.42 * 1.25 = 0.5250",
        result.adjusted_pos == expected,
        f"Expected: {expected}, Got: {result.adjusted_pos}"
    )
    
    return results


def test_audit_trail():
    """Test audit trail completeness."""
    results = TestResults()
    print("\n--- Audit Trail Tests ---")
    
    model = PosModelV2()
    
    result = model.calculate(
        ticker="AUDIT_TEST",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
        fda_designations=[FDADesignation.BREAKTHROUGH_THERAPY],
        trial_characteristics=[TrialCharacteristic.BIOMARKER_SELECTED],
    )
    
    # Convert to dict
    result_dict = result.to_dict()
    
    # Check required fields
    results.record(
        "Has ticker",
        "ticker" in result_dict and result_dict["ticker"] == "AUDIT_TEST",
        f"Got: {result_dict.get('ticker')}"
    )
    results.record(
        "Has inputs section",
        "inputs" in result_dict and isinstance(result_dict["inputs"], dict)
    )
    results.record(
        "Has base_rate section",
        "base_rate" in result_dict and "value" in result_dict["base_rate"]
    )
    results.record(
        "Has modifiers section with details",
        "modifiers" in result_dict and "details" in result_dict["modifiers"]
    )
    results.record(
        "Has result section",
        "result" in result_dict and "adjusted_pos" in result_dict["result"]
    )
    results.record(
        "Has metadata with timestamp",
        "metadata" in result_dict and "timestamp" in result_dict["metadata"]
    )
    results.record(
        "Has model version",
        result_dict.get("metadata", {}).get("model_version") == "2.0.0"
    )
    
    # Check audit hash
    hash1 = result.get_audit_hash()
    hash2 = result.get_audit_hash()
    results.record(
        "Audit hash is deterministic",
        hash1 == hash2,
        f"Hash1: {hash1[:16]}..., Hash2: {hash2[:16]}..."
    )
    results.record(
        "Audit hash is 64 char hex (SHA256)",
        len(hash1) == 64 and all(c in "0123456789abcdef" for c in hash1)
    )
    
    # Check modifier details
    modifier_details = result_dict["modifiers"]["details"]
    results.record(
        "Modifier details has 2 entries (1 FDA + 1 trial)",
        len(modifier_details) == 2,
        f"Got: {len(modifier_details)}"
    )
    
    # Check JSON serialization
    try:
        json_str = json.dumps(result_dict)
        reparsed = json.loads(json_str)
        results.record("JSON serialization round-trip works", True)
    except Exception as e:
        results.record("JSON serialization round-trip works", False, str(e))
    
    return results


def test_integration_helpers():
    """Test integration helper functions."""
    results = TestResults()
    print("\n--- Integration Helper Tests ---")
    
    # Test enhanced_stage_score
    # High PoS should boost score
    boosted = enhanced_stage_score(
        base_stage_score=Decimal("65"),
        adjusted_pos=Decimal("0.70"),
        median_pos=Decimal("0.50"),
    )
    expected_boost = Decimal("65") * (Decimal("0.70") / Decimal("0.50"))  # 91
    results.record(
        "High PoS boosts stage score",
        boosted == expected_boost,
        f"Expected: {expected_boost}, Got: {boosted}"
    )
    
    # Low PoS should penalize score
    penalized = enhanced_stage_score(
        base_stage_score=Decimal("65"),
        adjusted_pos=Decimal("0.30"),
        median_pos=Decimal("0.50"),
    )
    expected_pen = Decimal("65") * (Decimal("0.30") / Decimal("0.50"))  # 39
    results.record(
        "Low PoS penalizes stage score",
        penalized == expected_pen,
        f"Expected: {expected_pen}, Got: {penalized}"
    )
    
    # Test pos_to_catalyst_ev_weight
    ev_high = pos_to_catalyst_ev_weight(
        adjusted_pos=Decimal("1.0"),
        base_ev=Decimal("100"),
    )
    results.record(
        "EV weight at PoS=1.0 equals base_ev",
        ev_high == Decimal("100"),
        f"Got: {ev_high}"
    )
    
    ev_mid = pos_to_catalyst_ev_weight(
        adjusted_pos=Decimal("0.5"),
        base_ev=Decimal("100"),
    )
    # With blend_factor=0.70: 100 * (0.3 + 0.7 * 0.5) = 100 * 0.65 = 65
    results.record(
        "EV weight at PoS=0.5 = 65 (with default blend)",
        ev_mid == Decimal("65"),
        f"Got: {ev_mid}"
    )
    
    return results


def test_model_validation():
    """Test the model consistency validation."""
    results = TestResults()
    print("\n--- Model Validation Tests ---")
    
    validation = validate_model_consistency()
    
    results.record(
        "Model passes consistency validation",
        validation["valid"],
        f"Issues: {validation.get('issues', [])}"
    )
    results.record(
        "Has coverage stats",
        "coverage_stats" in validation
    )
    results.record(
        "Has stage averages",
        "stage_averages" in validation
    )
    
    return results


def test_string_input_convenience():
    """Test the calculate_from_strings convenience method."""
    results = TestResults()
    print("\n--- String Input Convenience Tests ---")
    
    model = PosModelV2()
    
    result = model.calculate_from_strings(
        ticker="STRING_TEST",
        stage="phase 3",
        therapeutic_area="oncology",
        mechanism_class="small molecule",
        fda_designations=["breakthrough_therapy"],
        trial_characteristics=["biomarker_selected"],
    )
    
    results.record(
        "Parses stage correctly",
        result.stage == ClinicalStage.PHASE_3
    )
    results.record(
        "Parses therapeutic area correctly",
        result.therapeutic_area == TherapeuticArea.ONCOLOGY
    )
    results.record(
        "Parses mechanism correctly",
        result.mechanism_class == MechanismClass.SMALL_MOLECULE
    )
    results.record(
        "Parses FDA designations",
        FDADesignation.BREAKTHROUGH_THERAPY in result.fda_designations
    )
    results.record(
        "Parses trial characteristics",
        TrialCharacteristic.BIOMARKER_SELECTED in result.trial_characteristics
    )
    
    return results


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Wake Robin PoS Model v2.0 - Test Suite")
    print("=" * 60)
    
    all_results = []
    
    all_results.append(test_enum_parsing())
    all_results.append(test_base_rate_lookup())
    all_results.append(test_modifier_calculations())
    all_results.append(test_boundary_conditions())
    all_results.append(test_decimal_precision())
    all_results.append(test_audit_trail())
    all_results.append(test_integration_helpers())
    all_results.append(test_model_validation())
    all_results.append(test_string_input_convenience())
    
    # Aggregate results
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        print("\nFailed Tests:")
        for r in all_results:
            for name, msg in r.errors:
                print(f"  - {name}: {msg}")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
