"""
Tests for AACT Enum Normalization

Run with: pytest test_aact_normalize.py -v
"""

import pytest
from aact_normalize import (
    normalize_phase, normalize_status, normalize_intervention_type,
    normalize_study_type, normalize_study_row,
    is_clinical_phase, is_active_trial, is_biotech_intervention,
    CLINICAL_PHASES, ACTIVE_STATUSES, BIOTECH_INTERVENTION_TYPES,
)


class TestPhaseNormalization:
    """Test phase normalization handles both old and new AACT formats."""
    
    @pytest.mark.parametrize("raw,expected", [
        # New SCREAMING_SNAKE_CASE format
        ('PHASE1', 'Phase 1'),
        ('PHASE2', 'Phase 2'),
        ('PHASE3', 'Phase 3'),
        ('PHASE4', 'Phase 4'),
        ('PHASE1/PHASE2', 'Phase 1/Phase 2'),
        ('PHASE2/PHASE3', 'Phase 2/Phase 3'),
        ('EARLY_PHASE1', 'Early Phase 1'),
        ('NA', 'Not Applicable'),
        
        # Old Title Case format
        ('Phase 1', 'Phase 1'),
        ('Phase 2', 'Phase 2'),
        ('Phase 3', 'Phase 3'),
        ('Phase 1/Phase 2', 'Phase 1/Phase 2'),
        ('Not Applicable', 'Not Applicable'),
        ('N/A', 'Not Applicable'),
        
        # Edge cases
        (None, None),
        ('', None),
        ('  ', None),
        ('UNKNOWN_PHASE', None),  # Unknown maps to None (kill switch)
    ])
    def test_normalize_phase(self, raw, expected):
        assert normalize_phase(raw) == expected
    
    def test_clinical_phases_complete(self):
        """Verify all clinical phases are in the set."""
        assert 'Phase 1' in CLINICAL_PHASES
        assert 'Phase 2' in CLINICAL_PHASES
        assert 'Phase 3' in CLINICAL_PHASES
        assert 'Phase 1/Phase 2' in CLINICAL_PHASES
        assert 'Phase 2/Phase 3' in CLINICAL_PHASES
        # Phase 4 is post-marketing, not typically "clinical trial" for biotech alpha
        assert 'Phase 4' not in CLINICAL_PHASES
    
    @pytest.mark.parametrize("phase,expected", [
        ('Phase 2', True),
        ('Phase 3', True),
        ('Phase 4', False),
        ('Not Applicable', False),
        (None, False),
    ])
    def test_is_clinical_phase(self, phase, expected):
        assert is_clinical_phase(phase) == expected


class TestStatusNormalization:
    """Test status normalization handles both formats."""
    
    @pytest.mark.parametrize("raw,expected", [
        # New format
        ('RECRUITING', 'Recruiting'),
        ('ACTIVE_NOT_RECRUITING', 'Active, not recruiting'),
        ('COMPLETED', 'Completed'),
        ('TERMINATED', 'Terminated'),
        ('NOT_YET_RECRUITING', 'Not yet recruiting'),
        
        # Old format
        ('Recruiting', 'Recruiting'),
        ('Active, not recruiting', 'Active, not recruiting'),
        ('Completed', 'Completed'),
        
        # Edge cases
        (None, None),
        ('', None),
    ])
    def test_normalize_status(self, raw, expected):
        assert normalize_status(raw) == expected
    
    @pytest.mark.parametrize("status,expected", [
        ('Recruiting', True),
        ('Active, not recruiting', True),
        ('Not yet recruiting', True),
        ('Completed', False),
        ('Terminated', False),
        (None, False),
    ])
    def test_is_active_trial(self, status, expected):
        assert is_active_trial(status) == expected


class TestInterventionTypeNormalization:
    """Test intervention type normalization."""
    
    @pytest.mark.parametrize("raw,expected", [
        # New format
        ('DRUG', 'Drug'),
        ('BIOLOGICAL', 'Biological'),
        ('DEVICE', 'Device'),
        ('COMBINATION_PRODUCT', 'Combination Product'),
        
        # Old format
        ('Drug', 'Drug'),
        ('Biological', 'Biological'),
        
        # Edge cases
        (None, None),
        ('', None),
    ])
    def test_normalize_intervention_type(self, raw, expected):
        assert normalize_intervention_type(raw) == expected
    
    @pytest.mark.parametrize("itype,expected", [
        ('Drug', True),
        ('Biological', True),
        ('Combination Product', True),
        ('Device', False),
        ('Behavioral', False),
        (None, False),
    ])
    def test_is_biotech_intervention(self, itype, expected):
        assert is_biotech_intervention(itype) == expected


class TestStudyTypeNormalization:
    """Test study type normalization."""
    
    @pytest.mark.parametrize("raw,expected", [
        ('INTERVENTIONAL', 'Interventional'),
        ('OBSERVATIONAL', 'Observational'),
        ('Interventional', 'Interventional'),
        ('Observational', 'Observational'),
        (None, None),
    ])
    def test_normalize_study_type(self, raw, expected):
        assert normalize_study_type(raw) == expected


class TestRowNormalization:
    """Test full row normalization."""
    
    def test_normalize_study_row_new_format(self):
        """Test normalizing a row with new AACT format."""
        raw = {
            'nct_id': 'NCT00000001',
            'phase': 'PHASE2',
            'overall_status': 'RECRUITING',
            'study_type': 'INTERVENTIONAL',
            'brief_title': 'Test Study',
        }
        
        normalized = normalize_study_row(raw)
        
        # Original unchanged
        assert raw['phase'] == 'PHASE2'
        
        # Normalized values
        assert normalized['phase'] == 'Phase 2'
        assert normalized['overall_status'] == 'Recruiting'
        assert normalized['study_type'] == 'Interventional'
        
        # Pass-through fields unchanged
        assert normalized['nct_id'] == 'NCT00000001'
        assert normalized['brief_title'] == 'Test Study'
    
    def test_normalize_study_row_with_nulls(self):
        """Test row normalization handles missing fields."""
        raw = {
            'nct_id': 'NCT00000002',
            'phase': None,
            'overall_status': 'COMPLETED',
        }
        
        normalized = normalize_study_row(raw)
        
        assert normalized['phase'] is None
        assert normalized['overall_status'] == 'Completed'
        assert normalized.get('study_type') is None


class TestDeterminism:
    """Verify normalization is deterministic (same input -> same output)."""
    
    def test_phase_deterministic(self):
        """Same input always produces same output."""
        inputs = ['PHASE2', 'Phase 2', None, '', 'PHASE1/PHASE2']
        
        for _ in range(3):
            results = [normalize_phase(x) for x in inputs]
            assert results == ['Phase 2', 'Phase 2', None, None, 'Phase 1/Phase 2']
    
    def test_row_normalization_deterministic(self):
        """Full row normalization is deterministic."""
        raw = {
            'nct_id': 'NCT12345678',
            'phase': 'PHASE3',
            'overall_status': 'ACTIVE_NOT_RECRUITING',
            'study_type': 'INTERVENTIONAL',
        }
        
        result1 = normalize_study_row(raw)
        result2 = normalize_study_row(raw)
        
        assert result1 == result2
