"""
Comprehensive tests for the Institutional-Grade Audit Framework.

Tests all tiers and validation logic.
"""

import json
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    AuditTier,
    ComplianceGrade,
    PassCriteria,
    TierResult,
    ValidationCategory,
    ValidationFinding,
    AuditReport,
    AuditMetrics,
)


class TestAuditTypes:
    """Test core audit type definitions."""

    def test_audit_severity_values(self):
        """Test AuditSeverity enum values."""
        assert AuditSeverity.CRITICAL.value == "critical"
        assert AuditSeverity.HIGH.value == "high"
        assert AuditSeverity.MEDIUM.value == "medium"
        assert AuditSeverity.LOW.value == "low"
        assert AuditSeverity.INFO.value == "info"

    def test_compliance_grade_values(self):
        """Test ComplianceGrade enum values."""
        assert ComplianceGrade.A.value == "A"
        assert ComplianceGrade.B.value == "B"
        assert ComplianceGrade.C.value == "C"
        assert ComplianceGrade.D.value == "D"
        assert ComplianceGrade.F.value == "F"

    def test_audit_tier_values(self):
        """Test AuditTier enum values."""
        assert len(AuditTier) == 6
        assert AuditTier.TIER_1_DETERMINISM.value == "tier_1_determinism"
        assert AuditTier.TIER_6_DEPLOYMENT.value == "tier_6_deployment"

    def test_validation_finding_creation(self):
        """Test ValidationFinding dataclass."""
        finding = ValidationFinding(
            finding_id="TEST_001",
            severity=AuditSeverity.HIGH,
            category=ValidationCategory.DETERMINISM,
            title="Test Finding",
            description="A test finding",
            location="test.py:10",
            evidence="x = float(1.5)",
            remediation="Use Decimal instead",
            compliance_impact="May cause non-determinism",
        )

        assert finding.finding_id == "TEST_001"
        assert finding.severity == AuditSeverity.HIGH
        assert finding.category == ValidationCategory.DETERMINISM

        # Test to_dict
        d = finding.to_dict()
        assert d["finding_id"] == "TEST_001"
        assert d["severity"] == "high"
        assert d["category"] == "determinism"

    def test_pass_criteria_defaults(self):
        """Test PassCriteria default values."""
        criteria = PassCriteria()

        assert criteria.determinism_runs_required == 10
        assert criteria.provenance_coverage_min == Decimal("0.95")
        assert criteria.runtime_320_tickers_max_seconds == 300
        assert criteria.unit_test_coverage_min == Decimal("0.80")

    def test_audit_result_add_finding(self):
        """Test AuditResult finding addition."""
        result = AuditResult(
            check_name="test_check",
            passed=True,
        )

        result.add_finding(
            severity=AuditSeverity.MEDIUM,
            category=ValidationCategory.DATA_QUALITY,
            title="Test",
            description="Test description",
            location="test.py",
            evidence="evidence",
            remediation="fix it",
        )

        assert len(result.findings) == 1
        assert result.findings[0].finding_id == "test_check_001"

    def test_tier_result_counts(self):
        """Test TierResult critical/high counts."""
        findings = [
            ValidationFinding(
                finding_id="T1",
                severity=AuditSeverity.CRITICAL,
                category=ValidationCategory.SECURITY,
                title="Critical",
                description="",
                location="",
                evidence="",
                remediation="",
            ),
            ValidationFinding(
                finding_id="T2",
                severity=AuditSeverity.HIGH,
                category=ValidationCategory.SECURITY,
                title="High",
                description="",
                location="",
                evidence="",
                remediation="",
            ),
            ValidationFinding(
                finding_id="T3",
                severity=AuditSeverity.HIGH,
                category=ValidationCategory.SECURITY,
                title="High 2",
                description="",
                location="",
                evidence="",
                remediation="",
            ),
            ValidationFinding(
                finding_id="T4",
                severity=AuditSeverity.MEDIUM,
                category=ValidationCategory.SECURITY,
                title="Medium",
                description="",
                location="",
                evidence="",
                remediation="",
            ),
        ]

        result = TierResult(
            tier=AuditTier.TIER_1_DETERMINISM,
            grade=ComplianceGrade.D,
            passed=False,
            findings=findings,
        )

        assert result.critical_count == 1
        assert result.high_count == 2

    def test_audit_report_properties(self):
        """Test AuditReport property calculations."""
        findings1 = [
            ValidationFinding(
                finding_id="T1",
                severity=AuditSeverity.CRITICAL,
                category=ValidationCategory.SECURITY,
                title="Critical",
                description="",
                location="",
                evidence="",
                remediation="",
            ),
        ]
        findings2 = [
            ValidationFinding(
                finding_id="T2",
                severity=AuditSeverity.HIGH,
                category=ValidationCategory.SECURITY,
                title="High",
                description="",
                location="",
                evidence="",
                remediation="",
            ),
        ]

        tier_results = [
            TierResult(
                tier=AuditTier.TIER_1_DETERMINISM,
                grade=ComplianceGrade.C,
                passed=True,
                findings=findings1,
            ),
            TierResult(
                tier=AuditTier.TIER_2_DATA_INTEGRITY,
                grade=ComplianceGrade.B,
                passed=True,
                findings=findings2,
            ),
        ]

        report = AuditReport(
            report_id="test123",
            generated_at="2026-01-24T12:00:00",
            codebase_version="1.0.0",
            audit_framework_version="1.0.0",
            overall_grade=ComplianceGrade.B,
            overall_passed=True,
            tier_results=tier_results,
        )

        assert report.total_findings == 2
        assert len(report.critical_findings) == 1
        assert len(report.high_findings) == 1

        # Test get_tier_result
        tier1 = report.get_tier_result(AuditTier.TIER_1_DETERMINISM)
        assert tier1 is not None
        assert tier1.grade == ComplianceGrade.C

    def test_audit_report_to_dict(self):
        """Test AuditReport serialization."""
        report = AuditReport(
            report_id="test123",
            generated_at="2026-01-24T12:00:00",
            codebase_version="1.0.0",
            audit_framework_version="1.0.0",
            overall_grade=ComplianceGrade.A,
            overall_passed=True,
            tier_results=[],
        )

        d = report.to_dict()
        assert d["report_id"] == "test123"
        assert d["overall_grade"] == "A"
        assert d["overall_passed"] is True

    def test_audit_report_generate_markdown(self):
        """Test markdown report generation."""
        report = AuditReport(
            report_id="test123",
            generated_at="2026-01-24T12:00:00",
            codebase_version="1.0.0",
            audit_framework_version="1.0.0",
            overall_grade=ComplianceGrade.A,
            overall_passed=True,
            tier_results=[],
            executive_summary="Test summary",
            recommendations=["Fix issues", "Deploy"],
        )

        md = report.generate_markdown()
        assert "# Institutional Audit Report" in md
        assert "test123" in md
        assert "Grade: A" in md
        assert "PASSED" in md
        assert "Test summary" in md
        assert "Fix issues" in md


class TestTier1Determinism:
    """Test Tier 1 Determinism validators."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create temporary codebase structure."""
        # Create module with float violation
        module_file = tmp_path / "module_2_financial.py"
        module_file.write_text("""
from decimal import Decimal

def calculate_score(value):
    # This has a float violation
    ratio = 1.5
    score = value * ratio
    return score

def proper_calculation(value):
    ratio = Decimal("1.5")
    return value * ratio
""")

        # Create common directory
        common_dir = tmp_path / "common"
        common_dir.mkdir()

        pit_file = common_dir / "pit_enforcement.py"
        pit_file.write_text("""
def compute_pit_cutoff(as_of_date):
    '''Compute PIT cutoff.'''
    return as_of_date
""")

        return tmp_path

    def test_decimal_compliance_validator(self, temp_codebase):
        """Test DecimalComplianceValidator."""
        from audit_framework.tier1_determinism.decimal_compliance import (
            DecimalComplianceValidator,
        )

        validator = DecimalComplianceValidator(str(temp_codebase))
        report = validator.scan_codebase()

        assert report.files_scanned > 0
        # Should find the float literal
        assert report.total_violations > 0

    def test_reproducibility_non_deterministic_scan(self, temp_codebase):
        """Test non-deterministic source detection."""
        # Create file with datetime.now()
        bad_file = temp_codebase / "bad_module.py"
        bad_file.write_text("""
from datetime import datetime

def get_data():
    current_time = datetime.now()
    return current_time
""")

        from audit_framework.tier1_determinism.reproducibility import (
            ReproducibilityValidator,
        )

        validator = ReproducibilityValidator(str(temp_codebase))
        sources = validator.scan_non_deterministic_sources()

        assert len(sources) > 0
        assert any("datetime.now" in s for s in sources)

    def test_pit_integrity_validator(self, temp_codebase):
        """Test PITIntegrityValidator."""
        from audit_framework.tier1_determinism.pit_integrity import (
            PITIntegrityValidator,
        )

        validator = PITIntegrityValidator(str(temp_codebase))
        report = validator.scan_codebase()

        assert report.files_scanned > 0


class TestTier2DataIntegrity:
    """Test Tier 2 Data Integrity validators."""

    @pytest.fixture
    def temp_codebase_with_data(self, tmp_path):
        """Create temporary codebase with data files."""
        # Create production_data directory
        data_dir = tmp_path / "production_data"
        data_dir.mkdir()

        # Create universe.json
        universe = [
            {"ticker": "ACME", "status": "active"},
            {"ticker": "BETA", "status": "active"},
        ]
        (data_dir / "universe.json").write_text(json.dumps(universe))

        # Create financial_records.json
        financial = [
            {"ticker": "ACME", "total_cash": "100000000", "source_date": "2026-01-01"},
        ]
        (data_dir / "financial_records.json").write_text(json.dumps(financial))

        return tmp_path

    def test_provenance_validator(self, temp_codebase_with_data):
        """Test ProvenanceValidator."""
        from audit_framework.tier2_data_integrity.provenance import (
            ProvenanceValidator,
        )

        validator = ProvenanceValidator(str(temp_codebase_with_data))
        report = validator.run_audit()

        # Should have analyzed records
        assert report.total_records >= 0

    def test_coverage_validator(self, temp_codebase_with_data):
        """Test CoverageValidator."""
        from audit_framework.tier2_data_integrity.coverage import (
            CoverageValidator,
        )

        validator = CoverageValidator(str(temp_codebase_with_data))
        report = validator.run_audit()

        assert report.universe_size == 2

    def test_failure_mode_validator(self, temp_codebase_with_data):
        """Test FailureModeValidator."""
        from audit_framework.tier2_data_integrity.failure_modes import (
            FailureModeValidator,
        )

        validator = FailureModeValidator(str(temp_codebase_with_data))
        report = validator.run_audit()

        assert report.tests_run > 0


class TestTier3Performance:
    """Test Tier 3 Performance validators."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create temporary codebase."""
        module_file = tmp_path / "module.py"
        module_file.write_text("""
import logging

logger = logging.getLogger(__name__)

def process():
    try:
        result = compute()
    except ValueError as e:
        logger.error(f"Error: {e}")
        raise
    return result

def compute():
    return 42
""")
        return tmp_path

    def test_performance_validator(self, temp_codebase):
        """Test PerformanceValidator."""
        from audit_framework.tier3_performance.profiling import (
            PerformanceValidator,
        )

        validator = PerformanceValidator(str(temp_codebase))
        report = validator.run_audit()

        assert len(report.benchmarks) > 0

    def test_resilience_validator(self, temp_codebase):
        """Test ResilienceValidator."""
        from audit_framework.tier3_performance.resilience import (
            ResilienceValidator,
        )

        validator = ResilienceValidator(str(temp_codebase))
        report = validator.run_audit()

        assert report.resilience_score >= 0

    def test_dependency_validator(self, temp_codebase):
        """Test DependencyValidator."""
        from audit_framework.tier3_performance.dependencies import (
            DependencyValidator,
        )

        validator = DependencyValidator(str(temp_codebase))
        report = validator.run_audit()

        assert report.local_modules is not None


class TestTier4Testing:
    """Test Tier 4 Testing validators."""

    @pytest.fixture
    def temp_codebase_with_tests(self, tmp_path):
        """Create temporary codebase with tests."""
        # Create tests directory
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file = tests_dir / "test_module.py"
        test_file.write_text("""
import pytest

def test_one():
    assert True

def test_two():
    assert 1 + 1 == 2

def test_determinism_check():
    # Golden test for determinism
    assert True
""")

        # Create backtest directory
        backtest_dir = tmp_path / "backtest"
        backtest_dir.mkdir()

        metrics_file = backtest_dir / "metrics.py"
        metrics_file.write_text("""
def calculate_sharpe_ratio(returns):
    '''Calculate Sharpe ratio.'''
    return 1.5

def calculate_ic(predictions, actuals):
    '''Calculate information coefficient.'''
    return 0.05
""")

        return tmp_path

    def test_test_coverage_validator(self, temp_codebase_with_tests):
        """Test TestCoverageValidator."""
        from audit_framework.tier4_testing.coverage import (
            TestCoverageValidator,
        )

        validator = TestCoverageValidator(str(temp_codebase_with_tests))
        report = validator.run_audit()

        assert report.total_tests >= 3
        assert report.has_golden_tests  # Has "determinism" in test name

    def test_backtest_validator(self, temp_codebase_with_tests):
        """Test BacktestValidator."""
        from audit_framework.tier4_testing.backtesting import (
            BacktestValidator,
        )

        validator = BacktestValidator(str(temp_codebase_with_tests))
        report = validator.run_audit()

        assert report.readiness_score > 0


class TestTier5Architecture:
    """Test Tier 5 Architecture validators."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create temporary codebase."""
        # Create a module
        module_file = tmp_path / "module.py"
        module_file.write_text('''
"""Module docstring."""

def simple_function():
    """Simple function."""
    return 42

def complex_function(x, y, z):
    """Complex function with nesting."""
    result = 0
    if x > 0:
        if y > 0:
            if z > 0:
                result = x + y + z
            else:
                result = x + y
        else:
            result = x
    return result
''')

        # Create governance directory
        gov_dir = tmp_path / "governance"
        gov_dir.mkdir()

        audit_log = gov_dir / "audit_log.py"
        audit_log.write_text("""
class AuditLog:
    '''Audit logging.'''
    pass
""")

        # Create README
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\nInstallation instructions.\n")

        return tmp_path

    def test_maintainability_validator(self, temp_codebase):
        """Test MaintainabilityValidator."""
        from audit_framework.tier5_architecture.maintainability import (
            MaintainabilityValidator,
        )

        validator = MaintainabilityValidator(str(temp_codebase))
        report = validator.run_audit()

        assert report.total_lines > 0
        assert report.maintainability_score >= 0

    def test_security_validator(self, temp_codebase):
        """Test SecurityValidator."""
        from audit_framework.tier5_architecture.security import (
            SecurityValidator,
        )

        validator = SecurityValidator(str(temp_codebase))
        report = validator.run_audit()

        assert report.security_score >= 0


class TestTier6Deployment:
    """Test Tier 6 Deployment validators."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create temporary codebase."""
        # Create pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[project]
name = "test-project"
version = "1.0.0"
requires-python = ">=3.10"
''')

        # Create README
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\nSetup and installation.\n")

        # Create governance
        gov_dir = tmp_path / "governance"
        gov_dir.mkdir()
        (gov_dir / "audit_log.py").write_text("class AuditLog: pass")
        (gov_dir / "schema_registry.py").write_text("SCHEMA_VERSION = '1.0'")

        # Create production data
        data_dir = tmp_path / "production_data"
        data_dir.mkdir()

        # Create state management
        (tmp_path / "state_management.py").write_text("class StateManager: pass")

        # Create run_screen
        (tmp_path / "run_screen.py").write_text("def main(): pass")

        # Create tests
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test(): pass")

        # Create .git
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        return tmp_path

    def test_deployment_validator(self, temp_codebase):
        """Test DeploymentValidator."""
        from audit_framework.tier6_deployment.readiness import (
            DeploymentValidator,
        )

        validator = DeploymentValidator(str(temp_codebase))
        report = validator.run_audit()

        assert report.items_total > 0
        assert report.readiness_score >= 0


class TestOrchestrator:
    """Test the Audit Orchestrator."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create minimal codebase for orchestrator testing."""
        # Create basic structure
        (tmp_path / "common").mkdir()
        (tmp_path / "governance").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "production_data").mkdir()

        # Minimal files
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "1.0.0"'
        )
        (tmp_path / "README.md").write_text("# Test\n\nSetup instructions.\n")
        (tmp_path / "governance" / "audit_log.py").write_text(
            "class AuditLog: pass"
        )
        (tmp_path / "tests" / "test_main.py").write_text(
            "def test_one(): pass"
        )
        (tmp_path / "production_data" / "universe.json").write_text("[]")

        return tmp_path

    def test_orchestrator_single_tier(self, temp_codebase):
        """Test running single tier."""
        from audit_framework.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator(str(temp_codebase))
        result = orchestrator.run_tier(AuditTier.TIER_6_DEPLOYMENT)

        assert result.tier == AuditTier.TIER_6_DEPLOYMENT
        assert result.grade is not None

    def test_orchestrator_full_audit(self, temp_codebase):
        """Test running full audit."""
        from audit_framework.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator(str(temp_codebase))
        report = orchestrator.run_full_audit()

        assert report.report_id is not None
        assert len(report.tier_results) == 6
        assert report.overall_grade is not None

    def test_orchestrator_save_json_report(self, temp_codebase):
        """Test saving JSON report."""
        from audit_framework.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator(str(temp_codebase))
        report = orchestrator.run_full_audit()

        output_path = orchestrator.save_report(report, format="json")
        assert Path(output_path).exists()

        with open(output_path) as f:
            data = json.load(f)
        assert data["report_id"] == report.report_id

    def test_orchestrator_save_markdown_report(self, temp_codebase):
        """Test saving Markdown report."""
        from audit_framework.orchestrator import AuditOrchestrator

        orchestrator = AuditOrchestrator(str(temp_codebase))
        report = orchestrator.run_full_audit()

        output_path = orchestrator.save_report(report, format="markdown")
        assert Path(output_path).exists()

        content = Path(output_path).read_text()
        assert "# Institutional Audit Report" in content


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def temp_codebase(self, tmp_path):
        """Create minimal codebase."""
        (tmp_path / "common").mkdir()
        (tmp_path / "governance").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "production_data").mkdir()
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "1.0.0"'
        )
        return tmp_path

    def test_run_full_audit_function(self, temp_codebase):
        """Test run_full_audit convenience function."""
        from audit_framework.orchestrator import run_full_audit

        report = run_full_audit(str(temp_codebase))
        assert report is not None
        assert report.overall_grade is not None

    def test_run_tier_audit_function(self, temp_codebase):
        """Test run_tier_audit convenience function."""
        from audit_framework.orchestrator import run_tier_audit

        result = run_tier_audit(
            str(temp_codebase),
            AuditTier.TIER_1_DETERMINISM,
        )
        assert result is not None
        assert result.tier == AuditTier.TIER_1_DETERMINISM
