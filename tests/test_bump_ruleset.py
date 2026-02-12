"""Tests for scripts/bump_ruleset.py — CLI parsers and end-to-end dry-run."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bump_ruleset import _parse_buckets, _parse_bool


# =============================================================================
# _parse_buckets
# =============================================================================

class TestParseBuckets:
    def test_valid_string(self):
        """Standard 3-bucket string parses to expected tuple-of-tuples."""
        result = _parse_buckets("400:1.0,1000:0.85,2000:0.70")
        assert result == ((400.0, 1.0), (1000.0, 0.85), (2000.0, 0.70))

    def test_single_bucket(self):
        """Single bucket is valid."""
        result = _parse_buckets("500:0.90")
        assert result == ((500.0, 0.90),)

    def test_whitespace_tolerance(self):
        """Spaces around commas are stripped."""
        result = _parse_buckets("100:1.0 , 200:0.80 , 300:0.60")
        assert result == ((100.0, 1.0), (200.0, 0.80), (300.0, 0.60))

    def test_missing_colon_rejected(self):
        """Token without ':' raises ArgumentTypeError."""
        from argparse import ArgumentTypeError
        with pytest.raises(ArgumentTypeError, match="threshold:mult"):
            _parse_buckets("400-1.0,1000:0.85")

    def test_non_numeric_rejected(self):
        """Non-numeric values raise ArgumentTypeError."""
        from argparse import ArgumentTypeError
        with pytest.raises(ArgumentTypeError, match="Non-numeric"):
            _parse_buckets("abc:1.0,1000:0.85")

    def test_descending_thresholds_rejected(self):
        """Descending thresholds raise ArgumentTypeError."""
        from argparse import ArgumentTypeError
        with pytest.raises(ArgumentTypeError, match="strictly ascending"):
            _parse_buckets("1000:1.0,400:0.85,2000:0.70")

    def test_mult_out_of_range_rejected(self):
        """Multiplier outside (0, 1] raises ArgumentTypeError."""
        from argparse import ArgumentTypeError
        with pytest.raises(ArgumentTypeError, match="Multiplier must be in"):
            _parse_buckets("400:1.5,1000:0.85")
        with pytest.raises(ArgumentTypeError, match="Multiplier must be in"):
            _parse_buckets("400:0.0,1000:0.85")


# =============================================================================
# _parse_bool
# =============================================================================

class TestParseBool:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes"])
    def test_truthy(self, val):
        assert _parse_bool(val) is True

    @pytest.mark.parametrize("val", ["false", "False", "FALSE", "0", "no"])
    def test_falsy(self, val):
        assert _parse_bool(val) is False

    def test_invalid_rejected(self):
        from argparse import ArgumentTypeError
        with pytest.raises(ArgumentTypeError):
            _parse_bool("maybe")


# =============================================================================
# End-to-end: --cost-haircut-buckets --dry-run
# =============================================================================

class TestBumpRulesetCLI:
    def test_cost_haircut_buckets_dry_run(self):
        """bump_ruleset.py --cost-haircut-buckets ... --dry-run produces expected output."""
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "scripts" / "bump_ruleset.py"),
                "--cost-haircut-buckets", "50:1.0,100:0.85,150:0.70",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "cost_haircut_buckets" in result.stdout
        assert "[dry-run]" in result.stdout

    def test_invalid_buckets_exits_nonzero(self):
        """bump_ruleset.py with bad bucket format exits non-zero."""
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "scripts" / "bump_ruleset.py"),
                "--cost-haircut-buckets", "bad_format",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0
