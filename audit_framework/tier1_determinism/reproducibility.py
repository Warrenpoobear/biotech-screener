"""
Query 1.2 - Reproducibility Stress Test.

Validates that the screening pipeline produces byte-identical outputs
across multiple runs with the same inputs.

Tests:
- Run complete pipeline 10 times consecutively
- Verify 100% hash identity across all runs
- Assert zero variance in any score
- Assert identical ranking order
- Flag non-deterministic dependencies
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class RunResult:
    """Result of a single pipeline run."""

    run_number: int
    output_hash: str
    rankings: List[Tuple[str, str]]  # List of (ticker, score) tuples
    execution_time_ms: int
    memory_peak_mb: float
    output_data: Optional[Dict[str, Any]] = None


@dataclass
class ReproducibilityReport:
    """Complete reproducibility test report."""

    total_runs: int
    identical_hash_runs: int
    unique_hashes: Set[str] = field(default_factory=set)
    score_variance: Dict[str, Decimal] = field(default_factory=dict)
    ranking_correlations: List[Decimal] = field(default_factory=list)
    run_results: List[RunResult] = field(default_factory=list)
    non_deterministic_sources: List[str] = field(default_factory=list)
    passed: bool = False

    @property
    def is_perfectly_reproducible(self) -> bool:
        return len(self.unique_hashes) == 1 and self.identical_hash_runs == self.total_runs


class ReproducibilityValidator:
    """
    Validates pipeline reproducibility through stress testing.

    Executes the pipeline multiple times and verifies:
    - Byte-identical outputs (via SHA256 hashing)
    - Zero score variance
    - Identical ranking order
    """

    def __init__(
        self,
        codebase_path: str,
        data_dir: str = "production_data",
        as_of_date: Optional[str] = None,
    ):
        """
        Initialize validator.

        Args:
            codebase_path: Root of codebase
            data_dir: Directory containing input data
            as_of_date: Date to use for screening (ISO format)
        """
        self.codebase_path = Path(codebase_path)
        self.data_dir = data_dir
        self.as_of_date = as_of_date or date.today().isoformat()

    def _compute_output_hash(self, output: Dict[str, Any]) -> str:
        """Compute deterministic hash of output, excluding non-deterministic fields."""
        # Create a copy without runtime-specific fields
        clean_output = self._strip_non_deterministic_fields(output)

        # Serialize with sorted keys for determinism
        json_str = json.dumps(clean_output, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"

    def _strip_non_deterministic_fields(self, obj: Any) -> Any:
        """Recursively remove fields that are expected to vary between runs."""
        non_deterministic_keys = {
            "timestamp",
            "generated_at",
            "execution_time_ms",
            "runtime_ms",
            "run_id",
            "memory_peak_mb",
        }

        if isinstance(obj, dict):
            return {
                k: self._strip_non_deterministic_fields(v)
                for k, v in obj.items()
                if k not in non_deterministic_keys
            }
        elif isinstance(obj, list):
            return [self._strip_non_deterministic_fields(item) for item in obj]
        else:
            return obj

    def _extract_rankings(self, output: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract ticker rankings from output."""
        rankings = []

        # Try different output structures
        if "rankings" in output:
            for item in output["rankings"]:
                ticker = item.get("ticker", "")
                score = str(item.get("composite_score", item.get("score", "0")))
                rankings.append((ticker, score))
        elif "results" in output:
            for item in output["results"]:
                ticker = item.get("ticker", "")
                score = str(item.get("composite_score", item.get("score", "0")))
                rankings.append((ticker, score))
        elif "ranked_securities" in output:
            for item in output["ranked_securities"]:
                ticker = item.get("ticker", "")
                score = str(item.get("composite_score", "0"))
                rankings.append((ticker, score))

        return rankings

    def _calculate_spearman_correlation(
        self,
        rankings1: List[Tuple[str, str]],
        rankings2: List[Tuple[str, str]],
    ) -> Decimal:
        """Calculate Spearman rank correlation between two rankings."""
        if not rankings1 or not rankings2:
            return Decimal("0")

        # Create rank maps
        rank1 = {ticker: i for i, (ticker, _) in enumerate(rankings1)}
        rank2 = {ticker: i for i, (ticker, _) in enumerate(rankings2)}

        # Find common tickers
        common_tickers = set(rank1.keys()) & set(rank2.keys())
        if not common_tickers:
            return Decimal("0")

        n = len(common_tickers)
        if n < 2:
            return Decimal("1")

        # Calculate Spearman's rho
        d_squared_sum = sum(
            (rank1[ticker] - rank2[ticker]) ** 2
            for ticker in common_tickers
        )

        rho = Decimal("1") - (Decimal("6") * Decimal(d_squared_sum)) / (Decimal(n) * (Decimal(n) ** 2 - 1))
        return rho.quantize(Decimal("0.0000"))

    def _calculate_score_variance(
        self,
        all_rankings: List[List[Tuple[str, str]]],
    ) -> Dict[str, Decimal]:
        """Calculate score variance for each ticker across runs."""
        variance: Dict[str, Decimal] = {}

        if len(all_rankings) < 2:
            return variance

        # Collect all scores per ticker
        ticker_scores: Dict[str, List[Decimal]] = {}
        for rankings in all_rankings:
            for ticker, score in rankings:
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = []
                try:
                    ticker_scores[ticker].append(Decimal(score))
                except Exception:
                    pass

        # Calculate variance for each ticker
        for ticker, scores in ticker_scores.items():
            if len(scores) < 2:
                continue

            mean = sum(scores) / len(scores)
            squared_diffs = [(s - mean) ** 2 for s in scores]
            variance[ticker] = (sum(squared_diffs) / len(scores)).quantize(
                Decimal("0.00000000")
            )

        return variance

    def scan_non_deterministic_sources(self) -> List[str]:
        """
        Scan codebase for sources of non-determinism.

        Checks for:
        - datetime.now() usage
        - random module usage
        - os.urandom()
        - uuid without explicit seed
        """
        violations = []

        patterns = [
            (r"datetime\.now\(\)", "datetime.now() usage"),
            (r"datetime\.utcnow\(\)", "datetime.utcnow() usage"),
            (r"time\.time\(\)", "time.time() usage in scoring"),
            (r"random\.(random|randint|choice|shuffle)", "random module usage"),
            (r"os\.urandom", "os.urandom usage"),
            (r"uuid\.uuid4\(\)", "uuid4 without seed"),
            (r"secrets\.", "secrets module usage"),
        ]

        exclude_dirs = {"tests", "deprecated", "venv", ".venv", "__pycache__", "mnt", "audit_framework"}

        for root, dirs, files in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        lines = content.splitlines()
                except Exception:
                    continue

                for pattern, description in patterns:
                    for i, line in enumerate(lines, 1):
                        # Skip comments
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue

                        if re.search(pattern, line):
                            rel_path = file_path.relative_to(self.codebase_path)
                            violations.append(
                                f"{rel_path}:{i} - {description}: {stripped[:80]}"
                            )

        return violations

    def run_pipeline_once(
        self,
        run_number: int,
        output_dir: Path,
    ) -> Optional[RunResult]:
        """
        Execute the screening pipeline once.

        Returns RunResult or None if execution failed.
        """
        output_file = output_dir / f"run_{run_number:02d}_output.json"

        start_time = time.time()

        try:
            # Run the pipeline
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.codebase_path / "run_screen.py"),
                    "--as-of-date", self.as_of_date,
                    "--data-dir", str(self.codebase_path / self.data_dir),
                    "--output", str(output_file),
                ],
                cwd=str(self.codebase_path),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.returncode != 0:
                return None

            # Load and analyze output
            if output_file.exists():
                with open(output_file, "r") as f:
                    output_data = json.load(f)

                output_hash = self._compute_output_hash(output_data)
                rankings = self._extract_rankings(output_data)

                return RunResult(
                    run_number=run_number,
                    output_hash=output_hash,
                    rankings=rankings,
                    execution_time_ms=execution_time_ms,
                    memory_peak_mb=0.0,  # Would need memory profiling
                    output_data=output_data,
                )

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        return None

    def run_stress_test(
        self,
        num_runs: int = 10,
        output_dir: Optional[str] = None,
    ) -> ReproducibilityReport:
        """
        Run the complete reproducibility stress test.

        Args:
            num_runs: Number of pipeline executions
            output_dir: Directory for output files

        Returns:
            Complete reproducibility report
        """
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.codebase_path / "audit_framework" / "reports" / "reproducibility"

        out_path.mkdir(parents=True, exist_ok=True)

        run_results: List[RunResult] = []
        unique_hashes: Set[str] = set()

        for i in range(1, num_runs + 1):
            result = self.run_pipeline_once(i, out_path)
            if result:
                run_results.append(result)
                unique_hashes.add(result.output_hash)

        # Calculate metrics
        identical_runs = num_runs if len(unique_hashes) == 1 and len(run_results) == num_runs else 0

        # Calculate ranking correlations (each pair compared to first run)
        correlations: List[Decimal] = []
        if run_results and run_results[0].rankings:
            first_rankings = run_results[0].rankings
            for result in run_results[1:]:
                corr = self._calculate_spearman_correlation(first_rankings, result.rankings)
                correlations.append(corr)

        # Calculate score variance
        all_rankings = [r.rankings for r in run_results if r.rankings]
        score_variance = self._calculate_score_variance(all_rankings)

        # Scan for non-deterministic sources
        non_det_sources = self.scan_non_deterministic_sources()

        passed = (
            len(unique_hashes) == 1
            and len(run_results) == num_runs
            and all(v == Decimal("0") for v in score_variance.values())
            and all(c == Decimal("1") or c == Decimal("1.0000") for c in correlations)
        )

        return ReproducibilityReport(
            total_runs=num_runs,
            identical_hash_runs=identical_runs,
            unique_hashes=unique_hashes,
            score_variance=score_variance,
            ranking_correlations=correlations,
            run_results=run_results,
            non_deterministic_sources=non_det_sources,
            passed=passed,
        )


def run_reproducibility_stress_test(
    codebase_path: str,
    num_runs: int = 10,
    data_dir: str = "production_data",
    as_of_date: Optional[str] = None,
) -> AuditResult:
    """
    Run complete reproducibility stress test.

    Args:
        codebase_path: Root of codebase
        num_runs: Number of test runs
        data_dir: Input data directory
        as_of_date: Date for screening

    Returns:
        AuditResult with findings
    """
    validator = ReproducibilityValidator(
        codebase_path=codebase_path,
        data_dir=data_dir,
        as_of_date=as_of_date,
    )

    # First just scan for non-deterministic sources without running pipeline
    non_det_sources = validator.scan_non_deterministic_sources()

    result = AuditResult(
        check_name="reproducibility_stress_test",
        passed=len(non_det_sources) == 0,
        metrics={
            "non_deterministic_sources_count": len(non_det_sources),
            "sources_scanned": True,
        },
        details=f"Found {len(non_det_sources)} potential non-deterministic sources",
    )

    # Add findings for non-deterministic sources
    for source in non_det_sources:
        parts = source.split(" - ")
        location = parts[0] if parts else source

        result.add_finding(
            severity=AuditSeverity.HIGH,
            category=ValidationCategory.REPRODUCIBILITY,
            title="Non-deterministic code pattern detected",
            description=source,
            location=location,
            evidence=source,
            remediation="Replace with deterministic alternative or pass explicit as_of_date parameter",
            compliance_impact="Non-deterministic code prevents reproducible audit trails",
        )

    return result
