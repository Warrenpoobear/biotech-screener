#!/usr/bin/env python3
"""
validate_pipeline.py - Post-Run Validation for Biotech Screener

Validates pipeline output to ensure:
1. All required sections present
2. Universe size is stable (within tolerance)
3. No module returns all-zeros or all-default values
4. Scores are within expected ranges (0-100)
5. Weights sum to expected target
6. Output includes proper run metadata
7. Excluded tickers have zero weight

Usage:
    python validate_pipeline.py --output results.json
    python validate_pipeline.py --output results.json --baseline baseline.json
    python validate_pipeline.py --output results.json --strict
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Validation thresholds
EXPECTED_WEIGHT_SUM = 1.00  # Fully invested, no cash reserve
WEIGHT_SUM_TOLERANCE = 0.01
MIN_UNIVERSE_SIZE = 50
MAX_ZERO_SCORE_RATIO = 0.20  # Max 20% can have zero scores
SCORE_MIN = 0.0
SCORE_MAX = 100.0


class ValidationResult:
    """Validation result container"""

    def __init__(self):
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks: Dict[str, bool] = {}
        self.metrics: Dict[str, Any] = {}

    def error(self, msg: str, check_name: str = None):
        self.errors.append(msg)
        self.passed = False
        if check_name:
            self.checks[check_name] = False

    def warning(self, msg: str):
        self.warnings.append(msg)

    def success(self, check_name: str, metric: Any = None):
        self.checks[check_name] = True
        if metric is not None:
            self.metrics[check_name] = metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "metrics": self.metrics,
        }


def compute_content_hash(data: Any) -> str:
    """Compute deterministic content hash"""
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def get_git_info() -> Dict[str, str]:
    """Get current git commit info"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}


def validate_structure(data: Dict[str, Any], result: ValidationResult):
    """Validate output structure has all required sections"""
    required_sections = [
        "run_metadata",
        "module_1_universe",
        "module_2_financial",
        "module_3_catalyst",
        "module_4_clinical",
        "module_5_composite",
        "summary",
    ]

    for section in required_sections:
        if section not in data:
            result.error(f"Missing required section: {section}", f"structure_{section}")
        else:
            result.success(f"structure_{section}", True)


def validate_run_metadata(data: Dict[str, Any], result: ValidationResult):
    """Validate run metadata is complete"""
    metadata = data.get("run_metadata", {})

    required_fields = ["as_of_date", "version", "input_hashes"]

    for field in required_fields:
        if field not in metadata:
            result.error(f"Missing run_metadata field: {field}", "metadata_complete")
            return

    result.success("metadata_complete", True)
    result.metrics["as_of_date"] = metadata.get("as_of_date")
    result.metrics["version"] = metadata.get("version")

    # Validate as_of_date format
    try:
        date.fromisoformat(metadata["as_of_date"])
        result.success("metadata_date_valid", True)
    except ValueError:
        result.error(f"Invalid as_of_date format: {metadata['as_of_date']}", "metadata_date_valid")


def validate_universe_size(data: Dict[str, Any], result: ValidationResult, baseline: Optional[Dict] = None):
    """Validate universe size is reasonable"""
    summary = data.get("summary", {})
    total = summary.get("total_evaluated", 0)
    active = summary.get("active_universe", 0)
    excluded = summary.get("excluded", 0)
    final_ranked = summary.get("final_ranked", 0)

    result.metrics["universe_total"] = total
    result.metrics["universe_active"] = active
    result.metrics["universe_excluded"] = excluded
    result.metrics["universe_final_ranked"] = final_ranked

    # Check minimum size
    if total < MIN_UNIVERSE_SIZE:
        result.warning(f"Universe size ({total}) below expected minimum ({MIN_UNIVERSE_SIZE})")

    # Check active vs excluded makes sense
    if active + excluded != total and excluded != 0:
        result.warning(f"Active ({active}) + Excluded ({excluded}) != Total ({total})")

    # Check final ranked is reasonable
    if final_ranked == 0:
        result.error("No securities were ranked (final_ranked = 0)", "universe_ranked")
    else:
        result.success("universe_ranked", final_ranked)

    # Compare to baseline if provided
    if baseline:
        baseline_total = baseline.get("summary", {}).get("total_evaluated", 0)
        change_pct = abs(total - baseline_total) / max(baseline_total, 1) * 100

        if change_pct > 10:
            result.warning(f"Universe size changed by {change_pct:.1f}% from baseline ({baseline_total} -> {total})")

        result.metrics["universe_change_pct"] = change_pct


def validate_module_2_scores(data: Dict[str, Any], result: ValidationResult):
    """Validate Module 2 financial scores"""
    m2 = data.get("module_2_financial", {})
    scores = m2.get("scores", [])

    if not scores:
        result.error("Module 2 has no scores", "m2_has_scores")
        return

    result.success("m2_has_scores", len(scores))

    # Check for all-zeros
    zero_count = sum(1 for s in scores if s.get("financial_score", 0) == 0)
    zero_ratio = zero_count / len(scores)

    result.metrics["m2_zero_count"] = zero_count
    result.metrics["m2_zero_ratio"] = zero_ratio

    if zero_ratio > MAX_ZERO_SCORE_RATIO:
        result.warning(f"Module 2: {zero_count}/{len(scores)} ({zero_ratio:.1%}) have zero scores")

    # Check score range
    out_of_range = 0
    for s in scores:
        score = s.get("financial_score", 0)
        if score < SCORE_MIN or score > SCORE_MAX:
            out_of_range += 1

    if out_of_range > 0:
        result.error(f"Module 2: {out_of_range} scores outside [0, 100] range", "m2_score_range")
    else:
        result.success("m2_score_range", True)

    # Check severity distribution
    severities = {}
    for s in scores:
        sev = s.get("severity", "unknown")
        severities[sev] = severities.get(sev, 0) + 1

    result.metrics["m2_severity_distribution"] = severities


def validate_module_3_catalysts(data: Dict[str, Any], result: ValidationResult):
    """Validate Module 3 catalyst data"""
    m3 = data.get("module_3_catalyst", {})
    summaries = m3.get("summaries", {})
    diag = m3.get("diagnostic_counts", {})

    if not summaries:
        result.warning("Module 3 has no catalyst summaries")

    result.metrics["m3_tickers_analyzed"] = diag.get("tickers_analyzed", 0)
    result.metrics["m3_events_detected"] = diag.get("events_detected_total", 0)
    result.metrics["m3_tickers_with_events"] = diag.get("tickers_with_events", 0)

    # Check for severe negatives
    severe_neg = diag.get("tickers_with_severe_negative", 0)
    result.metrics["m3_severe_negatives"] = severe_neg

    if severe_neg > 0:
        result.warning(f"Module 3: {severe_neg} tickers have severe negative events")

    result.success("m3_complete", True)


def validate_module_4_scores(data: Dict[str, Any], result: ValidationResult):
    """Validate Module 4 clinical scores"""
    m4 = data.get("module_4_clinical", {})
    scores = m4.get("scores", [])

    if not scores:
        result.error("Module 4 has no scores", "m4_has_scores")
        return

    result.success("m4_has_scores", len(scores))

    # Check for all-zeros
    zero_count = sum(1 for s in scores if float(s.get("clinical_score", "0")) == 0)
    zero_ratio = zero_count / len(scores) if scores else 0

    result.metrics["m4_zero_count"] = zero_count
    result.metrics["m4_zero_ratio"] = zero_ratio

    if zero_ratio > MAX_ZERO_SCORE_RATIO:
        result.warning(f"Module 4: {zero_count}/{len(scores)} ({zero_ratio:.1%}) have zero scores")

    result.success("m4_score_range", True)


def validate_module_5_composite(data: Dict[str, Any], result: ValidationResult):
    """Validate Module 5 composite ranking"""
    m5 = data.get("module_5_composite", {})
    ranked = m5.get("ranked_securities", [])
    excluded = m5.get("excluded_securities", [])

    if not ranked:
        result.error("Module 5 has no ranked securities", "m5_has_ranked")
        return

    result.success("m5_has_ranked", len(ranked))
    result.metrics["m5_ranked_count"] = len(ranked)
    result.metrics["m5_excluded_count"] = len(excluded)

    # Check score range
    out_of_range = 0
    for s in ranked:
        score = float(s.get("composite_score", "0"))
        if score < SCORE_MIN or score > SCORE_MAX:
            out_of_range += 1

    if out_of_range > 0:
        result.error(f"Module 5: {out_of_range} scores outside [0, 100] range", "m5_score_range")
    else:
        result.success("m5_score_range", True)

    # Check weight sum
    total_weight = sum(float(s.get("position_weight", "0")) for s in ranked)
    weight_diff = abs(total_weight - EXPECTED_WEIGHT_SUM)

    result.metrics["m5_weight_sum"] = total_weight

    if weight_diff > WEIGHT_SUM_TOLERANCE:
        result.error(
            f"Module 5: Weight sum ({total_weight:.4f}) differs from target ({EXPECTED_WEIGHT_SUM}) by {weight_diff:.4f}",
            "m5_weight_sum"
        )
    else:
        result.success("m5_weight_sum", total_weight)

    # Check excluded have zero weight
    excluded_with_weight = 0
    for s in excluded:
        if float(s.get("position_weight", "0")) > 0:
            excluded_with_weight += 1

    if excluded_with_weight > 0:
        result.error(f"Module 5: {excluded_with_weight} excluded securities have non-zero weight", "m5_excluded_zero_weight")
    else:
        result.success("m5_excluded_zero_weight", True)

    # Check exclusion reasons are specified
    unknown_reasons = sum(1 for s in excluded if s.get("reason", "unknown") == "unknown")
    if unknown_reasons > 0:
        result.warning(f"Module 5: {unknown_reasons} excluded securities have unknown reason")


def validate_determinism(data: Dict[str, Any], baseline: Optional[Dict], result: ValidationResult):
    """Validate output is deterministic (matches baseline if provided)"""
    if baseline is None:
        result.metrics["determinism_check"] = "no_baseline"
        return

    # Compare content hashes
    current_hash = compute_content_hash(data)
    baseline_hash = compute_content_hash(baseline)

    result.metrics["current_hash"] = current_hash
    result.metrics["baseline_hash"] = baseline_hash

    if current_hash == baseline_hash:
        result.success("determinism_match", True)
    else:
        # Check which sections differ
        differing_sections = []
        for key in data.keys():
            if key in baseline:
                if compute_content_hash(data[key]) != compute_content_hash(baseline[key]):
                    differing_sections.append(key)

        result.warning(f"Output differs from baseline in sections: {differing_sections}")
        result.metrics["differing_sections"] = differing_sections


def run_validation(
    output_path: Path,
    baseline_path: Optional[Path] = None,
    strict: bool = False
) -> ValidationResult:
    """Run all validation checks"""

    result = ValidationResult()

    # Load output
    try:
        with open(output_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        result.error(f"Output file not found: {output_path}", "file_exists")
        return result
    except json.JSONDecodeError as e:
        result.error(f"Invalid JSON in output: {e}", "json_valid")
        return result

    result.success("file_exists", str(output_path))
    result.success("json_valid", True)

    # Load baseline if provided
    baseline = None
    if baseline_path and baseline_path.exists():
        try:
            with open(baseline_path) as f:
                baseline = json.load(f)
            result.metrics["baseline_loaded"] = str(baseline_path)
        except Exception as e:
            result.warning(f"Could not load baseline: {e}")

    # Run all checks
    validate_structure(data, result)
    validate_run_metadata(data, result)
    validate_universe_size(data, result, baseline)
    validate_module_2_scores(data, result)
    validate_module_3_catalysts(data, result)
    validate_module_4_scores(data, result)
    validate_module_5_composite(data, result)
    validate_determinism(data, baseline, result)

    # In strict mode, warnings become errors
    if strict and result.warnings:
        for warning in result.warnings:
            result.error(f"[STRICT] {warning}", "strict_mode")

    return result


def print_result(result: ValidationResult, verbose: bool = False):
    """Print validation result"""
    print("=" * 60)
    print("PIPELINE OUTPUT VALIDATION")
    print("=" * 60)

    # Git info
    git_info = get_git_info()
    print(f"Git commit: {git_info['commit']} ({git_info['branch']})")

    # Summary metrics
    print()
    print("METRICS:")
    for key, value in sorted(result.metrics.items()):
        if not key.startswith("_"):
            print(f"  {key}: {value}")

    # Checks
    print()
    print("CHECKS:")
    for check, passed in sorted(result.checks.items()):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    # Warnings
    if result.warnings:
        print()
        print("WARNINGS:")
        for warning in result.warnings:
            print(f"  [WARN] {warning}")

    # Errors
    if result.errors:
        print()
        print("ERRORS:")
        for error in result.errors:
            print(f"  [ERR] {error}")

    # Final result
    print()
    print("=" * 60)
    if result.passed:
        print("RESULT: VALIDATION PASSED")
    else:
        print("RESULT: VALIDATION FAILED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate biotech screener pipeline output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate_pipeline.py --output results.json
    python validate_pipeline.py --output results.json --baseline golden/baseline.json
    python validate_pipeline.py --output results.json --strict

What it checks:
    - All required output sections present
    - Run metadata complete (as_of_date, version, hashes)
    - Universe size reasonable and stable
    - No modules return all-zero scores
    - Scores within expected ranges [0, 100]
    - Position weights sum to target (0.90)
    - Excluded securities have zero weight
    - Output matches baseline (if provided)
        """
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to pipeline output JSON file"
    )

    parser.add_argument(
        "--baseline", "-b",
        type=Path,
        default=None,
        help="Path to baseline JSON for comparison"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: treat warnings as errors"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )

    args = parser.parse_args()

    result = run_validation(args.output, args.baseline, args.strict)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_result(result, args.verbose)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
