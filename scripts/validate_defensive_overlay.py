#!/usr/bin/env python3
"""
Defensive Overlay Validator

Robust daily sanity check for position weights.
Handles empty buckets, type coercion, and edge cases.

Usage:
    python scripts/validate_defensive_overlay.py results_2026-01-29.json
    python scripts/validate_defensive_overlay.py  # uses latest results_*.json
"""

import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def safe_decimal(val) -> Decimal:
    """Safely convert any value to Decimal."""
    if val is None:
        return Decimal("0")
    if isinstance(val, Decimal):
        return val
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def safe_max(vals: List, default=None):
    """Return max of list, or default if empty."""
    return max(vals) if vals else default


def safe_min(vals: List, default=None):
    """Return min of list, or default if empty."""
    return min(vals) if vals else default


def calculate_dynamic_floor(n_securities: int) -> Decimal:
    """Expected floor based on universe size."""
    if n_securities <= 50:
        return Decimal("0.01")
    elif n_securities <= 100:
        return Decimal("0.005")
    elif n_securities <= 200:
        return Decimal("0.003")
    else:
        return Decimal("0.002")


def validate_defensive_overlay(results_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate defensive overlay weights.

    Returns:
        (passed: bool, messages: List[str])
    """
    messages = []
    errors = []

    with open(results_path) as f:
        results = json.load(f)

    m5 = results.get("module_5_composite", {})
    ranked = m5.get("ranked_securities", [])
    excluded = m5.get("excluded_securities", [])

    if not ranked:
        return False, ["ERROR: No ranked securities found"]

    messages.append(f"Validating {len(ranked)} ranked + {len(excluded)} excluded securities")
    messages.append("")

    # Check if position sizing is enabled (position_weight field exists)
    has_position_weight = any("position_weight" in r for r in ranked)

    if not has_position_weight:
        messages.append("[INFO] Position sizing: DISABLED (using expected_return instead)")
        messages.append("[INFO] Expected return fields (v3.4+):")

        # Check for expected return fields
        has_er = sum(1 for r in ranked if "expected_excess_return_annual" in r)
        has_z = sum(1 for r in ranked if "score_z" in r)
        messages.append(f"  - score_z: {has_z}/{len(ranked)} securities")
        messages.append(f"  - expected_excess_return_annual: {has_er}/{len(ranked)} securities")

        # Show ER distribution
        er_values = [float(r.get("expected_excess_return_annual", 0)) for r in ranked if "expected_excess_return_annual" in r]
        if er_values:
            messages.append(f"  - ER range: {min(er_values):.4f} to {max(er_values):.4f}")

        # Skip position weight validation
        messages.append("")
        messages.append("=" * 60)
        messages.append("VALIDATION PASSED: Expected return mode (no position weights)")
        return True, messages

    # Extract weights with safe conversion
    weights = [safe_decimal(r.get("position_weight", 0)) for r in ranked]
    nonzero_weights = [w for w in weights if w > 0]

    # 1. Sum exactly 1.0
    total_weight = sum(weights)
    sum_ok = abs(total_weight - Decimal("1")) < Decimal("0.0001")
    status = "OK" if sum_ok else "FAILED"
    messages.append(f"[{status}] Weight sum: {total_weight:.6f} (target: 1.000000)")
    if not sum_ok:
        errors.append(f"Weight sum {total_weight} != 1.0")

    # 2. All excluded have weight 0
    excluded_weights = [safe_decimal(e.get("position_weight", 0)) for e in excluded]
    excluded_nonzero = [w for w in excluded_weights if w > 0]
    excl_ok = len(excluded_nonzero) == 0
    status = "OK" if excl_ok else "FAILED"
    messages.append(f"[{status}] Excluded with zero weight: {len(excluded) - len(excluded_nonzero)}/{len(excluded)}")
    if not excl_ok:
        errors.append(f"{len(excluded_nonzero)} excluded securities have non-zero weight")

    # 3. Min weight equals dynamic floor
    expected_floor = calculate_dynamic_floor(len(nonzero_weights))
    actual_min = safe_min(nonzero_weights, Decimal("0"))
    floor_ok = actual_min is not None and abs(actual_min - expected_floor) < Decimal("0.0001")
    status = "OK" if floor_ok else "WARN"
    messages.append(f"[{status}] Min weight: {actual_min:.4f} (expected floor: {expected_floor:.4f})")
    if not floor_ok and actual_min is not None:
        if actual_min < expected_floor - Decimal("0.0001"):
            errors.append(f"Min weight {actual_min} below floor {expected_floor}")

    # 4. No negative weights
    negative_weights = [w for w in weights if w < 0]
    neg_ok = len(negative_weights) == 0
    status = "OK" if neg_ok else "FAILED"
    messages.append(f"[{status}] Negative weights: {len(negative_weights)}")
    if not neg_ok:
        errors.append(f"{len(negative_weights)} negative weights found")

    # 5. Max weight plausibility (should be < 10% unless bad vol data)
    actual_max = safe_max(nonzero_weights, Decimal("0"))
    max_ok = actual_max is not None and actual_max < Decimal("0.10")
    status = "OK" if max_ok else "WARN"
    messages.append(f"[{status}] Max weight: {actual_max:.4f} (plausible if < 10%)")
    if actual_max and actual_max >= Decimal("0.10"):
        errors.append(f"Max weight {actual_max} >= 10% - check for near-zero volatility data")

    # 6. Position count
    messages.append(f"[INFO] Positions with weight: {len(nonzero_weights)}/{len(ranked)}")

    # 7. Weight distribution by quintile
    messages.append("")
    messages.append("Weight distribution by score quintile:")
    sorted_by_score = sorted(ranked, key=lambda x: safe_decimal(x.get("composite_score", 0)), reverse=True)
    quintile_size = len(sorted_by_score) // 5
    for i in range(5):
        start = i * quintile_size
        end = start + quintile_size if i < 4 else len(sorted_by_score)
        q_weights = [safe_decimal(r.get("position_weight", 0)) for r in sorted_by_score[start:end]]
        q_total = sum(q_weights) * 100
        q_avg = sum(q_weights) / len(q_weights) * 100 if q_weights else Decimal("0")
        messages.append(f"  Q{i+1} (rank {start+1:3d}-{end:3d}): {q_total:5.1f}% total, {q_avg:.3f}% avg")

    # 8. Volatility bucket distribution
    messages.append("")
    vol_buckets = {}
    for r in ranked:
        vol_adj = r.get("volatility_adjustment", {})
        bucket = vol_adj.get("vol_bucket", "unknown") if vol_adj else "unknown"
        vol_buckets[bucket] = vol_buckets.get(bucket, 0) + 1
    messages.append("Volatility bucket distribution:")
    for bucket, count in sorted(vol_buckets.items()):
        messages.append(f"  {bucket}: {count}")

    # 9. Top 5 and bottom 5 weights
    messages.append("")
    messages.append("Top 5 weights:")
    weight_ranked = sorted(ranked, key=lambda x: safe_decimal(x.get("position_weight", 0)), reverse=True)
    for r in weight_ranked[:5]:
        ticker = r.get("ticker", "???")
        score = safe_decimal(r.get("composite_score", 0))
        weight = safe_decimal(r.get("position_weight", 0)) * 100
        messages.append(f"  {ticker:6s}  score={score:5.2f}  weight={weight:.2f}%")

    messages.append("Bottom 5 weights (non-zero):")
    nonzero_ranked = [r for r in weight_ranked if safe_decimal(r.get("position_weight", 0)) > 0]
    for r in nonzero_ranked[-5:]:
        ticker = r.get("ticker", "???")
        score = safe_decimal(r.get("composite_score", 0))
        weight = safe_decimal(r.get("position_weight", 0)) * 100
        messages.append(f"  {ticker:6s}  score={score:5.2f}  weight={weight:.2f}%")

    # Summary
    messages.append("")
    messages.append("=" * 60)
    if errors:
        messages.append(f"VALIDATION FAILED: {len(errors)} error(s)")
        for e in errors:
            messages.append(f"  - {e}")
        return False, messages
    else:
        messages.append("VALIDATION PASSED: Defensive overlay functioning correctly")
        return True, messages


def find_latest_results() -> Optional[Path]:
    """Find the most recent results_*.json file."""
    results_files = list(Path(".").glob("results_*.json"))
    if not results_files:
        return None
    return max(results_files, key=lambda p: p.stat().st_mtime)


def main():
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        results_path = find_latest_results()
        if not results_path:
            print("ERROR: No results_*.json found. Specify path as argument.")
            sys.exit(1)

    if not results_path.exists():
        print(f"ERROR: File not found: {results_path}")
        sys.exit(1)

    print(f"Validating: {results_path}")
    print("=" * 60)

    passed, messages = validate_defensive_overlay(results_path)

    for msg in messages:
        print(msg)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
