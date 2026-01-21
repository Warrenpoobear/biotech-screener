#!/usr/bin/env python3
"""
Module 5 Diagnostics (v3) - Momentum breakdown and health metrics.

Extracted from module_5_composite_v3.py for maintainability.

This module provides:
- compute_momentum_breakdown(): Single source of truth for momentum stats
- build_momentum_health(): Package breakdown for JSON output

Design invariant enforced here:
    applied = neg + pos + neutral = total_rankable - missing - low_conf

This prevents the "applied:0 but breakdown sums to 44" class of bug.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from src.modules.ic_enhancements import _to_decimal


def compute_momentum_breakdown(
    ranked_securities: List[Dict[str, Any]],
    diagnostic_counts: Dict[str, Any],
    total_rankable: int,
) -> Dict[str, Any]:
    """
    Compute momentum breakdown stats from diagnostic counts and securities.

    This is the SINGLE SOURCE OF TRUTH for momentum metrics. All consumers
    (guardrail check, log formatting, momentum_health JSON) use this output.

    Args:
        ranked_securities: List of scored security dicts with momentum_signal
        diagnostic_counts: Dict with momentum_* counters from build_diagnostic_counts
        total_rankable: Total number of rankable securities (passed explicitly)

    Returns:
        Dict with three groups of fields:

        A) Raw counters (from diagnostic_counts):
           - missing, low_conf, neg, pos
           - windows_used: {20d, 60d, 120d}
           - computable, meaningful, strong_signal, strong_and_effective
           - sources: {prices, 13f}

        B) Derived invariants (computed here, single definition):
           - total_rankable
           - applied (= neg + pos + neutral = total_rankable - missing - low_conf)
           - neutral (= applied - neg - pos, clamped >= 0)
           - coverage_pct (= applied / total_rankable * 100)

        C) Aggregates requiring iteration:
           - avg_weight (Decimal, average momentum weight across active securities)

    Invariant enforced:
        applied = neg + pos + neutral = total_rankable - missing - low_conf
    """
    # A) Raw counters from diagnostic_counts
    missing = diagnostic_counts.get("momentum_missing_prices", 0)
    low_conf = diagnostic_counts.get("momentum_computed_low_conf", 0)
    neg = diagnostic_counts.get("momentum_applied_negative", 0)
    pos = diagnostic_counts.get("momentum_applied_positive", 0)

    windows_used = {
        "20d": diagnostic_counts.get("momentum_window_20d", 0),
        "60d": diagnostic_counts.get("momentum_window_60d", 0),
        "120d": diagnostic_counts.get("momentum_window_120d", 0),
    }

    computable = diagnostic_counts.get("momentum_computable", 0)
    meaningful = diagnostic_counts.get("momentum_meaningful", 0)
    strong_signal = diagnostic_counts.get("momentum_strong_signal", 0)
    strong_and_effective = diagnostic_counts.get("momentum_strong_and_effective", 0)

    sources = {
        "prices": diagnostic_counts.get("momentum_source_prices", 0),
        "13f": diagnostic_counts.get("momentum_source_13f", 0),
    }

    # B) Derived invariants - SINGLE DEFINITION
    # with_data = sum of all windows
    with_data = windows_used["20d"] + windows_used["60d"] + windows_used["120d"]

    # applied = securities with momentum data that passed confidence gate
    # INVARIANT: applied = with_data - low_conf
    applied = max(0, with_data - low_conf)

    # neutral = applied but not strong positive or negative
    # INVARIANT: neutral = applied - neg - pos
    neutral = max(0, applied - neg - pos)

    # coverage_pct = fraction of universe with usable momentum
    if total_rankable > 0:
        coverage_pct = (Decimal(str(applied)) / Decimal(str(total_rankable)) * 100)
    else:
        coverage_pct = Decimal("0")

    # C) Aggregates requiring iteration over ranked_securities
    # avg_weight = average effective momentum weight across active securities
    avg_weight = Decimal("0")
    weight_count = 0

    for sec in ranked_securities:
        flags = sec.get("flags", [])
        # Include securities that have momentum data and passed confidence gate
        if "momentum_missing_prices" not in flags and "momentum_low_confidence" not in flags:
            eff_weights = sec.get("effective_weights", {})
            mom_weight = _to_decimal(eff_weights.get("momentum", "0")) or Decimal("0")
            avg_weight += mom_weight
            weight_count += 1

    if weight_count > 0:
        avg_weight = (avg_weight / Decimal(str(weight_count))).quantize(Decimal("0.001"))
    else:
        avg_weight = Decimal("0")

    return {
        # A) Raw counters
        "missing": missing,
        "low_conf": low_conf,
        "neg": neg,
        "pos": pos,
        "windows_used": windows_used,
        "computable": computable,
        "meaningful": meaningful,
        "strong_signal": strong_signal,
        "strong_and_effective": strong_and_effective,
        "sources": sources,
        # B) Derived invariants
        "total_rankable": total_rankable,
        "with_data": with_data,
        "applied": applied,
        "neutral": neutral,
        "coverage_pct": coverage_pct,
        # C) Aggregates
        "avg_weight": avg_weight,
    }


def build_momentum_health(
    breakdown: Dict[str, Any],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Build momentum_health dict for JSON output from breakdown.

    This packages the breakdown into the schema expected by results JSON,
    enabling scriptable A/B comparisons.

    Args:
        breakdown: Output from compute_momentum_breakdown()
        as_of_date: ISO date string for the scoring run

    Returns:
        Dict matching the momentum_health schema in results JSON
    """
    return {
        "coverage_applied": breakdown["applied"],
        "coverage_pct": round(float(breakdown["coverage_pct"]), 1),
        "computable": breakdown["computable"],
        "meaningful": breakdown["meaningful"],
        "strong_signal": breakdown["strong_signal"],
        "strong_and_effective": breakdown["strong_and_effective"],
        "avg_weight": str(breakdown["avg_weight"]),
        "windows_used": breakdown["windows_used"],
        "sources": breakdown["sources"],
        "total_rankable": breakdown["total_rankable"],
        "as_of_date": as_of_date,
    }


def format_momentum_log_lines(breakdown: Dict[str, Any]) -> List[str]:
    """
    Format momentum breakdown into log lines.

    Returns the same log line formats currently used in module_5_composite_v3.py
    for backwards compatibility with log parsing.

    Args:
        breakdown: Output from compute_momentum_breakdown()

    Returns:
        List of log line strings (caller should log at appropriate level)
    """
    lines = []

    total_rankable = breakdown["total_rankable"]
    missing = breakdown["missing"]
    low_conf = breakdown["low_conf"]
    neg = breakdown["neg"]
    pos = breakdown["pos"]
    neutral = breakdown["neutral"]
    applied = breakdown["applied"]
    windows = breakdown["windows_used"]
    coverage_pct = breakdown["coverage_pct"]
    avg_weight = breakdown["avg_weight"]
    computable = breakdown["computable"]
    meaningful = breakdown["meaningful"]
    strong_signal = breakdown["strong_signal"]
    strong_effective = breakdown["strong_and_effective"]
    sources = breakdown["sources"]

    # Line 1: Detailed breakdown
    lines.append(
        f"INFO: momentum breakdown - "
        f"missing:{missing}, low_conf:{low_conf}, "
        f"applied[neg:{neg}, pos:{pos}, neutral:{neutral}] | "
        f"windows[20d:{windows['20d']}, 60d:{windows['60d']}, 120d:{windows['120d']}] | "
        f"coverage={applied}/{total_rankable} ({coverage_pct:.1f}%), avg_weight={avg_weight}"
    )

    # Line 2: Stable metrics dashboard
    lines.append(
        f"INFO: momentum stable metrics - "
        f"computable:{computable}, meaningful:{meaningful}, "
        f"coverage_applied:{applied}/{total_rankable}, "
        f"strong:{strong_signal}, strong_and_effective:{strong_effective}, "
        f"sources[prices:{sources['prices']}, 13f:{sources['13f']}]"
    )

    return lines


def check_coverage_guardrail(
    breakdown: Dict[str, Any],
    threshold_pct: float = 20.0,
) -> Optional[str]:
    """
    Check if momentum coverage has collapsed below threshold.

    Args:
        breakdown: Output from compute_momentum_breakdown()
        threshold_pct: Coverage percentage threshold (default 20%)

    Returns:
        Warning message if coverage below threshold, None otherwise
    """
    coverage_pct = float(breakdown["coverage_pct"])

    if breakdown["total_rankable"] > 0 and coverage_pct < threshold_pct:
        return (
            f"WARN: momentum coverage collapsed ({coverage_pct:.1f}%) - "
            f"check enrich_market_data_momentum outputs / market_data.json keys"
        )

    return None
