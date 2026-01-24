"""
Query 7.3: Time Series Coherence Checks

Validates that rankings evolve logically over time:

1. Rank Velocity Checks
2. Score Decomposition Analysis
3. Catalyst Timeline Coherence
4. Forward-Looking Validation

These checks ensure rankings don't change erratically and that
changes are explainable by underlying data changes.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    RankingSnapshot,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class RankChange:
    """Record of a rank change between periods."""
    ticker: str
    old_rank: int
    new_rank: int
    rank_delta: int
    old_score: Optional[Decimal]
    new_score: Optional[Decimal]
    score_delta: Optional[Decimal]
    score_decomposition: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class CatalystEvent:
    """A catalyst event that should affect rankings."""
    ticker: str
    event_type: str
    event_date: str
    expected_direction: str  # "positive", "negative", "neutral"
    actual_rank_change: Optional[int] = None


class TimeSeriesChecker:
    """
    Time series coherence checker.

    Validates that ranking changes are logical and explainable
    over time.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        current_snapshot: RankingSnapshot,
        previous_snapshot: Optional[RankingSnapshot] = None,
        historical_snapshots: Optional[List[RankingSnapshot]] = None,
        catalyst_events: Optional[List[CatalystEvent]] = None,
    ) -> SanityCheckResult:
        """
        Run all time series checks.

        Args:
            current_snapshot: Current ranking snapshot
            previous_snapshot: Previous period snapshot (e.g., last week)
            historical_snapshots: Longer history for trend analysis
            catalyst_events: Known catalyst events to check

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []
        metrics: Dict[str, Any] = {}

        # 1. Rank Velocity Checks (requires previous snapshot)
        if previous_snapshot:
            velocity_flags, velocity_metrics = self._check_rank_velocity(
                current_snapshot, previous_snapshot
            )
            flags.extend(velocity_flags)
            metrics["rank_velocity"] = velocity_metrics

        # 2. Score Decomposition Analysis
        if previous_snapshot:
            decomp_flags = self._check_score_decomposition(
                current_snapshot, previous_snapshot
            )
            flags.extend(decomp_flags)

        # 3. Catalyst Timeline Coherence
        if catalyst_events:
            catalyst_flags = self._check_catalyst_coherence(
                current_snapshot, catalyst_events
            )
            flags.extend(catalyst_flags)

        # 4. Trend Stability (requires historical data)
        if historical_snapshots and len(historical_snapshots) >= 3:
            trend_flags, trend_metrics = self._check_trend_stability(
                current_snapshot, historical_snapshots
            )
            flags.extend(trend_flags)
            metrics["trend_stability"] = trend_metrics

        # Overall metrics
        metrics["total_flags"] = len(flags)
        metrics["by_severity"] = {
            "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
            "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
            "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
            "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
        }

        passed = not any(f.severity == FlagSeverity.CRITICAL for f in flags)

        return SanityCheckResult(
            check_name="time_series_coherence",
            category=CheckCategory.TIME_SERIES,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_rank_velocity(
        self,
        current: RankingSnapshot,
        previous: RankingSnapshot,
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """
        Check rank velocity (change rate between periods).

        Flags:
        - Ticker jumps >15 positions in one week
        - Large positive jump without catalyst explanation
        """
        flags: List[SanityFlag] = []

        # Build lookups
        current_ranks = {s.ticker: s for s in current.securities if s.rank is not None}
        prev_ranks = {s.ticker: s for s in previous.securities if s.rank is not None}

        # Track changes
        rank_changes: List[RankChange] = []
        large_jumps: List[RankChange] = []

        common_tickers = set(current_ranks.keys()) & set(prev_ranks.keys())

        for ticker in common_tickers:
            curr = current_ranks[ticker]
            prev = prev_ranks[ticker]

            rank_delta = prev.rank - curr.rank  # Positive = improved rank

            score_delta = None
            if curr.composite_score is not None and prev.composite_score is not None:
                score_delta = curr.composite_score - prev.composite_score

            change = RankChange(
                ticker=ticker,
                old_rank=prev.rank,
                new_rank=curr.rank,
                rank_delta=rank_delta,
                old_score=prev.composite_score,
                new_score=curr.composite_score,
                score_delta=score_delta,
            )
            rank_changes.append(change)

            # Check for large jumps
            if abs(rank_delta) > self.config.max_rank_jump_single_week:
                large_jumps.append(change)

                # Determine if this is a jump UP or DOWN
                if rank_delta > 0:  # Improved rank
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.TIME_SERIES,
                        ticker=ticker,
                        check_name="rank_jump_investigation",
                        message=f"{ticker} jumped from #{prev.rank} to #{curr.rank} in one week",
                        details={
                            "old_rank": prev.rank,
                            "new_rank": curr.rank,
                            "rank_delta": rank_delta,
                            "score_delta": float(score_delta) if score_delta else None,
                        },
                        recommendation="Investigate catalyst or data anomaly - generate Rank Jump Explanation Report",
                    ))
                else:  # Dropped rank
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.TIME_SERIES,
                        ticker=ticker,
                        check_name="rank_drop_investigation",
                        message=f"{ticker} dropped from #{prev.rank} to #{curr.rank} in one week",
                        details={
                            "old_rank": prev.rank,
                            "new_rank": curr.rank,
                            "rank_delta": rank_delta,
                            "score_delta": float(score_delta) if score_delta else None,
                        },
                        recommendation="Investigate negative catalyst or data issue",
                    ))

        # Calculate velocity metrics
        if rank_changes:
            abs_deltas = [abs(c.rank_delta) for c in rank_changes]
            metrics = {
                "avg_rank_change": mean(abs_deltas),
                "max_rank_change": max(abs_deltas),
                "large_jumps_count": len(large_jumps),
                "common_tickers": len(common_tickers),
            }
        else:
            metrics = {
                "avg_rank_change": 0,
                "max_rank_change": 0,
                "large_jumps_count": 0,
                "common_tickers": 0,
            }

        return flags, metrics

    def _check_score_decomposition(
        self,
        current: RankingSnapshot,
        previous: RankingSnapshot,
    ) -> List[SanityFlag]:
        """
        Analyze score decomposition for major rank changes.

        Flags:
        - Rank change without corresponding score component changes
        - Unexplained composite score delta
        """
        flags: List[SanityFlag] = []

        current_lookup = {s.ticker: s for s in current.securities if s.rank is not None}
        prev_lookup = {s.ticker: s for s in previous.securities if s.rank is not None}

        for ticker in current_lookup:
            if ticker not in prev_lookup:
                continue

            curr = current_lookup[ticker]
            prev = prev_lookup[ticker]

            rank_delta = abs(prev.rank - curr.rank)

            # Only analyze significant rank changes
            if rank_delta < 10:
                continue

            # Check if we can explain the change with component deltas
            explained_delta = Decimal("0")
            unexplained = False

            component_changes = {}

            # Clinical score change
            if curr.clinical_score is not None and prev.clinical_score is not None:
                delta = curr.clinical_score - prev.clinical_score
                component_changes["clinical"] = float(delta)
                explained_delta += delta * Decimal("0.4")  # ~40% weight

            # Financial score change
            if curr.financial_score is not None and prev.financial_score is not None:
                delta = curr.financial_score - prev.financial_score
                component_changes["financial"] = float(delta)
                explained_delta += delta * Decimal("0.25")  # ~25% weight

            # Catalyst score change
            if curr.catalyst_score is not None and prev.catalyst_score is not None:
                delta = curr.catalyst_score - prev.catalyst_score
                component_changes["catalyst"] = float(delta)
                explained_delta += delta * Decimal("0.15")  # ~15% weight

            # Check if composite delta is explained
            if curr.composite_score is not None and prev.composite_score is not None:
                actual_delta = curr.composite_score - prev.composite_score
                unexplained_delta = actual_delta - explained_delta

                if abs(unexplained_delta) > Decimal("5"):  # >5 point unexplained
                    unexplained = True

            if unexplained and rank_delta >= self.config.max_rank_jump_single_week:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.TIME_SERIES,
                    ticker=ticker,
                    check_name="unexplained_score_change",
                    message=f"{ticker} has unexplained score delta with {rank_delta} position change",
                    details={
                        "rank_delta": rank_delta,
                        "component_changes": component_changes,
                        "unexplained_delta": float(unexplained_delta) if unexplained_delta else None,
                    },
                    recommendation="Score decomposition cannot fully explain rank change",
                ))

        return flags

    def _check_catalyst_coherence(
        self,
        current: RankingSnapshot,
        catalyst_events: List[CatalystEvent],
    ) -> List[SanityFlag]:
        """
        Check catalyst timeline coherence.

        Flags:
        - Catalyst date passed but ranking unchanged
        - Negative catalyst but rank improved
        """
        flags: List[SanityFlag] = []

        current_lookup = {s.ticker: s for s in current.securities}

        for event in catalyst_events:
            sec = current_lookup.get(event.ticker)
            if not sec:
                continue

            # Check if catalyst should have affected ranking
            if event.expected_direction == "positive":
                if event.actual_rank_change is not None and event.actual_rank_change < 0:
                    # Rank should have improved (lower number) but didn't
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.TIME_SERIES,
                        ticker=event.ticker,
                        check_name="catalyst_not_incorporated",
                        message=f"Positive catalyst {event.event_type} on {event.event_date} but rank dropped",
                        details={
                            "event_type": event.event_type,
                            "event_date": event.event_date,
                            "expected_direction": event.expected_direction,
                            "actual_rank_change": event.actual_rank_change,
                        },
                        recommendation="Catalyst result may not be incorporated",
                    ))

            elif event.expected_direction == "negative":
                if event.actual_rank_change is not None and event.actual_rank_change > 5:
                    # Rank improved despite negative catalyst
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.TIME_SERIES,
                        ticker=event.ticker,
                        check_name="negative_catalyst_ignored",
                        message=f"Negative catalyst {event.event_type} on {event.event_date} but rank improved +{event.actual_rank_change}",
                        details={
                            "event_type": event.event_type,
                            "event_date": event.event_date,
                            "expected_direction": event.expected_direction,
                            "actual_rank_change": event.actual_rank_change,
                        },
                        recommendation="Negative catalyst may be ignored",
                    ))

        return flags

    def _check_trend_stability(
        self,
        current: RankingSnapshot,
        historical: List[RankingSnapshot],
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """
        Check trend stability over longer period.

        Flags:
        - Steady climber identified (positive pattern)
        - Erratic movement without trend
        """
        flags: List[SanityFlag] = []

        # Sort historical by date
        sorted_historical = sorted(historical, key=lambda x: x.as_of_date)

        # Get top 50 current for analysis
        current_top50 = {s.ticker for s in current.securities if s.rank and s.rank <= 50}

        # Track rank history for each ticker
        rank_histories: Dict[str, List[Tuple[str, int]]] = {}

        for snap in sorted_historical:
            for sec in snap.securities:
                if sec.ticker in current_top50 and sec.rank is not None:
                    if sec.ticker not in rank_histories:
                        rank_histories[sec.ticker] = []
                    rank_histories[sec.ticker].append((snap.as_of_date, sec.rank))

        # Analyze trends
        steady_climbers = []
        erratic_movers = []

        for ticker, history in rank_histories.items():
            if len(history) < 3:
                continue

            ranks = [r for _, r in history]

            # Check for steady improvement
            improvements = sum(1 for i in range(1, len(ranks)) if ranks[i] < ranks[i-1])
            if improvements >= len(ranks) - 1:
                steady_climbers.append(ticker)

            # Check for erratic movement (high variance)
            if len(ranks) > 2:
                rank_stdev = stdev(ranks)
                if rank_stdev > 20:  # High variance
                    erratic_movers.append((ticker, rank_stdev))

        # Flag erratic movers if they're in top 20
        current_lookup = {s.ticker: s for s in current.securities}
        for ticker, variance in erratic_movers:
            sec = current_lookup.get(ticker)
            if sec and sec.rank and sec.rank <= 20:
                flags.append(SanityFlag(
                    severity=FlagSeverity.LOW,
                    category=CheckCategory.TIME_SERIES,
                    ticker=ticker,
                    check_name="erratic_rank_movement",
                    message=f"{ticker} shows erratic rank movement (stdev={variance:.1f}) over analysis period",
                    details={
                        "current_rank": sec.rank,
                        "rank_stdev": variance,
                        "history_length": len(rank_histories.get(ticker, [])),
                    },
                    recommendation="Monitor for signal stability",
                ))

        metrics = {
            "steady_climbers": steady_climbers,
            "erratic_movers_count": len(erratic_movers),
            "tickers_analyzed": len(rank_histories),
        }

        return flags, metrics


def compute_rank_correlation(
    snapshot1: RankingSnapshot,
    snapshot2: RankingSnapshot,
) -> Optional[float]:
    """
    Compute Spearman rank correlation between two snapshots.

    Args:
        snapshot1: First ranking snapshot
        snapshot2: Second ranking snapshot

    Returns:
        Spearman correlation coefficient or None if insufficient data
    """
    ranks1 = {s.ticker: s.rank for s in snapshot1.securities if s.rank is not None}
    ranks2 = {s.ticker: s.rank for s in snapshot2.securities if s.rank is not None}

    common = set(ranks1.keys()) & set(ranks2.keys())
    if len(common) < 5:
        return None

    # Compute Spearman correlation
    x = [ranks1[t] for t in common]
    y = [ranks2[t] for t in common]

    n = len(x)
    d_squared = sum((xi - yi) ** 2 for xi, yi in zip(x, y))

    rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
    return rho


def generate_rank_jump_report(
    ticker: str,
    current: SecurityContext,
    previous: SecurityContext,
) -> Dict[str, Any]:
    """
    Generate a detailed rank jump explanation report.

    Args:
        ticker: Ticker symbol
        current: Current security context
        previous: Previous security context

    Returns:
        Detailed report dict
    """
    report = {
        "ticker": ticker,
        "old_rank": previous.rank,
        "new_rank": current.rank,
        "rank_change": previous.rank - current.rank if previous.rank and current.rank else None,
    }

    # Score decomposition
    components = {}

    if current.clinical_score is not None and previous.clinical_score is not None:
        components["clinical"] = {
            "old": float(previous.clinical_score),
            "new": float(current.clinical_score),
            "delta": float(current.clinical_score - previous.clinical_score),
        }

    if current.financial_score is not None and previous.financial_score is not None:
        components["financial"] = {
            "old": float(previous.financial_score),
            "new": float(current.financial_score),
            "delta": float(current.financial_score - previous.financial_score),
        }

    if current.catalyst_score is not None and previous.catalyst_score is not None:
        components["catalyst"] = {
            "old": float(previous.catalyst_score),
            "new": float(current.catalyst_score),
            "delta": float(current.catalyst_score - previous.catalyst_score),
        }

    if current.composite_score is not None and previous.composite_score is not None:
        components["composite"] = {
            "old": float(previous.composite_score),
            "new": float(current.composite_score),
            "delta": float(current.composite_score - previous.composite_score),
        }

    report["component_changes"] = components

    # Key changes
    report["key_changes"] = []

    if current.lead_phase != previous.lead_phase:
        report["key_changes"].append(f"Phase changed: {previous.lead_phase} -> {current.lead_phase}")

    if current.days_to_catalyst != previous.days_to_catalyst:
        report["key_changes"].append(
            f"Days to catalyst: {previous.days_to_catalyst} -> {current.days_to_catalyst}"
        )

    return report
