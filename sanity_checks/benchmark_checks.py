"""
Query 7.2: Benchmark Comparison Checks

Validates outputs against known industry benchmarks and peer comparisons:

1. Relative Ranking Sanity
2. Historical Pattern Validation
3. Peer Relative Checks
4. Sector Average Comparisons

These checks ensure the model produces sensible outputs relative to
market benchmarks and peer groups.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Set, Tuple

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
class PeerGroup:
    """A group of peer securities for comparison."""
    primary_ticker: str
    peer_tickers: List[str]
    indication: str
    stage: str


@dataclass
class SectorStats:
    """Statistics for a sector."""
    sector: str
    count: int
    mean_score: Decimal
    median_score: Decimal
    std_score: Decimal
    min_score: Decimal
    max_score: Decimal


class BenchmarkChecker:
    """
    Benchmark comparison sanity checker.

    Validates that rankings are sensible relative to:
    - Sector concentrations
    - Peer comparisons
    - Historical benchmarks
    - Industry patterns
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
        historical_snapshots: Optional[List[RankingSnapshot]] = None,
        peer_groups: Optional[List[PeerGroup]] = None,
    ) -> SanityCheckResult:
        """
        Run all benchmark checks.

        Args:
            securities: Current ranked securities
            historical_snapshots: Optional historical ranking data
            peer_groups: Optional peer group definitions

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []

        # 1. Relative Ranking Sanity
        flags.extend(self._check_relative_ranking(securities))

        # 2. Historical Pattern Validation
        if historical_snapshots:
            flags.extend(self._check_historical_patterns(securities, historical_snapshots))

        # 3. Peer Relative Checks
        if peer_groups:
            flags.extend(self._check_peer_relative(securities, peer_groups))

        # 4. Sector Average Comparisons
        flags.extend(self._check_sector_averages(securities))

        # Calculate metrics
        metrics = self._calculate_metrics(flags, securities)

        # Pass if no CRITICAL flags
        passed = not any(f.severity == FlagSeverity.CRITICAL for f in flags)

        return SanityCheckResult(
            check_name="benchmark_comparison",
            category=CheckCategory.BENCHMARK,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_relative_ranking(
        self,
        securities: List[SecurityContext],
    ) -> List[SanityFlag]:
        """
        Check relative ranking sanity.

        Flags:
        - Top 10 has >5 from same sub-sector (concentration risk)
        - Micro-cap dominates top 10 but large-caps excluded
        """
        flags: List[SanityFlag] = []

        # Get top 10 securities
        top10 = [s for s in securities if s.rank is not None and s.rank <= 10]

        if not top10:
            return flags

        # Check 1: Sector concentration in top 10
        sector_counts = Counter(s.sector for s in top10 if s.sector)
        for sector, count in sector_counts.items():
            if count > self.config.max_same_sector_in_top10:
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.BENCHMARK,
                    ticker=None,
                    check_name="sector_concentration_risk",
                    message=f"Top 10 has {count} names from {sector} sector (max: {self.config.max_same_sector_in_top10})",
                    details={
                        "sector": sector,
                        "count": count,
                        "threshold": self.config.max_same_sector_in_top10,
                        "tickers": [s.ticker for s in top10 if s.sector == sector],
                    },
                    recommendation="Insufficient diversification - sector concentration risk",
                ))

        # Check 2: Indication concentration in top 10
        indication_counts = Counter(s.indication for s in top10 if s.indication)
        for indication, count in indication_counts.items():
            if count > 4:  # More than 4 in same indication is concerning
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.BENCHMARK,
                    ticker=None,
                    check_name="indication_concentration_risk",
                    message=f"Top 10 has {count} names in {indication} indication",
                    details={
                        "indication": indication,
                        "count": count,
                        "tickers": [s.ticker for s in top10 if s.indication == indication],
                    },
                    recommendation="Regulatory/competitive risk correlation",
                ))

        # Check 3: Market cap distribution check
        micro_caps = [s for s in top10 if s.is_micro_cap]
        if len(micro_caps) >= 8:  # 80% micro-cap is concerning
            # Check if there are any large-caps in universe that were excluded
            all_large_caps = [
                s for s in securities
                if s.market_cap_mm and s.market_cap_mm >= Decimal("1000")
            ]
            if all_large_caps:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.BENCHMARK,
                    ticker=None,
                    check_name="liquidity_bias_not_reflected",
                    message=f"Micro-cap ({len(micro_caps)}/10) dominates top 10, {len(all_large_caps)} large-caps in universe",
                    details={
                        "micro_cap_count": len(micro_caps),
                        "large_cap_count": len(all_large_caps),
                    },
                    recommendation="Liquidity bias not reflected - small names may be uninvestable",
                ))

        return flags

    def _check_historical_patterns(
        self,
        securities: List[SecurityContext],
        historical_snapshots: List[RankingSnapshot],
    ) -> List[SanityFlag]:
        """
        Check historical pattern validation.

        Flags:
        - >50% turnover from last 4 weeks (high churn)
        - <10% turnover (stale signals)
        """
        flags: List[SanityFlag] = []

        if len(historical_snapshots) < 4:
            return flags

        # Get current top 10
        current_top10 = set(
            s.ticker for s in securities
            if s.rank is not None and s.rank <= 10
        )

        if not current_top10:
            return flags

        # Get top 10 from each historical snapshot
        weekly_top10s: List[Set[str]] = []
        for snap in sorted(historical_snapshots, key=lambda x: x.as_of_date)[-4:]:
            top10 = set(
                s.ticker for s in snap.securities
                if s.rank is not None and s.rank <= 10
            )
            weekly_top10s.append(top10)

        # Calculate turnover vs. each historical week
        turnovers = []
        for hist_top10 in weekly_top10s:
            if hist_top10:
                overlap = len(current_top10 & hist_top10)
                union = len(current_top10 | hist_top10)
                if union > 0:
                    turnover = 1 - (overlap / max(len(current_top10), len(hist_top10)))
                    turnovers.append(turnover)

        if not turnovers:
            return flags

        avg_turnover = mean(turnovers)

        # Check high churn
        if avg_turnover > 0.50:
            flags.append(SanityFlag(
                severity=FlagSeverity.HIGH,
                category=CheckCategory.BENCHMARK,
                ticker=None,
                check_name="high_ranking_churn",
                message=f"Top 10 turnover {avg_turnover:.0%} vs last 4 weeks (threshold: 50%)",
                details={
                    "avg_turnover": avg_turnover,
                    "weekly_turnovers": turnovers,
                },
                recommendation="High churn - investigate volatility in signals",
            ))
        # Check stale signals
        elif avg_turnover < 0.10:
            flags.append(SanityFlag(
                severity=FlagSeverity.MEDIUM,
                category=CheckCategory.BENCHMARK,
                ticker=None,
                check_name="stale_signals",
                message=f"Top 10 turnover {avg_turnover:.0%} vs last 4 weeks (expected: >10%)",
                details={
                    "avg_turnover": avg_turnover,
                    "weekly_turnovers": turnovers,
                },
                recommendation="Stale signals - check data freshness",
            ))

        return flags

    def _check_peer_relative(
        self,
        securities: List[SecurityContext],
        peer_groups: List[PeerGroup],
    ) -> List[SanityFlag]:
        """
        Check peer relative rankings.

        For each peer group, verify ranking differentials make sense.
        """
        flags: List[SanityFlag] = []

        # Build lookup
        rank_lookup = {s.ticker: s for s in securities if s.rank is not None}

        for group in peer_groups:
            primary = rank_lookup.get(group.primary_ticker)
            if not primary:
                continue

            peers = [rank_lookup.get(t) for t in group.peer_tickers]
            peers = [p for p in peers if p is not None]

            if not peers:
                continue

            # Check if primary is ranked significantly different from peers
            peer_ranks = [p.rank for p in peers]
            avg_peer_rank = mean(peer_ranks)
            primary_rank = primary.rank

            rank_diff = abs(primary_rank - avg_peer_rank)

            # Flag if >30 position difference from peer average
            if rank_diff > 30:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.BENCHMARK,
                    ticker=primary.ticker,
                    check_name="peer_rank_divergence",
                    message=f"{primary.ticker} ranked #{primary_rank} but peer avg is #{avg_peer_rank:.0f}",
                    details={
                        "ticker": primary.ticker,
                        "rank": primary_rank,
                        "peer_avg_rank": avg_peer_rank,
                        "peers": [(p.ticker, p.rank) for p in peers],
                        "indication": group.indication,
                        "stage": group.stage,
                    },
                    recommendation="Generate Differentiation Justification Report",
                ))

        return flags

    def _check_sector_averages(
        self,
        securities: List[SecurityContext],
    ) -> List[SanityFlag]:
        """
        Check sector average comparisons.

        Flags:
        - Top candidate score <1.2x sector median
        - Score >3x sector median (extreme outlier)
        """
        flags: List[SanityFlag] = []

        # Group by sector
        by_sector: Dict[str, List[SecurityContext]] = defaultdict(list)
        for sec in securities:
            if sec.sector and sec.composite_score is not None:
                by_sector[sec.sector].append(sec)

        # Calculate sector stats
        sector_stats: Dict[str, SectorStats] = {}
        for sector, secs in by_sector.items():
            if len(secs) < 3:
                continue

            scores = sorted([float(s.composite_score) for s in secs])
            median_idx = len(scores) // 2
            median = Decimal(str(scores[median_idx]))
            mean_score = Decimal(str(mean(scores)))
            std_score = Decimal(str(stdev(scores))) if len(scores) > 1 else Decimal("0")

            sector_stats[sector] = SectorStats(
                sector=sector,
                count=len(secs),
                mean_score=mean_score,
                median_score=median,
                std_score=std_score,
                min_score=Decimal(str(min(scores))),
                max_score=Decimal(str(max(scores))),
            )

        # Check top candidates relative to sector median
        for sec in securities:
            if sec.rank is None or sec.composite_score is None:
                continue

            if sec.sector not in sector_stats:
                continue

            stats = sector_stats[sec.sector]
            if stats.median_score == 0:
                continue

            ratio = sec.composite_score / stats.median_score

            # Top candidate with insufficient alpha
            if sec.rank <= 10:
                if ratio < Decimal("1.2"):
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.BENCHMARK,
                        ticker=sec.ticker,
                        check_name="insufficient_alpha_vs_sector",
                        message=f"Top {sec.rank} candidate score only {ratio:.2f}x sector median",
                        details={
                            "rank": sec.rank,
                            "score": float(sec.composite_score),
                            "sector_median": float(stats.median_score),
                            "ratio": float(ratio),
                        },
                        recommendation="Insufficient alpha vs. sector average",
                    ))

            # Extreme outlier check
            if ratio > Decimal("3.0"):
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.BENCHMARK,
                    ticker=sec.ticker,
                    check_name="extreme_outlier_score",
                    message=f"{sec.ticker} score {ratio:.2f}x sector median - extreme outlier",
                    details={
                        "rank": sec.rank,
                        "score": float(sec.composite_score),
                        "sector_median": float(stats.median_score),
                        "ratio": float(ratio),
                    },
                    recommendation="Extreme outlier - verify data quality",
                ))

        return flags

    def _calculate_metrics(
        self,
        flags: List[SanityFlag],
        securities: List[SecurityContext],
    ) -> Dict[str, Any]:
        """Calculate summary metrics."""
        # Sector distribution in top 20
        top20 = [s for s in securities if s.rank is not None and s.rank <= 20]
        sector_dist = Counter(s.sector for s in top20 if s.sector)

        # Market cap distribution in top 20
        cap_buckets = {"micro": 0, "small": 0, "mid": 0, "large": 0}
        for s in top20:
            if s.market_cap_mm is None:
                continue
            if s.market_cap_mm < 200:
                cap_buckets["micro"] += 1
            elif s.market_cap_mm < 1000:
                cap_buckets["small"] += 1
            elif s.market_cap_mm < 5000:
                cap_buckets["mid"] += 1
            else:
                cap_buckets["large"] += 1

        return {
            "total_flags": len(flags),
            "top20_sector_distribution": dict(sector_dist),
            "top20_cap_distribution": cap_buckets,
            "by_severity": {
                "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
                "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
                "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
                "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
            },
        }


def calculate_sector_stats(
    securities: List[SecurityContext],
) -> Dict[str, SectorStats]:
    """
    Calculate statistics for each sector.

    Args:
        securities: List of securities with scores

    Returns:
        Dict mapping sector to SectorStats
    """
    by_sector: Dict[str, List[float]] = defaultdict(list)

    for sec in securities:
        if sec.sector and sec.composite_score is not None:
            by_sector[sec.sector].append(float(sec.composite_score))

    stats: Dict[str, SectorStats] = {}
    for sector, scores in by_sector.items():
        if len(scores) < 2:
            continue

        sorted_scores = sorted(scores)
        median_idx = len(sorted_scores) // 2

        stats[sector] = SectorStats(
            sector=sector,
            count=len(scores),
            mean_score=Decimal(str(mean(scores))),
            median_score=Decimal(str(sorted_scores[median_idx])),
            std_score=Decimal(str(stdev(scores))),
            min_score=Decimal(str(min(scores))),
            max_score=Decimal(str(max(scores))),
        )

    return stats
