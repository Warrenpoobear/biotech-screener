"""
Query 8.1: Executive Dashboard Validation

Ensures outputs are consumable by investment committee:

1. One-Pager Quality Check
2. Ranking Stability Visualization
3. Portfolio Fit Analysis
4. Risk Summary Dashboard

These validations ensure the screening output is ready for
IC presentation and decision-making.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
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
class OnePagerContent:
    """One-pager content for a candidate."""
    ticker: str
    company_name: Optional[str]
    market_cap_mm: Optional[Decimal]
    sector: Optional[str]
    current_price: Optional[Decimal]
    price_52w_high: Optional[Decimal]
    price_52w_low: Optional[Decimal]
    thesis: Optional[str]
    catalyst_date: Optional[str]
    catalyst_type: Optional[str]
    top_bulls: List[str]
    top_bears: List[str]
    composite_score: Optional[Decimal]
    score_breakdown: Dict[str, Decimal]


@dataclass
class RankHistoryEntry:
    """Rank history entry for stability visualization."""
    as_of_date: str
    rank: int
    score: Optional[Decimal]


@dataclass
class RankingStability:
    """Ranking stability analysis for a ticker."""
    ticker: str
    current_rank: int
    rank_history: List[RankHistoryEntry]
    trend: str  # "rising", "stable", "falling"
    volatility: Decimal
    weeks_in_top20: int


@dataclass
class PortfolioFit:
    """Portfolio fit analysis for a candidate."""
    ticker: str
    overlaps_existing: bool
    complementary_sectors: List[str]
    redundant_with: List[str]
    incremental_contribution: Optional[Decimal]


@dataclass
class RiskSummary:
    """Aggregate risk summary for top candidates."""
    avg_runway_months: Decimal
    binary_catalyst_pct: Decimal
    sector_concentrations: Dict[str, Decimal]
    liquidity_weighted_adv: Decimal


class ExecutiveDashboardValidator:
    """
    Executive dashboard validator.

    Ensures all elements required for IC presentation are
    present and complete.
    """

    # Required one-pager fields
    REQUIRED_ONEPAGER_FIELDS = {
        "ticker",
        "company_name",
        "market_cap_mm",
        "sector",
        "thesis",
        "catalyst_date",
        "top_bulls",
        "top_bears",
        "composite_score",
        "score_breakdown",
    }

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
        one_pagers: Optional[Dict[str, OnePagerContent]] = None,
        historical_snapshots: Optional[List[RankingSnapshot]] = None,
        existing_holdings: Optional[Dict[str, Decimal]] = None,
    ) -> SanityCheckResult:
        """
        Run all executive dashboard validations.

        Args:
            securities: List of security contexts
            one_pagers: Optional one-pager content by ticker
            historical_snapshots: Optional historical data for stability
            existing_holdings: Optional existing portfolio holdings

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []
        metrics: Dict[str, Any] = {}

        # 1. One-Pager Quality Check
        if one_pagers:
            pager_flags, pager_metrics = self._check_one_pagers(securities, one_pagers)
            flags.extend(pager_flags)
            metrics["one_pager_quality"] = pager_metrics
        else:
            # Missing one-pagers for top 10 is a critical issue
            top10 = [s for s in securities if s.rank and s.rank <= 10]
            if top10:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.DASHBOARD,
                    ticker=None,
                    check_name="missing_all_one_pagers",
                    message=f"No one-pager content provided for {len(top10)} top 10 candidates",
                    details={"top10_tickers": [s.ticker for s in top10]},
                    recommendation="Generate one-pagers before IC presentation",
                ))

        # 2. Ranking Stability Visualization Data
        if historical_snapshots:
            stability_flags, stability_data = self._analyze_ranking_stability(
                securities, historical_snapshots
            )
            flags.extend(stability_flags)
            metrics["ranking_stability"] = stability_data

        # 3. Portfolio Fit Analysis
        if existing_holdings:
            fit_flags, fit_analysis = self._analyze_portfolio_fit(
                securities, existing_holdings
            )
            flags.extend(fit_flags)
            metrics["portfolio_fit"] = fit_analysis

        # 4. Risk Summary Dashboard
        risk_summary, risk_flags = self._generate_risk_summary(securities)
        flags.extend(risk_flags)
        metrics["risk_summary"] = risk_summary

        # Calculate overall metrics
        metrics["total_flags"] = len(flags)
        metrics["by_severity"] = {
            "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
            "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
            "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
            "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
        }

        passed = not any(f.severity == FlagSeverity.CRITICAL for f in flags)

        return SanityCheckResult(
            check_name="executive_dashboard",
            category=CheckCategory.DASHBOARD,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_one_pagers(
        self,
        securities: List[SecurityContext],
        one_pagers: Dict[str, OnePagerContent],
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """Check one-pager quality for top candidates."""
        flags: List[SanityFlag] = []

        complete_count = 0
        incomplete_count = 0

        for sec in securities:
            if sec.rank is None or sec.rank > 10:
                continue

            pager = one_pagers.get(sec.ticker)

            if pager is None:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.DASHBOARD,
                    ticker=sec.ticker,
                    check_name="missing_one_pager",
                    message=f"Top {sec.rank} candidate missing one-pager",
                    details={"rank": sec.rank},
                    recommendation="Create one-pager before IC presentation",
                ))
                incomplete_count += 1
                continue

            # Check for missing required fields
            missing = []
            if not pager.company_name:
                missing.append("company_name")
            if not pager.market_cap_mm:
                missing.append("market_cap_mm")
            if not pager.sector:
                missing.append("sector")
            if not pager.thesis:
                missing.append("thesis")
            if not pager.catalyst_date:
                missing.append("catalyst_date")
            if len(pager.top_bulls) < 3:
                missing.append("top_bulls (need 3)")
            if len(pager.top_bears) < 3:
                missing.append("top_bears (need 3)")
            if not pager.score_breakdown:
                missing.append("score_breakdown")

            if missing:
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.DASHBOARD,
                    ticker=sec.ticker,
                    check_name="incomplete_one_pager",
                    message=f"Top {sec.rank} one-pager missing: {', '.join(missing)}",
                    details={
                        "rank": sec.rank,
                        "missing_fields": missing,
                    },
                    recommendation="Complete one-pager fields",
                ))
                incomplete_count += 1
            else:
                complete_count += 1

        metrics = {
            "top10_complete": complete_count,
            "top10_incomplete": incomplete_count,
            "completeness_pct": complete_count / (complete_count + incomplete_count)
            if (complete_count + incomplete_count) > 0 else 0,
        }

        return flags, metrics

    def _analyze_ranking_stability(
        self,
        securities: List[SecurityContext],
        historical_snapshots: List[RankingSnapshot],
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """Analyze ranking stability for visualization."""
        flags: List[SanityFlag] = []

        # Sort snapshots by date
        sorted_snapshots = sorted(historical_snapshots, key=lambda x: x.as_of_date)

        # Build rank history for top 20
        stability_data: Dict[str, RankingStability] = {}

        current_top20 = {s.ticker for s in securities if s.rank and s.rank <= 20}

        for ticker in current_top20:
            history: List[RankHistoryEntry] = []

            for snap in sorted_snapshots:
                sec = snap.get_by_ticker(ticker)
                if sec and sec.rank:
                    history.append(RankHistoryEntry(
                        as_of_date=snap.as_of_date,
                        rank=sec.rank,
                        score=sec.composite_score,
                    ))

            if len(history) < 2:
                continue

            # Calculate trend
            ranks = [h.rank for h in history]
            first_half_avg = sum(ranks[:len(ranks)//2]) / max(1, len(ranks)//2)
            second_half_avg = sum(ranks[len(ranks)//2:]) / max(1, len(ranks) - len(ranks)//2)

            if second_half_avg < first_half_avg - 3:
                trend = "rising"
            elif second_half_avg > first_half_avg + 3:
                trend = "falling"
            else:
                trend = "stable"

            # Calculate volatility (rank standard deviation)
            mean_rank = sum(ranks) / len(ranks)
            variance = sum((r - mean_rank) ** 2 for r in ranks) / len(ranks)
            volatility = Decimal(str(variance ** 0.5))

            # Count weeks in top 20
            weeks_in_top20 = sum(1 for h in history if h.rank <= 20)

            current_sec = next((s for s in securities if s.ticker == ticker), None)
            current_rank = current_sec.rank if current_sec else 999

            stability_data[ticker] = RankingStability(
                ticker=ticker,
                current_rank=current_rank,
                rank_history=history,
                trend=trend,
                volatility=volatility,
                weeks_in_top20=weeks_in_top20,
            )

            # Flag high volatility in top 10
            if current_rank <= 10 and volatility > Decimal("10"):
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.DASHBOARD,
                    ticker=ticker,
                    check_name="high_rank_volatility",
                    message=f"Top {current_rank} has high rank volatility (stdev={volatility:.1f})",
                    details={
                        "rank": current_rank,
                        "volatility": float(volatility),
                        "trend": trend,
                        "weeks_analyzed": len(history),
                    },
                    recommendation="Highlight volatility in presentation",
                ))

        # Generate stability summary
        summary = {
            "tickers_analyzed": len(stability_data),
            "rising": sum(1 for s in stability_data.values() if s.trend == "rising"),
            "stable": sum(1 for s in stability_data.values() if s.trend == "stable"),
            "falling": sum(1 for s in stability_data.values() if s.trend == "falling"),
            "stability_by_ticker": {
                ticker: {
                    "current_rank": s.current_rank,
                    "trend": s.trend,
                    "volatility": float(s.volatility),
                    "weeks_in_top20": s.weeks_in_top20,
                }
                for ticker, s in stability_data.items()
            },
        }

        return flags, summary

    def _analyze_portfolio_fit(
        self,
        securities: List[SecurityContext],
        existing_holdings: Dict[str, Decimal],
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """Analyze portfolio fit with existing holdings."""
        flags: List[SanityFlag] = []

        existing_tickers = set(existing_holdings.keys())
        existing_sectors: Dict[str, List[str]] = {}

        # Build sector map for existing holdings (would need more data)
        # For now, track overlap

        overlaps = []
        new_names = []

        for sec in securities:
            if sec.rank is None or sec.rank > 20:
                continue

            if sec.ticker in existing_tickers:
                overlaps.append({
                    "ticker": sec.ticker,
                    "rank": sec.rank,
                    "current_position_mm": float(existing_holdings[sec.ticker]),
                })
            else:
                new_names.append({
                    "ticker": sec.ticker,
                    "rank": sec.rank,
                    "sector": sec.sector,
                })

        # Flag high overlap
        if len(overlaps) > 10:
            flags.append(SanityFlag(
                severity=FlagSeverity.MEDIUM,
                category=CheckCategory.DASHBOARD,
                ticker=None,
                check_name="high_portfolio_overlap",
                message=f"{len(overlaps)} of top 20 already held - limited new opportunities",
                details={
                    "overlap_count": len(overlaps),
                    "new_count": len(new_names),
                    "overlapping_tickers": [o["ticker"] for o in overlaps],
                },
                recommendation="Consider refreshing screening criteria",
            ))

        analysis = {
            "overlap_count": len(overlaps),
            "new_names_count": len(new_names),
            "overlaps": overlaps,
            "new_names": new_names,
        }

        return flags, analysis

    def _generate_risk_summary(
        self,
        securities: List[SecurityContext],
    ) -> Tuple[Dict[str, Any], List[SanityFlag]]:
        """Generate risk summary dashboard for top 20."""
        flags: List[SanityFlag] = []

        top20 = [s for s in securities if s.rank and s.rank <= 20]

        if not top20:
            return {"error": "No top 20 candidates"}, []

        # Calculate metrics
        runway_months = [
            float(s.runway_months) for s in top20
            if s.runway_months is not None
        ]
        avg_runway = sum(runway_months) / len(runway_months) if runway_months else 0

        # Binary catalysts (<3 months)
        binary_count = sum(
            1 for s in top20
            if s.days_to_catalyst is not None and s.days_to_catalyst <= 90
        )
        binary_pct = binary_count / len(top20) if top20 else 0

        # Sector concentration
        sector_counts: Dict[str, int] = {}
        for s in top20:
            if s.sector:
                sector_counts[s.sector] = sector_counts.get(s.sector, 0) + 1

        sector_concentrations = {
            sector: count / len(top20)
            for sector, count in sector_counts.items()
        }

        # Liquidity-weighted ADV
        adv_values = [
            float(s.adv_dollars) for s in top20
            if s.adv_dollars is not None
        ]
        avg_adv = sum(adv_values) / len(adv_values) if adv_values else 0

        # Flag concerning metrics
        if avg_runway < 9:
            flags.append(SanityFlag(
                severity=FlagSeverity.HIGH,
                category=CheckCategory.DASHBOARD,
                ticker=None,
                check_name="low_avg_runway",
                message=f"Top 20 average runway only {avg_runway:.1f} months",
                details={"avg_runway_months": avg_runway},
                recommendation="Portfolio has elevated dilution risk",
            ))

        if binary_pct > 0.40:
            flags.append(SanityFlag(
                severity=FlagSeverity.MEDIUM,
                category=CheckCategory.DASHBOARD,
                ticker=None,
                check_name="high_binary_concentration",
                message=f"{binary_pct:.0%} of top 20 have binary catalysts in 3 months",
                details={"binary_pct": binary_pct, "binary_count": binary_count},
                recommendation="High event risk concentration",
            ))

        for sector, concentration in sector_concentrations.items():
            if concentration > 0.40:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.DASHBOARD,
                    ticker=None,
                    check_name="sector_concentration",
                    message=f"{concentration:.0%} of top 20 in {sector}",
                    details={"sector": sector, "concentration": concentration},
                    recommendation="Sector concentration risk",
                ))

        summary = {
            "avg_runway_months": avg_runway,
            "binary_catalyst_pct": binary_pct,
            "binary_catalyst_count": binary_count,
            "sector_concentrations": sector_concentrations,
            "avg_adv_dollars": avg_adv,
            "top20_count": len(top20),
        }

        return summary, flags


def generate_ic_presentation_package(
    securities: List[SecurityContext],
    one_pagers: Dict[str, OnePagerContent],
    historical_snapshots: List[RankingSnapshot],
) -> Dict[str, Any]:
    """
    Generate complete IC presentation package.

    Args:
        securities: List of security contexts
        one_pagers: One-pager content by ticker
        historical_snapshots: Historical ranking data

    Returns:
        Complete presentation package
    """
    validator = ExecutiveDashboardValidator()

    result = validator.run_all_checks(
        securities=securities,
        one_pagers=one_pagers,
        historical_snapshots=historical_snapshots,
    )

    top10 = [s for s in securities if s.rank and s.rank <= 10]

    package = {
        "validation_result": result.to_dict(),
        "ic_ready": result.passed,
        "top10_summary": [
            {
                "ticker": s.ticker,
                "rank": s.rank,
                "composite_score": float(s.composite_score) if s.composite_score else None,
                "sector": s.sector,
                "lead_phase": s.lead_phase,
                "days_to_catalyst": s.days_to_catalyst,
            }
            for s in top10
        ],
        "risk_dashboard": result.metrics.get("risk_summary", {}),
        "ranking_stability": result.metrics.get("ranking_stability", {}),
        "flags_requiring_discussion": [
            f.to_dict() for f in result.flags
            if f.severity in (FlagSeverity.CRITICAL, FlagSeverity.HIGH)
        ],
    }

    return package
