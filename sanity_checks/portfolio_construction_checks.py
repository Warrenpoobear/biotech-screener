"""
Query 7.6: Portfolio Construction Reality Checks

Validates that top candidates are actually investable:

1. Liquidity Reality Checks
2. Risk Concentration Checks
3. Fund Mandate Compliance
4. Transaction Cost Estimation

These checks ensure the screening output is actionable for
a real investment portfolio.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FundMandate:
    """
    Fund investment mandate constraints.

    Immutable to prevent modification during validation.
    """
    name: str
    min_market_cap_mm: Decimal = Decimal("200")
    max_position_size_mm: Decimal = Decimal("50")
    max_adv_days_to_build: int = 5
    allowed_exchanges: frozenset = frozenset({"NYSE", "NASDAQ", "AMEX"})
    us_primary_only: bool = True
    max_sector_concentration_pct: Decimal = Decimal("0.30")
    aum_mm: Decimal = Decimal("775")  # Fund AUM in millions


@dataclass
class LiquidityAssessment:
    """Liquidity assessment for a position."""
    ticker: str
    adv_dollars: Decimal
    target_position_mm: Decimal
    days_to_build: Decimal
    estimated_impact_pct: Decimal
    investable: bool
    reason: Optional[str] = None


@dataclass
class ConcentrationRisk:
    """Concentration risk assessment."""
    concentration_type: str  # "sector", "indication", "catalyst_date"
    count: int
    tickers: List[str]
    exceeds_threshold: bool


class PortfolioConstructionChecker:
    """
    Portfolio construction reality checker.

    Validates that screening output is actionable for
    portfolio implementation.
    """

    # Impact model parameters
    IMPACT_CONSTANT = Decimal("0.1")  # 10 bps per ADV day
    SPREAD_ASSUMPTION = Decimal("0.001")  # 10 bps spread

    def __init__(
        self,
        mandate: Optional[FundMandate] = None,
        config: Optional[ThresholdConfig] = None,
    ) -> None:
        self.mandate = mandate or FundMandate(name="default")
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
        existing_holdings: Optional[Dict[str, Decimal]] = None,
    ) -> SanityCheckResult:
        """
        Run all portfolio construction checks.

        Args:
            securities: List of security contexts
            existing_holdings: Optional dict of ticker -> current position size

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []
        metrics: Dict[str, Any] = {}

        # 1. Liquidity Reality Checks
        liquidity_flags, liquidity_metrics = self._check_liquidity(securities)
        flags.extend(liquidity_flags)
        metrics["liquidity"] = liquidity_metrics

        # 2. Risk Concentration Checks
        concentration_flags, concentration_risks = self._check_concentration(securities)
        flags.extend(concentration_flags)
        metrics["concentration"] = {
            risk.concentration_type: {
                "count": risk.count,
                "tickers": risk.tickers,
                "exceeds_threshold": risk.exceeds_threshold,
            }
            for risk in concentration_risks
        }

        # 3. Fund Mandate Compliance
        mandate_flags = self._check_mandate_compliance(securities)
        flags.extend(mandate_flags)

        # 4. Transaction Cost Estimation
        cost_flags, cost_metrics = self._estimate_transaction_costs(securities)
        flags.extend(cost_flags)
        metrics["transaction_costs"] = cost_metrics

        # 5. Portfolio Fit Analysis (if existing holdings provided)
        if existing_holdings:
            fit_flags = self._check_portfolio_fit(securities, existing_holdings)
            flags.extend(fit_flags)

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
            check_name="portfolio_construction",
            category=CheckCategory.PORTFOLIO_CONSTRUCTION,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_liquidity(
        self,
        securities: List[SecurityContext],
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """
        Check liquidity reality.

        Flags:
        - Insufficient ADV for fund size
        - High market impact expected
        """
        flags: List[SanityFlag] = []
        assessments: List[LiquidityAssessment] = []

        # Calculate target position size (equal weight in top 20)
        target_position_mm = self.mandate.aum_mm / Decimal("20")

        for sec in securities:
            if sec.rank is None or sec.rank > 20:
                continue

            if sec.adv_dollars is None:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=sec.ticker,
                    check_name="missing_liquidity_data",
                    message=f"Top {sec.rank} candidate missing ADV data",
                    details={"rank": sec.rank},
                    recommendation="Cannot assess liquidity",
                ))
                continue

            # Calculate days to build position
            daily_capacity = sec.adv_dollars * Decimal("0.20")  # 20% of ADV
            if daily_capacity > 0:
                days_to_build = (target_position_mm * Decimal("1000000")) / daily_capacity
            else:
                days_to_build = Decimal("999")

            # Estimate market impact
            impact = self._estimate_impact(target_position_mm, sec.adv_dollars, days_to_build)

            investable = days_to_build <= self.mandate.max_adv_days_to_build
            reason = None if investable else f"Requires {days_to_build:.1f} days to build (max: {self.mandate.max_adv_days_to_build})"

            assessment = LiquidityAssessment(
                ticker=sec.ticker,
                adv_dollars=sec.adv_dollars,
                target_position_mm=target_position_mm,
                days_to_build=days_to_build,
                estimated_impact_pct=impact,
                investable=investable,
                reason=reason,
            )
            assessments.append(assessment)

            # Flag if not investable
            if not investable:
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=sec.ticker,
                    check_name="insufficient_liquidity",
                    message=f"Top {sec.rank} candidate: ADV ${sec.adv_dollars/1000000:.1f}M insufficient for ${self.mandate.aum_mm:.0f}M fund",
                    details={
                        "rank": sec.rank,
                        "adv_dollars": float(sec.adv_dollars),
                        "days_to_build": float(days_to_build),
                        "estimated_impact_pct": float(impact),
                    },
                    recommendation="Insufficient liquidity for fund size",
                ))

            # Flag high impact
            if impact > self.config.max_position_impact_pct:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=sec.ticker,
                    check_name="high_market_impact",
                    message=f"Estimated {impact:.1%} market impact for ${target_position_mm:.0f}M position",
                    details={
                        "rank": sec.rank,
                        "estimated_impact_pct": float(impact),
                        "threshold": float(self.config.max_position_impact_pct),
                    },
                    recommendation="High transaction cost expected",
                ))

        # Calculate metrics
        investable_count = sum(1 for a in assessments if a.investable)
        metrics = {
            "total_assessed": len(assessments),
            "investable_count": investable_count,
            "not_investable_count": len(assessments) - investable_count,
            "avg_days_to_build": float(sum(a.days_to_build for a in assessments) / len(assessments)) if assessments else 0,
        }

        return flags, metrics

    def _check_concentration(
        self,
        securities: List[SecurityContext],
    ) -> Tuple[List[SanityFlag], List[ConcentrationRisk]]:
        """
        Check risk concentration.

        Flags:
        - Sector concentration
        - Indication concentration
        - Catalyst date clustering
        """
        flags: List[SanityFlag] = []
        risks: List[ConcentrationRisk] = []

        top10 = [s for s in securities if s.rank is not None and s.rank <= 10]

        if not top10:
            return flags, risks

        # Sector concentration
        sector_counts: Dict[str, List[str]] = {}
        for sec in top10:
            if sec.sector:
                if sec.sector not in sector_counts:
                    sector_counts[sec.sector] = []
                sector_counts[sec.sector].append(sec.ticker)

        for sector, tickers in sector_counts.items():
            if len(tickers) > self.config.max_same_sector_in_top10:
                risk = ConcentrationRisk(
                    concentration_type="sector",
                    count=len(tickers),
                    tickers=tickers,
                    exceeds_threshold=True,
                )
                risks.append(risk)
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=None,
                    check_name="sector_concentration",
                    message=f"Top 10 has {len(tickers)} names in {sector} sector",
                    details={
                        "sector": sector,
                        "count": len(tickers),
                        "tickers": tickers,
                    },
                    recommendation="Regulatory/competitive risk correlation",
                ))

        # Indication concentration
        indication_counts: Dict[str, List[str]] = {}
        for sec in top10:
            if sec.indication:
                if sec.indication not in indication_counts:
                    indication_counts[sec.indication] = []
                indication_counts[sec.indication].append(sec.ticker)

        for indication, tickers in indication_counts.items():
            if len(tickers) >= 4:  # 4+ in same indication
                risk = ConcentrationRisk(
                    concentration_type="indication",
                    count=len(tickers),
                    tickers=tickers,
                    exceeds_threshold=True,
                )
                risks.append(risk)
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=None,
                    check_name="indication_concentration",
                    message=f"Top 10 has {len(tickers)} names in {indication}",
                    details={
                        "indication": indication,
                        "count": len(tickers),
                        "tickers": tickers,
                    },
                    recommendation="Regulatory/competitive risk correlation",
                ))

        # Catalyst date clustering
        catalyst_windows: Dict[str, List[str]] = {}  # window -> tickers
        for sec in top10:
            if sec.next_catalyst_date:
                # Round to 2-week windows
                window_key = sec.next_catalyst_date[:7]  # YYYY-MM approximation
                if window_key not in catalyst_windows:
                    catalyst_windows[window_key] = []
                catalyst_windows[window_key].append(sec.ticker)

        for window, tickers in catalyst_windows.items():
            if len(tickers) >= 3:
                risk = ConcentrationRisk(
                    concentration_type="catalyst_date",
                    count=len(tickers),
                    tickers=tickers,
                    exceeds_threshold=True,
                )
                risks.append(risk)
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=None,
                    check_name="catalyst_clustering",
                    message=f"Top 10 has {len(tickers)} names with catalysts in {window}",
                    details={
                        "window": window,
                        "count": len(tickers),
                        "tickers": tickers,
                    },
                    recommendation="Event risk clustering",
                ))

        return flags, risks

    def _check_mandate_compliance(
        self,
        securities: List[SecurityContext],
    ) -> List[SanityFlag]:
        """
        Check fund mandate compliance.

        Flags:
        - Market cap below mandate minimum
        - ADR restrictions
        """
        flags: List[SanityFlag] = []

        for sec in securities:
            if sec.rank is None or sec.rank > 20:
                continue

            # Market cap check
            if sec.market_cap_mm is not None:
                if sec.market_cap_mm < self.mandate.min_market_cap_mm:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                        ticker=sec.ticker,
                        check_name="below_mandate_threshold",
                        message=f"Market cap ${sec.market_cap_mm:.0f}M below mandate minimum ${self.mandate.min_market_cap_mm:.0f}M",
                        details={
                            "rank": sec.rank,
                            "market_cap_mm": float(sec.market_cap_mm),
                            "mandate_min": float(self.mandate.min_market_cap_mm),
                        },
                        recommendation="Below mandate threshold - not investable per mandate",
                    ))

        return flags

    def _estimate_transaction_costs(
        self,
        securities: List[SecurityContext],
    ) -> Tuple[List[SanityFlag], Dict[str, Any]]:
        """
        Estimate transaction costs for top 20.

        Returns cost metrics and flags for high-cost positions.
        """
        flags: List[SanityFlag] = []

        target_position_mm = self.mandate.aum_mm / Decimal("20")
        costs: List[Tuple[str, Decimal]] = []

        for sec in securities:
            if sec.rank is None or sec.rank > 20:
                continue

            if sec.adv_dollars is None:
                continue

            # Estimate total cost (spread + impact)
            daily_capacity = sec.adv_dollars * Decimal("0.20")
            if daily_capacity > 0:
                days_to_build = (target_position_mm * Decimal("1000000")) / daily_capacity
            else:
                days_to_build = Decimal("999")

            impact = self._estimate_impact(target_position_mm, sec.adv_dollars, days_to_build)
            total_cost = self.SPREAD_ASSUMPTION + impact

            costs.append((sec.ticker, total_cost))

            # Flag high cost
            if total_cost > Decimal("0.02"):  # >2% total cost
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=sec.ticker,
                    check_name="high_transaction_cost",
                    message=f"Estimated {total_cost:.1%} transaction cost for ${target_position_mm:.0f}M position",
                    details={
                        "rank": sec.rank,
                        "total_cost_pct": float(total_cost),
                        "spread_pct": float(self.SPREAD_ASSUMPTION),
                        "impact_pct": float(impact),
                    },
                    recommendation="High transaction cost - consider smaller position",
                ))

        metrics = {
            "avg_cost_pct": float(sum(c for _, c in costs) / len(costs)) if costs else 0,
            "max_cost_pct": float(max(c for _, c in costs)) if costs else 0,
            "positions_over_2pct": sum(1 for _, c in costs if c > Decimal("0.02")),
        }

        return flags, metrics

    def _check_portfolio_fit(
        self,
        securities: List[SecurityContext],
        existing_holdings: Dict[str, Decimal],
    ) -> List[SanityFlag]:
        """
        Check portfolio fit with existing holdings.

        Flags:
        - Overlap with existing positions
        - Redundant positions
        """
        flags: List[SanityFlag] = []

        for sec in securities:
            if sec.rank is None or sec.rank > 20:
                continue

            if sec.ticker in existing_holdings:
                current_position = existing_holdings[sec.ticker]
                flags.append(SanityFlag(
                    severity=FlagSeverity.LOW,
                    category=CheckCategory.PORTFOLIO_CONSTRUCTION,
                    ticker=sec.ticker,
                    check_name="existing_holding_overlap",
                    message=f"Top {sec.rank} candidate already held (${current_position:.1f}M)",
                    details={
                        "rank": sec.rank,
                        "current_position_mm": float(current_position),
                    },
                    recommendation="Existing holding - consider position sizing",
                ))

        return flags

    def _estimate_impact(
        self,
        position_mm: Decimal,
        adv_dollars: Decimal,
        days_to_build: Decimal,
    ) -> Decimal:
        """Estimate market impact using simple model."""
        if adv_dollars <= 0 or days_to_build <= 0:
            return Decimal("0.10")  # 10% default

        participation = (position_mm * Decimal("1000000")) / (adv_dollars * days_to_build)
        impact = self.IMPACT_CONSTANT * participation.sqrt() if participation > 0 else Decimal("0")
        return min(impact, Decimal("0.10"))


def generate_investable_capital_report(
    securities: List[SecurityContext],
    mandate: FundMandate,
) -> Dict[str, Any]:
    """
    Generate investable capital allocation report.

    Args:
        securities: List of security contexts
        mandate: Fund mandate constraints

    Returns:
        Detailed investability report
    """
    checker = PortfolioConstructionChecker(mandate=mandate)

    top20 = [s for s in securities if s.rank is not None and s.rank <= 20]

    report = {
        "mandate": {
            "name": mandate.name,
            "aum_mm": float(mandate.aum_mm),
            "min_market_cap_mm": float(mandate.min_market_cap_mm),
            "max_adv_days": mandate.max_adv_days_to_build,
        },
        "securities": [],
        "summary": {
            "total_candidates": len(top20),
            "investable": 0,
            "not_investable": 0,
        },
    }

    target_position_mm = mandate.aum_mm / Decimal("20")

    for sec in top20:
        sec_report: Dict[str, Any] = {
            "ticker": sec.ticker,
            "rank": sec.rank,
            "market_cap_mm": float(sec.market_cap_mm) if sec.market_cap_mm else None,
        }

        # Check investability
        investable = True
        issues = []

        if sec.market_cap_mm and sec.market_cap_mm < mandate.min_market_cap_mm:
            investable = False
            issues.append(f"Market cap ${sec.market_cap_mm:.0f}M below minimum")

        if sec.adv_dollars:
            daily_capacity = sec.adv_dollars * Decimal("0.20")
            if daily_capacity > 0:
                days_to_build = (target_position_mm * Decimal("1000000")) / daily_capacity
                if days_to_build > mandate.max_adv_days_to_build:
                    investable = False
                    issues.append(f"Requires {days_to_build:.1f} days to build position")
                sec_report["days_to_build"] = float(days_to_build)
        else:
            investable = False
            issues.append("Missing ADV data")

        sec_report["investable"] = investable
        sec_report["issues"] = issues
        report["securities"].append(sec_report)

        if investable:
            report["summary"]["investable"] += 1
        else:
            report["summary"]["not_investable"] += 1

    return report
