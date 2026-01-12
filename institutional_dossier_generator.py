#!/usr/bin/env python3
"""
institutional_dossier_generator.py - Institutional Investment Dossier Generator

Generates comprehensive institutional-grade investment dossiers following a
standardized template structure for single stock analysis.

Template Structure:
1. Executive Summary (recommendation, targets, probability-weighted analysis)
2. Investment Thesis & Variant Perception
3. Catalyst Analysis (timeline, near/mid/long term)
4. Scientific & Clinical Review
5. Commercial Opportunity Assessment
6. Risk Framework (probability-weighted scenarios)
7. Valuation & Price Target Analysis
8. Institutional Positioning & Ownership
9. Technical Setup & Entry Strategy
10. Final Recommendation

Version: 1.0.0
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceTarget:
    """Price target with probability weighting."""
    scenario: str  # "bull", "base", "bear"
    target_price: float
    probability: float  # 0.0 to 1.0
    upside_pct: float
    rationale: str


@dataclass
class Catalyst:
    """Individual catalyst event."""
    event: str
    expected_date: str
    timeframe: str  # "near", "mid", "long"
    probability_of_success: float
    impact_on_success: str  # e.g., "+150%" or "$20 PT"
    impact_on_failure: str  # e.g., "-60%"
    key_metrics: List[str]


@dataclass
class RiskFactor:
    """Individual risk factor."""
    category: str  # "clinical", "regulatory", "commercial", "financial", "competitive"
    description: str
    probability: float  # 0.0 to 1.0
    impact_severity: str  # "high", "medium", "low"
    mitigation: Optional[str] = None


@dataclass
class InstitutionalHolder:
    """Institutional ownership entry."""
    name: str
    shares: int
    market_value: float
    pct_outstanding: float
    change_qoq: float  # Percent change quarter-over-quarter


@dataclass
class TechnicalLevel:
    """Technical analysis level."""
    level_type: str  # "support", "resistance", "entry", "target"
    price: float
    strength: str  # "strong", "moderate", "weak"
    notes: str


@dataclass
class DossierContent:
    """Complete dossier content structure."""
    # Header
    ticker: str
    company_name: str
    generation_date: str
    analyst_type: str = "Algorithmic Biotech Screener"

    # Executive Summary
    recommendation: str = "HOLD"  # "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
    current_price: float = 0.0
    market_cap: float = 0.0
    price_targets: List[PriceTarget] = field(default_factory=list)
    probability_weighted_return: float = 0.0
    investment_horizon: str = "12-18 months"
    key_thesis_points: List[str] = field(default_factory=list)

    # Scores
    composite_score: float = 0.0
    catalyst_score: float = 0.0
    probability_score: float = 0.0
    timing_score: float = 0.0
    governance_score: float = 0.0

    # Investment Thesis
    variant_perception: str = ""
    market_misunderstanding: str = ""
    thesis_summary: str = ""

    # Catalysts
    catalysts: List[Catalyst] = field(default_factory=list)

    # Clinical/Scientific
    lead_program: str = ""
    mechanism_of_action: str = ""
    clinical_stage: str = ""
    active_trials: int = 0
    total_trials: int = 0
    key_trial_data: List[Dict[str, Any]] = field(default_factory=list)
    competitive_advantage: str = ""

    # Commercial
    target_market_size: str = ""
    peak_sales_estimate: str = ""
    competition_assessment: str = ""
    partnership_status: str = ""

    # Financials
    cash_position: float = 0.0
    runway_months: float = 0.0
    burn_rate: float = 0.0
    revenue_ttm: float = 0.0
    debt: float = 0.0

    # Risks
    risk_factors: List[RiskFactor] = field(default_factory=list)

    # Valuation
    valuation_method: str = ""
    valuation_summary: str = ""

    # Institutional
    institutional_ownership_pct: float = 0.0
    top_holders: List[InstitutionalHolder] = field(default_factory=list)
    insider_activity: str = ""

    # Technical
    technical_levels: List[TechnicalLevel] = field(default_factory=list)
    technical_trend: str = ""
    volume_analysis: str = ""

    # Data Quality
    data_coverage: float = 0.0
    data_freshness: str = ""
    data_warnings: List[str] = field(default_factory=list)


# =============================================================================
# DOSSIER GENERATOR
# =============================================================================

class InstitutionalDossierGenerator:
    """
    Generates institutional-grade investment dossiers for biotech stocks.

    Integrates with:
    - Enhanced catalyst scoring engines
    - Universe snapshot data
    - Clinical trial data
    - Financial data
    - Institutional ownership data
    """

    def __init__(
        self,
        snapshot_path: str = "outputs/universe_snapshot_latest.json",
        output_dir: str = "dossiers/institutional",
    ):
        self.snapshot_path = Path(snapshot_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load universe data if available
        self.companies: Dict[str, Dict] = {}
        if self.snapshot_path.exists():
            with open(self.snapshot_path) as f:
                data = json.load(f)
                for company in data:
                    self.companies[company.get("ticker", "")] = company

    def _get_recommendation(self, score: float) -> str:
        """Convert composite score to recommendation."""
        if score >= 80:
            return "STRONG BUY"
        elif score >= 65:
            return "BUY"
        elif score >= 50:
            return "HOLD"
        elif score >= 35:
            return "SELL"
        else:
            return "STRONG SELL"

    def _calculate_probability_weighted_return(
        self,
        price_targets: List[PriceTarget],
    ) -> float:
        """Calculate probability-weighted expected return."""
        if not price_targets:
            return 0.0

        weighted_return = sum(
            pt.upside_pct * pt.probability
            for pt in price_targets
        )
        return round(weighted_return, 1)

    def _determine_catalyst_timeframe(self, days: Optional[int]) -> str:
        """Determine catalyst timeframe from days until event."""
        if days is None:
            return "long"
        elif days <= 90:
            return "near"
        elif days <= 365:
            return "mid"
        else:
            return "long"

    def _format_currency(self, value: float, decimals: int = 1) -> str:
        """Format currency value."""
        if value >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        elif value >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        elif value >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        else:
            return f"${value:.{decimals}f}"

    def _estimate_runway(self, cash: float, burn_rate: float) -> float:
        """Estimate runway in months."""
        if burn_rate <= 0 or cash <= 0:
            return 0.0
        return cash / burn_rate

    def build_dossier_content(
        self,
        ticker: str,
        enhanced_score: Optional[Dict[str, Any]] = None,
        trial_data: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> DossierContent:
        """
        Build complete dossier content from available data sources.

        Args:
            ticker: Stock ticker symbol
            enhanced_score: Enhanced catalyst score data
            trial_data: Clinical trial data
            additional_context: Any additional context for the dossier
        """
        company = self.companies.get(ticker, {})
        enhanced_score = enhanced_score or {}
        trial_data = trial_data or []
        additional_context = additional_context or {}

        # Extract base data
        market_data = company.get("market_data", {})
        financials = company.get("financials", {})
        clinical = company.get("clinical", {})
        data_quality = company.get("data_quality", {})

        # Calculate scores
        composite_score = float(enhanced_score.get("enhanced_score", 50))
        catalyst_score = float(enhanced_score.get("base_scores", {}).get("catalyst_score", 50))

        # Get probability and timing from enhanced score
        prob_data = enhanced_score.get("probability", {})
        timing_data = enhanced_score.get("timing", {})
        gov_data = enhanced_score.get("governance", {})

        probability_score = float(prob_data.get("pos", 0.5)) * 100
        timing_score = 50 + float(timing_data.get("cluster_convexity_bonus", 0))
        governance_score = 50 - float(gov_data.get("penalty", 0))

        # Current market data
        current_price = market_data.get("price", 0)
        market_cap = market_data.get("market_cap", 0)

        # Build price targets (algorithmic estimation)
        price_targets = self._build_price_targets(
            current_price,
            composite_score,
            probability_score,
            additional_context.get("analyst_targets", {}),
        )

        # Calculate probability-weighted return
        prob_weighted_return = self._calculate_probability_weighted_return(price_targets)

        # Build catalysts from trial data and events
        catalysts = self._build_catalysts(
            trial_data,
            clinical,
            additional_context.get("events", []),
        )

        # Build risk factors
        risk_factors = self._build_risk_factors(
            company,
            enhanced_score,
            additional_context,
        )

        # Financials
        cash = financials.get("cash", 0) or 0
        debt = financials.get("debt", 0) or 0
        revenue = financials.get("revenue_ttm", 0) or 0
        burn_rate = revenue * 0.25 / 12 if revenue > 0 else 10e6 / 12
        runway = self._estimate_runway(cash, burn_rate)

        # Build thesis and variant perception
        thesis_points, variant_perception, market_misunderstanding = self._build_thesis(
            company,
            composite_score,
            catalysts,
            additional_context,
        )

        # Technical levels (simplified)
        technical_levels = self._build_technical_levels(
            current_price,
            market_data,
        )

        # Data warnings
        data_warnings = []
        if data_quality.get("sec_stale", True):
            data_warnings.append("SEC financial data may be stale")
        if data_quality.get("financial_coverage", 100) < 50:
            data_warnings.append(f"Low financial data coverage: {data_quality.get('financial_coverage', 0):.0f}%")

        return DossierContent(
            # Header
            ticker=ticker,
            company_name=market_data.get("company_name", ticker),
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # Executive Summary
            recommendation=self._get_recommendation(composite_score),
            current_price=current_price,
            market_cap=market_cap,
            price_targets=price_targets,
            probability_weighted_return=prob_weighted_return,
            key_thesis_points=thesis_points,

            # Scores
            composite_score=composite_score,
            catalyst_score=catalyst_score,
            probability_score=probability_score,
            timing_score=timing_score,
            governance_score=governance_score,

            # Investment Thesis
            variant_perception=variant_perception,
            market_misunderstanding=market_misunderstanding,
            thesis_summary=additional_context.get("thesis_summary", ""),

            # Catalysts
            catalysts=catalysts,

            # Clinical
            lead_program=additional_context.get("lead_program", clinical.get("lead_stage", "Unknown")),
            mechanism_of_action=additional_context.get("moa", ""),
            clinical_stage=clinical.get("lead_stage", "unknown"),
            active_trials=clinical.get("active_trials", 0),
            total_trials=clinical.get("total_trials", 0),
            competitive_advantage=additional_context.get("competitive_advantage", ""),

            # Commercial
            target_market_size=additional_context.get("tam", ""),
            peak_sales_estimate=additional_context.get("peak_sales", ""),
            competition_assessment=additional_context.get("competition", ""),
            partnership_status=additional_context.get("partnerships", ""),

            # Financials
            cash_position=cash,
            runway_months=runway,
            burn_rate=burn_rate,
            revenue_ttm=revenue,
            debt=debt,

            # Risks
            risk_factors=risk_factors,

            # Valuation
            valuation_method=additional_context.get("valuation_method", "Risk-adjusted DCF / Comparable Analysis"),
            valuation_summary=additional_context.get("valuation_summary", ""),

            # Institutional
            institutional_ownership_pct=additional_context.get("inst_ownership_pct", 0),
            insider_activity=additional_context.get("insider_activity", ""),

            # Technical
            technical_levels=technical_levels,
            technical_trend=additional_context.get("technical_trend", ""),
            volume_analysis=additional_context.get("volume_analysis", ""),

            # Data Quality
            data_coverage=data_quality.get("overall_coverage", 0),
            data_freshness=additional_context.get("data_freshness", ""),
            data_warnings=data_warnings,
        )

    def _build_price_targets(
        self,
        current_price: float,
        composite_score: float,
        probability_score: float,
        analyst_targets: Dict[str, Any],
    ) -> List[PriceTarget]:
        """Build price targets based on score and any analyst data."""
        if current_price <= 0:
            return []

        # Use analyst targets if provided
        if analyst_targets:
            targets = []
            for scenario in ["bull", "base", "bear"]:
                if scenario in analyst_targets:
                    t = analyst_targets[scenario]
                    target_price = t.get("price", current_price)
                    upside = ((target_price / current_price) - 1) * 100
                    targets.append(PriceTarget(
                        scenario=scenario,
                        target_price=target_price,
                        probability=t.get("probability", 0.33),
                        upside_pct=round(upside, 1),
                        rationale=t.get("rationale", ""),
                    ))
            return targets

        # Algorithmic estimation based on scores
        # Higher composite score = higher upside potential
        score_factor = composite_score / 50  # 1.0 at score 50
        prob_factor = probability_score / 50  # 1.0 at 50%

        bull_upside = 50 + (score_factor * 50) + (prob_factor * 30)
        base_upside = 10 + (score_factor * 20)
        bear_downside = -20 - ((1 - prob_factor) * 30)

        # Probability distribution based on score
        if composite_score >= 70:
            bull_prob, base_prob, bear_prob = 0.35, 0.45, 0.20
        elif composite_score >= 50:
            bull_prob, base_prob, bear_prob = 0.25, 0.50, 0.25
        else:
            bull_prob, base_prob, bear_prob = 0.20, 0.40, 0.40

        return [
            PriceTarget(
                scenario="bull",
                target_price=round(current_price * (1 + bull_upside/100), 2),
                probability=bull_prob,
                upside_pct=round(bull_upside, 1),
                rationale="Successful catalyst execution with strong clinical data",
            ),
            PriceTarget(
                scenario="base",
                target_price=round(current_price * (1 + base_upside/100), 2),
                probability=base_prob,
                upside_pct=round(base_upside, 1),
                rationale="Mixed results with moderate progress",
            ),
            PriceTarget(
                scenario="bear",
                target_price=round(current_price * (1 + bear_downside/100), 2),
                probability=bear_prob,
                upside_pct=round(bear_downside, 1),
                rationale="Clinical setback or regulatory delay",
            ),
        ]

    def _build_catalysts(
        self,
        trial_data: List[Dict],
        clinical: Dict,
        events: List[Dict],
    ) -> List[Catalyst]:
        """Build catalyst list from trial and event data."""
        catalysts = []

        # From events
        for event in events:
            timeframe = self._determine_catalyst_timeframe(
                event.get("days_until")
            )
            catalysts.append(Catalyst(
                event=event.get("description", event.get("event_type", "Unknown")),
                expected_date=event.get("event_date", "TBD"),
                timeframe=timeframe,
                probability_of_success=event.get("probability", 0.5),
                impact_on_success=event.get("upside", "Positive"),
                impact_on_failure=event.get("downside", "Negative"),
                key_metrics=event.get("key_metrics", []),
            ))

        # From trial data
        for trial in trial_data[:5]:  # Top 5 trials
            phase = trial.get("phase", "")
            completion = trial.get("primary_completion_date", "")
            catalysts.append(Catalyst(
                event=f"{phase} Trial: {trial.get('brief_title', 'Unknown')}",
                expected_date=completion or "TBD",
                timeframe=self._determine_catalyst_timeframe(
                    trial.get("days_to_completion")
                ),
                probability_of_success=trial.get("pos", 0.5),
                impact_on_success="+30-50%",
                impact_on_failure="-20-40%",
                key_metrics=[trial.get("primary_endpoint", "Primary endpoint")],
            ))

        return catalysts

    def _build_risk_factors(
        self,
        company: Dict,
        enhanced_score: Dict,
        additional_context: Dict,
    ) -> List[RiskFactor]:
        """Build risk factors from company data and scores."""
        risks = []

        clinical = company.get("clinical", {})
        financials = company.get("financials", {})
        data_quality = company.get("data_quality", {})

        # Clinical risk based on stage
        stage = clinical.get("lead_stage", "unknown")
        if stage in ["phase_1", "phase_2"]:
            risks.append(RiskFactor(
                category="clinical",
                description=f"Lead program in {stage.replace('_', ' ').title()} - high development risk",
                probability=0.6 if stage == "phase_1" else 0.45,
                impact_severity="high",
                mitigation="Diversified pipeline may offset single-program risk",
            ))

        # Financial risk - runway
        cash = financials.get("cash", 0) or 0
        if cash > 0:
            runway = self._estimate_runway(cash, 10e6/12)
            if runway < 12:
                risks.append(RiskFactor(
                    category="financial",
                    description=f"Limited cash runway (~{runway:.0f} months) - dilution risk",
                    probability=0.8,
                    impact_severity="high",
                    mitigation="May require near-term financing",
                ))
            elif runway < 24:
                risks.append(RiskFactor(
                    category="financial",
                    description=f"Moderate runway (~{runway:.0f} months) - monitor cash position",
                    probability=0.4,
                    impact_severity="medium",
                ))

        # Governance risk from black swans
        gov_data = enhanced_score.get("governance", {})
        if int(gov_data.get("n_black_swans", 0) or 0) > 0:
            risks.append(RiskFactor(
                category="regulatory",
                description="Recent adverse event detected (black swan signal)",
                probability=0.7,
                impact_severity="high",
            ))

        # Data quality risk
        if data_quality.get("overall_coverage", 100) < 60:
            risks.append(RiskFactor(
                category="data",
                description=f"Incomplete data coverage ({data_quality.get('overall_coverage', 0):.0f}%)",
                probability=0.3,
                impact_severity="medium",
                mitigation="Verify key assumptions with primary sources",
            ))

        # Add any custom risks from context
        for risk in additional_context.get("additional_risks", []):
            risks.append(RiskFactor(
                category=risk.get("category", "other"),
                description=risk.get("description", ""),
                probability=risk.get("probability", 0.5),
                impact_severity=risk.get("severity", "medium"),
                mitigation=risk.get("mitigation"),
            ))

        return risks

    def _build_thesis(
        self,
        company: Dict,
        composite_score: float,
        catalysts: List[Catalyst],
        additional_context: Dict,
    ) -> Tuple[List[str], str, str]:
        """Build thesis points and variant perception."""
        thesis_points = []

        clinical = company.get("clinical", {})
        market_data = company.get("market_data", {})

        # Score-based thesis
        if composite_score >= 70:
            thesis_points.append(f"Strong algorithmic score ({composite_score:.1f}) indicates favorable risk/reward")

        # Pipeline strength
        active = clinical.get("active_trials", 0)
        if active >= 5:
            thesis_points.append(f"Robust pipeline with {active} active trials")

        # Near-term catalysts
        near_catalysts = [c for c in catalysts if c.timeframe == "near"]
        if near_catalysts:
            thesis_points.append(f"{len(near_catalysts)} near-term catalysts within 90 days")

        # Custom thesis points
        thesis_points.extend(additional_context.get("thesis_points", []))

        # Variant perception
        variant = additional_context.get("variant_perception", "")
        if not variant and composite_score >= 65:
            variant = "Market may be underappreciating near-term catalyst optionality"

        # Market misunderstanding
        misunderstanding = additional_context.get("market_misunderstanding", "")
        if not misunderstanding and near_catalysts:
            misunderstanding = "Timing and probability of successful catalyst execution may be mispriced"

        return thesis_points, variant, misunderstanding

    def _build_technical_levels(
        self,
        current_price: float,
        market_data: Dict,
    ) -> List[TechnicalLevel]:
        """Build simplified technical levels."""
        if current_price <= 0:
            return []

        levels = []

        # Simple support/resistance based on round numbers and % moves
        support_1 = round(current_price * 0.90, 2)
        support_2 = round(current_price * 0.75, 2)
        resistance_1 = round(current_price * 1.15, 2)
        resistance_2 = round(current_price * 1.30, 2)

        levels = [
            TechnicalLevel("resistance", resistance_2, "moderate", "30% above current"),
            TechnicalLevel("resistance", resistance_1, "moderate", "15% above current"),
            TechnicalLevel("support", support_1, "moderate", "10% below current"),
            TechnicalLevel("support", support_2, "strong", "25% below current"),
        ]

        # 52-week levels if available
        high_52w = market_data.get("52_week_high")
        low_52w = market_data.get("52_week_low")

        if high_52w:
            levels.insert(0, TechnicalLevel("resistance", high_52w, "strong", "52-week high"))
        if low_52w:
            levels.append(TechnicalLevel("support", low_52w, "strong", "52-week low"))

        return levels

    def generate_markdown(self, content: DossierContent) -> str:
        """Generate markdown dossier from content."""
        lines = []

        # Header
        lines.extend([
            f"# INSTITUTIONAL INVESTMENT DOSSIER: {content.company_name} ({content.ticker})",
            "",
            f"**Report Type:** Algorithmic Biotech Analysis",
            f"**Generated:** {content.generation_date}",
            f"**Source:** {content.analyst_type}",
            "",
            "---",
            "",
        ])

        # Executive Summary
        lines.extend([
            "## 1. EXECUTIVE SUMMARY",
            "",
            f"### Recommendation: **{content.recommendation}**",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Current Price | ${content.current_price:.2f} |",
            f"| Market Cap | {self._format_currency(content.market_cap)} |",
            f"| Composite Score | {content.composite_score:.1f}/100 |",
            f"| Investment Horizon | {content.investment_horizon} |",
            "",
        ])

        # Price Targets
        if content.price_targets:
            lines.extend([
                "### Price Targets & Probability-Weighted Analysis",
                "",
                "| Scenario | Target | Upside | Probability | Weighted |",
                "|----------|--------|--------|-------------|----------|",
            ])
            for pt in content.price_targets:
                weighted = pt.upside_pct * pt.probability
                lines.append(
                    f"| {pt.scenario.title()} | ${pt.target_price:.2f} | "
                    f"{pt.upside_pct:+.1f}% | {pt.probability:.0%} | {weighted:+.1f}% |"
                )
            lines.extend([
                "",
                f"**Probability-Weighted Expected Return: {content.probability_weighted_return:+.1f}%**",
                "",
            ])

        # Key Thesis Points
        if content.key_thesis_points:
            lines.append("### Key Investment Thesis Points")
            lines.append("")
            for point in content.key_thesis_points:
                lines.append(f"- {point}")
            lines.append("")

        # Score Breakdown
        lines.extend([
            "### Score Breakdown",
            "",
            "| Component | Score | Weight |",
            "|-----------|-------|--------|",
            f"| Catalyst Quality | {content.catalyst_score:.1f} | 40% |",
            f"| Probability of Success | {content.probability_score:.1f} | 25% |",
            f"| Timing Optimization | {content.timing_score:.1f} | 20% |",
            f"| Governance/Risk | {content.governance_score:.1f} | 15% |",
            f"| **Composite** | **{content.composite_score:.1f}** | 100% |",
            "",
        ])

        # Investment Thesis
        lines.extend([
            "---",
            "",
            "## 2. INVESTMENT THESIS & VARIANT PERCEPTION",
            "",
        ])

        if content.variant_perception:
            lines.extend([
                "### Variant Perception",
                "",
                f"> {content.variant_perception}",
                "",
            ])

        if content.market_misunderstanding:
            lines.extend([
                "### What the Market is Missing",
                "",
                content.market_misunderstanding,
                "",
            ])

        if content.thesis_summary:
            lines.extend([
                "### Thesis Summary",
                "",
                content.thesis_summary,
                "",
            ])

        # Catalyst Analysis
        lines.extend([
            "---",
            "",
            "## 3. CATALYST ANALYSIS",
            "",
        ])

        if content.catalysts:
            # Group by timeframe
            near = [c for c in content.catalysts if c.timeframe == "near"]
            mid = [c for c in content.catalysts if c.timeframe == "mid"]
            long_term = [c for c in content.catalysts if c.timeframe == "long"]

            if near:
                lines.extend([
                    "### Near-Term Catalysts (0-90 days)",
                    "",
                ])
                for c in near:
                    lines.extend([
                        f"**{c.event}**",
                        f"- Expected: {c.expected_date}",
                        f"- Probability of Success: {c.probability_of_success:.0%}",
                        f"- Impact on Success: {c.impact_on_success}",
                        f"- Impact on Failure: {c.impact_on_failure}",
                        "",
                    ])

            if mid:
                lines.extend([
                    "### Mid-Term Catalysts (3-12 months)",
                    "",
                ])
                for c in mid:
                    lines.extend([
                        f"**{c.event}**",
                        f"- Expected: {c.expected_date}",
                        f"- Probability of Success: {c.probability_of_success:.0%}",
                        "",
                    ])

            if long_term:
                lines.extend([
                    "### Long-Term Catalysts (12+ months)",
                    "",
                ])
                for c in long_term:
                    lines.append(f"- {c.event} ({c.expected_date})")
                lines.append("")
        else:
            lines.extend(["*No specific catalysts identified*", ""])

        # Scientific & Clinical Review
        lines.extend([
            "---",
            "",
            "## 4. SCIENTIFIC & CLINICAL REVIEW",
            "",
            f"**Clinical Stage:** {content.clinical_stage.replace('_', ' ').title()}",
            f"**Active Trials:** {content.active_trials}/{content.total_trials}",
            "",
        ])

        if content.lead_program:
            lines.append(f"**Lead Program:** {content.lead_program}")
            lines.append("")

        if content.mechanism_of_action:
            lines.extend([
                "### Mechanism of Action",
                "",
                content.mechanism_of_action,
                "",
            ])

        if content.competitive_advantage:
            lines.extend([
                "### Competitive Advantage",
                "",
                content.competitive_advantage,
                "",
            ])

        # Commercial Opportunity
        lines.extend([
            "---",
            "",
            "## 5. COMMERCIAL OPPORTUNITY ASSESSMENT",
            "",
        ])

        if content.target_market_size:
            lines.append(f"**Target Addressable Market:** {content.target_market_size}")
        if content.peak_sales_estimate:
            lines.append(f"**Peak Sales Estimate:** {content.peak_sales_estimate}")
        if content.partnership_status:
            lines.append(f"**Partnership Status:** {content.partnership_status}")
        if content.competition_assessment:
            lines.extend([
                "",
                "### Competitive Landscape",
                "",
                content.competition_assessment,
            ])
        lines.append("")

        # Risk Framework
        lines.extend([
            "---",
            "",
            "## 6. RISK FRAMEWORK",
            "",
        ])

        if content.risk_factors:
            lines.extend([
                "| Risk | Category | Probability | Severity |",
                "|------|----------|-------------|----------|",
            ])
            for rf in content.risk_factors:
                lines.append(
                    f"| {rf.description[:50]}{'...' if len(rf.description) > 50 else ''} | "
                    f"{rf.category.title()} | {rf.probability:.0%} | {rf.impact_severity.upper()} |"
                )
            lines.append("")

            # Detailed risks
            for rf in content.risk_factors:
                lines.extend([
                    f"### {rf.category.title()} Risk",
                    "",
                    f"**{rf.description}**",
                    f"- Probability: {rf.probability:.0%}",
                    f"- Severity: {rf.impact_severity.upper()}",
                ])
                if rf.mitigation:
                    lines.append(f"- Mitigation: {rf.mitigation}")
                lines.append("")
        else:
            lines.extend(["*Risk assessment data not available*", ""])

        # Valuation
        lines.extend([
            "---",
            "",
            "## 7. VALUATION & PRICE TARGET ANALYSIS",
            "",
            f"**Methodology:** {content.valuation_method}",
            "",
        ])

        if content.valuation_summary:
            lines.extend([content.valuation_summary, ""])

        # Recap price targets
        if content.price_targets:
            lines.extend([
                "### Scenario Analysis",
                "",
            ])
            for pt in content.price_targets:
                lines.append(f"- **{pt.scenario.title()} Case (${pt.target_price:.2f}):** {pt.rationale}")
            lines.append("")

        # Institutional Positioning
        lines.extend([
            "---",
            "",
            "## 8. INSTITUTIONAL POSITIONING & OWNERSHIP",
            "",
        ])

        if content.institutional_ownership_pct > 0:
            lines.append(f"**Institutional Ownership:** {content.institutional_ownership_pct:.1f}%")
            lines.append("")

        if content.top_holders:
            lines.extend([
                "### Top Institutional Holders",
                "",
                "| Institution | Shares | Value | % Outstanding | QoQ Change |",
                "|-------------|--------|-------|---------------|------------|",
            ])
            for h in content.top_holders[:10]:
                lines.append(
                    f"| {h.name} | {h.shares:,} | {self._format_currency(h.market_value)} | "
                    f"{h.pct_outstanding:.2f}% | {h.change_qoq:+.1f}% |"
                )
            lines.append("")

        if content.insider_activity:
            lines.extend([
                "### Insider Activity",
                "",
                content.insider_activity,
                "",
            ])

        # Technical Setup
        lines.extend([
            "---",
            "",
            "## 9. TECHNICAL SETUP & ENTRY STRATEGY",
            "",
        ])

        if content.technical_trend:
            lines.append(f"**Trend:** {content.technical_trend}")
            lines.append("")

        if content.technical_levels:
            lines.extend([
                "### Key Levels",
                "",
                "| Level Type | Price | Strength | Notes |",
                "|------------|-------|----------|-------|",
            ])
            for tl in content.technical_levels:
                lines.append(
                    f"| {tl.level_type.title()} | ${tl.price:.2f} | {tl.strength.title()} | {tl.notes} |"
                )
            lines.append("")

        if content.volume_analysis:
            lines.extend([
                "### Volume Analysis",
                "",
                content.volume_analysis,
                "",
            ])

        # Final Recommendation
        lines.extend([
            "---",
            "",
            "## 10. FINAL RECOMMENDATION",
            "",
            f"### {content.recommendation}",
            "",
            f"**Target:** ${content.price_targets[0].target_price:.2f} (base case)" if content.price_targets else "",
            f"**Probability-Weighted Return:** {content.probability_weighted_return:+.1f}%",
            f"**Composite Score:** {content.composite_score:.1f}/100",
            "",
        ])

        # Data Quality Warning
        if content.data_warnings:
            lines.extend([
                "---",
                "",
                "## DATA QUALITY NOTES",
                "",
            ])
            for warning in content.data_warnings:
                lines.append(f"- {warning}")
            lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            f"*Generated by {content.analyst_type} on {content.generation_date}*",
            "",
            "*This report is algorithmically generated for informational purposes only. "
            "Not financial advice. Verify all data before making investment decisions.*",
        ])

        return "\n".join(lines)

    def generate_dossier(
        self,
        ticker: str,
        enhanced_score: Optional[Dict[str, Any]] = None,
        trial_data: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> str:
        """
        Generate complete institutional dossier for a ticker.

        Args:
            ticker: Stock ticker symbol
            enhanced_score: Enhanced catalyst score data
            trial_data: Clinical trial data
            additional_context: Additional context for customization
            save: Whether to save to file

        Returns:
            Markdown formatted dossier
        """
        content = self.build_dossier_content(
            ticker=ticker,
            enhanced_score=enhanced_score,
            trial_data=trial_data,
            additional_context=additional_context,
        )

        markdown = self.generate_markdown(content)

        if save:
            output_path = self.output_dir / f"{ticker}_institutional_dossier.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"Dossier saved: {output_path}")

        return markdown


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for dossier generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate institutional investment dossiers"
    )
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument(
        "--snapshot",
        default="outputs/universe_snapshot_latest.json",
        help="Path to universe snapshot",
    )
    parser.add_argument(
        "--output-dir",
        default="dossiers/institutional",
        help="Output directory for dossiers",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file, just print",
    )

    args = parser.parse_args()

    generator = InstitutionalDossierGenerator(
        snapshot_path=args.snapshot,
        output_dir=args.output_dir,
    )

    dossier = generator.generate_dossier(
        ticker=args.ticker.upper(),
        save=not args.no_save,
    )

    if args.no_save:
        print(dossier)


if __name__ == "__main__":
    main()
