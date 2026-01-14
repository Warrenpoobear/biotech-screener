"""
section_generators.py - Section generators for institutional dossiers

Generates each major section of the dossier:
1. Executive Summary
2. Investment Thesis
3. Catalyst Analysis
4. Scientific/Clinical Review
5. Financial Analysis
6. Competitive Landscape
7. Risk Assessment
8. Position Sizing
9. Final Recommendation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .data_fetchers import DossierData

logger = logging.getLogger(__name__)


@dataclass
class PriceTarget:
    """Price target with probability weighting."""
    scenario: str  # "bull", "base", "bear"
    target_price: float
    probability: float
    upside_pct: float
    rationale: str


@dataclass
class RiskFactor:
    """Risk factor for assessment."""
    category: str
    description: str
    probability: float
    severity: str  # "high", "medium", "low"
    mitigation: Optional[str] = None


class DossierSectionGenerator:
    """
    Generates each section of the institutional dossier.

    Takes DossierData as input and generates markdown sections.
    """

    def __init__(self):
        self.sections: Dict[str, str] = {}

    def _format_currency(self, value: float, decimals: int = 1) -> str:
        """Format currency value."""
        if value >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        elif value >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        elif value >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        else:
            return f"${value:.2f}"

    def _get_recommendation(self, score: float) -> str:
        """Convert score to recommendation."""
        # Lower score = better in this system (rank-based)
        if score <= 15:
            return "STRONG BUY"
        elif score <= 25:
            return "BUY"
        elif score <= 40:
            return "HOLD"
        elif score <= 60:
            return "SELL"
        else:
            return "STRONG SELL"

    def _build_price_targets(
        self,
        current_price: float,
        score: float,
        clinical_stage: str,
    ) -> List[PriceTarget]:
        """Build price targets based on score and stage."""
        if current_price <= 0:
            return []

        # Adjust multipliers based on clinical stage
        stage_multipliers = {
            "phase_3": (2.0, 1.3, 0.6),  # bull, base, bear
            "phase_2": (2.5, 1.4, 0.5),
            "phase_1": (3.0, 1.5, 0.4),
            "preclinical": (4.0, 1.6, 0.3),
        }
        bull_mult, base_mult, bear_mult = stage_multipliers.get(
            clinical_stage, (2.0, 1.3, 0.6)
        )

        # Adjust probabilities based on score (lower = better)
        if score <= 20:
            bull_prob, base_prob, bear_prob = 0.35, 0.45, 0.20
        elif score <= 35:
            bull_prob, base_prob, bear_prob = 0.25, 0.50, 0.25
        else:
            bull_prob, base_prob, bear_prob = 0.20, 0.40, 0.40

        bull_price = current_price * bull_mult
        base_price = current_price * base_mult
        bear_price = current_price * bear_mult

        return [
            PriceTarget(
                scenario="bull",
                target_price=round(bull_price, 2),
                probability=bull_prob,
                upside_pct=round((bull_mult - 1) * 100, 1),
                rationale="Successful Phase 3 data with FDA approval pathway",
            ),
            PriceTarget(
                scenario="base",
                target_price=round(base_price, 2),
                probability=base_prob,
                upside_pct=round((base_mult - 1) * 100, 1),
                rationale="Mixed trial results with moderate progress",
            ),
            PriceTarget(
                scenario="bear",
                target_price=round(bear_price, 2),
                probability=bear_prob,
                upside_pct=round((bear_mult - 1) * 100, 1),
                rationale="Clinical setback or regulatory delay",
            ),
        ]

    def _build_risk_factors(self, data: DossierData) -> List[RiskFactor]:
        """Build risk factors from data."""
        risks = []

        # Clinical stage risk
        stage = data.clinical.lead_stage
        if stage == "phase_1":
            risks.append(RiskFactor(
                category="clinical",
                description="Lead program in Phase 1 - high development risk with ~90% historical failure rate",
                probability=0.90,
                severity="high",
                mitigation="Diversified pipeline may offset single-program risk",
            ))
        elif stage == "phase_2":
            risks.append(RiskFactor(
                category="clinical",
                description="Lead program in Phase 2 - significant development risk with ~70% historical failure rate",
                probability=0.70,
                severity="high",
                mitigation="Phase 2 data quality and endpoint selection critical",
            ))
        elif stage == "phase_3":
            risks.append(RiskFactor(
                category="clinical",
                description="Lead program in Phase 3 - moderate development risk with ~40% historical failure rate",
                probability=0.40,
                severity="medium",
                mitigation="Larger trials provide more statistical power",
            ))

        # Financial risk - runway
        runway = data.financial.runway_months
        if runway < 12:
            risks.append(RiskFactor(
                category="financial",
                description=f"Limited cash runway (~{runway:.0f} months) - high dilution risk",
                probability=0.85,
                severity="high",
                mitigation="Near-term financing required; watch for equity offerings",
            ))
        elif runway < 24:
            risks.append(RiskFactor(
                category="financial",
                description=f"Moderate runway (~{runway:.0f} months) - potential dilution",
                probability=0.50,
                severity="medium",
                mitigation="Monitor cash position and financing activities",
            ))

        # Catalyst risk
        if data.catalyst.severe_negatives > 0:
            risks.append(RiskFactor(
                category="regulatory",
                description=f"Recent severe negative catalyst events detected ({data.catalyst.severe_negatives})",
                probability=0.70,
                severity="high",
                mitigation="Review specific event details; may require position adjustment",
            ))

        # Market cap / liquidity risk
        if data.market.market_cap < 100_000_000:
            risks.append(RiskFactor(
                category="liquidity",
                description="Micro-cap stock - high volatility and liquidity risk",
                probability=0.60,
                severity="medium",
                mitigation="Use limit orders; scale in/out of positions gradually",
            ))

        return risks

    def generate_executive_summary(self, data: DossierData) -> str:
        """Generate executive summary section."""
        price = data.market.price
        market_cap = data.market.market_cap
        score = data.ranking.composite_score
        rank = data.ranking.rank

        recommendation = self._get_recommendation(score)
        price_targets = self._build_price_targets(price, score, data.clinical.lead_stage)

        # Calculate probability-weighted return
        pw_return = sum(pt.upside_pct * pt.probability for pt in price_targets)

        lines = [
            "## EXECUTIVE SUMMARY",
            "",
            f"**Recommendation:** {recommendation}",
            f"**Screening Rank:** #{rank} of 160 ranked securities",
            "",
            "### Key Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Current Price | ${price:.2f} |",
            f"| Market Cap | {self._format_currency(market_cap)} |",
            f"| Composite Score | {score:.2f} |",
            f"| Clinical Stage | {data.clinical.lead_stage.replace('_', ' ').title()} |",
            f"| Active Trials | {data.clinical.active_trials} |",
            f"| Cash Position | {self._format_currency(data.financial.cash)} |",
            f"| Runway | {data.financial.runway_months:.0f} months |",
            "",
        ]

        if price_targets:
            lines.extend([
                "### Price Target Analysis",
                "",
                "| Scenario | Target | Upside | Probability | Weighted |",
                "|----------|--------|--------|-------------|----------|",
            ])
            for pt in price_targets:
                weighted = pt.upside_pct * pt.probability
                lines.append(
                    f"| {pt.scenario.title()} | ${pt.target_price:.2f} | "
                    f"{pt.upside_pct:+.0f}% | {pt.probability:.0%} | {weighted:+.1f}% |"
                )
            lines.extend([
                "",
                f"**Probability-Weighted Expected Return: {pw_return:+.1f}%**",
                "",
            ])

        return "\n".join(lines)

    def generate_investment_thesis(self, data: DossierData) -> str:
        """Generate investment thesis section."""
        lines = [
            "## INVESTMENT THESIS & VARIANT PERCEPTION",
            "",
        ]

        # Build thesis points based on data
        thesis_points = []

        if data.ranking.rank <= 10:
            thesis_points.append(
                f"Top 10 ranked security (#{data.ranking.rank}) in algorithmic screening - "
                "indicates favorable risk/reward profile"
            )

        if data.clinical.active_trials >= 5:
            thesis_points.append(
                f"Robust clinical pipeline with {data.clinical.active_trials} active trials"
            )

        if data.catalyst.near_term_events > 0:
            thesis_points.append(
                f"{data.catalyst.near_term_events} near-term catalyst events within 90 days"
            )

        if data.financial.runway_months >= 24:
            thesis_points.append(
                f"Strong financial position with ~{data.financial.runway_months:.0f} months runway"
            )

        if thesis_points:
            lines.extend([
                "### Key Thesis Points",
                "",
            ])
            for point in thesis_points:
                lines.append(f"- {point}")
            lines.append("")

        # Variant perception
        if data.ranking.rank <= 20:
            lines.extend([
                "### Variant Perception",
                "",
                "> Market may be underappreciating near-term catalyst optionality and pipeline value. "
                "Algorithmic screening identifies favorable asymmetric risk/reward profile.",
                "",
            ])

        return "\n".join(lines)

    def generate_catalyst_analysis(self, data: DossierData) -> str:
        """Generate catalyst analysis section."""
        lines = [
            "## CATALYST ANALYSIS",
            "",
            f"**Catalyst Score:** {data.catalyst.catalyst_score:.1f}",
            f"**Near-Term Events:** {data.catalyst.near_term_events}",
            f"**Severe Negatives:** {data.catalyst.severe_negatives}",
            "",
        ]

        if data.catalyst.events:
            lines.extend([
                "### Upcoming Catalysts",
                "",
                "| Event | Type | Severity | Days Until |",
                "|-------|------|----------|------------|",
            ])
            for event in data.catalyst.events[:10]:
                lines.append(
                    f"| {event.get('description', event.get('event_type', 'Unknown'))[:40]} | "
                    f"{event.get('event_type', '-')} | "
                    f"{event.get('severity', '-')} | "
                    f"{event.get('days_until', '-')} |"
                )
            lines.append("")
        else:
            lines.extend([
                "### Upcoming Catalysts",
                "",
                "*No specific catalyst events identified in current screening window.*",
                "",
            ])

        return "\n".join(lines)

    def generate_clinical_review(self, data: DossierData) -> str:
        """Generate clinical/scientific review section."""
        lines = [
            "## SCIENTIFIC & CLINICAL REVIEW",
            "",
            f"**Clinical Stage:** {data.clinical.lead_stage.replace('_', ' ').title()}",
            f"**Active Trials:** {data.clinical.active_trials}",
            f"**Completed Trials:** {data.clinical.completed_trials}",
            f"**Total Trials:** {data.clinical.total_trials}",
            "",
        ]

        if data.clinical.trials:
            lines.extend([
                "### Key Trials",
                "",
                "| NCT ID | Phase | Status | Title |",
                "|--------|-------|--------|-------|",
            ])
            for trial in data.clinical.trials[:10]:
                nct_id = trial.get("nct_id", "-")
                phase = trial.get("phase", "-")
                status = trial.get("status", "-")
                title = trial.get("brief_title", trial.get("title", "-"))[:50]
                lines.append(f"| {nct_id} | {phase} | {status} | {title}... |")
            lines.append("")

        return "\n".join(lines)

    def generate_financial_analysis(self, data: DossierData) -> str:
        """Generate financial analysis section."""
        fin = data.financial

        lines = [
            "## FINANCIAL ANALYSIS",
            "",
            "### Balance Sheet Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Cash & Equivalents | {self._format_currency(fin.cash)} |",
            f"| Total Debt | {self._format_currency(fin.debt)} |",
            f"| Net Cash | {self._format_currency(fin.cash - fin.debt)} |",
            "",
            "### Cash Flow Analysis",
            "",
            f"| Metric | Value |",
            "|--------|-------|",
            f"| Revenue (TTM) | {self._format_currency(fin.revenue_ttm)} |",
            f"| Est. Monthly Burn | {self._format_currency(fin.burn_rate_monthly)} |",
            f"| Est. Runway | {fin.runway_months:.0f} months |",
            "",
        ]

        # Runway assessment
        if fin.runway_months < 12:
            lines.extend([
                "### Runway Assessment",
                "",
                "**WARNING:** Limited runway (<12 months). High probability of near-term "
                "financing event. Monitor for equity offerings, ATM programs, or debt facilities.",
                "",
            ])
        elif fin.runway_months < 24:
            lines.extend([
                "### Runway Assessment",
                "",
                "**MODERATE:** Runway of 12-24 months. Company will likely need to raise "
                "capital before major catalysts. Watch for financing announcements.",
                "",
            ])
        else:
            lines.extend([
                "### Runway Assessment",
                "",
                "**ADEQUATE:** Runway >24 months. Well-capitalized to execute on near-term "
                "clinical milestones without immediate dilution risk.",
                "",
            ])

        return "\n".join(lines)

    def generate_risk_assessment(self, data: DossierData) -> str:
        """Generate risk assessment section."""
        risks = self._build_risk_factors(data)

        lines = [
            "## RISK ASSESSMENT & MITIGATION",
            "",
        ]

        if risks:
            lines.extend([
                "### Risk Summary",
                "",
                "| Risk | Category | Probability | Severity |",
                "|------|----------|-------------|----------|",
            ])
            for risk in risks:
                desc = risk.description[:45] + "..." if len(risk.description) > 45 else risk.description
                lines.append(
                    f"| {desc} | {risk.category.title()} | "
                    f"{risk.probability:.0%} | {risk.severity.upper()} |"
                )
            lines.append("")

            # Detailed risks
            lines.append("### Detailed Risk Analysis")
            lines.append("")
            for risk in risks:
                lines.extend([
                    f"**{risk.category.title()} Risk:** {risk.description}",
                    f"- Probability: {risk.probability:.0%}",
                    f"- Severity: {risk.severity.upper()}",
                ])
                if risk.mitigation:
                    lines.append(f"- Mitigation: {risk.mitigation}")
                lines.append("")
        else:
            lines.extend([
                "*Insufficient data for detailed risk assessment.*",
                "",
            ])

        # Kill criteria
        lines.extend([
            "### Kill Criteria (Exit Triggers)",
            "",
            "1. Phase 3 trial failure on primary endpoint",
            "2. FDA complete response letter (CRL) without clear path forward",
            "3. Cash runway drops below 6 months without financing visibility",
            "4. Material fraud or governance failure",
            "5. Competitive approval that materially impairs market opportunity",
            "",
        ])

        return "\n".join(lines)

    def generate_position_sizing(self, data: DossierData) -> str:
        """Generate position sizing section."""
        score = data.ranking.composite_score
        rank = data.ranking.rank

        # Base position size on rank
        if rank <= 10:
            base_size = 2.0
            max_size = 3.0
        elif rank <= 30:
            base_size = 1.5
            max_size = 2.5
        elif rank <= 60:
            base_size = 1.0
            max_size = 2.0
        else:
            base_size = 0.5
            max_size = 1.0

        lines = [
            "## POSITION SIZING & PORTFOLIO CONSTRUCTION",
            "",
            "### Recommended Position Size",
            "",
            f"| Parameter | Value |",
            "|-----------|-------|",
            f"| Base Position | {base_size:.1f}% |",
            f"| Maximum Position | {max_size:.1f}% |",
            f"| Entry Tranches | 3 |",
            "",
            "### Entry Strategy",
            "",
            "1. **Initial Entry (40%):** Establish core position at current levels",
            "2. **Scale In (30%):** Add on 10-15% pullback or catalyst confirmation",
            "3. **Final Tranche (30%):** Complete position on technical breakout or positive data",
            "",
            "### Exit Strategy",
            "",
            "| Trigger | Action |",
            "|---------|--------|",
            "| Bull case achieved | Trim 50%, trail stop on remainder |",
            "| Base case achieved | Reduce to core position (50%) |",
            "| Bear case triggered | Exit 100% |",
            "| Stop loss (-25%) | Exit 100% |",
            "",
        ]

        return "\n".join(lines)

    def generate_final_recommendation(self, data: DossierData) -> str:
        """Generate final recommendation section."""
        score = data.ranking.composite_score
        rank = data.ranking.rank
        recommendation = self._get_recommendation(score)
        price = data.market.price

        price_targets = self._build_price_targets(price, score, data.clinical.lead_stage)
        pw_return = sum(pt.upside_pct * pt.probability for pt in price_targets)

        lines = [
            "## FINAL RECOMMENDATION",
            "",
            f"### {recommendation}",
            "",
        ]

        # Build supporting factors
        factors = []
        if rank <= 20:
            factors.append(f"Strong algorithmic ranking (#{rank}) indicates favorable screening profile")
        if data.clinical.active_trials >= 3:
            factors.append(f"Active clinical pipeline with {data.clinical.active_trials} ongoing trials")
        if data.catalyst.near_term_events > 0:
            factors.append(f"Near-term catalyst optionality with {data.catalyst.near_term_events} events")
        if data.financial.runway_months >= 18:
            factors.append(f"Adequate runway (~{data.financial.runway_months:.0f} months) to execute")
        if pw_return > 20:
            factors.append(f"Attractive probability-weighted return ({pw_return:+.1f}%)")

        if factors:
            lines.append("**Supporting Factors:**")
            lines.append("")
            for i, factor in enumerate(factors, 1):
                lines.append(f"{i}. {factor}")
            lines.append("")

        if price_targets:
            base_target = price_targets[1].target_price if len(price_targets) > 1 else price
            lines.extend([
                "### Investment Summary",
                "",
                f"| Metric | Value |",
                "|--------|-------|",
                f"| Recommendation | {recommendation} |",
                f"| Current Price | ${price:.2f} |",
                f"| Base Case Target | ${base_target:.2f} |",
                f"| Probability-Weighted Return | {pw_return:+.1f}% |",
                f"| Screening Rank | #{rank} |",
                "",
            ])

        lines.extend([
            "---",
            "",
            "*This analysis is algorithmically generated based on quantitative screening factors. "
            "It should be used as one input in a comprehensive investment process. "
            "Verify all data and conduct additional due diligence before making investment decisions.*",
        ])

        return "\n".join(lines)

    def generate_all_sections(self, data: DossierData) -> Dict[str, str]:
        """Generate all dossier sections."""
        return {
            "executive_summary": self.generate_executive_summary(data),
            "investment_thesis": self.generate_investment_thesis(data),
            "catalyst_analysis": self.generate_catalyst_analysis(data),
            "clinical_review": self.generate_clinical_review(data),
            "financial_analysis": self.generate_financial_analysis(data),
            "risk_assessment": self.generate_risk_assessment(data),
            "position_sizing": self.generate_position_sizing(data),
            "final_recommendation": self.generate_final_recommendation(data),
        }


if __name__ == "__main__":
    # Quick test
    from .data_fetchers import DossierDataFetcher

    fetcher = DossierDataFetcher(data_dir="./production_data")
    fetcher.load_base_data()

    data = fetcher.fetch_all_data("KMDA", "2026-01-14", "results_fixed.json")

    generator = DossierSectionGenerator()
    sections = generator.generate_all_sections(data)

    for name, content in sections.items():
        print(f"\n{'='*60}")
        print(f"SECTION: {name}")
        print(f"{'='*60}")
        print(content[:500] + "..." if len(content) > 500 else content)
