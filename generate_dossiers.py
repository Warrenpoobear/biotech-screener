import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional
import math

class DossierGenerator:
    def __init__(self, snapshot_path: str = 'outputs/universe_snapshot_latest.json', as_of_date: Optional[date] = None):
        """
        Initialize DossierGenerator.

        Args:
            snapshot_path: Path to universe snapshot JSON
            as_of_date: Point-in-time date for calculations (REQUIRED for determinism)
        """
        if as_of_date is None:
            raise ValueError(
                "as_of_date is REQUIRED for determinism. "
                "Do not use date.today() - pass explicit date."
            )
        self.as_of_date = as_of_date
        with open(snapshot_path) as f:
            self.companies = json.load(f)
        Path('dossiers/top5').mkdir(parents=True, exist_ok=True)
    
    def calculate_runway_months(self, company: Dict) -> Optional[float]:
        cash = company['financials'].get('cash')
        revenue = company['financials'].get('revenue_ttm')
        if not cash:
            return None
        if revenue and revenue > 0:
            estimated_quarterly_burn = revenue * 0.25 / 4
        else:
            estimated_quarterly_burn = 10e6
        if estimated_quarterly_burn <= 0:
            return None
        runway_months = (cash / estimated_quarterly_burn) * 3
        return round(runway_months, 1)
    
    def get_filing_freshness_days(self, company: Dict) -> Optional[int]:
        """Calculate days since filing using as_of_date (deterministic)."""
        try:
            filing_ts = company['provenance']['sources']['sec_edgar'].get('timestamp')
            if not filing_ts:
                return None
            filing_date = datetime.fromisoformat(filing_ts).date()
            days = (self.as_of_date - filing_date).days
            return days
        except (KeyError, ValueError, TypeError):
            return None
    
    def score_with_caps(self, company: Dict) -> Dict:
        cash = company['financials'].get('cash', 0)
        mcap = company['market_data']['market_cap']
        if cash > 0 and mcap > 0:
            raw_ratio = (cash / mcap) * 100
            capped_ratio = min(raw_ratio, 30.0)
            if capped_ratio > 0:
                compressed_score = math.log(1 + capped_ratio) * 2.5
            else:
                compressed_score = 0
        else:
            compressed_score = 0
        runway_months = self.calculate_runway_months(company)
        if runway_months:
            if runway_months > 24:
                runway_score = 10
            elif runway_months > 18:
                runway_score = 8
            elif runway_months > 12:
                runway_score = 6
            elif runway_months > 6:
                runway_score = 3
            else:
                runway_score = 1
        else:
            runway_score = 5
        stage_scores = {'commercial': 10, 'phase_3': 8, 'phase_2': 5, 'phase_1': 2, 'unknown': 0}
        stage_score = stage_scores.get(company['clinical']['lead_stage'], 0)
        active = company['clinical']['active_trials']
        total = company['clinical']['total_trials']
        if total > 0:
            pipeline_score = min((active / total) * 10 + min(active / 3, 5), 10)
        else:
            pipeline_score = 0
        volume_30d = company['market_data'].get('volume_avg_30d', 0)
        price = company['market_data']['price']
        dollar_volume = volume_30d * price if price > 0 else 0
        is_liquid = mcap > 1e9 and dollar_volume > 5e6
        composite = (stage_score * 0.30 + compressed_score * 0.25 + runway_score * 0.25 + pipeline_score * 0.20)
        return {
            'composite_score': round(composite, 2),
            'stage_score': stage_score,
            'cash_score': round(compressed_score, 2),
            'runway_score': runway_score,
            'pipeline_score': round(pipeline_score, 2),
            'runway_months': runway_months,
            'is_liquid': is_liquid,
            'dollar_volume_daily': dollar_volume
        }
    def generate_dossier(self, company: Dict, rank: int) -> str:
        ticker = company['ticker']
        scores = self.score_with_caps(company)
        lines = []
        lines.append(f"# Investment Dossier: {ticker}")
        lines.append(f"**Rank: #{rank} | Score: {scores['composite_score']:.2f}/10**")
        lines.append(f"**Generated: {self.as_of_date.isoformat()}**")
        lines.append("---")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        price = company['market_data']['price']
        mcap = company['market_data']['market_cap'] / 1e9
        cash = company['financials'].get('cash', 0) / 1e6 if company['financials'].get('cash') else 0
        lines.append(f"**{company['market_data'].get('company_name', ticker)}**")
        lines.append(f"- Ticker: {ticker}")
        lines.append(f"- Price: ${price:.2f}")
        lines.append(f"- Market Cap: ${mcap:.2f}B")
        lines.append(f"- Sector: {company['market_data'].get('sector', 'N/A')}")
        lines.append(f"- Industry: {company['market_data'].get('industry', 'N/A')}")
        lines.append("")
        lines.append("## Score Breakdown")
        lines.append("")
        lines.append("| Component | Score | Weight | Contribution |")
        lines.append("|-----------|-------|--------|-------------|")
        lines.append(f"| Stage | {scores['stage_score']:.1f}/10 | 30% | {scores['stage_score'] * 0.30:.2f} |")
        lines.append(f"| Cash Position | {scores['cash_score']:.1f}/10 | 25% | {scores['cash_score'] * 0.25:.2f} |")
        lines.append(f"| Runway | {scores['runway_score']:.1f}/10 | 25% | {scores['runway_score'] * 0.25:.2f} |")
        lines.append(f"| Pipeline | {scores['pipeline_score']:.1f}/10 | 20% | {scores['pipeline_score'] * 0.20:.2f} |")
        lines.append(f"| **Composite** | **{scores['composite_score']:.2f}/10** | 100% | **{scores['composite_score']:.2f}** |")
        lines.append("")
        lines.append("## Financial Reality Check")
        lines.append("")
        lines.append("**Balance Sheet:**")
        lines.append(f"- Cash & Equivalents: ${cash:.0f}M")
        debt = company['financials'].get('debt')
        if debt:
            lines.append(f"- Debt: ${debt/1e6:.0f}M")
            net_debt = company['financials'].get('net_debt', 0)
            lines.append(f"- Net Debt: ${net_debt/1e6:.0f}M")
        else:
            lines.append("- Debt: N/A")
        revenue = company['financials'].get('revenue_ttm')
        if revenue:
            lines.append(f"- TTM Revenue: ${revenue/1e6:.0f}M")
        else:
            lines.append("- TTM Revenue: N/A (pre-revenue)")
        lines.append("")
        lines.append("**Runway Analysis:**")
        if scores['runway_months']:
            lines.append(f"- Estimated Runway: ~{scores['runway_months']:.1f} months")
            if scores['runway_months'] < 12:
                lines.append("- WARNING: DILUTION RISK (runway < 12 months)")
            elif scores['runway_months'] < 18:
                lines.append("- WARNING: Watch for financing needs within 18 months")
            else:
                lines.append("- PASS: Adequate runway (>18 months)")
        else:
            lines.append("- Runway: Unable to calculate (missing data)")
        lines.append("")
        lines.append("**Coverage:**")
        lines.append(f"- Financial Data Coverage: {company['data_quality']['financial_coverage']:.0f}%")
        lines.append(f"- Overall Data Quality: {company['data_quality']['overall_coverage']:.0f}%")
        lines.append("")
        lines.append("## Pipeline Snapshot")
        lines.append("")
        lines.append(f"**Lead Stage:** {company['clinical']['lead_stage']}")
        lines.append(f"**Total Trials:** {company['clinical']['total_trials']}")
        lines.append(f"**Active Trials:** {company['clinical']['active_trials']}")
        lines.append(f"**Completed Trials:** {company['clinical']['completed_trials']}")
        lines.append("")
        by_phase = company['clinical'].get('by_phase', {})
        if by_phase:
            lines.append("**Trials by Phase:**")
            for phase, count in sorted(by_phase.items(), reverse=True):
                lines.append(f"- {phase}: {count}")
            lines.append("")
        conditions = company['clinical'].get('conditions', [])
        if conditions:
            lines.append(f"**Therapeutic Areas:** {', '.join(conditions[:5])}")
            lines.append("")
        lines.append("## Tradability Assessment")
        lines.append("")
        lines.append("**Liquidity Metrics:**")
        lines.append(f"- Market Cap: ${mcap:.2f}B")
        lines.append(f"- Avg Daily Volume: {company['market_data'].get('volume_avg_30d', 0):,} shares")
        lines.append(f"- Avg Daily Dollar Volume: ${scores['dollar_volume_daily']/1e6:.2f}M")
        if scores['is_liquid']:
            lines.append("- PASS: Meets liquidity requirements (>$1B mcap, >$5M daily volume)")
        else:
            lines.append("- WARNING: FAILS liquidity requirements - position sizing limited")
        lines.append("")
        prov = company.get('provenance', {})
        sources = prov.get('sources', {})
        lines.append("## Data Provenance")
        lines.append("")
        yahoo = sources.get('yahoo_finance', {})
        lines.append("**Market Data (Yahoo Finance)**")
        lines.append(f"- Timestamp: {yahoo.get('timestamp', 'N/A')}")
        lines.append(f"- Source: {yahoo.get('url', 'N/A')}")
        lines.append("")
        sec = sources.get('sec_edgar', {})
        if sec:
            lines.append("**Financial Data (SEC EDGAR)**")
            lines.append(f"- Timestamp: {sec.get('timestamp', 'N/A')}")
            lines.append(f"- CIK: {sec.get('cik', 'N/A')}")
            lines.append(f"- Source: {sec.get('url', 'N/A')}")
            days_old = self.get_filing_freshness_days(company)
            if days_old:
                if days_old > 120:
                    lines.append(f"- WARNING: STALE DATA - Filing is {days_old} days old (>120 days)")
                elif days_old > 90:
                    lines.append(f"- WARNING: Filing is {days_old} days old (>90 days)")
                else:
                    lines.append(f"- PASS: Filing is {days_old} days old (fresh)")
            lines.append("")
        lines.append("## Key Risk Factors")
        lines.append("")
        if scores['runway_months'] and scores['runway_months'] < 12:
            lines.append("- CRITICAL: Dilution risk (runway < 12 months)")
        stage = company['clinical']['lead_stage']
        if stage in ['phase_1', 'phase_2']:
            lines.append(f"- Clinical stage risk (lead asset in {stage})")
        if company['data_quality']['financial_coverage'] < 75:
            lines.append(f"- Data quality concern (financial coverage {company['data_quality']['financial_coverage']:.0f}%)")
        if not scores['is_liquid']:
            lines.append("- Liquidity risk (limited tradability)")
        lines.append("")
        lines.append("---")
        lines.append("*This dossier is algorithmically generated. Verify all data before making investment decisions.*")
        return "\n".join(lines)
    
    def generate_top5_dossiers(self):
        scored = []
        for company in self.companies:
            scores = self.score_with_caps(company)
            scored.append({'company': company, 'composite_score': scores['composite_score']})
        ranked = sorted(scored, key=lambda x: x['composite_score'], reverse=True)
        print("\nGenerating IC dossiers for Top 5...\n")
        for i, item in enumerate(ranked[:5], 1):
            company = item['company']
            ticker = company['ticker']
            dossier_text = self.generate_dossier(company, i)
            output_file = Path(f'dossiers/top5/{i}_{ticker}_dossier.md')
            with open(output_file, 'w') as f:
                f.write(dossier_text)
            print(f"Generated: {output_file.name}")
        print("\nAll dossiers generated in: dossiers/top5/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate investment dossiers")
    parser.add_argument(
        "--as-of-date",
        type=str,
        required=True,
        help="Point-in-time date (YYYY-MM-DD) - REQUIRED for determinism"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default='outputs/universe_snapshot_latest.json',
        help="Path to universe snapshot JSON"
    )
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of_date)
    generator = DossierGenerator(snapshot_path=args.snapshot, as_of_date=as_of)
    generator.generate_top5_dossiers()