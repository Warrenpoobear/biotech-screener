import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import math

class DossierGenerator:
    def __init__(self, snapshot_path: str = 'outputs/universe_snapshot_latest.json'):
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
        try:
            filing_ts = company['provenance']['sources']['sec_edgar'].get('timestamp')
            if not filing_ts:
                return None
            filing_date = datetime.fromisoformat(filing_ts)
            days = (datetime.now() - filing_date).days
            return days
        except:
            return None
    
    def score_with_caps(self, company: Dict) -> Dict:
        cash = company['financials'].get('cash', 0)
        mcap = company['market_data']['market_cap']
        if cash > 0 and mcap > 0:
            raw_ratio = (cash / mcap) * 100
            capped_ratio = min(raw_ratio, 30.0)
            compressed_score = math.log(1 + capped_ratio) * 2.5 if capped_ratio > 0 else 0
        else:
            compressed_score = 0
        runway_months = self.calculate_runway_months(company)
        if runway_months:
            runway_score = 10 if runway_months > 24 else 8 if runway_months > 18 else 6 if runway_months > 12 else 3 if runway_months > 6 else 1
        else:
            runway_score = 5
        stage_scores = {'commercial': 10, 'phase_3': 8, 'phase_2': 5, 'phase_1': 2, 'unknown': 0}
        stage_score = stage_scores.get(company['clinical']['lead_stage'], 0)
        active = company['clinical']['active_trials']
        total = company['clinical']['total_trials']
        pipeline_score = min((active / total) * 10 + min(active / 3, 5), 10) if total > 0 else 0
        volume_30d = company['market_data'].get('volume_avg_30d', 0)
        price = company['market_data']['price']
        dollar_volume = volume_30d * price if price > 0 else 0
        is_liquid = mcap > 1e9 and dollar_volume > 5e6
        composite = stage_score * 0.30 + compressed_score * 0.25 + runway_score * 0.25 + pipeline_score * 0.20
        return {
            'composite_score': round(composite, 2), 'stage_score': stage_score,
            'cash_score': round(compressed_score, 2), 'runway_score': runway_score,
            'pipeline_score': round(pipeline_score, 2), 'runway_months': runway_months,
            'is_liquid': is_liquid, 'dollar_volume_daily': dollar_volume
        }
    
    def generate_dossier(self, company: Dict, rank: int) -> str:
        ticker = company['ticker']
        scores = self.score_with_caps(company)
        lines = [f"# Investment Dossier: {ticker}", f"**Rank: #{rank} | Score: {scores['composite_score']:.2f}/10**", 
                 f"**Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**", "---", "", "## Executive Summary", ""]
        price = company['market_data']['price']
        mcap = company['market_data']['market_cap'] / 1e9
        cash = company['financials'].get('cash', 0) / 1e6 if company['financials'].get('cash') else 0
        lines.extend([f"**{company['market_data'].get('company_name', ticker)}**", f"- Ticker: {ticker}",
                      f"- Price: ${price:.2f}", f"- Market Cap: ${mcap:.2f}B",
                      f"- Sector: {company['market_data'].get('sector', 'N/A')}",
                      f"- Industry: {company['market_data'].get('industry', 'N/A')}", "",
                      "## Score Breakdown", "", "| Component | Score | Weight | Contribution |",
                      "|-----------|-------|--------|-------------|",
                      f"| Stage | {scores['stage_score']:.1f}/10 | 30% | {scores['stage_score'] * 0.30:.2f} |",
                      f"| Cash Position | {scores['cash_score']:.1f}/10 | 25% | {scores['cash_score'] * 0.25:.2f} |",
                      f"| Runway | {scores['runway_score']:.1f}/10 | 25% | {scores['runway_score'] * 0.25:.2f} |",
                      f"| Pipeline | {scores['pipeline_score']:.1f}/10 | 20% | {scores['pipeline_score'] * 0.20:.2f} |",
                      f"| **Composite** | **{scores['composite_score']:.2f}/10** | 100% | **{scores['composite_score']:.2f}** |",
                      "", "## Financial Reality Check", "", "**Balance Sheet:**", f"- Cash: ${cash:.0f}M"])
        debt = company['financials'].get('debt')
        lines.append(f"- Debt: ${debt/1e6:.0f}M" if debt else "- Debt: N/A")
        revenue = company['financials'].get('revenue_ttm')
        lines.append(f"- Revenue: ${revenue/1e6:.0f}M" if revenue else "- Revenue: N/A (pre-revenue)")
        lines.extend(["", "**Runway:**"])
        if scores['runway_months']:
            lines.append(f"- ~{scores['runway_months']:.1f} months")
            if scores['runway_months'] < 12:
                lines.append("- WARNING: DILUTION RISK")
        else:
            lines.append("- Unable to calculate")
        lines.extend(["", f"**Data Quality: {company['data_quality']['overall_coverage']:.0f}%**", "",
                      "## Pipeline", "", f"- Lead: {company['clinical']['lead_stage']}",
                      f"- Active Trials: {company['clinical']['active_trials']}/{company['clinical']['total_trials']}", ""])
        lines.extend(["## Tradability", "", f"- Market Cap: ${mcap:.2f}B",
                      f"- Daily Volume: ${scores['dollar_volume_daily']/1e6:.2f}M",
                      f"- Status: {'PASS' if scores['is_liquid'] else 'FAIL (illiquid)'}", ""])
        sec = company.get('provenance', {}).get('sources', {}).get('sec_edgar', {})
        if sec:
            days_old = self.get_filing_freshness_days(company)
            if days_old and days_old > 90:
                lines.extend(["## Data Warnings", "", f"- SEC filing is {days_old} days old", ""])
        return "\n".join(lines)
    
    def generate_top5_dossiers(self):
        scored = [{'company': c, 'composite_score': self.score_with_caps(c)['composite_score']} for c in self.companies]
        # Sort ASCENDING: lower score = better = rank 1
        # Validation showed inverted ranking: high scores predicted underperformance
        ranked = sorted(scored, key=lambda x: x['composite_score'], reverse=False)
        print("\nGenerating IC dossiers for Top 5...\n")
        for i, item in enumerate(ranked[:5], 1):
            company = item['company']
            ticker = company['ticker']
            dossier_text = self.generate_dossier(company, i)
            output_file = Path(f'dossiers/top5/{i}_{ticker}_dossier.md')
            with open(output_file, 'w') as f:
                f.write(dossier_text)
            print(f"Generated: {output_file.name}")
        print("\nAll dossiers in: dossiers/top5/\n")

if __name__ == "__main__":
    DossierGenerator().generate_top5_dossiers()