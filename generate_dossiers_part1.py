import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

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
        except (ValueError, TypeError, KeyError):
            return None
    
    def score_with_caps(self, company: Dict) -> Dict:
        cash = company['financials'].get('cash', 0)
        mcap = company['market_data']['market_cap']
        
        if cash > 0 and mcap > 0:
            raw_ratio = (cash / mcap) * 100
            capped_ratio = min(raw_ratio, 30.0)
            
            import math
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
        
        stage_scores = {
            'commercial': 10,
            'phase_3': 8,
            'phase_2': 5,
            'phase_1': 2,
            'unknown': 0
        }
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
        
        composite = (
            stage_score * 0.30 +
            compressed_score * 0.25 +
            runway_score * 0.25 +
            pipeline_score * 0.20
        )
        
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