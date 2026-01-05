#!/usr/bin/env python3
"""
Demo mode for Wake Robin Data Pipeline

Simulates data collection to demonstrate pipeline functionality
when network access is restricted. In production, use collect_universe_data.py
with real data sources.
"""
import json
import random
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

def generate_mock_yahoo_data(ticker: str) -> dict:
    """Generate realistic mock Yahoo Finance data."""
    base_price = random.uniform(20, 500)
    market_cap = base_price * random.uniform(50e6, 500e6)
    
    return {
        "ticker": ticker,
        "success": True,
        "price": {
            "current": round(base_price, 2),
            "day_high": round(base_price * 1.02, 2),
            "day_low": round(base_price * 0.98, 2),
            "52_week_high": round(base_price * 1.3, 2),
            "52_week_low": round(base_price * 0.7, 2)
        },
        "market_cap": {
            "value": int(market_cap),
            "currency": "USD"
        },
        "shares_outstanding": int(market_cap / base_price),
        "volume": {
            "last": random.randint(100000, 5000000),
            "average_30d": random.randint(500000, 3000000)
        },
        "valuation": {
            "pe_ratio": random.uniform(15, 50) if random.random() > 0.3 else None,
            "forward_pe": random.uniform(10, 40) if random.random() > 0.3 else None,
            "price_to_book": random.uniform(2, 10) if random.random() > 0.5 else None
        },
        "company_info": {
            "name": f"{ticker} Pharmaceuticals Inc.",
            "sector": "Healthcare",
            "industry": "Biotechnology"
        },
        "provenance": {
            "source": "MOCK DATA (Demo Mode)",
            "timestamp": datetime.now().isoformat(),
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "note": "Simulated for demonstration - use real collectors in production"
        },
        "from_cache": False
    }

def generate_mock_sec_data(ticker: str) -> dict:
    """Generate realistic mock SEC financial data."""
    has_good_data = random.random() > 0.2  # 80% have good coverage
    
    if has_good_data:
        cash = random.uniform(50e6, 5e9)
        debt = random.uniform(0, cash * 0.5) if random.random() > 0.3 else 0
        revenue = random.uniform(100e6, 10e9) if random.random() > 0.4 else None
        assets = cash * random.uniform(2, 5)
        liabilities = assets * random.uniform(0.3, 0.7)
        
        return {
            "ticker": ticker,
            "cik": f"{random.randint(1000000, 9999999):010d}",
            "success": True,
            "financials": {
                "cash": cash,
                "debt": debt,
                "net_debt": debt - cash,
                "revenue_ttm": revenue,
                "assets": assets,
                "liabilities": liabilities,
                "equity": assets - liabilities,
                "currency": "USD"
            },
            "coverage": {
                "has_cash": True,
                "has_debt": debt > 0,
                "has_revenue": revenue is not None,
                "has_balance_sheet": True,
                "pct_complete": 90 if revenue else 75
            },
            "provenance": {
                "source": "MOCK DATA (Demo Mode)",
                "timestamp": datetime.now().isoformat(),
                "note": "Simulated for demonstration"
            },
            "from_cache": False
        }
    else:
        return {
            "ticker": ticker,
            "success": False,
            "error": "Company not found in SEC database (simulated)",
            "timestamp": datetime.now().isoformat(),
            "from_cache": False
        }

def generate_mock_trials_data(ticker: str, company_name: str) -> dict:
    """Generate realistic mock clinical trials data."""
    has_trials = random.random() > 0.1  # 90% have some trials
    
    if has_trials:
        num_trials = random.randint(2, 20)
        active_trials = random.randint(1, num_trials // 2)
        
        phases = ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
        phase_dist = {
            phase: random.randint(0, num_trials // 2) 
            for phase in random.sample(phases, k=random.randint(1, 3))
        }
        
        lead_stage_map = {
            "PHASE4": "commercial",
            "PHASE3": "phase_3",
            "PHASE2": "phase_2",
            "PHASE1": "phase_1"
        }
        
        lead_phase = max(phase_dist.keys(), key=lambda p: phases.index(p))
        lead_stage = lead_stage_map[lead_phase]
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "success": True,
            "trials": [
                {
                    "nct_id": f"NCT{random.randint(10000000, 99999999)}",
                    "title": f"Study of {ticker}-{i} in Disease X",
                    "status": random.choice(["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"]),
                    "phase": random.choice(list(phase_dist.keys())),
                    "condition": random.choice(["Oncology", "Rare Disease", "Autoimmune", "CNS Disorder"]),
                    "start_date": "2023-01-15",
                    "completion_date": "2025-12-31",
                    "enrollment": random.randint(50, 500),
                    "sponsor": company_name
                }
                for i in range(min(5, num_trials))
            ],
            "summary": {
                "total_trials": num_trials,
                "by_phase": phase_dist,
                "by_status": {
                    "RECRUITING": active_trials,
                    "COMPLETED": num_trials - active_trials
                },
                "active_trials": active_trials,
                "completed_trials": num_trials - active_trials,
                "lead_stage": lead_stage,
                "conditions": ["Oncology", "Rare Disease"]
            },
            "provenance": {
                "source": "MOCK DATA (Demo Mode)",
                "timestamp": datetime.now().isoformat(),
                "note": "Simulated for demonstration"
            },
            "from_cache": False
        }
    else:
        return {
            "ticker": ticker,
            "success": True,
            "trials": [],
            "summary": {
                "total_trials": 0,
                "active_trials": 0,
                "completed_trials": 0,
                "lead_stage": "unknown"
            },
            "provenance": {
                "source": "MOCK DATA (Demo Mode)",
                "timestamp": datetime.now().isoformat()
            },
            "from_cache": False
        }

def merge_data_sources(ticker: str, yahoo_data: dict, sec_data: dict, trials_data: dict) -> dict:
    """Merge data from all sources (same as production)."""
    record = {
        "ticker": ticker,
        "as_of_date": datetime.now().isoformat(),
        "data_quality": {
            "yahoo_success": yahoo_data.get('success', False),
            "sec_success": sec_data.get('success', False),
            "trials_success": trials_data.get('success', False),
            "overall_coverage": 0.0
        }
    }
    
    # Market data
    if yahoo_data.get('success'):
        record['market_data'] = {
            "price": yahoo_data['price']['current'],
            "market_cap": yahoo_data['market_cap']['value'],
            "shares_outstanding": yahoo_data.get('shares_outstanding', 0),
            "volume_avg_30d": yahoo_data['volume']['average_30d'],
            "company_name": yahoo_data['company_info']['name']
        }
        record['data_quality']['has_price'] = True
    else:
        record['market_data'] = {"error": yahoo_data.get('error')}
        record['data_quality']['has_price'] = False
    
    # Financial data
    if sec_data.get('success'):
        record['financials'] = sec_data['financials']
        record['data_quality']['financial_coverage'] = sec_data['coverage']['pct_complete']
        record['data_quality']['has_cash'] = sec_data['coverage']['has_cash']
    else:
        record['financials'] = {"error": sec_data.get('error')}
        record['data_quality']['financial_coverage'] = 0.0
        record['data_quality']['has_cash'] = False
    
    # Clinical data
    if trials_data.get('success'):
        record['clinical'] = {
            "total_trials": trials_data['summary']['total_trials'],
            "active_trials": trials_data['summary']['active_trials'],
            "lead_stage": trials_data['summary']['lead_stage'],
            "top_trials": trials_data['trials'][:3]
        }
        record['data_quality']['has_clinical'] = True
    else:
        record['clinical'] = {"error": trials_data.get('error')}
        record['data_quality']['has_clinical'] = False
    
    # Calculate coverage
    scores = []
    if record['data_quality']['has_price']:
        scores.append(100.0)
    if record['data_quality']['financial_coverage'] > 0:
        scores.append(record['data_quality']['financial_coverage'])
    if record['data_quality']['has_clinical']:
        scores.append(80.0)
    
    record['data_quality']['overall_coverage'] = sum(scores) / len(scores) if scores else 0.0
    
    return record

def main():
    """Run demo pipeline."""
    print("\n🎭 Wake Robin Data Pipeline - DEMO MODE")
    print("="*60)
    print("⚠️  Using simulated data for demonstration")
    print("    In production, use: python collect_universe_data.py\n")
    
    # Load universe
    universe_path = Path(__file__).parent / "universe" / "pilot_universe.json"
    with open(universe_path) as f:
        universe = json.load(f)
    
    tickers_data = universe['tickers']
    print(f"Processing {len(tickers_data)} tickers from pilot universe...\n")
    
    records = []
    for i, ticker_info in enumerate(tickers_data, 1):
        ticker = ticker_info['ticker']
        company = ticker_info['company']
        
        print(f"[{i}/{len(tickers_data)}] {ticker} - {company}")
        
        # Generate mock data
        yahoo_data = generate_mock_yahoo_data(ticker)
        sec_data = generate_mock_sec_data(ticker)
        trials_data = generate_mock_trials_data(ticker, company)
        
        # Merge
        record = merge_data_sources(ticker, yahoo_data, sec_data, trials_data)
        records.append(record)
        
        # Show summary
        price = record['market_data'].get('price', 'N/A')
        coverage = record['data_quality']['overall_coverage']
        print(f"   ✓ Price: ${price} | Coverage: {coverage:.0f}%\n")
    
    # Save outputs
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = output_dir / f"demo_snapshot_{timestamp}.json"
    
    with open(snapshot_file, 'w') as f:
        json.dump({
            "mode": "DEMO",
            "note": "This is simulated data. Use real collectors in production.",
            "timestamp": datetime.now().isoformat(),
            "records": records
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print(f"\n📁 Output saved to: {snapshot_file.name}")
    print("\n💡 Next steps:")
    print("   1. Review the demo output to understand data structure")
    print("   2. In your production environment, run: python collect_universe_data.py")
    print("   3. The real collectors will fetch live data from public APIs")
    print("\n")

if __name__ == "__main__":
    main()
