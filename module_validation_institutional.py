"""
module_validation_institutional.py

Pipeline Wrapper for Institutional Validation
Integrates institutional_validation_v1_2.py into Wake Robin screening system.

Author: Wake Robin Capital Management
Date: 2026-01-09
"""

from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Optional
import json

# Import core validation module
from institutional_validation_v1_2 import (
    validate_institutional_activity,
    ManagerQuarterSnapshot,
    FilingMetadata,
    SnapshotState,
    InstitutionalValidationOutput,
    check_institutional_alerts,
    ActivityFeatures,
    BreadthFeatures
)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_manager_registry(registry_path: Path) -> Dict:
    """
    Load manager registry with elite/conditional tiers.
    
    Returns:
        {
          'elite_core': [cik1, cik2, ...],
          'conditional': [cik3, cik4, ...],
          'tiers': {cik: 'ELITE' | 'COND'},
          'metadata': {cik: {'name': str, 'aum_b': float}}
        }
    """
    with open(registry_path) as f:
        raw = json.load(f)
    
    elite_core = [m['cik'] for m in raw['elite_core']]
    conditional = [m['cik'] for m in raw.get('conditional', [])]
    
    tiers = {}
    metadata = {}
    
    for m in raw['elite_core']:
        tiers[m['cik']] = 'ELITE'
        metadata[m['cik']] = {
            'name': m['name'],
            'aum_b': m.get('aum_b', 0)
        }
    
    for m in raw.get('conditional', []):
        tiers[m['cik']] = 'COND'
        metadata[m['cik']] = {
            'name': m['name'],
            'aum_b': m.get('aum_b', 0)
        }
    
    return {
        'elite_core': elite_core,
        'conditional': conditional,
        'tiers': tiers,
        'metadata': metadata
    }


def load_holdings_snapshots(
    holdings_path: Path,
    ticker: str
) -> Optional[Dict]:
    """
    Load holdings snapshots for specific ticker.
    
    Returns None if ticker not tracked by elite managers.
    """
    if not holdings_path.exists():
        return None
    
    with open(holdings_path) as f:
        data = json.load(f)
    
    return data.get(ticker)


def build_snapshots_from_json(
    holdings_json: Dict[str, Dict]
) -> Dict[str, ManagerQuarterSnapshot]:
    """
    Convert JSON holdings to ManagerQuarterSnapshot objects.
    
    Args:
        holdings_json: {cik: {quarter_end, state, shares, value_kusd, put_call}}
        
    Returns:
        {cik: ManagerQuarterSnapshot}
    """
    snapshots = {}
    
    for cik, data in holdings_json.items():
        snapshots[cik] = ManagerQuarterSnapshot(
            cik=cik,
            quarter_end=datetime.strptime(data['quarter_end'], '%Y-%m-%d').date(),
            state=SnapshotState(data['state']),
            shares=data['shares'],
            value_kusd=data['value_kusd'],
            put_call=data.get('put_call', '')
        )
    
    return snapshots


def build_filings_metadata_from_json(
    metadata_json: Dict[str, Dict]
) -> Dict[str, FilingMetadata]:
    """
    Convert JSON metadata to FilingMetadata objects.
    
    Args:
        metadata_json: {cik: {quarter_end, accession, total_value_kusd, ...}}
        
    Returns:
        {cik: FilingMetadata}
    """
    metadata = {}
    
    for cik, data in metadata_json.items():
        metadata[cik] = FilingMetadata(
            cik=cik,
            quarter_end=datetime.strptime(data['quarter_end'], '%Y-%m-%d').date(),
            accession=data['accession'],
            total_value_kusd=data['total_value_kusd'],
            filed_at=datetime.fromisoformat(data['filed_at']),
            is_amendment=data.get('is_amendment', False)
        )
    
    return metadata


# ==============================================================================
# MAIN VALIDATION WRAPPER
# ==============================================================================

def compute_institutional_validation(
    ticker: str,
    as_of_date: date,
    market_cap_usd: float,
    data_dir: Path
) -> Optional[InstitutionalValidationOutput]:
    """
    Compute institutional validation for one ticker.
    
    This is the main integration point for run_screen.py.
    
    Args:
        ticker: Stock ticker
        as_of_date: Point-in-time date for evaluation
        market_cap_usd: Current market cap
        data_dir: Path to production_data/ directory
        
    Returns:
        InstitutionalValidationOutput or None if no data
    """
    # Load manager registry
    registry_path = data_dir / 'manager_registry.json'
    if not registry_path.exists():
        print(f"Warning: Manager registry not found at {registry_path}")
        return None
    
    registry = load_manager_registry(registry_path)
    
    # Load holdings snapshots
    holdings_path = data_dir / 'holdings_snapshots.json'
    ticker_data = load_holdings_snapshots(holdings_path, ticker)
    
    if not ticker_data:
        # Ticker not tracked by elite managers - this is NORMAL
        return None
    
    # Convert JSON to dataclass objects
    current_holdings = build_snapshots_from_json(
        ticker_data['holdings']['current']
    )
    
    prior_holdings = build_snapshots_from_json(
        ticker_data['holdings']['prior']
    )
    
    filings_metadata = build_filings_metadata_from_json(
        ticker_data['filings_metadata']
    )
    
    # Call core validation module
    result = validate_institutional_activity(
        ticker=ticker,
        as_of_date=datetime.combine(as_of_date, datetime.min.time()),
        current_holdings=current_holdings,
        prior_holdings=prior_holdings,
        filings_metadata=filings_metadata,
        manager_tiers=registry['tiers'],
        elite_ciks=registry['elite_core'],
        market_cap_usd=market_cap_usd
    )
    
    return result


def format_institutional_summary_for_report(
    inst_result: Optional[InstitutionalValidationOutput]
) -> str:
    """
    Format institutional validation for Top 60 report.
    
    Returns:
        " │ INST: 0.72 [4H, 2N, 1A] ⚠️" or " │ INST: --- [NO DATA]"
    """
    if inst_result is None:
        return " │ INST: --- [NO DATA]"
    
    score = inst_result.inst_validation_score
    holders = inst_result.elite_holders_n
    new = inst_result.elite_new_n
    adds = inst_result.elite_material_add_n
    
    crowding = " ⚠️" if inst_result.crowding_flag else ""
    severe_crowding = " ⚠️⚠️" if inst_result.severe_crowding_flag else ""
    
    return f" │ INST: {score:.2f} [{holders}H, {new}N, {adds}A]{crowding}{severe_crowding}"


def convert_to_dict(
    inst_result: Optional[InstitutionalValidationOutput]
) -> Dict:
    """
    Convert InstitutionalValidationOutput to dict for CSV export.
    
    Returns dict with keys:
    - inst_score, inst_state, elite_holders, elite_new, elite_adds,
      elite_exits, crowding_flag, severe_crowding_flag
    """
    if inst_result is None:
        return {
            'inst_score': None,
            'inst_state': 'NO_DATA',
            'elite_holders': 0,
            'elite_new': 0,
            'elite_adds': 0,
            'elite_exits': 0,
            'elite_concentration': 0.0,
            'crowding_flag': False,
            'severe_crowding_flag': False
        }
    
    return {
        'inst_score': round(inst_result.inst_validation_score, 4),
        'inst_state': inst_result.inst_state.value,
        'elite_holders': inst_result.elite_holders_n,
        'elite_new': inst_result.elite_new_n,
        'elite_adds': inst_result.elite_material_add_n,
        'elite_exits': inst_result.elite_exit_n,
        'elite_concentration': round(inst_result.elite_concentration_sum, 4),
        'crowding_flag': inst_result.crowding_flag,
        'severe_crowding_flag': inst_result.severe_crowding_flag
    }


# ==============================================================================
# ALERT GENERATION
# ==============================================================================

def generate_institutional_alerts(
    summaries: List[Dict],
    output_dir: Path,
    timestamp: str,
    report_time: Optional[str] = None
) -> None:
    """
    Generate institutional activity alert report.

    Args:
        summaries: List of ticker summaries with institutional data
        output_dir: Path to outputs/ directory
        timestamp: Timestamp string for filename
        report_time: Explicit report generation timestamp for PIT safety.
                    If None, uses the filename timestamp for reproducibility.
    """
    # PIT SAFETY: Use explicit timestamp, not wall-clock
    if report_time is None:
        # Use the provided timestamp parameter for reproducibility
        report_time = timestamp
    high_priority_alerts = []
    medium_priority_alerts = []
    
    for s in summaries:
        ticker = s['ticker']
        
        # Skip tickers with no institutional data
        if s.get('inst_state') == 'NO_DATA':
            continue
        
        # HIGH PRIORITY: Cluster buy (2+ elite initiations)
        if s.get('elite_new', 0) >= 2:
            high_priority_alerts.append({
                'ticker': ticker,
                'type': 'CLUSTER_BUY',
                'message': f"{s['elite_new']} elite funds initiated new positions",
                'score': s.get('inst_score', 0),
                'final_score': s.get('final_score', 0)
            })
        
        # HIGH PRIORITY: Multiple exits (2+ elite exits)
        if s.get('elite_exits', 0) >= 2:
            high_priority_alerts.append({
                'ticker': ticker,
                'type': 'MULTIPLE_EXITS',
                'message': f"{s['elite_exits']} elite funds exited - NEGATIVE SIGNAL",
                'score': s.get('inst_score', 0),
                'final_score': s.get('final_score', 0)
            })
        
        # MEDIUM PRIORITY: Sustained accumulation
        if s.get('elite_holders', 0) >= 3 and s.get('elite_adds', 0) >= 2:
            medium_priority_alerts.append({
                'ticker': ticker,
                'type': 'SUSTAINED_ACCUMULATION',
                'message': f"{s['elite_adds']} elite funds adding to existing {s['elite_holders']}-holder base",
                'score': s.get('inst_score', 0),
                'final_score': s.get('final_score', 0)
            })
        
        # MEDIUM PRIORITY: Crowding risk
        if s.get('crowding_flag', False):
            medium_priority_alerts.append({
                'ticker': ticker,
                'type': 'CROWDING_RISK',
                'message': f"{s['elite_holders']} elite funds in small/mid cap - monitor for exits",
                'score': s.get('inst_score', 0),
                'final_score': s.get('final_score', 0)
            })
        
        # MEDIUM PRIORITY: Severe crowding
        if s.get('severe_crowding_flag', False):
            medium_priority_alerts.append({
                'ticker': ticker,
                'type': 'SEVERE_CROWDING',
                'message': f"{s['elite_holders']} elite funds - REFLEXIVITY DANGER",
                'score': s.get('inst_score', 0),
                'final_score': s.get('final_score', 0)
            })
    
    # Write alert report
    alert_path = output_dir / f'institutional_alerts_{timestamp}.txt'
    
    with open(alert_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WAKE ROBIN - INSTITUTIONAL ACTIVITY ALERTS\n")
        f.write("=" * 80 + "\n")
        # PIT SAFETY: Use explicit timestamp, never wall-clock
        f.write(f"Generated: {report_time}\n")
        f.write(f"Alert Count: {len(high_priority_alerts) + len(medium_priority_alerts)}\n")
        f.write("=" * 80 + "\n\n")
        
        # HIGH PRIORITY
        if high_priority_alerts:
            f.write("HIGH PRIORITY ALERTS\n")
            f.write("-" * 80 + "\n\n")
            
            for alert in sorted(high_priority_alerts, key=lambda x: -x['score']):
                f.write(f"[{alert['ticker']}] {alert['type']}\n")
                f.write(f"  → {alert['message']}\n")
                f.write(f"  → Inst Score: {alert['score']:.2f} | Composite Score: {alert['final_score']:.2f}\n")
                f.write("\n")
        
        # MEDIUM PRIORITY
        if medium_priority_alerts:
            f.write("\nMEDIUM PRIORITY ALERTS\n")
            f.write("-" * 80 + "\n\n")
            
            for alert in sorted(medium_priority_alerts, key=lambda x: -x['score']):
                f.write(f"[{alert['ticker']}] {alert['type']}\n")
                f.write(f"  → {alert['message']}\n")
                f.write(f"  → Inst Score: {alert['score']:.2f} | Composite Score: {alert['final_score']:.2f}\n")
                f.write("\n")
        
        # SUMMARY
        f.write("=" * 80 + "\n")
        f.write("ALERT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"High Priority:   {len(high_priority_alerts)}\n")
        f.write(f"Medium Priority: {len(medium_priority_alerts)}\n")
        f.write(f"Total Alerts:    {len(high_priority_alerts) + len(medium_priority_alerts)}\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ {alert_path.name}")


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def compute_institutional_validation_batch(
    tickers: List[str],
    as_of_date: date,
    universe: List[Dict],
    data_dir: Path
) -> Dict[str, Optional[InstitutionalValidationOutput]]:
    """
    Compute institutional validation for batch of tickers.
    
    Args:
        tickers: List of tickers to process
        as_of_date: Point-in-time date
        universe: List of universe records (for market cap)
        data_dir: Path to production_data/
        
    Returns:
        {ticker: InstitutionalValidationOutput or None}
    """
    results = {}
    
    # Create ticker → market cap lookup
    ticker_to_mcap = {
        s['ticker']: s.get('market_cap_usd', 0)
        for s in universe
        if s.get('ticker') != '_XBI_BENCHMARK_'
    }
    
    for ticker in tickers:
        market_cap = ticker_to_mcap.get(ticker, 0)
        
        result = compute_institutional_validation(
            ticker=ticker,
            as_of_date=as_of_date,
            market_cap_usd=market_cap,
            data_dir=data_dir
        )
        
        results[ticker] = result
    
    return results


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Example: Process single ticker
    """
    from pathlib import Path
    from datetime import date
    
    ticker = "ARGX"
    as_of_date = date(2024, 11, 15)
    market_cap_usd = 12_000_000_000
    data_dir = Path('production_data')
    
    result = compute_institutional_validation(
        ticker=ticker,
        as_of_date=as_of_date,
        market_cap_usd=market_cap_usd,
        data_dir=data_dir
    )
    
    if result:
        print(f"\nInstitutional Validation: {ticker}")
        print(f"Score: {result.inst_validation_score:.3f}")
        print(f"State: {result.inst_state.value}")
        print(f"Elite Holders: {result.elite_holders_n}")
        print(f"New Initiations: {result.elite_new_n}")
        print(f"Material Adds: {result.elite_material_add_n}")
        print(f"Exits: {result.elite_exit_n}")
        print(f"Crowding: {result.crowding_flag}")
        
        # Format for report
        summary = format_institutional_summary_for_report(result)
        print(f"\nReport Format: {summary}")
    else:
        print(f"\n{ticker}: NO INSTITUTIONAL DATA (not tracked by elite managers)")
