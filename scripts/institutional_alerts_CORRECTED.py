"""
institutional_alerts.py (DETERMINISM FIX)

Generate institutional activity alerts with deterministic timestamps.

CRITICAL FIX: Use pipeline timestamp instead of datetime.now() for byte-identical outputs.

Author: Wake Robin Capital Management
Date: 2026-01-09 (Determinism-Corrected)
"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime

def generate_institutional_alerts(
    summaries: List[Dict],
    output_dir: Path,
    timestamp: str,  # CRITICAL: Accept pipeline timestamp for determinism
    as_of_date: str = None  # Optional: explicit as_of_date for report header
) -> None:
    """
    Generate institutional activity alert report with deterministic timestamps.
    
    CRITICAL FIX: Uses pipeline's deterministic timestamp instead of datetime.now()
    
    Args:
        summaries: List of ticker summaries with institutional data
        output_dir: Path to outputs/ directory
        timestamp: Deterministic timestamp from pipeline (e.g., "20260109_143022")
        as_of_date: Optional explicit date for report (e.g., "2026-01-09")
    """
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
    
    # CRITICAL FIX: Use deterministic timestamp formatting
    if as_of_date:
        report_date_str = as_of_date
    else:
        # Parse timestamp (format: YYYYMMDD_HHMMSS)
        try:
            dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
            report_date_str = dt.strftime('%Y-%m-%d')
        except ValueError:
            # Fallback if timestamp format unexpected
            report_date_str = timestamp[:8]  # Extract YYYYMMDD
    
    with open(alert_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WAKE ROBIN - INSTITUTIONAL ACTIVITY ALERTS\n")
        f.write("=" * 80 + "\n")
        
        # CRITICAL FIX: Use deterministic timestamp, not datetime.now()
        f.write(f"Report Date: {report_date_str}\n")
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
# INTEGRATION WITH GENERATE_ALL_REPORTS.PY
# ==============================================================================

def integrate_into_report_generator():
    """
    Example: How to call from generate_all_reports.py
    """
    
    # Inside generate_all_reports.py main():
    
    # Load results
    with open(results_path) as f:
        summaries = json.load(f)
    
    # Get timestamp from results filename or generate deterministically
    # Extract from results_YYYYMMDD_HHMMSS.json
    timestamp = "20260109_143022"  # From results filename
    as_of_date = "2026-01-09"  # From results metadata
    
    # Generate reports [1-6]...
    
    # [7/7] Generate institutional alerts with DETERMINISTIC timestamp
    print("[7/7] Generating institutional alerts...")
    generate_institutional_alerts(
        summaries=summaries,
        output_dir=output_dir,
        timestamp=timestamp,  # Pass pipeline timestamp
        as_of_date=as_of_date  # Optional: for cleaner report header
    )


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Standalone test of alert generation.
    """
    from pathlib import Path
    
    # Sample summaries
    summaries = [
        {
            'ticker': 'ARGX',
            'final_score': 84.5,
            'inst_score': 0.72,
            'inst_state': 'KNOWN',
            'elite_holders': 4,
            'elite_new': 2,
            'elite_adds': 1,
            'elite_exits': 0,
            'crowding_flag': False,
            'severe_crowding_flag': False
        },
        {
            'ticker': 'NVAX',
            'final_score': 82.3,
            'inst_score': 0.15,
            'inst_state': 'KNOWN',
            'elite_holders': 1,
            'elite_new': 0,
            'elite_adds': 0,
            'elite_exits': 2,  # RED FLAG
            'crowding_flag': False,
            'severe_crowding_flag': False
        },
        {
            'ticker': 'SMOL',
            'final_score': 75.0,
            'inst_score': None,
            'inst_state': 'NO_DATA',
            'elite_holders': 0,
            'elite_new': 0,
            'elite_adds': 0,
            'elite_exits': 0,
            'crowding_flag': False,
            'severe_crowding_flag': False
        }
    ]
    
    # Generate alerts with deterministic timestamp
    generate_institutional_alerts(
        summaries=summaries,
        output_dir=Path('outputs'),
        timestamp='20260109_143022',  # Deterministic from pipeline
        as_of_date='2026-01-09'
    )
    
    print("\nGenerated institutional_alerts_20260109_143022.txt")
    print("✓ Uses deterministic timestamp (no datetime.now())")
    print("✓ Byte-identical outputs for identical inputs")
