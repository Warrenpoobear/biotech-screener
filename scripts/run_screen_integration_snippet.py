"""
run_screen.py Integration Snippet

This shows exactly where to add institutional validation to your existing
run_screen.py workflow. Add after Module 5 completes.
"""

# ==============================================================================
# EXISTING CODE (Modules 1-5 complete)
# ==============================================================================

# ... your existing imports ...

from module_validation_institutional import (
    compute_institutional_validation,
    convert_to_dict,
    format_institutional_summary_for_report,
    generate_institutional_alerts
)


def main():
    # ... existing setup ...
    
    # Modules 1-5 run as normal
    # ... existing module execution ...
    
    # Module 5 completes, you have:
    # - ranked_summaries: List[Dict] with composite scores
    # - Each summary has: ticker, final_score, m1_universe, m2_financial, etc.
    
    # ==============================================================================
    # NEW: MODULE VALIDATION - INSTITUTIONAL (Post-Composite Layer)
    # ==============================================================================
    
    print("\n" + "="*80)
    print("INSTITUTIONAL VALIDATION LAYER")
    print("="*80)
    print("Computing institutional signals for confirmation (not scoring)...")
    
    # Add institutional validation to each summary
    for summary in ranked_summaries:
        ticker = summary['ticker']
        
        # Skip benchmark
        if ticker == '_XBI_BENCHMARK_':
            summary.update({
                'inst_score': None,
                'inst_state': 'N/A',
                'elite_holders': 0,
                'elite_new': 0,
                'elite_adds': 0,
                'elite_exits': 0,
                'elite_concentration': 0.0,
                'crowding_flag': False,
                'severe_crowding_flag': False
            })
            continue
        
        # Get market cap from summary (already computed in Module 1)
        market_cap_usd = summary.get('market_cap_usd', 0)
        
        # Compute institutional validation
        inst_result = compute_institutional_validation(
            ticker=ticker,
            as_of_date=as_of_date,
            market_cap_usd=market_cap_usd,
            data_dir=production_data_dir
        )
        
        # Convert to dict and merge into summary
        inst_dict = convert_to_dict(inst_result)
        summary.update(inst_dict)
        
        # Optional: Print progress for high-signal tickers
        if inst_result and inst_result.inst_validation_score > 0.60:
            print(f"  {ticker}: Inst={inst_result.inst_validation_score:.2f} "
                  f"[{inst_result.elite_holders_n}H, {inst_result.elite_new_n}N]")
    
    # Count institutional coverage
    covered = sum(1 for s in ranked_summaries 
                  if s.get('inst_state') not in ('NO_DATA', 'N/A'))
    total = len([s for s in ranked_summaries if s['ticker'] != '_XBI_BENCHMARK_'])
    
    print(f"\nInstitutional Coverage: {covered}/{total} tickers ({covered/total*100:.1f}%)")
    print("="*80 + "\n")
    
    # ==============================================================================
    # EXISTING: Save results
    # ==============================================================================
    
    # Save to JSON (now includes institutional fields)
    with open(output_path, 'w') as f:
        json.dump(ranked_summaries, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # ==============================================================================
    # EXISTING: Generate reports (now with institutional columns)
    # ==============================================================================
    
    # Your existing report generation...
    # The generate_all_reports.py script will automatically pick up
    # the new institutional fields from ranked_summaries


# ==============================================================================
# generate_all_reports.py Modifications
# ==============================================================================

def generate_csv_export(summaries, output_path):
    """
    Modified to include institutional validation columns.
    """
    import csv
    
    # UPDATED fieldnames to include institutional columns
    fieldnames = [
        'ticker', 'rank', 'final_score',
        'm1_universe', 'm2_financial', 'm3_catalyst', 'm4_clinical',
        'market_cap_usd', 'sector', 'clinical_stage',
        # NEW INSTITUTIONAL COLUMNS
        'inst_score', 'inst_state', 'elite_holders',
        'elite_new', 'elite_adds', 'elite_exits',
        'elite_concentration', 'crowding_flag', 'severe_crowding_flag'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for summary in summaries:
            # Filter to only include defined fieldnames
            row = {k: summary.get(k) for k in fieldnames}
            writer.writerow(row)


def generate_top60_report(summaries, output_path):
    """
    Modified to include institutional summary in right column.
    """
    from module_validation_institutional import format_institutional_summary_for_report
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WAKE ROBIN - TOP 60 CONVICTION LONGS\n")
        f.write("="*80 + "\n\n")
        
        top60 = summaries[:60]
        
        for i, s in enumerate(top60, 1):
            ticker = s['ticker']
            score = s['final_score']
            stage = s.get('clinical_stage', 'unknown')
            
            # NEW: Get institutional summary from validation result
            # (inst fields already in summary dict)
            inst_summary = format_institutional_summary_for_report_from_dict(s)
            
            # Format with institutional data in right column
            f.write(f"{i:4d}. {ticker:6s} ({score:5.2f}) - {stage:12s}{inst_summary}\n")


def format_institutional_summary_for_report_from_dict(summary: dict) -> str:
    """
    Format institutional data from summary dict (not full result object).
    """
    if summary.get('inst_state') in ('NO_DATA', 'N/A'):
        return " │ INST: --- [NO DATA]"
    
    score = summary.get('inst_score', 0)
    holders = summary.get('elite_holders', 0)
    new = summary.get('elite_new', 0)
    adds = summary.get('elite_adds', 0)
    
    crowding = " ⚠️" if summary.get('crowding_flag', False) else ""
    severe_crowding = " ⚠️⚠️" if summary.get('severe_crowding_flag', False) else ""
    
    return f" │ INST: {score:.2f} [{holders}H, {new}N, {adds}A]{crowding}{severe_crowding}"


def generate_executive_summary(summaries, output_path):
    """
    Modified to include institutional coverage stats.
    """
    # ... existing executive summary code ...
    
    # NEW: Add institutional coverage section
    covered = sum(1 for s in summaries if s.get('inst_state') not in ('NO_DATA', 'N/A'))
    total = len([s for s in summaries if s['ticker'] != '_XBI_BENCHMARK_'])
    
    high_conviction = sum(1 for s in summaries 
                          if s.get('inst_score') and s['inst_score'] > 0.70)
    
    cluster_buys = sum(1 for s in summaries 
                       if s.get('elite_new', 0) >= 2)
    
    exits = sum(1 for s in summaries 
                if s.get('elite_exits', 0) >= 2)
    
    f.write("\nINSTITUTIONAL VALIDATION\n")
    f.write(f"  Coverage: {covered}/{total} tickers ({covered/total*100:.1f}%)\n")
    f.write(f"  High Conviction: {high_conviction} tickers (score > 0.70)\n")
    f.write(f"  Cluster Buys: {cluster_buys} tickers (2+ elite initiations)\n")
    f.write(f"  Multiple Exits: {exits} tickers (2+ elite exits - RED FLAG)\n")


# ==============================================================================
# NEW: generate_institutional_alerts.py Integration
# ==============================================================================

def main():
    # ... after all reports generated ...
    
    # Generate institutional alerts
    from module_validation_institutional import generate_institutional_alerts
    
    print("[7/7] Generating institutional alerts...")
    generate_institutional_alerts(
        summaries=ranked_summaries,
        output_dir=output_dir,
        timestamp=timestamp
    )
    print(f"  ✓ institutional_alerts_{timestamp}.txt")


# ==============================================================================
# Example: Complete run_screen.py execution flow
# ==============================================================================

"""
BEFORE (5 modules):
1. Module 1: Universe filtering
2. Module 2: Financial health
3. Module 3: Catalyst detection
4. Module 4: Clinical development
5. Module 5: Composite scoring
→ Save results
→ Generate reports

AFTER (5 modules + validation layer):
1. Module 1: Universe filtering
2. Module 2: Financial health
3. Module 3: Catalyst detection
4. Module 4: Clinical development
5. Module 5: Composite scoring
→ [NEW] Institutional validation layer (confirmation only)
→ Save results (now with inst columns)
→ Generate reports (now with inst columns)
→ [NEW] Generate institutional alerts

Key Points:
- Institutional validation runs AFTER Module 5
- Does NOT modify composite scores
- Adds columns to output only
- ~45% coverage is expected (elites are selective)
- NO_DATA is neutral, not negative
"""


# ==============================================================================
# Testing the Integration
# ==============================================================================

def test_institutional_integration():
    """
    Test that institutional validation integrates correctly.
    """
    from pathlib import Path
    from datetime import date
    import json
    
    # Load sample results
    with open('results_ZERO_BUG_FIXED.json') as f:
        results = json.load(f)
    
    # Add institutional validation to first 10 tickers
    for summary in results[:10]:
        ticker = summary['ticker']
        
        inst_result = compute_institutional_validation(
            ticker=ticker,
            as_of_date=date.today(),
            market_cap_usd=summary.get('market_cap_usd', 0),
            data_dir=Path('production_data')
        )
        
        inst_dict = convert_to_dict(inst_result)
        summary.update(inst_dict)
        
        print(f"\n{ticker}:")
        print(f"  Composite Score: {summary['final_score']:.2f}")
        print(f"  Inst Score: {inst_dict['inst_score']}")
        print(f"  Inst State: {inst_dict['inst_state']}")
        print(f"  Elite Holders: {inst_dict['elite_holders']}")
        print(f"  Activity: {inst_dict['elite_new']}N, {inst_dict['elite_adds']}A, {inst_dict['elite_exits']}E")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_institutional_integration()
