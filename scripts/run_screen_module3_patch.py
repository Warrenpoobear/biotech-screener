# run_screen.py Integration Patch for Module 3A
# Add this EXACT code after Module 2 (Financial Health)

# ============================================================================
# Module 3: Catalyst Detection
# ============================================================================

print("\n[3/7] Module 3: Catalyst detection...")

from module_3_catalyst import compute_module_3_catalyst, Module3Config
from event_detector import SimpleMarketCalendar

# Run Module 3A
m3_result = compute_module_3_catalyst(
    trial_records_path=data_dir / "trial_records.json",
    state_dir=data_dir / "ctgov_state",
    active_tickers=set(active_tickers),
    as_of_date=as_of_date,
    market_calendar=SimpleMarketCalendar(),
    config=Module3Config(),
    output_dir=data_dir
)

# Extract results (summaries is already a dict keyed by ticker)
catalyst_summaries = m3_result["summaries"]
diag3 = m3_result.get("diagnostic_counts", {})

# Print diagnostics
print(f"  Events detected: {diag3.get('events_detected', 0)}, "
      f"Tickers with events: {diag3.get('tickers_with_events', 0)}/{diag3.get('tickers_analyzed', 0)}, "
      f"Severe negatives: {diag3.get('severe_negatives', 0)}")

# ============================================================================
# Continue to Module 4...
# ============================================================================
