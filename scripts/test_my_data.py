import json
from module_5_composite_with_defensive import compute_module_5_composite_with_defensive

# Load your actual universe snapshot
with open('wake_robin_data_pipeline/outputs/universe_snapshot_latest.json') as f:
    data = json.load(f)

# Create minimal module results
universe_result = {"active_securities": data[:5]}  # Just 5 tickers for test
financial_result = {"scores": [{"ticker": s["ticker"], "financial_score": "80", "severity": "none"} for s in data[:5]]}
catalyst_result = {"scores": [{"ticker": s["ticker"], "catalyst_score": "75", "severity": "none"} for s in data[:5]]}
clinical_result = {"scores": [{"ticker": s["ticker"], "clinical_score": "85", "severity": "none", "lead_phase": "phase_3"} for s in data[:5]]}

# Run with defensive overlays
output = compute_module_5_composite_with_defensive(
    universe_result,
    financial_result,
    catalyst_result,
    clinical_result,
    as_of_date="2026-01-06",
    validate=True
)

print("\n✅ Defensive overlays working!")
print(f"Top ticker: {output['ranked_securities'][0]['ticker']}")
print(f"Position weight: {output['ranked_securities'][0].get('position_weight', 'N/A')}")
print(f"Defensive notes: {output['ranked_securities'][0].get('defensive_notes', [])}")