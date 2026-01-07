# Module 3A Integration with run_screen.py
## Drop-In Code for Existing Pipeline

---

## 🔧 Integration Code

Add this to your `run_screen.py` after Module 2 (Financial Health):

```python
# ============================================================================
# Module 3: Catalyst Detection
# ============================================================================

print("\n[3/7] Module 3: Catalyst detection...")

# Import Module 3
from module_3_catalyst import compute_module_3_catalyst, Module3Config
from event_detector import SimpleMarketCalendar

# Initialize market calendar (or use your existing one)
market_calendar = SimpleMarketCalendar()

# Initialize Module 3 config (or load from file)
m3_config = Module3Config()

# Run Module 3
m3_result = compute_module_3_catalyst(
    trial_records_path=data_dir / "trial_records.json",
    state_dir=data_dir / "ctgov_state",
    active_tickers=set(active_tickers),
    as_of_date=as_of_date,
    market_calendar=market_calendar,
    config=m3_config,
    output_dir=data_dir
)

# Extract results
catalyst_summaries = m3_result['summaries']  # Dict[ticker, TickerCatalystSummary]
diag = m3_result.get('diagnostic_counts', {})

# Print diagnostics
print(f"  Events detected: {diag.get('events_detected', 0)}, "
      f"Tickers with events: {diag.get('tickers_with_events', 0)}/{diag.get('tickers_analyzed', 0)}, "
      f"Severe negatives: {diag.get('severe_negatives', 0)}")
```

---

## 📊 Module 5 Integration

In your Module 5 composite scoring, add catalyst scores:

```python
# ============================================================================
# Module 5: Composite Ranking with Catalyst Integration
# ============================================================================

for ticker in active_tickers:
    # Get module scores
    m2_score = m2_result['securities'].get(ticker, {}).get('health_score', 0)
    m4_score = m4_result['scores'].get(ticker, {}).get('clinical_quality_score', 0)
    
    # Get catalyst score (NEW)
    catalyst_summary = catalyst_summaries.get(ticker)
    m3_score = 0.0
    severe_negative = False
    
    if catalyst_summary:
        m3_score = catalyst_summary.catalyst_score_net
        severe_negative = catalyst_summary.severe_negative_flag
    
    # Kill switch: exclude if severe negative
    if severe_negative:
        logger.warning(f"Excluding {ticker}: severe negative catalyst event")
        continue
    
    # Composite score with catalyst
    composite_score = (
        0.25 * m2_score +
        0.15 * m3_score +  # NEW: Catalyst contribution
        0.40 * m4_score +
        0.20 * other_factors
    )
    
    # Store result
    results.append({
        'ticker': ticker,
        'composite_score': composite_score,
        'catalyst_score_net': m3_score,
        'severe_negative_flag': severe_negative,
        # ... other fields
    })
```

---

## 🔍 Accessing Catalyst Details

To get detailed event information:

```python
# Get catalyst summary for a specific ticker
summary = catalyst_summaries.get('VRTX')

if summary:
    print(f"\n{summary.ticker} Catalyst Summary:")
    print(f"  Net Score: {summary.catalyst_score_net:.2f}")
    print(f"  Positive: +{summary.catalyst_score_pos:.2f}")
    print(f"  Negative: -{summary.catalyst_score_neg:.2f}")
    print(f"  Severe Negative: {summary.severe_negative_flag}")
    print(f"  Events: {len(summary.events)}")
    
    for event in summary.events:
        print(f"    - {event.event_type.value}: {event.nct_id}")
        print(f"      Impact: {event.impact}, Direction: {event.direction}")
        print(f"      Disclosed: {event.disclosed_at}")
```

---

## 📝 Updated Pipeline Order

Your complete pipeline will now be:

```
[1/7] Module 1: Universe filtering
[2/7] Module 2: Financial health
[3/7] Module 3: Catalyst detection     ← NEW
[4/7] Module 4: Clinical development
[5/7] Module 5: Composite ranking      ← Enhanced with catalyst
[6/7] Defensive overlay
[7/7] Top-N selection
```

---

## 🎯 Expected Output

When you run `run_screen.py`, you should see:

```
[3/7] Module 3: Catalyst detection...
  Events detected: 12, Tickers with events: 8/98, Severe negatives: 2
```

---

## 📊 Output Files

Module 3 will create:

```
production_data/
├── ctgov_state/
│   ├── state_2024-01-15.jsonl         # State snapshot
│   ├── state_2024-01-22.jsonl
│   └── manifest.json                   # Snapshot inventory
├── catalyst_events_2024-01-15.json     # Deterministic output
└── run_log_2024-01-15.json             # Non-deterministic log
```

---

## ⚙️ Configuration (Optional)

Create `config/module_3a_config.json`:

```json
{
  "noise_band_days": 14,
  "recency_threshold_days": 90,
  "decay_constant": 30.0,
  "confidence_scores": {
    "CT_STATUS_SEVERE_NEG": 0.95,
    "CT_STATUS_DOWNGRADE": 0.85,
    "CT_STATUS_UPGRADE": 0.80,
    "CT_TIMELINE_PUSHOUT": 0.75,
    "CT_TIMELINE_PULLIN": 0.70,
    "CT_DATE_CONFIRMED_ACTUAL": 0.85,
    "CT_RESULTS_POSTED": 0.90
  }
}
```

Load it in run_screen.py:

```python
# Load Module 3 config
import json
with open('config/module_3a_config.json') as f:
    m3_config_dict = json.load(f)
m3_config = Module3Config.from_dict(m3_config_dict)
```

---

## 🚨 Important Notes

### Kill Switch
Always check `severe_negative_flag` before including a ticker:

```python
if catalyst_summary and catalyst_summary.severe_negative_flag:
    # Trial terminated/suspended - exclude from ranking
    continue
```

### First Run
On the first run, Module 3 will:
- Create initial state snapshot
- Detect 0 events (no prior state to compare)
- Subsequent runs will detect deltas

### State Management
- State snapshots accumulate over time
- Each snapshot is ~2-5 MB
- Keep last 52 snapshots (1 year of weekly runs)

---

## ✅ Verification

After integration, verify it works:

```powershell
# Run full screening with Module 3
python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_with_catalyst.json

# Check catalyst events were created
ls production_data/catalyst_events_*.json
ls production_data/ctgov_state/state_*.jsonl
```

---

## 🔧 Troubleshooting

**Issue: "No module named 'module_3_catalyst'"**
- Solution: Ensure all Module 3 files are in your project directory

**Issue: "No prior snapshot found"**
- Expected on first run
- Will create initial snapshot
- Events will be detected on subsequent runs

**Issue: "Severe negatives: 0"**
- Normal if no trials were terminated/suspended
- Severe negatives are rare events

---

## 📈 Next Steps

After integration:
1. Run first screening (creates initial snapshot)
2. Wait 1 week
3. Run second screening (will detect deltas)
4. Review catalyst_events_*.json for detected events
5. Validate Module 5 composite scores include catalyst

---

This completes the Module 3A integration!
