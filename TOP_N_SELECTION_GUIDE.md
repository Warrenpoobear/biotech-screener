# Top-N Selection Implementation Guide

## 🎯 **What This Does:**

Instead of sizing all 98 stocks, we **only invest in the top N** after ranking:
- Top 60: Balanced conviction (recommended for first live version)
- Top 40: High conviction (sharper, higher max weights)

**Result:** Max weights naturally rise to 4-6% without changing math!

---

## ✅ **Implementation Complete:**

I've added `top_n` parameter to:
1. `apply_caps_and_renormalize()` - Does the surgical cut
2. `enrich_with_defensive_overlays()` - Passes it through

---

## 🔧 **How to Enable (Two Steps):**

### **Step 1: Modify Your Calling Code**

In `module_5_composite_with_defensive.py`, find where you call `enrich_with_defensive_overlays()` and add the `top_n` parameter:

```python
# BEFORE (invests in all 98 stocks):
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
)

# AFTER (invests in top 60 only):
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
    top_n=60,  # NEW: Only size top 60 names
)
```

### **Step 2: Run Screener**

```powershell
python run_screen.py `
    --as-of-date 2026-01-06 `
    --data-dir production_data `
    --output screening_TOP60.json
```

---

## 📊 **Expected Results:**

### **Top-60 (Balanced):**

```
Universe: 98 stocks evaluated
Invested: 60 stocks (top 60 by composite rank)
Excluded: 38 stocks (flagged "NOT_IN_TOP_N")

Expected weights:
  Max: 4.8-5.5% (was 3.7%)
  Min: 0.7-0.9% (higher floor due to fewer stocks)
  Range: 6-8:1
  Top 10: 30-35% (was 24.7%)
```

### **Top-40 (High Conviction):**

```
Universe: 98 stocks evaluated
Invested: 40 stocks (top 40 by composite rank)
Excluded: 58 stocks

Expected weights:
  Max: 6.0-7.0% (at cap!)
  Min: 1.0-1.3%
  Range: 5-7:1
  Top 10: 40-45%
```

---

## 💡 **Why This Works:**

### **The Math:**

**Before (98 stocks):**
```
Investable: 90%
Stocks: 98
Average: 0.92%
Max with vol^2.0: ~3.7% (normalization compression)
```

**After (60 stocks):**
```
Investable: 90%
Stocks: 60  ← 38% fewer!
Average: 1.50%
Max with vol^2.0: ~5.0% (less compression!)
```

**The key:** Fewer stocks → Less total "weight mass" → Less normalization compression → Higher max weights!

---

## 🎯 **Recommendation Matrix:**

### **Choose Your N:**

| Top-N | Max Weight | Top 10 Conc. | Use Case |
|-------|------------|--------------|----------|
| **60** | 4.8-5.5% | 30-35% | **Recommended for launch** ✅ |
| **50** | 5.5-6.0% | 33-38% | Higher conviction |
| **40** | 6.0-7.0% | 40-45% | Sharp conviction |
| **98** | 3.7% | 25% | Full universe (current) |

**For biotech's idiosyncratic nature, I recommend starting with N=60.**

---

## 🔍 **How Selection Works:**

### **Step-by-Step:**

1. **Score all 98 stocks** (modules 1-4 + defensive multiplier)
2. **Rank by composite score** (1-98)
3. **Select top N** (e.g., ranks 1-60)
4. **Set ranks 61-98 to zero weight**
5. **Flag them:** `"NOT_IN_TOP_N"`
6. **Normalize weights** over only the 60 invested stocks

### **Example Output:**

```
Rank 1: ARGX   Score: 44.10, Weight: 5.20% ✓ Invested
Rank 2: GILD   Score: 44.10, Weight: 4.60% ✓ Invested
...
Rank 60: XYZ   Score: 31.50, Weight: 1.10% ✓ Invested (last)
Rank 61: ABC   Score: 31.20, Weight: 0.00% ✗ NOT_IN_TOP_N
Rank 62: DEF   Score: 30.80, Weight: 0.00% ✗ NOT_IN_TOP_N
...
Rank 98: ZZZ   Score: 15.00, Weight: 0.00% ✗ NOT_IN_TOP_N
```

---

## 📋 **Validation After Running:**

### **Check the Results:**

```powershell
python -c "
import json
data = json.load(open('screening_TOP60.json'))
ranked = data['module_5_composite']['ranked_securities']

# Count invested vs excluded
invested = [r for r in ranked if float(r['position_weight']) > 0]
excluded_topn = [r for r in ranked if 'NOT_IN_TOP_N' in r.get('position_flags', [])]

print(f'Invested: {len(invested)} stocks')
print(f'Excluded by top-N: {len(excluded_topn)} stocks')
print()

# Check weights
weights = [float(r['position_weight']) for r in invested]
print(f'Max weight: {max(weights)*100:.2f}%')
print(f'Min weight: {min(weights)*100:.2f}%')
print(f'Range: {max(weights)/min(weights):.1f}:1')
print(f'Top 10: {sum(sorted(weights, reverse=True)[:10])*100:.1f}%')
print()

# Show cutoff
last_invested = [r for r in ranked if float(r['position_weight']) > 0][-1]
first_excluded = [r for r in ranked if 'NOT_IN_TOP_N' in r.get('position_flags', [])][0]
print(f'Cutoff:')
print(f'  Last invested: Rank {last_invested[\"composite_rank\"]}, {last_invested[\"ticker\"]}, Score {last_invested[\"composite_score\"]}')
print(f'  First excluded: Rank {first_excluded[\"composite_rank\"]}, {first_excluded[\"ticker\"]}, Score {first_excluded[\"composite_score\"]}')
"
```

**Expected output:**
```
Invested: 60 stocks
Excluded by top-N: 38 stocks

Max weight: 5.20%
Min weight: 0.85%
Range: 6.1:1
Top 10: 32.5%

Cutoff:
  Last invested: Rank 60, TECH, Score 31.50
  First excluded: Rank 61, RARE, Score 31.20
```

---

## 🎊 **Benefits of Top-N Selection:**

### **1. Higher Max Weights (4-6%)**
- Natural result of fewer stocks
- No math hacking needed
- Stays under 7% cap

### **2. More Conviction Per Name**
- Only invest in highest-conviction ideas
- Drop the marginal/questionable names
- Focus research resources

### **3. Lower Transaction Costs**
- Fewer names = fewer trades
- Lower turnover
- More manageable for execution

### **4. Interpretability**
- Clear narrative: "We own the top 60"
- Easy to explain to IC
- Audit trail clear

### **5. Scales Naturally**
- Can adjust N up/down based on market conditions
- Easy to make more/less aggressive
- No code changes needed

---

## ⚙️ **Advanced: Make N Dynamic (Optional)**

If you want to vary N based on market conditions:

```python
def calculate_top_n(market_conditions: str) -> int:
    """
    Dynamically adjust N based on market conditions.
    """
    if market_conditions == "high_conviction":
        return 40  # Concentrated
    elif market_conditions == "balanced":
        return 60  # Normal
    elif market_conditions == "defensive":
        return 80  # Diversified
    else:
        return None  # Full universe
```

---

## 🚀 **Next Steps:**

### **For Your First Live Run:**

1. **Set N=60** in `module_5_composite_with_defensive.py`
2. **Run screener** with your 98-stock universe
3. **Validate** results using the script above
4. **Expected:** Max ~5%, Top 10 ~32%, Range ~6:1

### **Tuning Over Time:**

```
Week 1-2: Run with N=60, observe
Week 3-4: Try N=50 or N=70, compare
Week 5+:  Settle on optimal N for your strategy
```

---

## 📊 **Comparison: Full Universe vs Top-60**

| Metric | Full 98 | Top 60 | Improvement |
|--------|---------|--------|-------------|
| **Stocks invested** | 98 | 60 | -39% positions |
| **Max weight** | 3.7% | 5.2% | **+41%** |
| **Top 10 conc.** | 24.7% | 32.5% | **+32%** |
| **Min weight** | 0.45% | 0.85% | +89% |
| **Range** | 8.2:1 | 6.1:1 | More balanced |
| **Execution cost** | Higher | Lower | Fewer trades |

---

## 🎯 **Final Configuration:**

```python
# In module_5_composite_with_defensive.py:

TOP_N = 60  # Start here for balanced conviction

output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
    top_n=TOP_N,  # Enable top-N selection
)
```

```python
# Core parameters (already set):
VOLATILITY_EXPONENT = 2.0    # Variance weighting
ELITE_BONUS = 1.40           # 40% for elite diversifiers
GOOD_BONUS = 1.10            # 10% for good diversifiers
MAX_POSITION = 0.07          # 7% safety cap
CASH_TARGET = 0.10           # 10% cash buffer
```

---

## ✅ **This Is The Clean Solution!**

Top-N selection:
- ✅ Increases max weights naturally (no math hacking)
- ✅ More conviction per name
- ✅ Lower transaction costs
- ✅ Interpretable ("we own the top 60")
- ✅ Biotech-appropriate (idiosyncratic, catalyst-driven)
- ✅ Scales easily (adjust N as needed)

**This is the institutional-grade approach for conviction portfolios!** 🎉
