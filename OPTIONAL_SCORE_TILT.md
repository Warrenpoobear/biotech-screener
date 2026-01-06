# Optional: Score-Tilted Weighting

## 🎯 **What This Does:**

Makes the **defensive multiplier affect weights** (not just ranking).

Currently:
- Elite 1.40x → Changes score → Changes rank → **Doesn't change weight**
- Weight = pure inverse-volatility (independent of score)

With score-tilt:
- Elite 1.40x → Higher score → Higher rank → **ALSO higher weight**
- Weight = inverse-volatility × score_rank_percentile

---

## 📊 **The Concept:**

### **Current (Pure Inverse-Vol):**

```python
raw_weight = 1 / (vol^2.0)

ARGX (22% vol, elite):
  Score: 44.10 (elite bonus applied)
  Rank: 1
  Weight: 1/0.22^2 = 20.7 raw → 5.2% final

GILD (24% vol, elite):
  Score: 44.10 (same as ARGX)
  Rank: 3
  Weight: 1/0.24^2 = 17.4 raw → 4.6% final

Ratio: 5.2% / 4.6% = 1.13x (purely from volatility)
```

### **With Score-Tilt:**

```python
raw_weight = (1 / vol^2.0) × score_tilt_factor

score_tilt_factor = 0.7 + (0.8 × rank_percentile)
  # Rank 1 (100th %ile): 0.7 + 0.8 = 1.5x
  # Rank 50 (50th %ile): 0.7 + 0.4 = 1.1x
  # Rank 98 (0th %ile): 0.7 + 0.0 = 0.7x

ARGX (22% vol, elite, rank 1):
  Base: 1/0.22^2 = 20.7
  Tilt: 1.5x (top rank)
  Total: 20.7 × 1.5 = 31.1 raw → 6.2% final

GILD (24% vol, elite, rank 3):
  Base: 1/0.24^2 = 17.4
  Tilt: 1.47x (rank 3)
  Total: 17.4 × 1.47 = 25.6 raw → 5.1% final

Ratio: 6.2% / 5.1% = 1.22x (from vol + rank)
```

**Result:** Elite bonus now affects weights indirectly through ranking!

---

## 🔧 **Implementation:**

### **Step 1: Add Score-Tilt Function**

Add this to `defensive_overlay_adapter.py`:

```python
def calculate_score_tilt_factor(rank: int, total_rankable: int) -> Decimal:
    """
    Calculate score-based tilt factor for position weighting.
    
    Maps rank to a multiplier:
    - Best rank (1): 1.5x
    - Median rank: 1.1x
    - Worst rank: 0.7x
    
    This makes higher-scored names (after defensive multiplier)
    get proportionally more weight beyond just inverse-vol.
    
    Args:
        rank: Composite rank (1 = best)
        total_rankable: Total number of rankable securities
    
    Returns:
        Tilt factor between 0.7 and 1.5
    """
    if total_rankable <= 1:
        return Decimal("1.0")
    
    # Calculate rank percentile (0 = worst, 1 = best)
    rank_percentile = Decimal(total_rankable - rank) / Decimal(total_rankable - 1)
    
    # Tilt factor: 0.7 (worst) to 1.5 (best)
    # Formula: 0.7 + 0.8 * rank_percentile
    tilt = Decimal("0.7") + (rank_percentile * Decimal("0.8"))
    
    return tilt
```

### **Step 2: Apply Tilt in Enrichment**

Modify `enrich_with_defensive_overlays()` after step 1 (score adjustment):

```python
# After applying defensive multiplier and re-ranking...

# NEW: Apply score-tilt to weights (if enabled)
if apply_position_sizing:
    total_rankable = len([r for r in ranked if r.get("rankable", True)])
    
    for rec in ranked:
        if not rec.get("rankable", True):
            continue
            
        ticker = rec["ticker"]
        ticker_data = scores_by_ticker.get(ticker, {})
        defensive_features = ticker_data.get("defensive_features", {})
        
        # Get base inverse-vol weight
        base_weight = raw_inv_vol_weight(defensive_features or {})
        
        # Apply score tilt
        rank = rec.get("composite_rank", 999)
        tilt_factor = calculate_score_tilt_factor(rank, total_rankable)
        
        # Combine
        if base_weight:
            tilted_weight = base_weight * tilt_factor
            rec["_position_weight_raw"] = tilted_weight
            rec["_score_tilt_factor"] = str(tilt_factor)  # For diagnostics
        else:
            rec["_position_weight_raw"] = None
```

### **Step 3: Use It**

In `module_5_composite_with_defensive.py`:

```python
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
    top_n=60,
    use_score_tilt=True,  # NEW: Enable score-tilted weighting
)
```

---

## 📊 **Expected Impact (Top-60 + Score-Tilt):**

### **Without Score-Tilt (Current):**

```
Top 10 (all similar scores, differentiated by vol):
1. ARGX   5.20%  (22% vol, elite, rank 1)
2. GILD   4.60%  (24% vol, elite, rank 3)
3. HALO   3.80%  (28% vol, elite, rank 5)
4. GLPG   3.20%  (31% vol, elite, rank 4)
5. OPCH   3.15%  (30% vol, elite, rank 7)
...

Range: 6.1:1
Differentiation: Purely from volatility
```

### **With Score-Tilt:**

```
Top 10 (differentiated by BOTH vol and rank):
1. ARGX   6.20%  (22% vol, elite, rank 1) ← +19% from tilt!
2. GILD   5.40%  (24% vol, elite, rank 3) ← +17% from tilt
3. HALO   4.35%  (28% vol, elite, rank 5) ← +14% from tilt
4. GLPG   3.60%  (31% vol, elite, rank 4) ← +13% from tilt
5. OPCH   3.50%  (30% vol, elite, rank 7) ← +11% from tilt
...

Range: 7.8:1  ← Increased from 6.1:1
Differentiation: From vol + score/rank
```

**Key difference:** 
- Rank 1 gets 1.5x tilt
- Rank 60 gets 0.72x tilt
- Creates **additional** 2.1x spread on top of vol weighting!

---

## 💡 **When to Use Score-Tilt:**

### **Use It If:**

- ✅ You want **elite bonus to affect weights**
- ✅ You want **more differentiation** between top names
- ✅ You believe **scores contain alpha** beyond just vol
- ✅ You're comfortable with **~30% more to top name**

### **Skip It If:**

- ❌ You want **pure risk-based weighting**
- ❌ Top-N selection alone gives enough concentration
- ❌ You prefer **simpler, more interpretable** weights
- ❌ Multiplier-affecting-scores-only is sufficient

---

## 🎯 **Recommendation:**

### **My Suggestion:**

**Start with Top-N alone** (no score-tilt):
1. Clean, interpretable
2. Simpler to explain
3. Pure risk-based sizing
4. Achieves 4-5% max weights

**Add score-tilt later if needed:**
- After a few weeks of live data
- If you want more top-name conviction
- If you find scores predictive

---

## 📋 **Comparison Matrix:**

| Configuration | Max Weight | How It Works | Complexity |
|---------------|------------|--------------|------------|
| **Full 98** | 3.7% | All stocks, pure inv-vol | Simple ✅ |
| **Top-60** | 5.2% | Fewer stocks, pure inv-vol | Simple ✅ |
| **Top-60 + Tilt** | 6.2% | Fewer stocks + score tilt | Moderate ⚠️ |
| **Top-40 + Tilt** | 7.0% (capped) | Very selective + tilt | Complex ⚠️ |

---

## 🚀 **Implementation Priority:**

```
Priority 1: Top-N selection (60)
  → Clean, effective, interpretable
  → Achieves 4-5% max weights
  → Recommended starting point

Priority 2: Monitor live performance
  → See if top-N alone is sufficient
  → Evaluate if more concentration needed

Priority 3: Add score-tilt (optional)
  → Only if you want multipliers to affect weights
  → Only if you want more top-name conviction
  → Can always add later
```

---

## ✅ **Recommendation:**

**For your first live version:**

```python
# In module_5_composite_with_defensive.py:

output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,      # ✅ Keep this
    apply_position_sizing=True,  # ✅ Keep this
    top_n=60,                    # ✅ ADD THIS (priority 1)
    # use_score_tilt=False,     # ⏸️  Skip for now (priority 3)
)
```

**This gives you:**
- Clean implementation
- 5% max weights
- Interpretable ("top 60")
- Easy to explain
- Room to enhance later

---

## 🎊 **Summary:**

**Top-N Selection:**
- ✅ Must have
- ✅ Clean solution
- ✅ 4-5% max weights
- ✅ Ready to ship

**Score-Tilt:**
- ⏸️  Nice to have
- ⏸️  Adds complexity
- ⏸️  6-7% max weights
- ⏸️  Consider after live testing

**Start simple, enhance later!** 🎯
