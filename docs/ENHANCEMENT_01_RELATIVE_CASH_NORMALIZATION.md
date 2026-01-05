# Enhancement #01: Relative Cash Normalization (Market-Cap Scaled)

**Status:** ✅ Implemented (2026-01-05)  
**Type:** Quick Win - Financial Health Signal Enhancement  
**Impact:** High (eliminates large-cap bias, improves IC interpretability)  
**Complexity:** Low (30 minutes implementation + testing)

---

## Executive Summary

Replaced absolute cash threshold (\cash > \\) with market-cap normalized cash strength signal. This creates a cross-cap comparable "financial runway" measure that's more interpretable for IC review.

**Key Metric:** Net Cash Ratio = \
et_cash_usd / market_cap_usd\

---

## What Changed

### Before (Absolute Threshold)
`python
# Module 2: Financial Health
if net_cash > 1_000_000_000:
    cash_fortress = True
    score_boost = +15
`

**Problem:** Biased toward large-caps. No distinction between:
- Company A: \ cash on \ cap (50% ratio - very strong)
- Company B: \ cash on \ cap (10% ratio - weak)

### After (Relative Scoring)
`python
# extensions/market_cap_normalization.py
net_cash_ratio = net_cash_usd / market_cap_usd

if net_cash_ratio >= 0.50:    # ≥50%
    status = 'FORTRESS'
    score_boost = +15
elif net_cash_ratio >= 0.30:  # 30-50%
    status = 'STRONG'
    score_boost = +10
elif net_cash_ratio >= 0.15:  # 15-30%
    status = 'ADEQUATE'
    score_boost = +5
else:                          # <15%
    status = 'WEAK'
    score_boost = +0
`

**Solution:** Proportional scoring creates fair comparison across all market caps.

---

## Status Bands & Score Boosts

| Status | Net Cash Ratio | Score Boost | Interpretation |
|--------|---------------|-------------|----------------|
| **FORTRESS** | ≥50% | +15 points | Exceptional balance sheet, minimal dilution risk |
| **STRONG** | 30-50% | +10 points | Healthy cash position, low dilution risk |
| **ADEQUATE** | 15-30% | +5 points | Standard balance sheet, moderate dilution risk |
| **WEAK** | <15% | +0 points | Elevated dilution risk, near-term funding needed |
| **UNKNOWN** | Missing data | +0 points | Insufficient data for assessment |

---

## Why It Matters

### 1. **Eliminates Large-Cap Bias**
- Old system: \ cash = fortress (regardless of company size)
- New system: \ cash evaluated relative to market cap

### 2. **Cross-Cap Comparability**
- Small-cap with 40% net cash ratio > Large-cap with 15% ratio
- IC can compare financial strength across size segments

### 3. **Interpretable IC Signal**
- "They're net-cash rich relative to size"
- Direct translation to dilution risk assessment

### 4. **Maintains Determinism**
- Decimal-only arithmetic
- Stable thresholds
- Auditable calculations
- Reproducible outputs

---

## Output Schema (Per Security)

### Fields Added to Composite Results
`json
{
  "ticker": "REGN",
  "composite_score_original": "65.5",    // Before enhancement
  "composite_score": "75.5",             // After enhancement
  "relative_cash_boost": "10",           // +10 from STRONG status
  
  "relative_cash_analysis": {
    "status": "STRONG",
    "net_cash_ratio": "0.35",            // 35% of market cap
    "net_cash_usd": "7000000000",        // \
    "market_cap_usd": "20000000000",     // \
    "score_boost": "10",
    "audit_hash": "34cbbb5c6f48bff9"    // Deterministic verification
  },
  
  "module_2_financial": {
    "cash_usd": "7500000000",
    "debt_usd": "500000000"
  }
}
`

### Summary Metadata
`json
{
  "market_cap_enhancement": {
    "enhanced_count": 2,
    "missing_market_cap": [],
    "total_securities": 2
  }
}
`

---

## Determinism & PIT Checklist

### ✅ Guaranteed Properties

1. **Decimal-Only Math**
   - All ratio calculations use \Decimal\ type
   - No float contamination in threshold comparisons
   - Quantized to 2 decimal places for stable JSON serialization

2. **Stable Rounding**
`python
   DECIMAL_PRECISION = Decimal('0.01')
   cash_ratio = (net_cash / market_cap).quantize(
       DECIMAL_PRECISION, 
       rounding=ROUND_HALF_UP
   )
`

3. **PIT-Safe Inputs**
   - \market_cap_usd\ must be point-in-time as of \s_of_date\
   - \cash_usd\ and \debt_usd\ from most recent 10-Q before \s_of_date\
   - No forward-looking data contamination

4. **Audit Trails**
`python
   audit_hash = SHA256({
       'net_cash': str(net_cash),
       'market_cap': str(market_cap),
       'ratio': str(ratio),
       'status': status
   })[:16]
`

5. **Toggle Design**
   - Enhancement is **optional** via flag
   - Original scores always preserved
   - Can be enabled/disabled per run without affecting base pipeline

---

## Integration Guide

### Runtime Usage

#### Option 1: Standalone Enhancement (Current Implementation)
`powershell
# Run base screening
python run_screen.py \
  --as-of-date 2024-12-15 \
  --data-dir ./data \
  --output results.json

# Enhance with relative cash (post-processing)
python scripts/integrate_market_cap_enhancement.py \
  --input results.json \
  --output results_enhanced.json \
  --market-caps market_caps.json
`

#### Option 2: Integrated Flag (Future Production)
`powershell
# Enhanced run (when integrated into run_screen.py)
python run_screen.py \
  --as-of-date 2024-12-15 \
  --data-dir ./data \
  --output results.json \
  --enable-relative-cash \
  --market-caps market_caps.json
`

### Output Contract

**Always output both:**
- \composite_score_original\ - Baseline (Module 5 output)
- \composite_score\ - Enhanced (baseline + enhancements if enabled)

**This guarantees:**
- Easy A/B comparisons
- Transparent change attribution
- Rollback capability (use original scores)
- Audit trail for IC review

---

## Data Requirements

### Required Inputs

| Field | Source | Format | PIT Requirement |
|-------|--------|--------|-----------------|
| \market_cap_usd\ | Yahoo Finance / Bloomberg | Decimal string | As of \s_of_date\ |
| \cash_usd\ | SEC 10-Q (Module 2) | Decimal string | Most recent before \s_of_date\ |
| \debt_usd\ | SEC 10-Q (Module 2) | Decimal string | Most recent before \s_of_date\ |

### Market Cap Data File Format
`json
{
  "REGN": "20000000000",
  "VRTX": "8000000000",
  "ALNY": "3500000000"
}
`

### Handling Missing Data
`python
if market_cap_usd is None or market_cap_usd <= 0:
    return {
        'status': 'UNKNOWN',
        'score_boost': '0',
        'missing_data': 'market_cap'
    }
`

**IC Interpretation:** UNKNOWN status flags securities requiring manual review.

---

## Testing & Validation

### Test Results (2026-01-05)

**Test Case 1: Large Cap with Strong Cash**
`
Company: REGN (Large Cap)
Cash: \.5B, Debt: \.5B, Market Cap: \
Net Cash Ratio: 35%
Status: STRONG
Score Boost: +10 points
✅ PASS
`

**Test Case 2: Mid Cap with Adequate Cash**
`
Company: VRTX (Mid Cap)
Cash: \, Debt: \.5B, Market Cap: \
Net Cash Ratio: 19%
Status: ADEQUATE
Score Boost: +5 points
✅ PASS
`

**Test Case 3: Small Cap with Weak Cash**
`
Company: Small Biotech
Cash: \, Debt: \, Market Cap: \
Net Cash Ratio: 10%
Status: WEAK
Score Boost: +0 points
✅ PASS
`

### Determinism Verification
`powershell
# Run twice with identical inputs
python scripts/integrate_market_cap_enhancement.py \
  --input test.json --output run1.json --market-caps caps.json
  
python scripts/integrate_market_cap_enhancement.py \
  --input test.json --output run2.json --market-caps caps.json

# Compare outputs
fc /b run1.json run2.json
# Result: Files are identical (byte-for-byte)
✅ Determinism verified
`

---

## Implementation Files

### Created Files

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| \extensions/market_cap_normalization.py\ | Core logic | 232 | ✅ Complete |
| \scripts/integrate_market_cap_enhancement.py\ | Integration script | 149 | ✅ Complete |
| \	est_data/market_caps.json\ | Example data | 4 | ✅ Complete |
| \docs/ENHANCEMENT_01_RELATIVE_CASH_NORMALIZATION.md\ | This document | 450+ | ✅ Complete |

### Modified Files

| File | Change | Status |
|------|--------|--------|
| None | Post-processing enhancement | ✅ No core changes |

**Advantage:** Zero disruption to existing pipeline. Can be tested independently.

---

## Production Integration Roadmap

### Phase 1: Post-Processing (Current) ✅ Complete
- Standalone enhancement script
- Run after main screening pipeline
- Full testing and validation

### Phase 2: Direct Integration (Next)
Integrate into \un_screen.py\:
`python
# In run_screen.py main() function
parser.add_argument(
    '--enable-relative-cash',
    action='store_true',
    help='Enable market cap normalization for financial scoring'
)
parser.add_argument(
    '--market-caps',
    type=Path,
    help='Market cap data JSON file'
)

# After module_5_composite
if args.enable_relative_cash and args.market_caps:
    from extensions.market_cap_normalization import enhance_financial_score_with_relative_cash
    # Apply enhancement to ranked_securities
`

### Phase 3: Automated Market Cap Fetching
Replace manual JSON file with live data:
`python
def fetch_market_caps_yahoo(tickers: List[str], as_of_date: str) -> Dict[str, Decimal]:
    \"\"\"Fetch historical market caps from Yahoo Finance.\"\"\"
    # Use yfinance or similar API
    # Ensure PIT discipline (as of specific date)
`

---

## Next Enhancement Priority: Coverage Gates

### Enhancement #02: Data Coverage Confidence Scoring

**Problem:** 60% missing financial data creates false confidence in scores.

**Solution:** Add \coverage_confidence\ signal:
`python
coverage_confidence = calculate_coverage_confidence({
    'financial': ['cash_usd', 'debt_usd', 'market_cap_usd'],
    'clinical': ['lead_stage', 'next_catalyst_date']
})

if coverage_confidence < 0.6:
    # Flag for IC review
    security['data_quality_flag'] = 'SPARSE_DATA'
    # Optionally downweight composite score
    security['composite_score_adjusted'] = composite_score * coverage_confidence
`

**Impact:**
- Eliminates false confidence on incomplete data
- Makes data quality transparent in dossiers
- Provides clear "data sufficiency" flag for IC

**Estimated Time:** 1 hour implementation + testing

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-01-05 | 1.0 | Initial implementation | DS + Claude |
| 2026-01-05 | 1.1 | Documentation complete | DS + Claude |

---

## Appendix: Example Output

### Before Enhancement
`json
{
  "ticker": "REGN",
  "composite_score": "65.5",
  "module_2_financial": {
    "financial_health_score": "75",
    "cash_usd": "7500000000",
    "debt_usd": "500000000"
  }
}
`

### After Enhancement
`json
{
  "ticker": "REGN",
  "composite_score_original": "65.5",
  "composite_score": "75.5",
  "relative_cash_boost": "10",
  "relative_cash_analysis": {
    "status": "STRONG",
    "net_cash_ratio": "0.35",
    "net_cash_usd": "7000000000",
    "market_cap_usd": "20000000000",
    "score_boost": "10",
    "audit_hash": "34cbbb5c6f48bff9"
  },
  "module_2_financial": {
    "financial_health_score": "75",
    "cash_usd": "7500000000",
    "debt_usd": "500000000"
  }
}
`

**Interpretation for IC:**
- REGN has net cash of \ (cash \.5B - debt \.5B)
- On \ market cap, this is 35% (STRONG status)
- Composite score boosted from 65.5 → 75.5 (+10 points)
- Low dilution risk, healthy balance sheet relative to size

---

## Questions & Support

**For IC Questions:**
- What does STRONG vs ADEQUATE mean? → See Status Bands table
- Why did score change? → Check \elative_cash_boost\ field
- Is this data trustworthy? → Verify \udit_hash\ matches

**For Engineering Questions:**
- How to enable/disable? → Use \--enable-relative-cash\ flag
- How to verify determinism? → Run twice, compare audit hashes
- Where is data sourced? → See Data Requirements section

---

**END OF DOCUMENT**