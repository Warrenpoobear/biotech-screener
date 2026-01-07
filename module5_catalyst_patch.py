# Module 5 Integration Patch for Module 3A Catalyst Scores
# Add this to your Module 5 composite scoring loop

# ============================================================================
# Module 5: Composite Ranking with Module 3A Integration
# ============================================================================

for ticker in active_tickers:
    # Get existing module scores
    m2_score = m2_result['securities'].get(ticker, {}).get('health_score', 0.0)
    m4_score = m4_result['scores'].get(ticker, {}).get('clinical_quality_score', 0.0)
    
    # ========================================================================
    # Module 3A: Catalyst Integration
    # ========================================================================
    
    # Get catalyst summary for this ticker
    catalyst_summary = catalyst_summaries.get(ticker)
    
    severe_negative = False
    m3_net_score = 0.0
    
    if catalyst_summary:
        # Check for severe negative events (trial terminated/suspended)
        severe_negative = catalyst_summary.severe_negative_flag
        
        # Get net catalyst score (positive - negative)
        m3_net_score = catalyst_summary.catalyst_score_net
    
    # ========================================================================
    # KILL SWITCH: Exclude tickers with severe negative catalyst events
    # ========================================================================
    
    if severe_negative:
        logger.warning(f"Excluding {ticker}: severe negative catalyst event (trial terminated/suspended)")
        continue  # Skip this ticker entirely
    
    # ========================================================================
    # CONSERVATIVE PENALTY: Apply only negative catalyst impact (for now)
    # ========================================================================
    
    # Extract only the negative component (don't reward positives until calibrated)
    m3_penalty = max(0.0, -m3_net_score)
    
    # Apply conservative weight (10% penalty weight)
    # You can tune this after seeing backtests
    m3_weighted_penalty = 0.10 * m3_penalty
    
    # ========================================================================
    # Composite Score Calculation
    # ========================================================================
    
    # Original weights adjusted for Module 3A penalty
    composite_score = (
        0.25 * m2_score +           # Financial health
        0.15 * 0.0 +                # Module 3A positive signals (disabled for v1)
        0.40 * m4_score +           # Clinical development
        0.20 * other_factors        # Other factors
        - m3_weighted_penalty       # Module 3A negative penalty
    )
    
    # Alternative: if you want to use full catalyst signal (after calibration):
    # composite_score = (
    #     0.25 * m2_score +
    #     0.15 * m3_net_score +     # Full catalyst signal
    #     0.40 * m4_score +
    #     0.20 * other_factors
    # )
    
    # Store result with catalyst metadata
    results.append({
        'ticker': ticker,
        'composite_score': composite_score,
        'm3_catalyst_net': m3_net_score,
        'm3_penalty_applied': m3_weighted_penalty,
        'severe_negative_flag': severe_negative,
        # ... other fields
    })

# ============================================================================
# Why This Approach?
# ============================================================================
# 
# 1. KILL SWITCH: Severe negative events (terminated/suspended trials) are
#    high-confidence negative signals. Exclude immediately.
#
# 2. PENALTY ONLY: Until you backtest and calibrate, only apply downside.
#    This prevents overweighting an uncalibrated feature.
#
# 3. CONSERVATIVE WEIGHT: 10% penalty weight is intentionally small.
#    Increase after validation.
#
# 4. TUNING PATH: After 2-3 months of production data:
#    a. Measure precision/recall of catalyst signals
#    b. Adjust penalty weight (0.05-0.20 range)
#    c. Consider enabling positive signals (0.10-0.15 weight)
#
# ============================================================================
