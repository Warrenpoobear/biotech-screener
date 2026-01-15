#!/usr/bin/env python3
"""
test_module_3_scoring.py - Unit tests for Module 3 catalyst scoring

Run: python test_module_3_scoring.py
"""

from module_3_scoring import (
    classify_event_severity,
    calculate_ticker_catalyst_score,
    score_catalyst_events
)


def test_severity_classification():
    """Test event severity classification"""
    
    print("\n" + "="*60)
    print("TEST 1: Event Severity Classification")
    print("="*60)
    
    # Test 1: Trial termination (severe negative)
    severity = classify_event_severity('TRIAL_TERMINATED', {})
    assert severity == 'SEVERE_NEGATIVE'
    print("✅ Trial termination → SEVERE_NEGATIVE")
    
    # Test 2: Phase advance (critical positive)
    severity = classify_event_severity(
        'PHASE_CHANGE',
        {'old_phase': 'PHASE2', 'new_phase': 'PHASE3'}
    )
    assert severity == 'CRITICAL_POSITIVE'
    print("✅ Phase advance P2→P3 → CRITICAL_POSITIVE")
    
    # Test 3: Enrollment complete (positive)
    severity = classify_event_severity(
        'STATUS_CHANGE',
        {'old_status': 'RECRUITING', 'new_status': 'ACTIVE_NOT_RECRUITING'}
    )
    assert severity == 'POSITIVE'
    print("✅ Enrollment complete → POSITIVE")
    
    # Test 4: Major delay (negative)
    severity = classify_event_severity(
        'TIMELINE_CHANGE',
        {'shift_days': 120}
    )
    assert severity == 'NEGATIVE'
    print("✅ 120-day delay → NEGATIVE")
    
    print("✅ All severity tests passed!")


def test_ticker_scoring():
    """Test ticker-level scoring"""
    
    print("\n" + "="*60)
    print("TEST 2: Ticker Scoring")
    print("="*60)
    
    # Test 1: Severe negative (trial termination)
    summary = {
        'events': [
            {'event_type': 'TRIAL_TERMINATED', 'details': {}}
        ]
    }
    result = calculate_ticker_catalyst_score('TEST', summary)
    assert result['catalyst_normalized'] == 20.0
    assert result['catalyst_sentiment'] == 'SEVERE_NEGATIVE'
    print(f"✅ Trial termination → Score {result['catalyst_normalized']} (expected 20)")
    
    # Test 2: Phase advance (critical positive)
    summary = {
        'events': [
            {
                'event_type': 'PHASE_CHANGE',
                'details': {'old_phase': 'PHASE2', 'new_phase': 'PHASE3'}
            }
        ]
    }
    result = calculate_ticker_catalyst_score('TEST', summary)
    assert result['catalyst_normalized'] >= 70.0
    assert result['catalyst_sentiment'] == 'CRITICAL_POSITIVE'
    print(f"✅ Phase advance → Score {result['catalyst_normalized']} (expected 75-80)")
    
    # Test 3: Positive events
    summary = {
        'events': [
            {
                'event_type': 'STATUS_CHANGE',
                'details': {'old_status': 'RECRUITING', 'new_status': 'ACTIVE_NOT_RECRUITING'}
            },
            {
                'event_type': 'RESULTS_POSTED',
                'details': {}
            }
        ]
    }
    result = calculate_ticker_catalyst_score('TEST', summary)
    assert result['catalyst_normalized'] > 50.0
    assert result['catalyst_sentiment'] == 'POSITIVE'
    print(f"✅ 2 positive events → Score {result['catalyst_normalized']} (expected 60-70)")
    
    # Test 4: Negative events
    summary = {
        'events': [
            {
                'event_type': 'TIMELINE_CHANGE',
                'details': {'shift_days': 120}
            }
        ]
    }
    result = calculate_ticker_catalyst_score('TEST', summary)
    assert result['catalyst_normalized'] < 50.0
    assert result['catalyst_sentiment'] == 'NEGATIVE'
    print(f"✅ Negative event → Score {result['catalyst_normalized']} (expected 40-45)")
    
    # Test 5: No events (neutral)
    summary = {'events': []}
    result = calculate_ticker_catalyst_score('TEST', summary)
    assert result['catalyst_normalized'] == 50.0
    assert result['catalyst_sentiment'] == 'NEUTRAL'
    print(f"✅ No events → Score {result['catalyst_normalized']} (expected 50)")
    
    print("✅ All ticker scoring tests passed!")


def test_batch_scoring():
    """Test scoring multiple tickers"""
    
    print("\n" + "="*60)
    print("TEST 3: Batch Scoring")
    print("="*60)
    
    summaries = {
        'GOOD': {
            'events': [
                {
                    'event_type': 'PHASE_CHANGE',
                    'details': {'old_phase': 'PHASE2', 'new_phase': 'PHASE3'}
                }
            ]
        },
        'BAD': {
            'events': [
                {'event_type': 'TRIAL_TERMINATED', 'details': {}}
            ]
        },
        'NEUTRAL': {
            'events': []
        }
    }
    
    active_tickers = ['GOOD', 'BAD', 'NEUTRAL', 'MISSING']
    
    results = score_catalyst_events(summaries, active_tickers)
    
    assert len(results) == 4
    print(f"✅ Scored {len(results)} tickers")
    
    # Check individual scores
    scores = {r['ticker']: r['catalyst_normalized'] for r in results}
    
    assert scores['GOOD'] > 70.0, "GOOD ticker should have high score"
    assert scores['BAD'] == 20.0, "BAD ticker should have severe negative score"
    assert scores['NEUTRAL'] == 50.0, "NEUTRAL ticker should have neutral score"
    assert scores['MISSING'] == 50.0, "MISSING ticker should default to neutral"
    
    print(f"✅ GOOD: {scores['GOOD']:.1f} (>70)")
    print(f"✅ BAD: {scores['BAD']:.1f} (=20)")
    print(f"✅ NEUTRAL: {scores['NEUTRAL']:.1f} (=50)")
    print(f"✅ MISSING: {scores['MISSING']:.1f} (=50)")
    
    # Check all scores are different
    unique_scores = len(set(scores.values()))
    assert unique_scores >= 3, "Should have at least 3 unique scores"
    print(f"✅ {unique_scores} unique scores (differentiation working)")
    
    print("✅ All batch scoring tests passed!")


def main():
    """Run all tests"""
    
    print("\n" + "="*80)
    print("MODULE 3 SCORING - TEST SUITE")
    print("="*80)
    
    try:
        test_severity_classification()
        test_ticker_scoring()
        test_batch_scoring()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nModule 3 scoring is ready for integration!")
        print("="*80 + "\n")
        
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
