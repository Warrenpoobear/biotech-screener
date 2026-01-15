#!/usr/bin/env python3
"""
verify_module3_scores.py - Verify Module 3 catalyst scoring is working
"""

import json
from collections import Counter

print("\n" + "="*80)
print("MODULE 3 VERIFICATION - Catalyst Scoring")
print("="*80 + "\n")

# Load results
try:
    with open('test_module3.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ test_module3.json not found")
    print("   Run: python run_screen.py --as-of-date 2026-01-07 --data-dir production_data --output test_module3.json")
    exit(1)

# Check if Module 3 scores exist
m3_data = data.get('module_3_catalyst', {})
scores = m3_data.get('scores', [])

if not scores:
    print("❌ No Module 3 scores found in output")
    print("   Module 3 scoring may not be integrated")
    exit(1)

print(f"✅ Found Module 3 scores for {len(scores)} tickers")

# Extract catalyst scores
catalyst_scores = [s['catalyst_normalized'] for s in scores]

print(f"\nCatalyst Score Statistics:")
print(f"  Min score: {min(catalyst_scores):.2f}")
print(f"  Max score: {max(catalyst_scores):.2f}")
print(f"  Mean score: {sum(catalyst_scores) / len(catalyst_scores):.2f}")
print(f"  Unique scores: {len(set(catalyst_scores))}")

# Check distribution
all_fifty = all(s == 50.0 for s in catalyst_scores)
print(f"\nAll scores 50.0? {all_fifty}")

if all_fifty:
    print("\n✅ This is EXPECTED if no events detected (same-day re-run)")
    print("   Module 3A delta detection: today vs today = 0 events")
    print("   All tickers get neutral score (50.0)")
    print("\n📅 To see varied scores:")
    print("   1. Wait for next week's data collection (new trial updates)")
    print("   2. OR run with different --as-of-date (historical comparison)")
else:
    print(f"\n✅ SUCCESS: Module 3 scoring is working!")
    print(f"   Scores vary from {min(catalyst_scores):.1f} to {max(catalyst_scores):.1f}")

# Show event detection stats
diag = m3_data.get('diagnostic_counts', {})
print(f"\nEvent Detection Statistics:")
print(f"  Events detected: {diag.get('events_detected', 0)}")
print(f"  Tickers with events: {diag.get('tickers_with_events', 0)}")
print(f"  Severe negatives: {diag.get('severe_negatives', 0)}")

# Sentiment distribution
sentiments = [s.get('catalyst_sentiment', 'UNKNOWN') for s in scores]
sentiment_counts = Counter(sentiments)

print(f"\nSentiment Distribution:")
for sentiment, count in sentiment_counts.most_common():
    pct = (count / len(scores)) * 100
    print(f"  {sentiment:20s}: {count:3d} ({pct:5.1f}%)")

# Show examples
print(f"\nExample Scores:")
for score_data in scores[:10]:
    ticker = score_data['ticker']
    score = score_data['catalyst_normalized']
    sentiment = score_data.get('catalyst_sentiment', 'UNKNOWN')
    event_count = score_data.get('event_count', 0)
    print(f"  {ticker:6s}: {score:5.1f}  |  {sentiment:20s}  |  {event_count} events")

# Check if scores changed vs Module 2
print("\n" + "="*80)
print("COMPARISON: Module 2 vs Module 3")
print("="*80 + "\n")

m2_scores = [s['financial_normalized'] for s in data['module_2_financial']['scores']]

print(f"Module 2 (Financial):")
print(f"  Range: {min(m2_scores):.1f} - {max(m2_scores):.1f}")
print(f"  Unique: {len(set(m2_scores))}")

print(f"\nModule 3 (Catalyst):")
print(f"  Range: {min(catalyst_scores):.1f} - {max(catalyst_scores):.1f}")
print(f"  Unique: {len(set(catalyst_scores))}")

# Composite score check
composite_scores = [r['composite_score'] for r in data['module_5_composite']['ranked_securities']]
print(f"\nComposite (Final):")
print(f"  Range: {min(composite_scores):.2f} - {max(composite_scores):.2f}")
print(f"  Expected: Wider than Module 2 only (15-70)")

print("\n" + "="*80)
print("✅ VERIFICATION COMPLETE")
print("="*80)

# Summary
print(f"\n📊 SUMMARY:")
if all_fifty and diag.get('events_detected', 0) == 0:
    print("  ✅ Module 3 scoring integrated correctly")
    print("  ✅ No events detected (expected for same-day re-run)")
    print("  ✅ All tickers neutral at 50.0 (correct behavior)")
    print("\n  📅 Next: Run with different date to see event detection")
elif not all_fifty:
    print("  ✅ Module 3 scoring integrated correctly")
    print(f"  ✅ {diag.get('events_detected', 0)} events detected")
    print(f"  ✅ Scores vary: {min(catalyst_scores):.1f} - {max(catalyst_scores):.1f}")
    print("  ✅ Week 2 Day 1 COMPLETE!")
else:
    print("  ⚠️  Module 3 might not be fully integrated")
    print("  Check WEEK_2_DAY_1_INTEGRATION.md for troubleshooting")

print()
