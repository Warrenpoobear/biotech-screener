#!/usr/bin/env python3
"""
Analyze how momentum enrichment changes your rankings.

Shows which tickers benefit/suffer from momentum integration.
"""

import json
import sys
from pathlib import Path
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_momentum_impact():
    """Compare base composite scores vs momentum signals."""
    print("=" * 70)
    print("MOMENTUM IMPACT ANALYSIS")
    print("=" * 70)

    # Load momentum signals
    momentum_file = Path("outputs/momentum_signals.json")
    if not momentum_file.exists():
        print("\nERROR: Run momentum calculation first")
        print("Run: python scripts/calculate_momentum_batch.py")
        return 1

    with open(momentum_file) as f:
        momentum_data = json.load(f)

    signals = momentum_data.get('signals', {})
    ranked_list = momentum_data.get('ranked_list', [])

    # Load regime info
    from src.scoring.integrate_momentum_regime_adaptive import calculate_xbi_regime, load_xbi_returns

    xbi_returns = load_xbi_returns()
    regime, xbi_90d = calculate_xbi_regime(xbi_returns, '2026-01-14')

    print(f"\nMarket Regime: {regime.upper()}")
    print(f"XBI 90-day return: {float(xbi_90d):.2%}")
    print(f"Momentum weight applied: {'25%' if regime == 'risk_on' else '15%' if regime == 'neutral' else '5%'}")
    print(f"Tickers analyzed: {len(signals)}")

    # Get momentum weights
    REGIME_WEIGHTS = {
        "risk_on": Decimal("0.25"),
        "neutral": Decimal("0.15"),
        "risk_off": Decimal("0.05")
    }
    mom_weight = REGIME_WEIGHTS[regime]

    # Analyze momentum distribution
    scores = [s['composite_momentum_score'] for s in signals.values()]
    avg_score = sum(scores) / len(scores)
    high_momentum = [t for t, s in signals.items() if s['composite_momentum_score'] > 80]
    low_momentum = [t for t, s in signals.items() if s['composite_momentum_score'] < 20]

    print(f"\nMomentum Score Distribution:")
    print(f"  Average: {avg_score:.1f}")
    print(f"  High momentum (>80): {len(high_momentum)} tickers")
    print(f"  Low momentum (<20): {len(low_momentum)} tickers")

    # Top momentum tickers
    print("\n" + "=" * 70)
    print("TOP 20 MOMENTUM TICKERS")
    print("(These get the biggest score boosts)")
    print("=" * 70)

    for i, item in enumerate(ranked_list[:20], 1):
        ticker = item['ticker']
        score = item['momentum_score']
        confidence = item['confidence']
        # Calculate approximate boost (score * weight * confidence_mult)
        conf_mult = signals[ticker]['confidence_multiplier']
        boost = score * float(mom_weight) * conf_mult
        print(f"{i:2d}. {ticker:6s}  Score: {score:5.1f}  Conf: {confidence:4s}  Est. Boost: +{boost:.1f} pts")

    # Bottom momentum tickers
    print("\n" + "=" * 70)
    print("BOTTOM 20 MOMENTUM TICKERS")
    print("(These get the biggest score drags)")
    print("=" * 70)

    for i, item in enumerate(ranked_list[-20:], 1):
        ticker = item['ticker']
        score = item['momentum_score']
        confidence = item['confidence']
        conf_mult = signals[ticker]['confidence_multiplier']
        # Drag = (score - 50) * weight * confidence (compared to neutral 50)
        drag = (score - 50) * float(mom_weight) * conf_mult
        print(f"{i:2d}. {ticker:6s}  Score: {score:5.1f}  Conf: {confidence:4s}  Est. Drag: {drag:+.1f} pts")

    # Show expected score impact for different base score scenarios
    print("\n" + "=" * 70)
    print("EXPECTED SCORE IMPACT EXAMPLES")
    print(f"(With {regime.upper()} regime = {float(mom_weight)*100:.0f}% momentum weight)")
    print("=" * 70)

    examples = [
        ("High Quality + High Momentum", 80, 95),
        ("High Quality + Low Momentum", 80, 15),
        ("Med Quality + High Momentum", 60, 95),
        ("Med Quality + Low Momentum", 60, 15),
        ("Low Quality + High Momentum", 40, 95),
        ("Low Quality + Low Momentum", 40, 15),
    ]

    print(f"\n{'Scenario':<35} {'Base':>6} {'Mom':>6} {'Final':>8} {'Change':>8}")
    print("-" * 70)

    for name, base, mom in examples:
        # Final = base * (1 - mom_weight) + mom * mom_weight
        base_weight = Decimal("1.0") - mom_weight
        final = float(base) * float(base_weight) + float(mom) * float(mom_weight)
        change = final - float(base)
        print(f"{name:<35} {base:>6.1f} {mom:>6.1f} {final:>8.1f} {change:>+8.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print(f"""
1. RISK-ON REGIME ACTIVE
   - XBI up {float(xbi_90d):.1%} over 90 days = strong bull market
   - Momentum gets full 25% weight
   - Your validation showed 87% spread in similar conditions

2. INTEGRATION IMPACT
   - {len(high_momentum)} tickers get significant momentum boosts
   - {len(low_momentum)} tickers get significant momentum drags
   - Average impact: ±{float(mom_weight) * 25:.1f} points for extreme momentum

3. SWEET SPOT CANDIDATES
   - Look for tickers with BOTH high base scores AND high momentum
   - These have fundamental quality + market confirmation

4. WATCH LIST
   - Low momentum tickers may be value traps or early-stage recovery
   - Consider as contrarian plays only with strong fundamentals
""")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(analyze_momentum_impact())
