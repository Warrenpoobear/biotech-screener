#!/usr/bin/env python3
"""
Run momentum integration on Module 5 ranked output.

Usage:
    python scripts/run_momentum_integration.py
    python scripts/run_momentum_integration.py --ranked-path outputs/ranked_with_real_defensive_FINAL.json
"""

import argparse
import json
import sys
from pathlib import Path
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Integrate momentum signals into Module 5 output")
    parser.add_argument(
        "--ranked-path",
        default="outputs/ranked_with_real_defensive_FINAL.json",
        help="Path to Module 5 ranked output JSON file"
    )
    parser.add_argument(
        "--output-path",
        default="outputs/ranked_with_momentum.json",
        help="Path for enriched output"
    )
    args = parser.parse_args()

    # Load momentum signals
    momentum_path = Path("outputs/momentum_signals.json")
    if not momentum_path.exists():
        print("Error: Momentum signals not found. Run calculate_momentum_batch.py first.")
        return 1

    with open(momentum_path) as f:
        momentum_data = json.load(f)

    signals = momentum_data.get("signals", {})
    print(f"Loaded momentum signals for {len(signals)} tickers")

    # Calculate regime
    returns_path = Path("data/returns/returns_db_daily.json")
    if returns_path.exists():
        from src.scoring.integrate_momentum_regime_adaptive import calculate_xbi_regime, load_xbi_returns

        xbi_returns = load_xbi_returns(str(returns_path))
        calc_date = momentum_data.get("metadata", {}).get("calculation_date", "2026-01-14")
        regime, xbi_return = calculate_xbi_regime(xbi_returns, calc_date)

        regime_weights = {"risk_on": 0.25, "neutral": 0.15, "risk_off": 0.05}
        momentum_weight = regime_weights.get(regime, 0.15)

        print(f"\n🎯 REGIME: {regime.upper()}")
        print(f"   XBI 90-day return: {float(xbi_return)*100:.1f}%")
        print(f"   Momentum weight: {momentum_weight*100:.0f}%")
    else:
        regime = "neutral"
        momentum_weight = 0.15
        print(f"\n⚠️  No returns data - using neutral regime (15% weight)")

    # Check if ranked file exists
    ranked_path = Path(args.ranked_path)
    if not ranked_path.exists():
        print(f"\n⚠️  Ranked file not found at {ranked_path}")
        print("Running standalone momentum analysis...\n")

        # Show top/bottom momentum
        ranked_momentum = sorted(signals.items(), key=lambda x: x[1].get("composite_momentum_score", 0), reverse=True)

        print("📈 TOP 10 MOMENTUM:")
        for ticker, data in ranked_momentum[:10]:
            score = data.get("composite_momentum_score", 0)
            print(f"  {ticker:6s}: {score:5.1f}")

        print("\n📉 BOTTOM 10 MOMENTUM:")
        for ticker, data in ranked_momentum[-10:]:
            score = data.get("composite_momentum_score", 0)
            print(f"  {ticker:6s}: {score:5.1f}")

        return 0

    # Load ranked output
    print(f"\nLoading ranked output from {ranked_path}...")
    with open(ranked_path) as f:
        ranked_data = json.load(f)

    # Get ranked securities
    ranked_securities = ranked_data.get("module_5_output", {}).get("ranked_securities", [])
    if not ranked_securities:
        print("Error: No ranked_securities found in file")
        return 1

    print(f"Found {len(ranked_securities)} ranked securities")

    # Integrate momentum
    print(f"\nIntegrating momentum (weight={momentum_weight*100:.0f}%)...")

    enriched_securities = []
    for sec in ranked_securities:
        ticker = sec.get("ticker", "")
        original_score = float(sec.get("composite_score", 0))

        # Get momentum score (default to 50 = neutral)
        momentum_info = signals.get(ticker, {})
        momentum_score = momentum_info.get("composite_momentum_score", 50)

        # Calculate adjustment: (momentum_score - 50) * weight
        # This gives +/- up to 12.5 points at 25% weight
        momentum_adjustment = (momentum_score - 50) * momentum_weight

        # Final score
        final_score = original_score + momentum_adjustment

        # Create enriched record
        enriched = dict(sec)
        enriched["momentum_score"] = round(momentum_score, 2)
        enriched["momentum_adjustment"] = round(momentum_adjustment, 2)
        enriched["composite_score_with_momentum"] = f"{final_score:.2f}"
        enriched["composite_score_original"] = sec.get("composite_score")
        enriched_securities.append(enriched)

    # Re-sort by new score
    enriched_securities.sort(key=lambda x: float(x.get("composite_score_with_momentum", 0)), reverse=True)

    # Update ranks
    for i, sec in enumerate(enriched_securities, 1):
        sec["composite_rank_with_momentum"] = i
        sec["composite_rank_original"] = sec.get("composite_rank")

    # Update ranked_data
    ranked_data["module_5_output"]["ranked_securities"] = enriched_securities
    ranked_data["momentum_metadata"] = {
        "regime": regime,
        "xbi_return_90d": str(xbi_return) if 'xbi_return' in dir() else "0",
        "momentum_weight": momentum_weight,
        "calculation_date": momentum_data.get("metadata", {}).get("calculation_date"),
        "tickers_with_momentum": len([s for s in enriched_securities if s.get("momentum_score", 50) != 50])
    }

    # Save enriched output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ranked_data, f, indent=2)

    print(f"\n✅ Saved enriched rankings to {output_path}")

    # Show top 20 with momentum impact
    print("\n" + "="*70)
    print("TOP 20 RANKINGS (WITH MOMENTUM)")
    print("="*70)
    print(f"{'Rank':>4} {'Ticker':>8} {'Original':>10} {'Momentum':>10} {'Adj':>8} {'Final':>10}")
    print("-"*70)

    for sec in enriched_securities[:20]:
        rank = sec.get("composite_rank_with_momentum", 0)
        ticker = sec.get("ticker", "")
        original = float(sec.get("composite_score_original", 0))
        momentum = sec.get("momentum_score", 50)
        adj = sec.get("momentum_adjustment", 0)
        final = float(sec.get("composite_score_with_momentum", 0))

        adj_str = f"+{adj:.1f}" if adj >= 0 else f"{adj:.1f}"
        print(f"{rank:4d} {ticker:>8} {original:10.2f} {momentum:10.1f} {adj_str:>8} {final:10.2f}")

    # Show biggest movers
    print("\n" + "="*70)
    print("BIGGEST MOMENTUM BOOSTS")
    print("="*70)
    boosted = sorted(enriched_securities, key=lambda x: x.get("momentum_adjustment", 0), reverse=True)[:10]
    for sec in boosted:
        ticker = sec.get("ticker", "")
        adj = sec.get("momentum_adjustment", 0)
        momentum = sec.get("momentum_score", 50)
        print(f"  {ticker:6s}: +{adj:.1f} pts (momentum={momentum:.0f})")

    print("\n" + "="*70)
    print("BIGGEST MOMENTUM DRAGS")
    print("="*70)
    dragged = sorted(enriched_securities, key=lambda x: x.get("momentum_adjustment", 0))[:10]
    for sec in dragged:
        ticker = sec.get("ticker", "")
        adj = sec.get("momentum_adjustment", 0)
        momentum = sec.get("momentum_score", 50)
        print(f"  {ticker:6s}: {adj:.1f} pts (momentum={momentum:.0f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
