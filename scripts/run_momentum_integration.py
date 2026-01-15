#!/usr/bin/env python3
"""
Run momentum integration on Module 5 output.

Usage:
    python scripts/run_momentum_integration.py
    python scripts/run_momentum_integration.py --module5-path outputs/module5_output.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.integrate_momentum_regime_adaptive import enrich_module5_with_momentum


def main():
    parser = argparse.ArgumentParser(description="Integrate momentum signals into Module 5 output")
    parser.add_argument(
        "--module5-path",
        default="outputs/module5_output.json",
        help="Path to Module 5 output JSON file"
    )
    parser.add_argument(
        "--output-path",
        default="outputs/module5_with_momentum.json",
        help="Path for enriched output"
    )
    args = parser.parse_args()

    # Check if Module 5 output exists
    module5_path = Path(args.module5_path)
    if not module5_path.exists():
        print(f"Error: Module 5 output not found at {module5_path}")
        print("\nTo use this script, first run your Module 5 composite scoring.")
        print("Or provide the path with: --module5-path <path>")

        # Show what we can do without Module 5
        print("\n" + "="*60)
        print("STANDALONE MOMENTUM ANALYSIS")
        print("="*60)

        # Load momentum signals directly
        momentum_path = Path("outputs/momentum_signals.json")
        if momentum_path.exists():
            with open(momentum_path) as f:
                momentum_data = json.load(f)

            signals = momentum_data.get("signals", {})
            print(f"\nMomentum signals available for {len(signals)} tickers")

            # Show top/bottom momentum
            ranked = sorted(signals.items(), key=lambda x: x[1].get("composite_momentum_score", 0), reverse=True)

            print("\n📈 TOP 10 MOMENTUM:")
            for ticker, data in ranked[:10]:
                score = data.get("composite_momentum_score", 0)
                print(f"  {ticker:6s}: {score:5.1f}")

            print("\n📉 BOTTOM 10 MOMENTUM:")
            for ticker, data in ranked[-10:]:
                score = data.get("composite_momentum_score", 0)
                print(f"  {ticker:6s}: {score:5.1f}")

            # Show regime
            returns_path = Path("data/returns/returns_db_daily.json")
            if returns_path.exists():
                from src.scoring.integrate_momentum_regime_adaptive import calculate_xbi_regime, load_xbi_returns
                from decimal import Decimal

                xbi_returns = load_xbi_returns(str(returns_path))
                calc_date = momentum_data.get("metadata", {}).get("calculation_date", "2026-01-14")
                regime, xbi_return = calculate_xbi_regime(xbi_returns, calc_date)

                # Regime weights
                regime_weights = {"risk_on": 0.25, "neutral": 0.15, "risk_off": 0.05}
                weight = regime_weights.get(regime, 0.15)

                print(f"\n🎯 CURRENT REGIME: {regime.upper()}")
                print(f"   XBI 90-day return: {float(xbi_return)*100:.1f}%")
                print(f"   Momentum weight: {weight*100:.0f}%")
        else:
            print("\nNo momentum signals found. Run calculate_momentum_batch.py first.")

        return 1

    # Load Module 5 output
    print(f"Loading Module 5 output from {module5_path}...")
    with open(module5_path) as f:
        module5_output = json.load(f)

    # Enrich with momentum
    print("Integrating momentum signals...")
    enriched = enrich_module5_with_momentum(module5_output)

    # Save enriched output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"\n✅ Enriched output saved to {output_path}")

    # Show summary
    regime = enriched.get("momentum_metadata", {}).get("regime", "unknown")
    weight = enriched.get("momentum_metadata", {}).get("momentum_weight", 0)

    print(f"\n📊 SUMMARY:")
    print(f"   Regime: {regime}")
    print(f"   Momentum weight: {weight*100:.0f}%")

    # Show top changes
    rankings = enriched.get("rankings", [])
    if rankings:
        print(f"\n   Top 10 tickers (with momentum adjustment):")
        for i, ticker_data in enumerate(rankings[:10], 1):
            ticker = ticker_data.get("ticker", "?")
            final = ticker_data.get("final_score", 0)
            momentum = ticker_data.get("momentum_score", 50)
            print(f"   {i:2d}. {ticker:6s}: {final:5.1f} (momentum: {momentum:.0f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
