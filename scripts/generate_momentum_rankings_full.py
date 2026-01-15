#!/usr/bin/env python3
"""
Generate full momentum rankings for all 308 tickers.

This creates a standalone ranking based on momentum signals alone,
useful when you don't have a full Module 5 output for all tickers.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Load momentum signals
    momentum_path = Path("outputs/momentum_signals.json")
    if not momentum_path.exists():
        print("Error: Run calculate_momentum_batch.py first")
        return 1

    with open(momentum_path) as f:
        momentum_data = json.load(f)

    signals = momentum_data.get("signals", {})
    print(f"Loaded momentum signals for {len(signals)} tickers")

    # Calculate regime
    returns_path = Path("data/returns/returns_db_daily.json")
    regime = "neutral"
    xbi_return = 0.0
    momentum_weight = 0.15

    if returns_path.exists():
        from src.scoring.integrate_momentum_regime_adaptive import calculate_xbi_regime, load_xbi_returns

        xbi_returns = load_xbi_returns(str(returns_path))
        calc_date = momentum_data.get("metadata", {}).get("calculation_date", "2026-01-14")
        regime, xbi_ret = calculate_xbi_regime(xbi_returns, calc_date)
        xbi_return = float(xbi_ret)

        regime_weights = {"risk_on": 0.25, "neutral": 0.15, "risk_off": 0.05}
        momentum_weight = regime_weights.get(regime, 0.15)

        print(f"\n🎯 REGIME: {regime.upper()}")
        print(f"   XBI 90-day return: {xbi_return*100:.1f}%")
        print(f"   Momentum weight: {momentum_weight*100:.0f}%")

    # Build rankings
    rankings = []
    for ticker, data in signals.items():
        if ticker.startswith("_"):  # Skip benchmarks
            continue

        momentum_score = data.get("composite_momentum_score", 50)
        confidence = data.get("confidence_tier", "UNKNOWN")

        # For standalone ranking, use momentum score directly
        # Normalize to 0-100 scale (it already is)
        rankings.append({
            "ticker": ticker,
            "momentum_score": round(momentum_score, 2),
            "confidence": confidence,
            "sharpe_score": data.get("sharpe_score", 50),
            "relative_strength_score": data.get("relative_strength_score", 50),
            "idio_momentum_score": data.get("idio_momentum_score", 50),
            "drawdown_gate_score": data.get("drawdown_gate_score", 50),
        })

    # Sort by momentum score
    rankings.sort(key=lambda x: x["momentum_score"], reverse=True)

    # Add ranks
    for i, r in enumerate(rankings, 1):
        r["momentum_rank"] = i

    # Create output
    output = {
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "regime": regime,
        "xbi_return_90d": xbi_return,
        "momentum_weight": momentum_weight,
        "total_tickers": len(rankings),
        "rankings": rankings,
        "metadata": {
            "calculation_date": momentum_data.get("metadata", {}).get("calculation_date"),
            "source": "momentum_signals.json",
            "note": "Standalone momentum ranking - not integrated with Module 5"
        }
    }

    # Save
    output_path = Path("outputs/momentum_rankings_full.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved rankings to {output_path}")

    # Display results
    print("\n" + "="*70)
    print(f"FULL MOMENTUM RANKINGS ({len(rankings)} tickers)")
    print("="*70)

    print("\n📈 TOP 30 MOMENTUM:")
    print(f"{'Rank':>4} {'Ticker':>8} {'Score':>8} {'Sharpe':>8} {'RelStr':>8} {'Idio':>8} {'DD':>8}")
    print("-"*70)
    for r in rankings[:30]:
        print(f"{r['momentum_rank']:4d} {r['ticker']:>8} {r['momentum_score']:8.1f} "
              f"{r['sharpe_score']:8.1f} {r['relative_strength_score']:8.1f} "
              f"{r['idio_momentum_score']:8.1f} {r['drawdown_gate_score']:8.1f}")

    print("\n📉 BOTTOM 20 MOMENTUM:")
    print(f"{'Rank':>4} {'Ticker':>8} {'Score':>8} {'Sharpe':>8} {'RelStr':>8} {'Idio':>8} {'DD':>8}")
    print("-"*70)
    for r in rankings[-20:]:
        print(f"{r['momentum_rank']:4d} {r['ticker']:>8} {r['momentum_score']:8.1f} "
              f"{r['sharpe_score']:8.1f} {r['relative_strength_score']:8.1f} "
              f"{r['idio_momentum_score']:8.1f} {r['drawdown_gate_score']:8.1f}")

    # Summary stats
    scores = [r["momentum_score"] for r in rankings]
    print("\n📊 DISTRIBUTION:")
    print(f"   High momentum (>80): {len([s for s in scores if s > 80])} tickers")
    print(f"   Neutral (40-60):     {len([s for s in scores if 40 <= s <= 60])} tickers")
    print(f"   Low momentum (<20):  {len([s for s in scores if s < 20])} tickers")
    print(f"   Mean score: {sum(scores)/len(scores):.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
