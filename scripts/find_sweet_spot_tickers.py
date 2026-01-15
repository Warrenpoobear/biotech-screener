#!/usr/bin/env python3
"""
Find 'sweet spot' tickers: High fundamental quality + High momentum.

These are your highest conviction plays - the market is confirming
your fundamental analysis with strong price action.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_sweet_spot_tickers():
    """
    Identify tickers with both:
    1. High base scores (>70) - strong fundamentals
    2. High momentum (>80) - market confirmation
    """
    print("=" * 70)
    print("SWEET SPOT ANALYSIS: HIGH QUALITY + HIGH MOMENTUM")
    print("=" * 70)

    # Load momentum signals
    momentum_file = Path("outputs/momentum_signals.json")

    if not momentum_file.exists():
        print("\nERROR: Momentum signals not found")
        print("Run: python scripts/calculate_momentum_batch.py")
        return 1

    with open(momentum_file, 'r') as f:
        momentum_data = json.load(f)

    signals = momentum_data.get("signals", {})
    print(f"\nLoaded momentum signals for {len(signals)} tickers")

    # Check for Module 5 base scores in various locations
    module5_files = [
        "outputs/ranked_full_308.json",
        "outputs/ranked_with_real_defensive_FINAL.json",
        "outputs/ranked_20260106.json",
        "outputs/ranked_full_universe_with_defensive.json",
        "outputs/module5.json",
        "outputs/module5_output.json",
    ]

    base_scores = {}
    module5_file_used = None

    for f_path in module5_files:
        p = Path(f_path)
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Try different structures
                ranked_securities = (
                    data.get("module_5_composite", {}).get("ranked_securities", []) or
                    data.get("module_5_output", {}).get("ranked_securities", []) or
                    data.get("securities", [])
                )

                if ranked_securities:
                    for sec in ranked_securities:
                        ticker = sec.get("ticker", "")
                        if ticker and not ticker.startswith("_"):
                            score = float(sec.get("composite_score", 50))
                            base_scores[ticker] = {
                                "score": score,
                                "rank": sec.get("composite_rank", 0),
                                "financial": float(sec.get("financial_normalized", 50)),
                                "clinical": float(sec.get("clinical_dev_normalized", 50)),
                                "catalyst": float(sec.get("catalyst_normalized", 50)),
                            }
                    module5_file_used = f_path
                    break
            except Exception as e:
                continue

    if base_scores:
        print(f"Loaded Module 5 base scores for {len(base_scores)} tickers from {module5_file_used}")

        # Find sweet spot tickers
        sweet_spot = []
        contrarian = []
        momentum_only = []
        avoid = []

        for ticker, mom_signals in signals.items():
            if ticker.startswith("_"):
                continue

            momentum_score = mom_signals.get("composite_momentum_score", 50)

            if ticker in base_scores:
                base_score = base_scores[ticker]["score"]

                # Categorize
                if base_score > 30 and momentum_score > 80:
                    sweet_spot.append({
                        "ticker": ticker,
                        "base": base_score,
                        "momentum": momentum_score,
                        "combined": (base_score + momentum_score) / 2,
                        "financial": base_scores[ticker]["financial"],
                        "clinical": base_scores[ticker]["clinical"],
                        "catalyst": base_scores[ticker]["catalyst"]
                    })
                elif base_score > 30 and momentum_score < 30:
                    contrarian.append({
                        "ticker": ticker,
                        "base": base_score,
                        "momentum": momentum_score
                    })
                elif base_score < 25 and momentum_score > 80:
                    momentum_only.append({
                        "ticker": ticker,
                        "base": base_score,
                        "momentum": momentum_score
                    })
                elif base_score < 25 and momentum_score < 30:
                    avoid.append({
                        "ticker": ticker,
                        "base": base_score,
                        "momentum": momentum_score
                    })

        # Display results
        print("\n" + "=" * 70)
        print(f"SWEET SPOT TICKERS: {len(sweet_spot)}")
        print("(High Base + High Momentum >80)")
        print("=" * 70)

        sweet_spot.sort(key=lambda x: x["combined"], reverse=True)

        if sweet_spot:
            print(f"\n{'Rank':>4} {'Ticker':>8} {'Combined':>10} {'Base':>8} {'Momentum':>10}")
            print("-" * 50)
            for i, t in enumerate(sweet_spot[:20], 1):
                print(f"{i:4d} {t['ticker']:>8} {t['combined']:10.1f} {t['base']:8.1f} {t['momentum']:10.1f}")
        else:
            print("\nNo tickers found in sweet spot with current thresholds")

        print("\n" + "=" * 70)
        print(f"CONTRARIAN OPPORTUNITIES: {len(contrarian)}")
        print("(High Base BUT Low Momentum <30)")
        print("=" * 70)

        if contrarian:
            contrarian.sort(key=lambda x: x["base"], reverse=True)
            print("\nGood fundamentals but weak momentum - value traps or turnarounds?")
            for i, t in enumerate(contrarian[:10], 1):
                print(f"  {i:2d}. {t['ticker']:6s}  Base: {t['base']:5.1f}, Mom: {t['momentum']:5.1f}")

        print("\n" + "=" * 70)
        print(f"MOMENTUM SPECULATION: {len(momentum_only)}")
        print("(Low Base BUT High Momentum >80)")
        print("=" * 70)

        if momentum_only:
            momentum_only.sort(key=lambda x: x["momentum"], reverse=True)
            print("\nHigh momentum without fundamental support - speculation risk")
            for i, t in enumerate(momentum_only[:10], 1):
                print(f"  {i:2d}. {t['ticker']:6s}  Mom: {t['momentum']:5.1f}, Base: {t['base']:5.1f}")

        print("\n" + "=" * 70)
        print(f"AVOID: {len(avoid)}")
        print("(Low Base AND Low Momentum <30)")
        print("=" * 70)

        if avoid:
            avoid.sort(key=lambda x: x["base"] + x["momentum"])
            print("\nWeak fundamentals + weak momentum = high risk")
            for i, t in enumerate(avoid[:10], 1):
                print(f"  {i:2d}. {t['ticker']:6s}  Base: {t['base']:5.1f}, Mom: {t['momentum']:5.1f}")

        # Export sweet spot to CSV
        if sweet_spot:
            output_file = Path("outputs/sweet_spot_tickers.csv")
            with open(output_file, 'w') as f:
                f.write("Rank,Ticker,Base_Score,Momentum_Score,Combined_Score,Financial,Clinical,Catalyst\n")
                for i, t in enumerate(sweet_spot, 1):
                    f.write(f"{i},{t['ticker']},{t['base']:.1f},{t['momentum']:.1f},{t['combined']:.1f},"
                            f"{t['financial']:.1f},{t['clinical']:.1f},{t['catalyst']:.1f}\n")

            print(f"\n📊 Sweet spot tickers exported to: {output_file}")

    else:
        # No Module 5 - show momentum-only analysis
        print("\n⚠️  Module 5 base scores not found")
        print("Showing momentum-only analysis (without fundamental scores)\n")

        # Extract momentum scores
        momentum_scores = [
            {
                "ticker": ticker,
                "score": data.get("composite_momentum_score", 50),
            }
            for ticker, data in signals.items()
            if not ticker.startswith("_")
        ]

        momentum_scores.sort(key=lambda x: x["score"], reverse=True)

        print("=" * 70)
        print("TOP 30 MOMENTUM TICKERS")
        print("(Strongest price action - awaiting fundamental analysis)")
        print("=" * 70)

        for i, t in enumerate(momentum_scores[:30], 1):
            print(f"{i:2d}. {t['ticker']:6s}  Score: {t['score']:5.1f}")

        print("\n" + "=" * 70)
        print("BOTTOM 20 MOMENTUM TICKERS")
        print("(Weakest price action - potential value traps)")
        print("=" * 70)

        for i, t in enumerate(momentum_scores[-20:], 1):
            idx = len(momentum_scores) - 19 + i - 1
            print(f"{idx:3d}. {t['ticker']:6s}  Score: {t['score']:5.1f}")

        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Generate Module 5 base scores for all 308 tickers")
        print("2. Run integration to find sweet spot (high base + high momentum)")
        print("3. Build IC dossiers for sweet spot candidates")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(find_sweet_spot_tickers())
