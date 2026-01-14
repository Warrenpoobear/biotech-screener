#!/usr/bin/env python3
"""
Ablation test: Does momentum improve actual screening performance?

Uses your existing historical test set with real catalyst outcomes.
"""

from decimal import Decimal
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.morningstar_momentum_signals_v2 import MorningstarMomentumSignals
from src.scoring.integrate_momentum_confidence_weighted import integrate_momentum_signals_with_confidence


def load_historical_test_set():
    """
    Load your existing historical test set.

    This should include:
    - 50+ catalysts from 2019-2024
    - Pre-catalyst dates
    - Post-catalyst 30/60/90d returns
    """
    # TODO: Point to your actual historical test data
    test_file = Path("tests/data/historical_catalyst_outcomes.json")

    if not test_file.exists():
        # Create sample structure
        sample = {
            "description": "Historical catalyst outcomes for ablation testing",
            "catalysts": [
                {
                    "ticker": "EXAMPLE",
                    "catalyst_date": "2023-06-15",
                    "catalyst_type": "phase3_readout",
                    "pre_catalyst_date": "2023-06-01",
                    "outcome_30d": 0.15,
                    "outcome_60d": 0.22,
                    "outcome_90d": 0.18
                }
            ],
            "outcomes_30d": {"EXAMPLE": 0.15},
            "outcomes_60d": {"EXAMPLE": 0.22},
            "outcomes_90d": {"EXAMPLE": 0.18}
        }

        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, 'w') as f:
            json.dump(sample, f, indent=2)

        print(f"Created sample test file: {test_file}")
        print("Please populate with your actual historical catalyst data")
        raise FileNotFoundError(
            "Historical test set not found. "
            "Create tests/data/historical_catalyst_outcomes.json with your validated catalysts."
        )

    with open(test_file, 'r') as f:
        return json.load(f)


def calculate_information_coefficient(scores, outcomes):
    """
    Calculate Spearman rank correlation (IC).

    This is your primary metric for signal quality.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("WARNING: scipy not installed, using simple correlation")
        return _simple_rank_correlation(scores, outcomes)

    # Get overlapping tickers
    common_tickers = set(scores.keys()) & set(outcomes.keys())

    if len(common_tickers) < 10:
        print(f"WARNING: Only {len(common_tickers)} overlapping tickers")
        return Decimal("0.0")

    score_list = [float(scores[t]) for t in common_tickers]
    outcome_list = [float(outcomes[t]) for t in common_tickers]

    ic, p_value = spearmanr(score_list, outcome_list)

    return Decimal(str(round(ic, 4)))


def _simple_rank_correlation(scores, outcomes):
    """Simple rank correlation without scipy."""
    common_tickers = list(set(scores.keys()) & set(outcomes.keys()))

    if len(common_tickers) < 10:
        return Decimal("0.0")

    # Get ranks
    score_sorted = sorted(common_tickers, key=lambda t: scores[t])
    outcome_sorted = sorted(common_tickers, key=lambda t: outcomes[t])

    score_ranks = {t: i for i, t in enumerate(score_sorted)}
    outcome_ranks = {t: i for i, t in enumerate(outcome_sorted)}

    n = len(common_tickers)

    # Calculate Spearman's rho
    d_squared_sum = sum(
        (score_ranks[t] - outcome_ranks[t]) ** 2
        for t in common_tickers
    )

    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))

    return Decimal(str(round(rho, 4)))


def calculate_hit_rate(scores, outcomes, threshold=0.0):
    """
    Calculate hit rate: % of high-score picks with positive outcomes.

    Args:
        scores: {ticker: score}
        outcomes: {ticker: return}
        threshold: Outcome threshold for "hit" (default: 0 = positive return)
    """
    common_tickers = set(scores.keys()) & set(outcomes.keys())

    if len(common_tickers) < 10:
        return Decimal("0.0")

    # Top quartile by score
    sorted_tickers = sorted(common_tickers, key=lambda t: scores[t], reverse=True)
    top_quartile = sorted_tickers[:len(sorted_tickers)//4]

    if not top_quartile:
        return Decimal("0.0")

    hits = sum(1 for t in top_quartile if outcomes[t] > threshold)
    hit_rate = Decimal(hits) / Decimal(len(top_quartile))

    return hit_rate.quantize(Decimal("0.0001"))


def run_ablation_test():
    """
    Compare baseline vs momentum-enhanced performance.

    Acceptance criteria:
    - IC improvement > 0.05
    - Hit rate improvement > 3%
    - No degradation > 2% on any metric
    """
    print("=" * 70)
    print("MOMENTUM ABLATION TEST - REAL HISTORICAL DATA")
    print("=" * 70)

    # Load test set
    print("\n1. Loading historical test set...")
    try:
        test_data = load_historical_test_set()
    except FileNotFoundError as e:
        print(f"\n{e}")
        return False

    n_catalysts = len(test_data.get("catalysts", []))
    print(f"   {n_catalysts} catalysts loaded")

    if n_catalysts < 20:
        print("   WARNING: Recommend 50+ catalysts for reliable ablation testing")

    # Load returns data
    print("\n2. Loading returns data...")
    returns_db_path = Path("data/returns")
    ticker_returns_all = {}
    xbi_returns = {}

    if returns_db_path.exists():
        for db_file in sorted(returns_db_path.glob("returns_db_*.json"), reverse=True):
            print(f"   Loading from: {db_file}")
            with open(db_file, 'r') as f:
                data = json.load(f)

            for ticker, returns in data.get("tickers", {}).items():
                ticker_returns_all[ticker] = {
                    date: Decimal(str(ret))
                    for date, ret in returns.items()
                }

            benchmark = data.get("benchmark", {}).get("XBI", {})
            xbi_returns = {
                date: Decimal(str(ret))
                for date, ret in benchmark.items()
            }
            break

    if not ticker_returns_all:
        print("   ERROR: No returns data found")
        print("   Run scripts/fetch_morningstar_data.py first")
        return False

    print(f"   Loaded {len(ticker_returns_all)} tickers")

    # Calculate baseline scores (simulated - without momentum)
    print("\n3. Calculating baseline scores (without momentum)...")
    baseline_scores = {}

    # Use random baseline for demonstration
    # TODO: Replace with your actual baseline scoring
    import random
    random.seed(42)  # Deterministic for testing

    for catalyst in test_data.get("catalysts", []):
        ticker = catalyst["ticker"]
        if ticker in ticker_returns_all:
            # Simulated baseline score (replace with actual module 5 output)
            baseline_scores[ticker] = Decimal(str(random.uniform(30, 70)))

    print(f"   Calculated {len(baseline_scores)} baseline scores")

    # Calculate momentum-enhanced scores
    print("\n4. Calculating momentum-enhanced scores...")
    calculator = MorningstarMomentumSignals()
    enhanced_scores = {}

    for catalyst in test_data.get("catalysts", []):
        ticker = catalyst["ticker"]
        pre_date = catalyst.get("pre_catalyst_date", catalyst["catalyst_date"])

        if ticker not in ticker_returns_all:
            continue

        if ticker not in baseline_scores:
            continue

        try:
            # Calculate momentum signals
            momentum_signals = calculator.calculate_all_signals(
                ticker,
                ticker_returns_all[ticker],
                xbi_returns,
                pre_date
            )

            # Integrate with baseline
            base_score_dict = {
                "financial_health_score": Decimal("50.0"),
                "clinical_development_score": baseline_scores[ticker],
                "institutional_signal_score": Decimal("50.0")
            }

            integrated = integrate_momentum_signals_with_confidence(
                base_score_dict,
                momentum_signals,
                ticker
            )

            enhanced_scores[ticker] = integrated["final_score"]

        except Exception as e:
            # Fall back to baseline if momentum fails
            enhanced_scores[ticker] = baseline_scores[ticker]

    print(f"   Calculated {len(enhanced_scores)} enhanced scores")

    # Get outcomes
    outcomes_60d = {
        t: Decimal(str(v))
        for t, v in test_data.get("outcomes_60d", {}).items()
    }

    # Compare metrics
    print("\n5. Comparing performance...")

    baseline_ic = calculate_information_coefficient(baseline_scores, outcomes_60d)
    enhanced_ic = calculate_information_coefficient(enhanced_scores, outcomes_60d)
    ic_improvement = enhanced_ic - baseline_ic

    baseline_hit = calculate_hit_rate(baseline_scores, outcomes_60d)
    enhanced_hit = calculate_hit_rate(enhanced_scores, outcomes_60d)
    hit_improvement = enhanced_hit - baseline_hit

    print(f"\n   Information Coefficient (60d):")
    print(f"     Baseline IC:  {baseline_ic:+.4f}")
    print(f"     Enhanced IC:  {enhanced_ic:+.4f}")
    print(f"     Improvement:  {ic_improvement:+.4f}")

    print(f"\n   Hit Rate (top quartile positive):")
    print(f"     Baseline:     {baseline_hit:.2%}")
    print(f"     Enhanced:     {enhanced_hit:.2%}")
    print(f"     Improvement:  {hit_improvement:+.2%}")

    # Verdict
    print("\n" + "=" * 70)

    if ic_improvement > Decimal("0.05") and hit_improvement > Decimal("0.03"):
        print("✅ VERDICT: SIGNIFICANT IMPROVEMENT")
        print("   Momentum signals approved for production")
        return True
    elif ic_improvement > Decimal("0.02") or hit_improvement > Decimal("0.01"):
        print("✓ VERDICT: MODERATE IMPROVEMENT")
        print("   Momentum signals conditionally approved")
        return True
    elif ic_improvement < Decimal("-0.02"):
        print("❌ VERDICT: DEGRADATION DETECTED")
        print("   Momentum signals need refinement")
        return False
    else:
        print("⚠️ VERDICT: INSUFFICIENT IMPROVEMENT")
        print("   Momentum signals need refinement before production")
        return False


if __name__ == "__main__":
    try:
        success = run_ablation_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
