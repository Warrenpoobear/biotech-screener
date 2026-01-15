#!/usr/bin/env python3
"""
Test Module 5 v2 with production data.

Loads existing module outputs from results_2026-01-07.json and runs
the v2 composite module to verify it works with real data.
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

# Import v2 module
from module_5_composite_v2 import compute_module_5_composite_v2

def load_production_data():
    """Load the production results file."""
    results_path = Path("/home/user/biotech-screener/results_2026-01-07.json")
    with open(results_path) as f:
        return json.load(f)

def main():
    print("=" * 70)
    print("MODULE 5 V2 PRODUCTION TEST")
    print("=" * 70)

    # Load production data
    print("\n1. Loading production data from results_2026-01-07.json...")
    data = load_production_data()

    universe = data["module_1_universe"]
    financial = data["module_2_financial"]
    catalyst = data["module_3_catalyst"]
    clinical = data["module_4_clinical"]

    print(f"   Universe: {len(universe.get('active_securities', []))} active securities")
    print(f"   Financial: {len(financial.get('scores', []))} scored")
    print(f"   Clinical: {len(clinical.get('scores', []))} scored")
    print(f"   Catalyst: {len(catalyst.get('summaries', {}))} summaries")

    # Run Module 5 v2
    print("\n2. Running Module 5 v2 composite scoring...")
    result = compute_module_5_composite_v2(
        universe_result=universe,
        financial_result=financial,
        catalyst_result=catalyst,
        clinical_result=clinical,
        as_of_date="2026-01-07",
    )

    # Print results
    print("\n3. Results Summary:")
    print(f"   Scoring mode: {result['scoring_mode']}")
    print(f"   Schema version: {result['schema_version']}")
    print(f"   Ranked: {len(result['ranked_securities'])} securities")
    print(f"   Excluded: {len(result['excluded_securities'])} securities")

    diag = result["diagnostic_counts"]
    print(f"\n   Diagnostics:")
    print(f"     Total input: {diag['total_input']}")
    print(f"     Rankable: {diag['rankable']}")
    print(f"     Excluded: {diag['excluded']}")
    print(f"     With caps applied: {diag['with_caps_applied']}")

    # Print top 10 ranked
    print("\n4. Top 10 Ranked Securities (v2):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Ticker':<8} {'Score':<8} {'Conf_F':<8} {'Conf_C':<8} {'Caps':<10} {'Hash':<16}")
    print("-" * 70)

    for sec in result["ranked_securities"][:10]:
        caps = "Yes" if sec.get("monotonic_caps_applied") else "No"
        print(f"{sec['composite_rank']:<5} {sec['ticker']:<8} {sec['composite_score']:<8} "
              f"{sec['confidence_financial']:<8} {sec['confidence_clinical']:<8} "
              f"{caps:<10} {sec['determinism_hash']}")

    # Show breakdown for top ticker
    if result["ranked_securities"]:
        top = result["ranked_securities"][0]
        print(f"\n5. Score Breakdown for {top['ticker']}:")
        bd = top["score_breakdown"]
        print(f"   Mode: {bd['mode']}")
        print(f"   Normalization: {bd['normalization_method']}")
        print(f"\n   Effective weights: {bd['effective_weights']}")
        print(f"\n   Components:")
        for comp in bd["components"]:
            print(f"     {comp['name']}: raw={comp['raw']}, norm={comp['normalized']}, "
                  f"weight={comp['weight_effective']}, contrib={comp['contribution']}")
        print(f"\n   Hybrid aggregation:")
        print(f"     Alpha: {bd['hybrid_aggregation']['alpha']}")
        print(f"     Weighted sum: {bd['hybrid_aggregation']['weighted_sum']}")
        print(f"     Min critical: {bd['hybrid_aggregation']['min_critical']}")
        print(f"\n   Final scores:")
        for k, v in bd["final"].items():
            print(f"     {k}: {v}")

    # Show any excluded
    if result["excluded_securities"]:
        print(f"\n6. Excluded Securities:")
        for exc in result["excluded_securities"][:5]:
            print(f"   {exc['ticker']}: {exc['reason']}")

    # Show cohort stats
    print(f"\n7. Cohort Statistics:")
    for cohort, stats in result["cohort_stats"].items():
        print(f"   {cohort}: {stats['count']} securities")

    # Compare with v1 if available
    v1_ranked = data["module_5_composite"]["ranked_securities"]
    print(f"\n8. Comparison with v1:")
    print(f"   v1 ranked: {len(v1_ranked)} securities")
    print(f"   v2 ranked: {len(result['ranked_securities'])} securities")

    # Show rank changes
    v1_ranks = {s["ticker"]: s["composite_rank"] for s in v1_ranked}
    v2_ranks = {s["ticker"]: s["composite_rank"] for s in result["ranked_securities"]}

    print(f"\n   Top 10 rank comparison:")
    print(f"   {'Ticker':<8} {'v1 Rank':<10} {'v2 Rank':<10} {'Change':<10}")
    print(f"   {'-'*38}")
    for sec in result["ranked_securities"][:10]:
        ticker = sec["ticker"]
        v1_rank = v1_ranks.get(ticker, "N/A")
        v2_rank = sec["composite_rank"]
        if isinstance(v1_rank, int):
            change = v1_rank - v2_rank
            change_str = f"+{change}" if change > 0 else str(change)
        else:
            change_str = "NEW"
        print(f"   {ticker:<8} {str(v1_rank):<10} {v2_rank:<10} {change_str:<10}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Save results
    output_path = Path("/home/user/biotech-screener/results_v2_2026-01-07.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
