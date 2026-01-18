#!/usr/bin/env python3
"""
score_ranking_display.py

Score Ranking Display Utility for Biotech Screener

Displays ranked securities with comparison between enhanced composite scores
and base scores, showing the delta contribution from enhancements.

Design Philosophy:
- DETERMINISTIC: Same inputs produce identical outputs
- STDLIB-ONLY: No external dependencies
- DECIMAL-ONLY: Pure Decimal arithmetic for precision
- FAIL LOUDLY: Clear error states with validation
- AUDITABLE: Full breakdown of score contributions

Usage:
    python score_ranking_display.py --file results.json
    python score_ranking_display.py --file results.json --top 100
    python score_ranking_display.py --file results.json --format csv --output rankings.csv

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import csv
import json
import sys

__version__ = "1.0.0"


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TOP_N = 60
SCORE_PRECISION = Decimal("0.01")


# =============================================================================
# DATA CLASSES
# =============================================================================

class RankedSecurity:
    """Represents a ranked security with score breakdown."""

    def __init__(
        self,
        ticker: str,
        enhanced_score: Decimal,
        base_score: Decimal,
        rank: int,
        severity: str = "none",
        stage_bucket: str = "unknown",
        market_cap_bucket: str = "unknown",
        score_breakdown: Optional[Dict[str, Any]] = None,
    ):
        self.ticker = ticker
        self.enhanced_score = enhanced_score
        self.base_score = base_score
        self.rank = rank
        self.severity = severity
        self.stage_bucket = stage_bucket
        self.market_cap_bucket = market_cap_bucket
        self.score_breakdown = score_breakdown or {}

    @property
    def delta(self) -> Decimal:
        """Calculate delta between enhanced and base scores."""
        return self.enhanced_score - self.base_score

    @property
    def delta_pct(self) -> Decimal:
        """Calculate delta as percentage of base score."""
        if self.base_score == Decimal("0"):
            return Decimal("0")
        return ((self.enhanced_score - self.base_score) / self.base_score * Decimal("100")).quantize(SCORE_PRECISION)

    @property
    def gov_flags(self) -> str:
        """Get governance flags applied to this security."""
        flags = []

        # Severity flags
        if self.severity == "sev1":
            flags.append("SEV1")
        elif self.severity == "sev2":
            flags.append("SEV2")
        elif self.severity == "sev3":
            flags.append("SEV3")

        # Check for caps and penalties from breakdown
        penalties = self.score_breakdown.get("penalties_and_gates", {})
        if penalties:
            uncertainty = penalties.get("uncertainty_penalty_pct", "0")
            if float(uncertainty) > 5:
                flags.append("UNC")
            if penalties.get("monotonic_caps_applied"):
                flags.append("CAP")

        return ",".join(flags) if flags else "PASS"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_decimal(value: Any) -> Optional[Decimal]:
    """Convert value to Decimal safely."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _quantize_score(score: Decimal) -> Decimal:
    """Quantize score to 2 decimal places."""
    return score.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load Module 5 results from JSON file.

    Args:
        filepath: Path to results JSON file

    Returns:
        Loaded results dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid JSON or missing required fields
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")

    # Support both direct module 5 output and full pipeline output
    if "module_5_composite" in data:
        return data["module_5_composite"]
    elif "ranked_securities" in data:
        return data
    else:
        raise ValueError(f"No ranked_securities found in {filepath}")


def extract_scores(
    results: Dict[str, Any],
) -> Tuple[Dict[str, Decimal], Dict[str, Decimal], Dict[str, Dict[str, Any]]]:
    """
    Extract enhanced and base scores from Module 5 results.

    The enhanced score is the final composite_score.
    The base score is the pre_penalty_score (before enhancements and penalties).

    Args:
        results: Module 5 results dictionary

    Returns:
        Tuple of (enhanced_scores, base_scores, security_metadata)
    """
    ranked = results.get("ranked_securities", [])

    enhanced_scores: Dict[str, Decimal] = {}
    base_scores: Dict[str, Decimal] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    for sec in ranked:
        ticker = sec.get("ticker")
        if not ticker:
            continue

        # Enhanced score is the final composite score
        enhanced = _to_decimal(sec.get("composite_score"))
        if enhanced is None:
            continue
        enhanced_scores[ticker] = _quantize_score(enhanced)

        # Base score from score_breakdown.final.pre_penalty_score
        breakdown = sec.get("score_breakdown", {})
        final = breakdown.get("final", {})
        base = _to_decimal(final.get("pre_penalty_score"))

        if base is not None:
            base_scores[ticker] = _quantize_score(base)
        else:
            # Fallback: use composite score as base (no enhancements)
            base_scores[ticker] = enhanced_scores[ticker]

        # Collect metadata
        metadata[ticker] = {
            "severity": sec.get("severity", "none"),
            "stage_bucket": sec.get("stage_bucket", "unknown"),
            "market_cap_bucket": sec.get("market_cap_bucket", "unknown"),
            "composite_rank": sec.get("composite_rank", 0),
            "score_breakdown": breakdown,
        }

    return enhanced_scores, base_scores, metadata


def rank_securities(
    enhanced_scores: Dict[str, Decimal],
    base_scores: Dict[str, Decimal],
    metadata: Dict[str, Dict[str, Any]],
) -> List[RankedSecurity]:
    """
    Create ranked list of securities sorted by enhanced score.

    Tiebreaker logic:
    1. Enhanced score (descending)
    2. Base score (descending)
    3. Ticker (ascending, for determinism)

    Args:
        enhanced_scores: {ticker: enhanced_score}
        base_scores: {ticker: base_score}
        metadata: {ticker: metadata_dict}

    Returns:
        List of RankedSecurity sorted by enhanced score
    """
    tickers = list(enhanced_scores.keys())

    # Sort with deterministic tiebreakers
    sorted_tickers = sorted(
        tickers,
        key=lambda t: (
            -enhanced_scores[t],
            -base_scores.get(t, Decimal("0")),
            t,  # Alphabetical tiebreaker for determinism
        ),
    )

    ranked = []
    for rank, ticker in enumerate(sorted_tickers, start=1):
        meta = metadata.get(ticker, {})
        sec = RankedSecurity(
            ticker=ticker,
            enhanced_score=enhanced_scores[ticker],
            base_score=base_scores.get(ticker, enhanced_scores[ticker]),
            rank=rank,
            severity=meta.get("severity", "none"),
            stage_bucket=meta.get("stage_bucket", "unknown"),
            market_cap_bucket=meta.get("market_cap_bucket", "unknown"),
            score_breakdown=meta.get("score_breakdown", {}),
        )
        ranked.append(sec)

    return ranked


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_rankings(
    ranked: List[RankedSecurity],
    top_n: int = DEFAULT_TOP_N,
    show_breakdown: bool = False,
) -> None:
    """
    Print formatted ranking table to console.

    Args:
        ranked: List of RankedSecurity
        top_n: Number of top securities to display
        show_breakdown: Show detailed score breakdown
    """
    display_list = ranked[:top_n]

    print(f"\nTop {len(display_list)} Securities (Ranked by Final Score):")
    print(f"{'Rank':<6}{'Ticker':<10}{'Final':<10}{'Pre-Gov':<10}{'Gov Impact':<12}{'Flags':<12}{'Stage':<8}")
    print("-" * 78)

    for sec in display_list:
        print(
            f"{sec.rank:<6}"
            f"{sec.ticker:<10}"
            f"{sec.enhanced_score:<10.2f}"
            f"{sec.base_score:<10.2f}"
            f"{sec.delta:+10.2f}  "
            f"{sec.gov_flags:<12}"
            f"{sec.stage_bucket:<8}"
        )

    # Summary statistics
    print("-" * 78)
    if display_list:
        avg_final = sum(s.enhanced_score for s in display_list) / len(display_list)
        avg_pregov = sum(s.base_score for s in display_list) / len(display_list)
        avg_impact = sum(s.delta for s in display_list) / len(display_list)
        max_impact = max(s.delta for s in display_list)
        min_impact = min(s.delta for s in display_list)

        print(f"\nScore Statistics (Top {len(display_list)}):")
        print(f"  Average Final Score:    {avg_final:.2f}")
        print(f"  Average Pre-Gov Score:  {avg_pregov:.2f}")
        print(f"  Average Gov Impact:     {avg_impact:+.2f}")
        print(f"  Gov Impact Range:       [{min_impact:+.2f}, {max_impact:+.2f}]")

        # Count by governance status
        clean_count = sum(1 for s in display_list if s.gov_flags == "PASS")
        penalized_count = len(display_list) - clean_count
        print(f"  Clean (no penalties):   {clean_count} ({clean_count/len(display_list)*100:.1f}%)")
        print(f"  Penalized:              {penalized_count} ({penalized_count/len(display_list)*100:.1f}%)")

        # Governance flag breakdown
        sev1_count = sum(1 for s in display_list if "SEV1" in s.gov_flags)
        sev2_count = sum(1 for s in display_list if "SEV2" in s.gov_flags)
        unc_count = sum(1 for s in display_list if "UNC" in s.gov_flags)
        cap_count = sum(1 for s in display_list if "CAP" in s.gov_flags)

        if penalized_count > 0:
            print(f"\n  Governance Breakdown:")
            if sev1_count > 0:
                print(f"    SEV1 (10% penalty):   {sev1_count}")
            if sev2_count > 0:
                print(f"    SEV2 (50% penalty):   {sev2_count}")
            if unc_count > 0:
                print(f"    UNC (uncertainty):    {unc_count}")
            if cap_count > 0:
                print(f"    CAP (monotonic cap):  {cap_count}")


def print_detailed_breakdown(ranked: List[RankedSecurity], top_n: int = 10) -> None:
    """
    Print detailed score breakdown for top securities.

    Args:
        ranked: List of RankedSecurity
        top_n: Number to show detailed breakdown
    """
    print(f"\n{'='*70}")
    print("DETAILED SCORE BREAKDOWN")
    print(f"{'='*70}")

    for sec in ranked[:top_n]:
        print(f"\n{sec.rank}. {sec.ticker} [{sec.gov_flags}]")
        print(f"   Final: {sec.enhanced_score:.2f} | Pre-Gov: {sec.base_score:.2f} | Gov Impact: {sec.delta:+.2f}")

        breakdown = sec.score_breakdown
        if breakdown:
            # Show component contributions
            components = breakdown.get("components", [])
            if components:
                print("   Components:")
                for comp in components:
                    name = comp.get("name", "unknown")
                    contrib = comp.get("contribution", "0")
                    weight = comp.get("weight_effective", "0")
                    print(f"     - {name}: {contrib} (weight: {weight})")

            # Show enhancement effects
            enhancements = breakdown.get("enhancements", {})
            if enhancements:
                print("   Enhancements:")
                for name, data in enhancements.items():
                    if isinstance(data, dict):
                        score = data.get("score", data.get("factor", "N/A"))
                        print(f"     - {name}: {score}")

            # Show final pipeline
            final = breakdown.get("final", {})
            if final:
                print("   Score Pipeline:")
                print(f"     Pre-penalty:     {final.get('pre_penalty_score', 'N/A')}")
                print(f"     Post-uncertainty: {final.get('post_uncertainty_score', 'N/A')}")
                print(f"     Post-severity:   {final.get('post_severity_score', 'N/A')}")
                print(f"     Post-cap:        {final.get('post_cap_score', 'N/A')}")
                print(f"     Final:           {final.get('composite_score', 'N/A')}")


def export_to_csv(
    ranked: List[RankedSecurity],
    filepath: Union[str, Path],
    top_n: Optional[int] = None,
) -> None:
    """
    Export rankings to CSV file.

    Args:
        ranked: List of RankedSecurity
        filepath: Output CSV path
        top_n: Limit to top N (None for all)
    """
    display_list = ranked[:top_n] if top_n else ranked

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank",
            "Ticker",
            "Final_Score",
            "PreGov_Score",
            "Gov_Impact",
            "Gov_Impact_Pct",
            "Gov_Flags",
            "Severity",
            "Stage",
            "Market_Cap_Bucket",
        ])

        for sec in display_list:
            writer.writerow([
                sec.rank,
                sec.ticker,
                f"{sec.enhanced_score:.2f}",
                f"{sec.base_score:.2f}",
                f"{sec.delta:+.2f}",
                f"{sec.delta_pct:+.1f}%",
                sec.gov_flags,
                sec.severity,
                sec.stage_bucket,
                sec.market_cap_bucket,
            ])

    print(f"\nRankings exported to: {filepath}")


def export_to_json(
    ranked: List[RankedSecurity],
    filepath: Union[str, Path],
    top_n: Optional[int] = None,
) -> None:
    """
    Export rankings to JSON file.

    Args:
        ranked: List of RankedSecurity
        filepath: Output JSON path
        top_n: Limit to top N (None for all)
    """
    display_list = ranked[:top_n] if top_n else ranked

    data = {
        "version": __version__,
        "count": len(display_list),
        "rankings": [
            {
                "rank": sec.rank,
                "ticker": sec.ticker,
                "final_score": str(sec.enhanced_score),
                "pre_gov_score": str(sec.base_score),
                "gov_impact": str(sec.delta),
                "gov_impact_pct": str(sec.delta_pct),
                "gov_flags": sec.gov_flags,
                "severity": sec.severity,
                "stage_bucket": sec.stage_bucket,
                "market_cap_bucket": sec.market_cap_bucket,
            }
            for sec in display_list
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nRankings exported to: {filepath}")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_enhancement_impact(ranked: List[RankedSecurity]) -> Dict[str, Any]:
    """
    Analyze the impact of enhancements on rankings.

    Args:
        ranked: List of RankedSecurity

    Returns:
        Dictionary with analysis results
    """
    if not ranked:
        return {"error": "No securities to analyze"}

    deltas = [sec.delta for sec in ranked]
    enhanced_scores = [sec.enhanced_score for sec in ranked]
    base_scores = [sec.base_score for sec in ranked]

    # Basic statistics
    avg_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    min_delta = min(deltas)

    # Count by delta direction
    positive_count = sum(1 for d in deltas if d > 0)
    negative_count = sum(1 for d in deltas if d < 0)
    zero_count = sum(1 for d in deltas if d == 0)

    # Score ranges
    avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
    avg_base = sum(base_scores) / len(base_scores)

    # Rank correlation check (are high base scores still high enhanced?)
    # Simple spearman proxy: check if top 10 base scores are in top 20 enhanced
    top_10_by_base = sorted(ranked, key=lambda x: x.base_score, reverse=True)[:10]
    top_10_base_tickers = {s.ticker for s in top_10_by_base}
    top_20_enhanced_tickers = {s.ticker for s in ranked[:20]}
    overlap_count = len(top_10_base_tickers & top_20_enhanced_tickers)

    # Count by governance flags
    clean_count = sum(1 for s in ranked if s.gov_flags == "PASS")
    penalized_count = len(ranked) - clean_count

    return {
        "total_securities": len(ranked),
        "average_gov_impact": str(_quantize_score(avg_delta)),
        "max_gov_impact": str(max_delta),
        "min_gov_impact": str(min_delta),
        "clean_count": clean_count,
        "penalized_count": penalized_count,
        "average_final_score": str(_quantize_score(avg_enhanced)),
        "average_pre_gov_score": str(_quantize_score(avg_base)),
        "top10_pregov_in_top20_final": overlap_count,
        "ranking_stability": "high" if overlap_count >= 7 else "medium" if overlap_count >= 4 else "low",
    }


def find_biggest_movers(
    ranked: List[RankedSecurity],
    top_n: int = 10,
) -> Tuple[List[RankedSecurity], List[RankedSecurity]]:
    """
    Find securities with biggest positive and negative deltas.

    Args:
        ranked: List of RankedSecurity
        top_n: Number of movers to return

    Returns:
        Tuple of (biggest_gainers, biggest_losers)
    """
    sorted_by_delta = sorted(ranked, key=lambda x: x.delta, reverse=True)

    gainers = sorted_by_delta[:top_n]
    losers = sorted_by_delta[-top_n:][::-1]  # Reverse to show biggest losers first

    return gainers, losers


def print_movers(
    ranked: List[RankedSecurity],
    top_n: int = 10,
) -> None:
    """
    Print biggest movers (gainers and losers).

    Args:
        ranked: List of RankedSecurity
        top_n: Number of movers to show
    """
    gainers, losers = find_biggest_movers(ranked, top_n)

    print(f"\n{'='*60}")
    print(f"LEAST PENALIZED (Smallest Gov Impact)")
    print(f"{'='*60}")
    print(f"{'Ticker':<10}{'Final':<12}{'Pre-Gov':<12}{'Impact':<12}{'Flags':<10}")
    print("-" * 60)
    for sec in gainers:
        print(f"{sec.ticker:<10}{sec.enhanced_score:<12.2f}{sec.base_score:<12.2f}{sec.delta:+10.2f}  {sec.gov_flags:<10}")

    print(f"\n{'='*60}")
    print(f"MOST PENALIZED (Largest Gov Impact)")
    print(f"{'='*60}")
    print(f"{'Ticker':<10}{'Final':<12}{'Pre-Gov':<12}{'Impact':<12}{'Flags':<10}")
    print("-" * 60)
    for sec in losers:
        print(f"{sec.ticker:<10}{sec.enhanced_score:<12.2f}{sec.base_score:<12.2f}{sec.delta:+10.2f}  {sec.gov_flags:<10}")


# =============================================================================
# SELF-CHECKS
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify module correctness."""
    errors = []

    # CHECK 1: Decimal conversion
    d = _to_decimal("42.567")
    if d != Decimal("42.567"):
        errors.append(f"CHECK1 FAIL: _to_decimal('42.567') = {d}")

    # CHECK 2: Score quantization
    q = _quantize_score(Decimal("42.567"))
    if q != Decimal("42.57"):
        errors.append(f"CHECK2 FAIL: _quantize_score(42.567) = {q}")

    # CHECK 3: RankedSecurity delta calculation
    sec = RankedSecurity(
        ticker="TEST",
        enhanced_score=Decimal("75.00"),
        base_score=Decimal("70.00"),
        rank=1,
    )
    if sec.delta != Decimal("5.00"):
        errors.append(f"CHECK3 FAIL: delta = {sec.delta}, expected 5.00")

    # CHECK 4: Delta percentage
    if sec.delta_pct != Decimal("7.14"):
        errors.append(f"CHECK4 FAIL: delta_pct = {sec.delta_pct}, expected 7.14")

    # CHECK 5: Ranking determinism
    enhanced = {"A": Decimal("80"), "B": Decimal("80"), "C": Decimal("70")}
    base = {"A": Decimal("75"), "B": Decimal("70"), "C": Decimal("65")}
    metadata: Dict[str, Dict[str, Any]] = {}

    ranked1 = rank_securities(enhanced, base, metadata)
    ranked2 = rank_securities(enhanced, base, metadata)

    tickers1 = [s.ticker for s in ranked1]
    tickers2 = [s.ticker for s in ranked2]

    if tickers1 != tickers2:
        errors.append(f"CHECK5 FAIL: Non-deterministic ranking: {tickers1} vs {tickers2}")

    # CHECK 6: Tiebreaker order (A has higher base score, should rank before B)
    if tickers1[0] != "A" or tickers1[1] != "B":
        errors.append(f"CHECK6 FAIL: Tiebreaker failed: {tickers1}, expected ['A', 'B', 'C']")

    return errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display score rankings with enhanced vs base comparison"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to Module 5 results JSON file",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top securities to display (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (for csv/json formats)",
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed score breakdown",
    )
    parser.add_argument(
        "--movers", "-m",
        action="store_true",
        help="Show biggest movers (gainers/losers)",
    )
    parser.add_argument(
        "--analysis", "-a",
        action="store_true",
        help="Show enhancement impact analysis",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run self-checks and exit",
    )

    args = parser.parse_args()

    # Run self-checks if requested
    if args.self_check:
        errors = _run_self_checks()
        if errors:
            print("SELF-CHECK FAILURES:")
            for e in errors:
                print(f"  {e}")
            sys.exit(1)
        else:
            print("All self-checks passed!")
            sys.exit(0)

    # Validate --file is provided for normal operation
    if not args.file:
        parser.error("--file/-f is required")

    # Load and process results
    try:
        results = load_results(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Extract scores and build rankings
    enhanced_scores, base_scores, metadata = extract_scores(results)

    if not enhanced_scores:
        print("Error: No securities found in results")
        sys.exit(1)

    ranked = rank_securities(enhanced_scores, base_scores, metadata)

    # Output based on format
    if args.format == "text":
        print_rankings(ranked, args.top)

        if args.detailed:
            print_detailed_breakdown(ranked, min(args.top, 10))

        if args.movers:
            print_movers(ranked)

        if args.analysis:
            analysis = analyze_enhancement_impact(ranked)
            print(f"\n{'='*50}")
            print("ENHANCEMENT IMPACT ANALYSIS")
            print(f"{'='*50}")
            for key, value in analysis.items():
                print(f"  {key}: {value}")

    elif args.format == "csv":
        if not args.output:
            print("Error: --output required for csv format")
            sys.exit(1)
        export_to_csv(ranked, args.output, args.top)

    elif args.format == "json":
        if not args.output:
            print("Error: --output required for json format")
            sys.exit(1)
        export_to_json(ranked, args.output, args.top)

    print(f"\nProcessed {len(ranked)} securities")


if __name__ == "__main__":
    main()
