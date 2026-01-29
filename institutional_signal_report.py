"""
Institutional Signal Analysis & Reporting

Generates actionable investment signals from 13F holdings data.
Implements deterministic scoring with risk gates and audit trail.

All functions are deterministic (no datetime.now() in scoring).
Report generation uses provided as_of_date for timestamps.
"""
import argparse
import hashlib
import json
from typing import Any, Dict, List, Optional

# Import risk gates for fail-closed filtering
from risk_gates import (
    load_market_data,
    load_financial_data,
    apply_all_gates,
    compute_parameters_hash as compute_risk_gates_hash,
)

# Import from canonical manager registry
from elite_managers import (
    get_elite_managers,
    get_all_managers,
    get_elite_ciks,
)

# =============================================================================
# VERSION TRACKING
# =============================================================================

SIGNAL_SCORE_VERSION = "2.1.0"  # Updated: now loads from registry


# =============================================================================
# MANAGER REGISTRY (loaded from canonical source)
# =============================================================================

def _build_manager_names() -> Dict[str, str]:
    """Build CIK -> name mapping from registry."""
    return {m['cik']: m['name'] for m in get_all_managers()}


def _build_elite_managers() -> set:
    """Build set of Elite Core manager CIKs from registry."""
    return set(get_elite_ciks())


# Lazy-loaded manager data (cached on first access)
_manager_names_cache = None
_elite_managers_cache = None


def get_manager_names() -> Dict[str, str]:
    """Get CIK -> name mapping (cached)."""
    global _manager_names_cache
    if _manager_names_cache is None:
        _manager_names_cache = _build_manager_names()
    return _manager_names_cache


def get_elite_manager_ciks() -> set:
    """Get set of Elite Core CIKs (cached)."""
    global _elite_managers_cache
    if _elite_managers_cache is None:
        _elite_managers_cache = _build_elite_managers()
    return _elite_managers_cache


# Note: Use get_manager_names() and get_elite_manager_ciks() functions
# instead of direct dict/set access to ensure fresh data from registry


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str = "production_data/holdings_snapshots.json") -> Dict:
    """Load holdings data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_coverage(data: Dict) -> Dict[str, Any]:
    """
    Analyze manager coverage statistics.

    Args:
        data: Holdings data dict

    Returns:
        Coverage statistics dict
    """
    mgr_counts = [len(info["holdings"]["current"]) for info in data.values()]
    n = len(data)
    coverage = {
        "total_tickers": n,
        "avg_managers": round(sum(mgr_counts) / n, 2) if n > 0 else 0,
        "distribution": {}
    }
    for threshold in [1, 2, 3, 4, 5, 6, 7, 8]:
        count = sum(1 for c in mgr_counts if c >= threshold)
        pct = (count / n * 100) if n > 0 else 0
        coverage["distribution"][f"{threshold}+"] = {"count": count, "pct": pct}
    return coverage


# =============================================================================
# SIGNAL SCORING
# =============================================================================

def calculate_signal_score(
    ticker: str,
    info: Dict,
    market_data: Optional[Dict[str, Dict[str, Any]]] = None,
    financial_data: Optional[Dict[str, Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculate comprehensive signal score for a ticker.

    Enhanced scoring includes:
    - Net buyers (new + increased - decreased positions)
    - Net dollar flow (change in total value)
    - Conviction proxy (position size relative to typical)
    - Elite manager initiations (2x bonus)
    - Consensus boost (3 at 6+ mgrs, 2 at 5+ mgrs)
    - Coordination boost (1 if 50%+ are buying)
    - Dollar boost (up to 3 based on net flow)
    - Staleness penalty (placeholder = 1.0)

    Risk Gate Integration:
    - When market_data and financial_data provided, applies risk gates
    - Failed gates result in passes_gates=False and signal_score=0
    - The signal is still returned (not None) for reporting killed signals

    Args:
        ticker: Stock ticker symbol
        info: Holdings info dict with current/prior holdings
        market_data: Optional market data dict for risk gates
        financial_data: Optional financial data dict for risk gates

    Returns:
        Signal score dict, or None if mgr_count < 4
    """
    curr = info["holdings"]["current"]
    prior = info["holdings"]["prior"]
    mgr_count = len(curr)

    # Gate: minimum manager count
    if mgr_count < 4:
        return None

    # Calculate totals
    total_curr = sum(pos["value_kusd"] for pos in curr.values())
    total_prior = sum(pos["value_kusd"] for pos in prior.values() if pos["value_kusd"] > 0)

    # Net flow in thousands USD
    net_flow_kusd = total_curr - total_prior

    # Percentage change
    pct_change = ((total_curr - total_prior) / total_prior * 100) if total_prior > 0 else 0

    # Track manager activity
    up_list = []
    new_list = []
    down_list = []
    conviction_values = []

    for mgr_cik in curr.keys():
        curr_val = curr[mgr_cik]["value_kusd"]
        prior_val = prior.get(mgr_cik, {}).get("value_kusd", 0)

        # Conviction proxy: position size / $50M normalized
        if curr_val > 0:
            conviction_proxy = min(curr_val / 50_000, 2.0)
            conviction_values.append(conviction_proxy)

        if prior_val == 0:
            new_list.append(mgr_cik)
        elif curr_val > prior_val * 1.2:
            up_list.append(mgr_cik)
        elif curr_val < prior_val * 0.8:
            down_list.append(mgr_cik)

    # Average conviction
    avg_conviction = sum(conviction_values) / len(conviction_values) if conviction_values else 1.0
    conviction_multiplier = 1.0 + (avg_conviction - 1.0) * 0.3

    # Score components
    net_buyers = len(up_list) + len(new_list) - len(down_list)

    # Elite managers get 2x bonus for new positions
    elite_ciks = get_elite_manager_ciks()
    elite_new_boost = sum(2 for cik in new_list if cik in elite_ciks)

    # Consensus boost
    if mgr_count >= 6:
        consensus_boost = 3
    elif mgr_count >= 5:
        consensus_boost = 2
    else:
        consensus_boost = 0

    # Coordination boost: 50%+ of managers buying
    coord_boost = 1 if (len(up_list) + len(new_list)) >= mgr_count * 0.5 else 0

    # Dollar flow boost: up to 3 points based on net flow
    if net_flow_kusd > 0:
        dollar_boost = min(int(net_flow_kusd / 100_000), 3)
    else:
        dollar_boost = 0

    # Staleness penalty (placeholder for future implementation)
    staleness_penalty = 1.0

    # Calculate base score
    base_score = net_buyers + elite_new_boost + consensus_boost + coord_boost + dollar_boost

    # Apply multipliers
    signal_score = int(base_score * conviction_multiplier * staleness_penalty)

    # Build result dict
    result = {
        "ticker": ticker,
        "mgrs": mgr_count,
        "pct_change": pct_change,
        "net_buyers": net_buyers,
        "signal_score": signal_score,
        "net_flow_kusd": net_flow_kusd,
        "conviction_avg": round(avg_conviction, 2),
        "new": [get_manager_names().get(cik, cik) for cik in sorted(new_list)],
        "increased": [get_manager_names().get(cik, cik) for cik in sorted(up_list)],
        "decreased": [get_manager_names().get(cik, cik) for cik in sorted(down_list)],
        # Risk gate fields (defaults before applying gates)
        "passes_gates": True,
        "risk_flags": [],
        "gate_results": {},
        # Version tracking
        "score_version": SIGNAL_SCORE_VERSION,
        "parameters_hash": compute_signal_parameters_hash(),
    }

    # Apply risk gates if data provided
    if market_data is not None or financial_data is not None:
        gate_result = apply_all_gates(ticker, market_data, financial_data)

        result["passes_gates"] = gate_result["passes"]
        result["risk_flags"] = gate_result["risk_flags"]
        result["gate_results"] = {
            "adv_usd": gate_result.get("adv_usd", 0.0),
            "price": gate_result.get("price"),
            "market_cap": gate_result.get("market_cap"),
            "runway_months": gate_result.get("runway_months"),
        }

        # Kill switch: failed gates = score 0
        if not gate_result["passes"]:
            result["signal_score"] = 0

    return result


# =============================================================================
# PARAMETER MANAGEMENT
# =============================================================================

def get_signal_parameters() -> Dict[str, Any]:
    """
    Get current signal scoring parameters.

    Returns:
        Dict of all scoring parameters
    """
    return {
        "version": SIGNAL_SCORE_VERSION,
        "min_manager_count": 4,
        "conviction_base_kusd": 50_000,
        "conviction_max": 2.0,
        "conviction_weight": 0.3,
        "elite_new_bonus": 2,
        "consensus_6_boost": 3,
        "consensus_5_boost": 2,
        "coord_threshold": 0.5,
        "dollar_flow_divisor": 100_000,
        "dollar_boost_max": 3,
        "staleness_penalty": 1.0,
        "elite_managers": sorted(list(get_elite_manager_ciks())),
    }


def compute_signal_parameters_hash() -> str:
    """
    Compute SHA256 hash of signal parameters.

    Returns:
        First 16 characters of SHA256 hash
    """
    params = get_signal_parameters()
    canonical = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    data: Dict,
    market_data: Optional[Dict[str, Dict[str, Any]]] = None,
    financial_data: Optional[Dict[str, Dict[str, Any]]] = None,
    output_file: Optional[str] = None,
    as_of_date: Optional[str] = None
) -> List[Dict]:
    """
    Generate institutional signal report.

    Produces two sections:
    1. TOP 25 INSTITUTIONAL SIGNALS (PASSING RISK GATES)
    2. SIGNALS KILLED BY RISK GATES (top 10)

    Args:
        data: Holdings data dict
        market_data: Market data dict for risk gates (optional)
        financial_data: Financial data dict for risk gates (optional)
        output_file: Path to write report (optional)
        as_of_date: Analysis date for report header (optional)

    Returns:
        List of all signal dicts
    """
    coverage = analyze_coverage(data)

    # Score all tickers
    all_signals = []
    for ticker in sorted(data.keys()):  # Sort for determinism
        info = data[ticker]
        score = calculate_signal_score(ticker, info, market_data, financial_data)
        if score:
            all_signals.append(score)

    # Separate passing and killed signals
    passing = [s for s in all_signals if s["passes_gates"]]
    killed = [s for s in all_signals if not s["passes_gates"]]

    # Sort passing: by (-signal_score, -mgrs, -pct_change, ticker)
    passing.sort(key=lambda x: (-x["signal_score"], -x["mgrs"], -x["pct_change"], x["ticker"]))

    # Sort killed: by (-mgrs, -pct_change, ticker)
    killed.sort(key=lambda x: (-x["mgrs"], -x["pct_change"], x["ticker"]))

    # Build report
    report = []
    report.append("=" * 120)
    report.append("WAKE ROBIN INSTITUTIONAL SIGNAL REPORT")
    if as_of_date:
        report.append(f"As Of Date: {as_of_date}")
    report.append(f"Score Version: {SIGNAL_SCORE_VERSION}")
    report.append(f"Parameters Hash: {compute_signal_parameters_hash()}")
    report.append("=" * 120)
    report.append("")

    # Coverage summary
    report.append("COVERAGE SUMMARY:")
    report.append(f"  Total Tickers: {coverage['total_tickers']}")
    report.append(f"  Avg Managers/Ticker: {coverage['avg_managers']}")
    report.append(f"  Signals Passing Gates: {len(passing)}")
    report.append(f"  Signals Killed by Gates: {len(killed)}")
    report.append("")
    for key, val in sorted(coverage["distribution"].items()):
        report.append(f"  {key} managers: {val['count']:3} tickers ({val['pct']:5.1f}%)")
    report.append("")

    # Passing signals table
    report.append("TOP 25 INSTITUTIONAL SIGNALS (PASSING RISK GATES):")
    report.append("-" * 120)
    header = f"{'Ticker':<7} {'Score':<6} {'Mgrs':<5} {'Q/Q':<7} {'Net':<4} {'$Flow':<10} {'Conv':<5} {'Activity Summary'}"
    report.append(header)
    report.append("-" * 120)

    for r in passing[:25]:
        activity = []
        if r["new"]:
            new_str = ", ".join(r["new"][:3])
            if len(r["new"]) > 3:
                new_str += f" +{len(r['new'])-3}"
            activity.append(f"NEW({len(r['new'])}): {new_str}")
        if r["increased"]:
            up_str = ", ".join(r["increased"][:3])
            if len(r["increased"]) > 3:
                up_str += f" +{len(r['increased'])-3}"
            activity.append(f"UP({len(r['increased'])}): {up_str}")
        if r["decreased"]:
            activity.append(f"DN({len(r['decreased'])})")
        act_str = " | ".join(activity) if activity else "Stable"

        # Format dollar flow
        flow_kusd = r.get("net_flow_kusd", 0)
        if flow_kusd >= 1_000_000:
            flow_str = f"+${flow_kusd/1_000_000:.1f}B"
        elif flow_kusd >= 1_000:
            flow_str = f"+${flow_kusd/1_000:.0f}M"
        elif flow_kusd <= -1_000_000:
            flow_str = f"-${abs(flow_kusd)/1_000_000:.1f}B"
        elif flow_kusd <= -1_000:
            flow_str = f"-${abs(flow_kusd)/1_000:.0f}M"
        elif flow_kusd > 0:
            flow_str = f"+${flow_kusd:.0f}K"
        else:
            flow_str = f"${flow_kusd:.0f}K"

        conv_str = f"{r.get('conviction_avg', 1.0):.1f}"

        line = f"{r['ticker']:<7} {r['signal_score']:<6} {r['mgrs']:<5} {r['pct_change']:>+6.0f}% {r['net_buyers']:>3}  {flow_str:<10} {conv_str:<5} {act_str}"
        report.append(line)

    report.append("")

    # Killed signals section
    if killed:
        report.append("SIGNALS KILLED BY RISK GATES (Top 10):")
        report.append("-" * 120)
        report.append(f"{'Ticker':<7} {'Mgrs':<5} {'Q/Q':<8} {'Risk Flags'}")
        report.append("-" * 120)

        for r in killed[:10]:
            flags_str = ", ".join(r.get("risk_flags", []))
            line = f"{r['ticker']:<7} {r['mgrs']:<5} {r['pct_change']:>+6.0f}%  {flags_str}"
            report.append(line)

        report.append("")

    # Methodology section
    report.append("=" * 120)
    report.append("SCORING METHODOLOGY:")
    report.append("")
    report.append("  Signal Score = (Net Buyers + Elite Bonus + Consensus + Coordination + Dollar Boost)")
    report.append("                 * Conviction Multiplier * Staleness Penalty")
    report.append("")
    report.append("  Components:")
    report.append("    Net Buyers = (New + Increased) - Decreased positions")
    report.append("    Elite Bonus = 2 points per elite manager initiating new position")
    report.append("    Consensus = +3 if 6+ managers, +2 if 5+ managers")
    report.append("    Coordination = +1 if 50%+ of managers are buying")
    report.append("    Dollar Boost = min(net_flow / $100M, 3)")
    report.append("    Conviction = 1.0 + (avg_position / $50M - 1) * 0.3")
    report.append("")
    report.append("  Risk Gates (signals killed if any fail):")
    report.append("    ADV_UNKNOWN - Average dollar volume cannot be computed")
    report.append("    LOW_LIQUIDITY - ADV below tier threshold")
    report.append("    PENNY_STOCK - Price below $2.00")
    report.append("    MICRO_CAP - Market cap below $50M")
    report.append("    CASH_RISK - Cash runway below 6 months")
    report.append("")
    # Build elite manager list dynamically from registry
    elite_mgrs = get_elite_managers()
    elite_names = [m['short_name'] for m in elite_mgrs[:9]]
    remaining = len(elite_mgrs) - 9
    elite_str = ", ".join(elite_names)
    if remaining > 0:
        elite_str += f" +{remaining} more"
    report.append(f"  Elite Core Managers ({len(elite_mgrs)}): {elite_str}")
    report.append("=" * 120)

    # Write output
    output = "\n".join(report)
    if output_file:
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(output)
        print(f"Report saved to: {output_file}")

    print(output)
    return all_signals


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate Institutional Signal Report"
    )
    parser.add_argument(
        "--holdings",
        type=str,
        default="production_data/holdings_snapshots.json",
        help="Path to holdings_snapshots.json"
    )
    parser.add_argument(
        "--market-data",
        type=str,
        default=None,
        help="Path to market_data.json for risk gates"
    )
    parser.add_argument(
        "--financial-data",
        type=str,
        default=None,
        help="Path to financial_data.json for risk gates"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="INSTITUTIONAL_SIGNAL_REPORT.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Analysis date (YYYY-MM-DD) for report header"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading holdings: {args.holdings}")
    data = load_data(args.holdings)

    market_data = None
    if args.market_data:
        print(f"Loading market data: {args.market_data}")
        market_data = load_market_data(args.market_data)

    financial_data = None
    if args.financial_data:
        print(f"Loading financial data: {args.financial_data}")
        financial_data = load_financial_data(args.financial_data)

    # Generate report
    signals = generate_report(
        data,
        market_data=market_data,
        financial_data=financial_data,
        output_file=args.output,
        as_of_date=args.as_of
    )


if __name__ == "__main__":
    main()
