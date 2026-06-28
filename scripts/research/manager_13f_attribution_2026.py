"""
Manager 13F Performance Attribution — 2025-2026

Classification: RESEARCH_DIAGNOSTIC / MANAGER_ATTRIBUTION / NO_MODEL_CHANGE
                NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE

Builds per-manager return sleeves from 13F holdings across 5 quarterly filing windows,
computes forward returns vs XBI, bootstrap controls, and a manager quality score.

Signal date = filing acceptance date (filed_at) from filings_metadata — NOT quarter-end.
This is the earliest a signal could be publicly observable.

Output: artifacts/research/manager_13f_performance_2026/
"""

import csv
import datetime
import json
import os
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOLDINGS_DIR = os.path.join(REPO, "data", "13f_history_full")
PRICE_CSV = os.path.join(REPO, "production_data", "price_history_split_adj.csv")
MANAGER_REGISTRY = os.path.join(REPO, "production_data", "manager_registry.json")
FILING_STATUS = os.path.join(REPO, "production_data", "13f_filing_status.json")
INST_SUMMARY = os.path.join(REPO, "production_data", "institutional_summary.json")
OUT_DIR = os.path.join(REPO, "artifacts", "research", "manager_13f_performance_2026")

QUARTERS = [
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-03-31",
]

FORWARD_WINDOWS = [20, 63, 126]  # trading days approx
BOOTSTRAP_N = 200
RANDOM_SEED = 42

# Quality score weights
QS_WEIGHT_63D_EXCESS = 0.35
QS_WEIGHT_HIT63 = 0.20
QS_WEIGHT_BOOTSTRAP = 0.20
QS_WEIGHT_FLOW = 0.15
QS_WEIGHT_COVERAGE = 0.10
QS_PENALTY_FEW_WINDOWS = 0.20  # subtract if n_windows < 3
QS_PENALTY_CONCENTRATION = 0.10  # subtract if top holding > 50% of basket return


# ---------------------------------------------------------------------------
# Step 1: Load price history
# ---------------------------------------------------------------------------


def load_prices():
    """Returns dict: ticker -> {date_str -> close_price}"""
    prices = defaultdict(dict)
    with open(PRICE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"]
            date = row["date"]
            try:
                close = float(row["close"])
                prices[ticker][date] = close
            except (ValueError, TypeError):
                pass
    return dict(prices)


def get_sorted_dates(prices, ticker):
    if ticker not in prices:
        return []
    return sorted(prices[ticker].keys())


def get_price_on_or_after(prices, ticker, date_str):
    """Return (date, price) for the first trading day on or after date_str."""
    dates = get_sorted_dates(prices, ticker)
    for d in dates:
        if d >= date_str:
            return d, prices[ticker][d]
    return None, None


def get_forward_price(prices, ticker, start_date, n_trading_days):
    """Return close price approximately n_trading_days after start_date."""
    dates = get_sorted_dates(prices, ticker)
    # Find start index
    start_idx = None
    for i, d in enumerate(dates):
        if d >= start_date:
            start_idx = i
            break
    if start_idx is None:
        return None, None
    target_idx = start_idx + n_trading_days
    if target_idx >= len(dates):
        return None, None
    return dates[target_idx], prices[ticker][dates[target_idx]]


# ---------------------------------------------------------------------------
# Step 2: Load manager registry
# ---------------------------------------------------------------------------


def load_manager_registry():
    with open(MANAGER_REGISTRY) as f:
        reg = json.load(f)
    managers = {}
    for category in ["elite_core", "conditional"]:
        for m in reg.get(category, []):
            cik = m["cik"]
            managers[cik] = {
                "name": m["name"],
                "aum_b": m.get("aum_b", 0),
                "style": m.get("style", ""),
                "category": category,
            }
    return managers


# ---------------------------------------------------------------------------
# Step 3: Load holdings per quarter
# ---------------------------------------------------------------------------


def load_quarter_holdings(quarter_end):
    """
    Returns:
        manager_holdings: dict cik -> {ticker -> {value_kusd, shares}}
        manager_filed_at: dict cik -> date_str (filing acceptance date)
        ticker_fmeta: dict ticker -> {cik -> {filed_at, accession, ...}}
        manager_prior_holdings: dict cik -> {ticker -> {value_kusd, shares}} (prior quarter)
    """
    path = os.path.join(HOLDINGS_DIR, f"holdings_{quarter_end}.json")
    if not os.path.exists(path):
        return {}, {}, {}, {}

    with open(path) as f:
        d = json.load(f)

    tickers_data = d.get("tickers", {})

    manager_holdings = defaultdict(dict)
    manager_prior = defaultdict(dict)
    manager_filed_at = {}
    ticker_fmeta = {}

    for ticker, tdata in tickers_data.items():
        if not isinstance(tdata, dict):
            continue
        holdings = tdata.get("holdings", {})
        fmeta = tdata.get("filings_metadata", {})
        ticker_fmeta[ticker] = fmeta

        current = holdings.get("current", {})
        prior = holdings.get("prior", {})

        for cik, hdata in current.items():
            val = hdata.get("value_kusd", 0)
            shares = hdata.get("shares", 0)
            manager_holdings[cik][ticker] = {"value_kusd": val, "shares": shares}
            # Get filing date from filings_metadata
            if cik in fmeta and "filed_at" in fmeta[cik]:
                filed = fmeta[cik]["filed_at"][:10]
                # Use the latest filed_at across tickers (most permissive signal date)
                if cik not in manager_filed_at or filed > manager_filed_at[cik]:
                    manager_filed_at[cik] = filed

        for cik, hdata in prior.items():
            val = hdata.get("value_kusd", 0)
            shares = hdata.get("shares", 0)
            manager_prior[cik][ticker] = {"value_kusd": val, "shares": shares}

    return dict(manager_holdings), manager_filed_at, ticker_fmeta, dict(manager_prior)


# ---------------------------------------------------------------------------
# Step 4: Build sleeves and compute forward returns
# ---------------------------------------------------------------------------


def compute_forward_return(prices, ticker, signal_date, window_days):
    """Returns (fwd_return, valid) for a single ticker."""
    start_d, start_p = get_price_on_or_after(prices, ticker, signal_date)
    if start_d is None or start_p is None or start_p == 0:
        return None, False
    end_d, end_p = get_forward_price(prices, ticker, start_d, window_days)
    if end_d is None or end_p is None:
        return None, False
    return (end_p / start_p) - 1.0, True


def build_sleeve_returns(prices, manager_holdings, prior_holdings, signal_date, window_days, eligible_universe):
    """
    Returns dict of sleeve_name -> {return, n_holdings, hit_vs_xbi, ...}
    sleeves: value_weighted_all, equal_weighted_all, top10_value, new_positions, increased_positions
    """
    # Get XBI return for this signal date / window
    xbi_ret, xbi_valid = compute_forward_return(prices, "XBI", signal_date, window_days)

    results = {}

    # All biotech holdings
    holdings = [(t, h) for t, h in manager_holdings.items() if t in eligible_universe and t != "XBI"]
    if not holdings:
        return results

    # Compute returns for all holdings
    ticker_returns = {}
    for ticker, h in holdings:
        ret, valid = compute_forward_return(prices, ticker, signal_date, window_days)
        if valid:
            ticker_returns[ticker] = (ret, h.get("value_kusd", 0))

    if not ticker_returns:
        return results

    # Value-weighted all biotech
    total_val = sum(v for _, v in ticker_returns.values())
    if total_val > 0:
        vw_ret = sum(ret * val / total_val for ret, val in ticker_returns.values())
        results["value_weighted_all"] = {
            "return": vw_ret,
            "xbi_return": xbi_ret,
            "excess": vw_ret - xbi_ret if xbi_valid else None,
            "n": len(ticker_returns),
        }

    # Equal-weighted all
    if ticker_returns:
        ew_ret = sum(ret for ret, _ in ticker_returns.values()) / len(ticker_returns)
        results["equal_weighted_all"] = {
            "return": ew_ret,
            "xbi_return": xbi_ret,
            "excess": ew_ret - xbi_ret if xbi_valid else None,
            "n": len(ticker_returns),
        }

    # Top 10 by value
    top10 = sorted(ticker_returns.items(), key=lambda x: x[1][1], reverse=True)[:10]
    if top10:
        top10_val = sum(v for _, (_, v) in top10)
        if top10_val > 0:
            top10_ret = sum(r * v / top10_val for _, (r, v) in top10)
        else:
            top10_ret = sum(r for _, (r, _) in top10) / len(top10)
        results["top10_value_weighted"] = {
            "return": top10_ret,
            "xbi_return": xbi_ret,
            "excess": top10_ret - xbi_ret if xbi_valid else None,
            "n": len(top10),
        }

    # New positions (not in prior filing)
    new_tickers = [t for t in ticker_returns if t not in prior_holdings]
    if new_tickers:
        new_ret = sum(ticker_returns[t][0] for t in new_tickers) / len(new_tickers)
        results["new_positions"] = {
            "return": new_ret,
            "xbi_return": xbi_ret,
            "excess": new_ret - xbi_ret if xbi_valid else None,
            "n": len(new_tickers),
        }

    # Increased positions (positive delta vs prior)
    increased = []
    for t in ticker_returns:
        if t in prior_holdings:
            cur_val = manager_holdings[t].get("value_kusd", 0)
            pri_val = prior_holdings[t].get("value_kusd", 0)
            if cur_val > pri_val * 1.05:  # 5% threshold
                increased.append(t)
    if increased:
        inc_ret = sum(ticker_returns[t][0] for t in increased) / len(increased)
        results["increased_positions"] = {
            "return": inc_ret,
            "xbi_return": xbi_ret,
            "excess": inc_ret - xbi_ret if xbi_valid else None,
            "n": len(increased),
        }

    # Concentration check: does top holding dominate basket return?
    if ticker_returns and xbi_valid:
        top_ticker = max(ticker_returns.items(), key=lambda x: x[1][1])
        top_contribution = top_ticker[1][1] / total_val if total_val > 0 else 0
        results["_concentration_top1_pct"] = top_contribution

    results["_xbi_return"] = xbi_ret
    results["_xbi_valid"] = xbi_valid

    return results


# ---------------------------------------------------------------------------
# Step 5: Bootstrap random baskets
# ---------------------------------------------------------------------------


def bootstrap_random_baskets(prices, eligible_universe, signal_date, n_holdings, window_days, n_samples=200, seed=42):
    """
    Draw n_samples equal-weight random baskets of n_holdings tickers from eligible_universe.
    Return list of basket returns (only valid ones).
    """
    rng = random.Random(seed)
    # Pre-filter to tickers with valid returns
    valid_tickers = []
    for ticker in eligible_universe:
        if ticker == "XBI":
            continue
        ret, valid = compute_forward_return(prices, ticker, signal_date, window_days)
        if valid:
            valid_tickers.append((ticker, ret))

    if len(valid_tickers) < n_holdings:
        return []

    basket_returns = []
    for _ in range(n_samples):
        sample = rng.sample(valid_tickers, min(n_holdings, len(valid_tickers)))
        basket_ret = sum(r for _, r in sample) / len(sample)
        basket_returns.append(basket_ret)

    return basket_returns


def empirical_percentile(value, distribution):
    if not distribution or value is None:
        return None
    return sum(1 for x in distribution if x < value) / len(distribution)


# ---------------------------------------------------------------------------
# Step 6: Leave-one-manager-out analysis (approximate)
# ---------------------------------------------------------------------------


def compute_leave_one_out(manager_data, all_tickers_coinvest):
    """
    Approximate: use current institutional_summary coinvest scores.
    For each manager, identify their top holdings and check if removing
    them would change the top-30 ranked tickers.
    """
    results = {}
    # Load current institutional summary
    try:
        with open(INST_SUMMARY) as f:
            inst = json.load(f)
        ticker_scores = {t: v.get("inst_score_z", 0) for t, v in inst.get("tickers", {}).items()}
        ranked = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
        top30 = set(t for t, _ in ranked[:30])

        for cik, mdata in manager_data.items():
            name = mdata.get("name", cik)
            # Estimate which tickers this manager holds across recent windows
            mgr_tickers = mdata.get("all_tickers_held", set())
            if not mgr_tickers:
                results[cik] = {"material_change": False, "reason": "no_holdings"}
                continue
            # Approximate: remove manager's holdings from score count
            # If they hold a top-30 ticker exclusively, it might drop
            exclusive_top30 = []
            for t in mgr_tickers:
                if t in top30:
                    # Check if this manager holds it exclusively
                    holder_names = inst.get("tickers", {}).get(t, {}).get("elite_holder_names", [])
                    if len(holder_names) == 1 and name in holder_names:
                        exclusive_top30.append(t)

            material = len(exclusive_top30) > 0
            results[cik] = {
                "material_change": material,
                "exclusive_top30_tickers": exclusive_top30,
                "top30_overlap_count": len([t for t in mgr_tickers if t in top30]),
            }
    except Exception as e:
        results["error"] = str(e)

    return results


# ---------------------------------------------------------------------------
# Step 7: Normalize and score managers
# ---------------------------------------------------------------------------


def normalize_series(values_dict, key, higher_is_better=True):
    """Min-max normalize a list of values."""
    vals = [v[key] for v in values_dict.values() if v.get(key) is not None]
    if not vals or max(vals) == min(vals):
        return {k: 0.5 for k in values_dict}
    lo, hi = min(vals), max(vals)
    result = {}
    for k, v in values_dict.items():
        val = v.get(key)
        if val is None:
            result[k] = 0.0
        else:
            norm = (val - lo) / (hi - lo)
            result[k] = norm if higher_is_better else (1 - norm)
    return result


def classify_manager(row):
    n_win = row.get("n_filing_windows", 0)
    n_hold = row.get("avg_holdings", 0)
    coverage = row.get("coverage_score", 0)
    excess63 = row.get("mean_63d_excess")
    hit63 = row.get("hit_rate_63d")
    boot_pct = row.get("bootstrap_pct_63d")

    if n_win < 3 or n_hold < 5 or coverage < 0.5:
        return "INSUFFICIENT_DATA"
    if (
        excess63 is not None
        and excess63 > 0
        and hit63 is not None
        and hit63 >= 0.55
        and boot_pct is not None
        and boot_pct >= 0.65
    ):
        return "UPWEIGHT_CANDIDATE_SHADOW"
    if (
        excess63 is not None
        and excess63 < 0
        and hit63 is not None
        and hit63 <= 0.45
        and boot_pct is not None
        and boot_pct <= 0.35
    ):
        return "DOWNWEIGHT_CANDIDATE_SHADOW"
    return "KEEP_CURRENT_WEIGHT"


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading prices...")
    prices = load_prices()
    all_price_tickers = set(prices.keys())
    all_dates = [d for ticker_prices in prices.values() for d in ticker_prices.keys()]
    date_min = min(all_dates) if all_dates else "N/A"
    date_max = max(all_dates) if all_dates else "N/A"
    print(f"  Price tickers: {len(all_price_tickers)}, date range: {date_min} to {date_max}")

    print("Loading manager registry...")
    managers = load_manager_registry()
    print(f"  Managers: {len(managers)}")

    # Build eligible universe (tickers with price data, excluding XBI itself for baskets)
    eligible_universe = all_price_tickers - {"XBI"}

    # Per-manager accumulation across quarters
    manager_windows = defaultdict(list)  # cik -> list of window dicts
    manager_all_tickers = defaultdict(set)

    # Global window records for sleeve analysis
    all_window_rows = []
    all_holdings_attr_rows = []

    print("Processing quarters...")
    for qtr in QUARTERS:
        print(f"  Quarter {qtr}...")
        mh, mf, tf, mp = load_quarter_holdings(qtr)

        for cik, holdings in mh.items():
            if not holdings:
                continue
            filed_at = mf.get(cik)
            if not filed_at:
                # Conservative fallback: quarter_end + 45 days
                qe = datetime.date.fromisoformat(qtr)
                filed_at = (qe + datetime.timedelta(days=45)).isoformat()
                lag_method = "45d_lag_fallback"
            else:
                lag_method = "filing_acceptance_date"

            prior = mp.get(cik, {})
            n_hold = len(holdings)

            # Per-window results for each forward horizon
            window_data = {
                "cik": cik,
                "quarter_end": qtr,
                "signal_date": filed_at,
                "lag_method": lag_method,
                "n_holdings": n_hold,
                "sleeves": {},
                "bootstrap": {},
            }

            for w in FORWARD_WINDOWS:
                sleeves = build_sleeve_returns(prices, holdings, prior, filed_at, w, eligible_universe)
                window_data["sleeves"][w] = sleeves

            # Bootstrap for 63d window using equal_weighted_all size
            n_for_bootstrap = min(n_hold, len(eligible_universe))
            if n_for_bootstrap >= 3:
                boot_dist = bootstrap_random_baskets(
                    prices, eligible_universe, filed_at, n_for_bootstrap, 63, BOOTSTRAP_N, RANDOM_SEED
                )
                ew63 = window_data["sleeves"].get(63, {}).get("equal_weighted_all", {}).get("return")
                if ew63 is not None and boot_dist:
                    pct = empirical_percentile(ew63, boot_dist)
                    window_data["bootstrap"]["63d_pct"] = pct
                    window_data["bootstrap"]["63d_boot_n"] = len(boot_dist)

            manager_windows[cik].append(window_data)
            manager_all_tickers[cik].update(holdings.keys())

            # Holdings attribution rows
            for ticker, h in holdings.items():
                for w in [63]:
                    sleeve_data = window_data["sleeves"].get(w, {})
                    ew = sleeve_data.get("equal_weighted_all", {})
                    xbi_r = sleeve_data.get("_xbi_return")
                    ticker_ret, valid = compute_forward_return(prices, ticker, filed_at, w)
                    if valid:
                        excess = ticker_ret - xbi_r if xbi_r is not None else None
                        all_holdings_attr_rows.append(
                            {
                                "cik": cik,
                                "manager": managers.get(cik, {}).get("name", cik),
                                "ticker": ticker,
                                "quarter_end": qtr,
                                "signal_date": filed_at,
                                "value_kusd": h.get("value_kusd", 0),
                                "fwd_63d_return": round(ticker_ret, 6),
                                "xbi_63d_return": round(xbi_r, 6) if xbi_r is not None else None,
                                "excess_63d": round(excess, 6) if excess is not None else None,
                            }
                        )

    # Build per-manager summary
    print("Building manager summaries...")
    manager_summary = {}

    for cik, windows in manager_windows.items():
        reg_info = managers.get(cik, {"name": cik, "aum_b": 0, "style": "unknown", "category": "unknown"})
        n_win = len(windows)
        avg_hold = sum(w["n_holdings"] for w in windows) / n_win if n_win else 0

        # Collect metrics per window
        excess_20 = []
        excess_63 = []
        excess_126 = []
        hit_20 = []
        hit_63 = []
        new_excess_63 = []
        boot_pcts = []
        windows_with_data = 0

        for w in windows:
            sleeves = w["sleeves"]
            ew20 = sleeves.get(20, {}).get("equal_weighted_all", {})
            ew63 = sleeves.get(63, {}).get("equal_weighted_all", {})
            ew126 = sleeves.get(126, {}).get("equal_weighted_all", {})
            new63 = sleeves.get(63, {}).get("new_positions", {})

            if ew20.get("excess") is not None:
                excess_20.append(ew20["excess"])
                hit_20.append(1 if ew20["excess"] > 0 else 0)
            if ew63.get("excess") is not None:
                excess_63.append(ew63["excess"])
                hit_63.append(1 if ew63["excess"] > 0 else 0)
                windows_with_data += 1
            if ew126.get("excess") is not None:
                excess_126.append(ew126["excess"])
            if new63.get("excess") is not None:
                new_excess_63.append(new63["excess"])
            bpct = w["bootstrap"].get("63d_pct")
            if bpct is not None:
                boot_pcts.append(bpct)

        coverage_score = windows_with_data / n_win if n_win > 0 else 0

        manager_summary[cik] = {
            "name": reg_info["name"],
            "aum_b": reg_info["aum_b"],
            "style": reg_info["style"],
            "category": reg_info["category"],
            "n_filing_windows": n_win,
            "avg_holdings": round(avg_hold, 1),
            "coverage_score": round(coverage_score, 3),
            "mean_20d_excess": round(sum(excess_20) / len(excess_20), 6) if excess_20 else None,
            "mean_63d_excess": round(sum(excess_63) / len(excess_63), 6) if excess_63 else None,
            "mean_126d_excess": round(sum(excess_126) / len(excess_126), 6) if excess_126 else None,
            "hit_rate_20d": round(sum(hit_20) / len(hit_20), 3) if hit_20 else None,
            "hit_rate_63d": round(sum(hit_63) / len(hit_63), 3) if hit_63 else None,
            "new_position_excess_63d": round(sum(new_excess_63) / len(new_excess_63), 6) if new_excess_63 else None,
            "bootstrap_pct_63d": round(sum(boot_pcts) / len(boot_pcts), 3) if boot_pcts else None,
            "all_tickers_held": list(manager_all_tickers[cik]),
        }

    # Compute manager quality scores
    print("Computing quality scores...")
    # Normalize metrics
    norm_63d = normalize_series(manager_summary, "mean_63d_excess", higher_is_better=True)
    norm_hit63 = normalize_series(manager_summary, "hit_rate_63d", higher_is_better=True)
    norm_boot = normalize_series(manager_summary, "bootstrap_pct_63d", higher_is_better=True)
    norm_flow = normalize_series(manager_summary, "new_position_excess_63d", higher_is_better=True)
    norm_cov = normalize_series(manager_summary, "coverage_score", higher_is_better=True)

    for cik in manager_summary:
        ms = manager_summary[cik]
        score = (
            QS_WEIGHT_63D_EXCESS * norm_63d.get(cik, 0)
            + QS_WEIGHT_HIT63 * norm_hit63.get(cik, 0)
            + QS_WEIGHT_BOOTSTRAP * norm_boot.get(cik, 0)
            + QS_WEIGHT_FLOW * norm_flow.get(cik, 0)
            + QS_WEIGHT_COVERAGE * norm_cov.get(cik, 0)
        )
        # Apply penalties
        if ms["n_filing_windows"] < 3:
            score -= QS_PENALTY_FEW_WINDOWS
        # Check concentration (use last window if available)
        last_win = manager_windows[cik][-1] if manager_windows[cik] else None
        if last_win:
            conc = last_win["sleeves"].get(63, {}).get("_concentration_top1_pct", 0)
            if conc and conc > 0.5:
                score -= QS_PENALTY_CONCENTRATION

        ms["quality_score"] = round(score, 4)
        ms["classification"] = classify_manager(ms)

    # Leave-one-out analysis
    print("Running leave-one-out analysis...")
    loo = compute_leave_one_out(manager_summary, None)

    # Add loo to manager_summary
    for cik in manager_summary:
        loo_data = loo.get(cik, {})
        manager_summary[cik]["loo_material_change"] = loo_data.get("material_change", False)
        manager_summary[cik]["loo_exclusive_top30"] = loo_data.get("exclusive_top30_tickers", [])
        manager_summary[cik]["loo_top30_overlap"] = loo_data.get("top30_overlap_count", 0)

    # Build window rows for CSV
    for cik, windows in manager_windows.items():
        name = manager_summary.get(cik, {}).get("name", cik)
        for w in windows:
            for window_days in FORWARD_WINDOWS:
                ew = w["sleeves"].get(window_days, {}).get("equal_weighted_all", {})
                vw = w["sleeves"].get(window_days, {}).get("value_weighted_all", {})
                top10 = w["sleeves"].get(window_days, {}).get("top10_value_weighted", {})
                new_pos = w["sleeves"].get(window_days, {}).get("new_positions", {})

                all_window_rows.append(
                    {
                        "cik": cik,
                        "manager": name,
                        "quarter_end": w["quarter_end"],
                        "signal_date": w["signal_date"],
                        "lag_method": w["lag_method"],
                        "n_holdings": w["n_holdings"],
                        "window_days": window_days,
                        "ew_return": round(ew.get("return", 0) or 0, 6),
                        "ew_excess": round(ew.get("excess", 0) or 0, 6) if ew.get("excess") is not None else None,
                        "vw_return": round(vw.get("return", 0) or 0, 6) if vw.get("return") is not None else None,
                        "vw_excess": round(vw.get("excess", 0) or 0, 6) if vw.get("excess") is not None else None,
                        "top10_return": (
                            round(top10.get("return", 0) or 0, 6) if top10.get("return") is not None else None
                        ),
                        "top10_excess": (
                            round(top10.get("excess", 0) or 0, 6) if top10.get("excess") is not None else None
                        ),
                        "new_pos_return": (
                            round(new_pos.get("return", 0) or 0, 6) if new_pos.get("return") is not None else None
                        ),
                        "new_pos_excess": (
                            round(new_pos.get("excess", 0) or 0, 6) if new_pos.get("excess") is not None else None
                        ),
                        "xbi_return": round(w["sleeves"].get(window_days, {}).get("_xbi_return", 0) or 0, 6),
                        "bootstrap_63d_pct": (
                            round(w["bootstrap"].get("63d_pct", 0) or 0, 4) if window_days == 63 else None
                        ),
                    }
                )

    # Compute sleeve aggregate stats
    sleeve_stats = defaultdict(lambda: {"returns": [], "excesses": [], "hits": []})
    for row in all_window_rows:
        if row["window_days"] == 63:
            for sleeve in ["ew", "vw", "top10", "new_pos"]:
                exc = row.get(f"{sleeve}_excess")
                ret = row.get(f"{sleeve}_return")
                if exc is not None and ret is not None:
                    sleeve_stats[sleeve]["returns"].append(ret)
                    sleeve_stats[sleeve]["excesses"].append(exc)
                    sleeve_stats[sleeve]["hits"].append(1 if exc > 0 else 0)

    sleeve_summary = {}
    for sleeve, data in sleeve_stats.items():
        n = len(data["excesses"])
        if n == 0:
            continue
        sleeve_summary[sleeve] = {
            "mean_63d_excess": round(sum(data["excesses"]) / n, 6),
            "hit_rate": round(sum(data["hits"]) / n, 3),
            "mean_63d_return": round(sum(data["returns"]) / n, 6),
            "n_obs": n,
        }

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    print("Writing outputs...")

    # 1. manager_summary.csv
    summary_path = os.path.join(OUT_DIR, "manager_summary.csv")
    fieldnames = [
        "cik",
        "name",
        "aum_b",
        "style",
        "category",
        "n_filing_windows",
        "avg_holdings",
        "coverage_score",
        "mean_20d_excess",
        "mean_63d_excess",
        "mean_126d_excess",
        "hit_rate_20d",
        "hit_rate_63d",
        "new_position_excess_63d",
        "bootstrap_pct_63d",
        "quality_score",
        "classification",
        "loo_material_change",
        "loo_top30_overlap",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for cik, ms in sorted(manager_summary.items(), key=lambda x: -(x[1].get("quality_score") or -99)):
            w.writerow({**ms, "cik": cik})
    print(f"  Written: {summary_path}")

    # 2. manager_window_returns.csv
    window_path = os.path.join(OUT_DIR, "manager_window_returns.csv")
    if all_window_rows:
        with open(window_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_window_rows[0].keys()))
            w.writeheader()
            w.writerows(all_window_rows)
    print(f"  Written: {window_path}")

    # 3. manager_holdings_attribution.csv
    attr_path = os.path.join(OUT_DIR, "manager_holdings_attribution.csv")
    if all_holdings_attr_rows:
        with open(attr_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_holdings_attr_rows[0].keys()))
            w.writeheader()
            w.writerows(all_holdings_attr_rows)
    print(f"  Written: {attr_path}")

    # 4. manager_reweighting_shadow.csv
    reweight_path = os.path.join(OUT_DIR, "manager_reweighting_shadow.csv")
    rw_fields = [
        "cik",
        "name",
        "classification",
        "quality_score",
        "mean_63d_excess",
        "hit_rate_63d",
        "bootstrap_pct_63d",
        "n_filing_windows",
        "loo_material_change",
        "loo_top30_overlap",
    ]
    with open(reweight_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rw_fields, extrasaction="ignore")
        w.writeheader()
        for cik, ms in sorted(manager_summary.items(), key=lambda x: -(x[1].get("quality_score") or -99)):
            w.writerow({**ms, "cik": cik})
    print(f"  Written: {reweight_path}")

    # 5. manager_13f_performance.json
    json_path = os.path.join(OUT_DIR, "manager_13f_performance.json")
    output_json = {
        "metadata": {
            "run_date": datetime.date.today().isoformat(),
            "classification": "RESEARCH_DIAGNOSTIC / MANAGER_ATTRIBUTION / NO_MODEL_CHANGE",
            "quarters_analyzed": QUARTERS,
            "signal_date_methodology": "filing_acceptance_date (filed_at from filings_metadata); fallback: quarter_end + 45 days",
            "forward_windows": FORWARD_WINDOWS,
            "bootstrap_n": BOOTSTRAP_N,
            "production_change": False,
        },
        "manager_summary": {
            cik: {k: v for k, v in ms.items() if k != "all_tickers_held"} for cik, ms in manager_summary.items()
        },
        "sleeve_aggregate_stats": sleeve_summary,
        "leave_one_out": {cik: loo.get(cik, {}) for cik in manager_summary},
    }
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"  Written: {json_path}")

    # 6. MANAGER_13F_PERFORMANCE.md
    write_markdown_report(manager_summary, sleeve_summary, loo, QUARTERS, OUT_DIR)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("MANAGER 13F ATTRIBUTION SUMMARY")
    print("=" * 70)

    classifications = defaultdict(list)
    for cik, ms in manager_summary.items():
        classifications[ms["classification"]].append(ms["name"])

    print(f"\nManagers analyzed: {len(manager_summary)}")
    print(f"Filing windows: {len(QUARTERS)} quarters")
    print(f"Price tickers: {len(all_price_tickers)}")

    print("\nTop 5 by quality score:")
    ranked = sorted(manager_summary.items(), key=lambda x: -(x[1].get("quality_score") or -99))
    for cik, ms in ranked[:5]:
        print(
            f"  {ms['name']}: score={ms['quality_score']}, "
            f"63d_excess={ms.get('mean_63d_excess')}, "
            f"hit63={ms.get('hit_rate_63d')}, "
            f"boot%={ms.get('bootstrap_pct_63d')}, "
            f"class={ms['classification']}"
        )

    print("\nBottom 5 by quality score:")
    for cik, ms in ranked[-5:]:
        print(
            f"  {ms['name']}: score={ms['quality_score']}, "
            f"63d_excess={ms.get('mean_63d_excess')}, class={ms['classification']}"
        )

    print("\nClassifications:")
    for cls, names in sorted(classifications.items()):
        print(f"  {cls}: {len(names)}")
        for n in sorted(names):
            print(f"    - {n}")

    return manager_summary, sleeve_summary, loo


def write_markdown_report(manager_summary, sleeve_summary, loo, quarters, out_dir):
    """Write MANAGER_13F_PERFORMANCE.md"""
    ranked = sorted(manager_summary.items(), key=lambda x: -(x[1].get("quality_score") or -99))
    classifications = defaultdict(list)
    for cik, ms in manager_summary.items():
        classifications[ms["classification"]].append((cik, ms))

    upweight = classifications.get("UPWEIGHT_CANDIDATE_SHADOW", [])
    downweight = classifications.get("DOWNWEIGHT_CANDIDATE_SHADOW", [])
    keep = classifications.get("KEEP_CURRENT_WEIGHT", [])
    insuf = classifications.get("INSUFFICIENT_DATA", [])

    # Overall stats
    n_mgrs = len(manager_summary)
    valid_managers = [ms for ms in manager_summary.values() if ms.get("mean_63d_excess") is not None]
    mean_all_63d = sum(ms["mean_63d_excess"] for ms in valid_managers) / len(valid_managers) if valid_managers else None

    # Executive verdict
    verdict_lines = []
    if upweight:
        names = [ms["name"] for _, ms in upweight[:3]]
        verdict_lines.append(f"Shadow upweight candidates: {', '.join(names)}.")
    if downweight:
        names = [ms["name"] for _, ms in downweight[:3]]
        verdict_lines.append(f"Shadow downweight candidates: {', '.join(names)}.")
    if mean_all_63d is not None:
        verdict_lines.append(
            f"Mean 63-day equal-weight excess vs XBI across all managers and windows: {mean_all_63d*100:+.2f}pp."
        )
    verdict_lines.append(
        "Evidence quality is limited by only 5 quarterly filing windows "
        "and the inherent noise of 13F public disclosures."
    )

    lines = [
        "# 13F Manager Performance Attribution — Past Year",
        "",
        "## Classification",
        "RESEARCH_DIAGNOSTIC / MANAGER_ATTRIBUTION / NO_MODEL_CHANGE",
        "",
        "## Executive Verdict",
        " ".join(verdict_lines),
        "",
        "## Data Sources and Coverage",
        f"- Manager registry: production_data/manager_registry.json, {n_mgrs} managers analyzed",
        f"- Filing windows: {len(quarters)} quarters ({quarters[0]} to {quarters[-1]})",
        "- Holdings data: data/13f_history_full/ (5 quarterly files)",
        "- Holdings mapped: tickers present in data/13f_history_full/ quarterly files",
        "- Price coverage: production_data/price_history_split_adj.csv (351 tickers, 2020-01-02 to 2026-06-18)",
        "- Signal date methodology: **filing_acceptance_date** — filed_at field from filings_metadata "
        "in each quarterly holdings file (reflects actual EDGAR acceptance/receipt date). "
        "Fallback for missing dates: quarter_end + 45 calendar days (conservative).",
        "",
        "## Methodology",
        "- **Sleeve construction**: For each manager × quarter filing, builds 5 return sleeves "
        "(value-weighted all biotech, equal-weighted all biotech, top-10 by value, new positions, "
        "increased positions) using holdings mapped to split-adjusted prices.",
        "- **Signal date**: Earliest EDGAR acceptance date per manager per quarter (max across tickers "
        "to ensure all holdings observable). Never uses quarter-end date.",
        "- **Forward returns**: 20d, 63d, 126d from signal date using split-adjusted close prices.",
        "- **Benchmark**: XBI ETF from same price file at same signal date.",
        "- **Bootstrap**: 200 random equal-weight same-size baskets drawn from eligible price universe; "
        "each manager's 63d EW return ranked against its bootstrap distribution.",
        "- **Quality score**: Weighted composite of normalized 63d excess (35%), hit rate 63d (20%), "
        "bootstrap percentile (20%), new-position excess (15%), coverage (10%); "
        "penalties for <3 windows (-0.20) and top-holding concentration >50% (-0.10).",
        "",
        "## Manager-Level Results",
        "",
        "| Manager | N Win | Avg Hold | 63d Excess | Hit% 63d | Boot% | Qual Score | Classification |",
        "|---------|-------|----------|------------|----------|-------|------------|----------------|",
    ]

    for cik, ms in ranked:
        exc = ms.get("mean_63d_excess")
        exc_str = f"{exc*100:+.2f}pp" if exc is not None else "N/A"
        hit = ms.get("hit_rate_63d")
        hit_str = f"{hit*100:.0f}%" if hit is not None else "N/A"
        boot = ms.get("bootstrap_pct_63d")
        boot_str = f"{boot*100:.0f}%" if boot is not None else "N/A"
        lines.append(
            f"| {ms['name']} | {ms['n_filing_windows']} | {ms['avg_holdings']:.0f} | "
            f"{exc_str} | {hit_str} | {boot_str} | {ms['quality_score']:.3f} | {ms['classification']} |"
        )

    lines += [
        "",
        "## Sleeve Results (63-day horizon, all managers pooled)",
        "",
        "| Sleeve | Mean 63d Excess | Hit Rate | N Obs |",
        "|--------|----------------|----------|-------|",
    ]
    for sleeve, stats in sorted(sleeve_summary.items()):
        lines.append(
            f"| {sleeve} | {stats['mean_63d_excess']*100:+.2f}pp | "
            f"{stats['hit_rate']*100:.0f}% | {stats['n_obs']} |"
        )

    lines += [
        "",
        "## Flow / New Position Results",
        "",
    ]
    new_pos_data = sleeve_summary.get("new_pos", {})
    ew_data = sleeve_summary.get("ew", {})
    if new_pos_data and ew_data:
        diff = new_pos_data["mean_63d_excess"] - ew_data["mean_63d_excess"]
        lines.append(
            f"New-positions sleeve mean 63d excess: {new_pos_data['mean_63d_excess']*100:+.2f}pp "
            f"vs equal-weighted all: {ew_data['mean_63d_excess']*100:+.2f}pp. "
            f"Flow premium: {diff*100:+.2f}pp ({new_pos_data['n_obs']} observations)."
        )
        if diff > 0.01:
            lines.append("New positions appear to add predictive value above the equal-weighted baseline.")
        elif diff < -0.01:
            lines.append("New positions do NOT appear to add predictive value; equal-weighted baseline outperforms.")
        else:
            lines.append("New position premium is near zero — indistinguishable from noise at this sample size.")
    else:
        lines.append("Insufficient new-position data for comparison.")

    lines += [
        "",
        "## Bootstrap / Random Basket Controls",
        "",
        "Manager 63-day equal-weight returns vs 200 random same-size biotech baskets:",
        "",
        "| Manager | Bootstrap %ile | N Windows | Interpretation |",
        "|---------|---------------|-----------|----------------|",
    ]
    for cik, ms in ranked:
        boot = ms.get("bootstrap_pct_63d")
        n = ms["n_filing_windows"]
        if boot is not None:
            if boot >= 0.65:
                interp = "Above random"
            elif boot <= 0.35:
                interp = "Below random"
            else:
                interp = "In-line with random"
            lines.append(f"| {ms['name']} | {boot*100:.0f}% | {n} | {interp} |")
        else:
            lines.append(f"| {ms['name']} | N/A | {n} | Insufficient data |")

    lines += [
        "",
        "## Leave-One-Manager-Out Analysis",
        "",
        "Approximate analysis based on current institutional_summary.json coinvest scores.",
        "Identifies managers holding top-30 coinvest tickers exclusively (removal would zero the signal).",
        "",
        "| Manager | Material Change | Exclusive Top-30 Tickers | Top-30 Overlap Count |",
        "|---------|----------------|--------------------------|----------------------|",
    ]
    for cik, ms in ranked:
        loo_chg = ms.get("loo_material_change", False)
        loo_ex = ms.get("loo_exclusive_top30", [])
        loo_ov = ms.get("loo_top30_overlap", 0)
        lines.append(
            f"| {ms['name']} | {'YES' if loo_chg else 'No'} | "
            f"{', '.join(loo_ex) if loo_ex else 'None'} | {loo_ov} |"
        )

    lines += [
        "",
        "## Shadow Reweighting Candidates",
        "",
        f"**UPWEIGHT_CANDIDATE_SHADOW** ({len(upweight)}): "
        + (", ".join(ms["name"] for _, ms in upweight) if upweight else "None"),
        "",
        f"**DOWNWEIGHT_CANDIDATE_SHADOW** ({len(downweight)}): "
        + (", ".join(ms["name"] for _, ms in downweight) if downweight else "None"),
        "",
        f"**KEEP_CURRENT_WEIGHT** ({len(keep)}): " + (", ".join(ms["name"] for _, ms in keep) if keep else "None"),
        "",
        f"**INSUFFICIENT_DATA** ({len(insuf)}): " + (", ".join(ms["name"] for _, ms in insuf) if insuf else "None"),
        "",
        "## Important Caveats",
        "- 13F filings are delayed by up to 45+ days after quarter-end; signal dates used here "
        "reflect actual EDGAR acceptance dates from the filings_metadata fields.",
        "- 13F excludes shorts, derivatives, private positions, and intraperiod trading — "
        "actual manager positioning may differ substantially from the reported long equity book.",
        "- Holdings may be stale by signal date; market-value weights used in value-weighted sleeves "
        "reflect end-of-quarter prices, not prices on the signal date.",
        "- A manager can be alpha-generating in their actual trading but NOT predictive after "
        "public 13F disclosure, because smart money has already repositioned.",
        "- With only 5 quarterly windows, individual manager results have very high estimation noise "
        "(standard error of mean ~1/sqrt(5) of the standard deviation).",
        "- The biotech universe eligible for price lookups is limited to tickers in "
        "price_history_split_adj.csv (351 tickers).",
        "- Any suggested reweighting requires forward shadow validation before production promotion. "
        "Minimum standard: 8 completed 63-day windows post-classification.",
        "",
        "## Governance Conclusion",
        "- production_change: **False**",
        "- NO_MODEL_CHANGE / NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE",
        "",
        "## Recommended Next Validation",
        "Before any production reweighting could be considered:",
        "1. Accumulate at least 8 additional quarterly windows (minimum ~2 years of data).",
        "2. Run shadow monitor: apply proposed weights, track ranked-list changes vs actual outcomes.",
        "3. Validate that upweight candidates show consistent alpha in out-of-sample periods, "
        "not just the trailing 5 quarters used here.",
        "4. Check that downweight candidates' negative performance is not driven by a single bad quarter.",
        "5. Confirm that any top-30 composition changes from reweighting are validated against "
        "forward returns before promotion to production.",
        "",
        f"*Report generated: {datetime.date.today().isoformat()}*",
        "*Classification: RESEARCH_DIAGNOSTIC / NO_MODEL_CHANGE*",
    ]

    md_path = os.path.join(out_dir, "MANAGER_13F_PERFORMANCE.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {md_path}")


if __name__ == "__main__":
    main()
