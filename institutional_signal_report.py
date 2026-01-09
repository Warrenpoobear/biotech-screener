"""
Institutional Signal Analysis & Reporting
Generates actionable investment signals from 13F holdings data
"""
import json
from collections import Counter
from datetime import datetime

def load_data(filepath="production_data/holdings_snapshots.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

MANAGER_NAMES = {
    "0001263508": "Baker Bros Advisors",
    "0001346824": "RA Capital Management", 
    "0001224962": "Perceptive Advisors",
    "0001009258": "Deerfield Management",
    "0001425738": "Redmile Group",
    "0001055951": "OrbiMed Advisors",
    "0001493215": "RTW Investments",
    "0001232621": "Tang Capital Partners",
    "0000909661": "Farallon Capital",
    "0001776382": "Venbio Partners",
    "0001822947": "Ally Bridge Group",
    "0001703031": "Bain Capital Life Sciences"
}

ELITE_MANAGERS = {
    "0001263508", "0001346824", "0001224962", "0001055951",
    "0001232621", "0001493215", "0000909661", "0001425738", "0001009258"
}

def analyze_coverage(data):
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

def calculate_signal_score(ticker, info):
    curr = info["holdings"]["current"]
    prior = info["holdings"]["prior"]
    mgr_count = len(curr)
    if mgr_count < 4:
        return None
    total_curr = sum(pos["value_kusd"] for pos in curr.values())
    total_prior = sum(pos["value_kusd"] for pos in prior.values() if pos["value_kusd"] > 0)
    pct_change = ((total_curr - total_prior) / total_prior * 100) if total_prior > 0 else 0
    up_list = []
    new_list = []
    down_list = []
    for mgr_cik in curr.keys():
        curr_val = curr[mgr_cik]["value_kusd"]
        prior_val = prior.get(mgr_cik, {}).get("value_kusd", 0)
        if prior_val == 0:
            new_list.append(mgr_cik)
        elif curr_val > prior_val * 1.2:
            up_list.append(mgr_cik)
        elif curr_val < prior_val * 0.8:
            down_list.append(mgr_cik)
    net_buyers = len(up_list) + len(new_list) - len(down_list)
    elite_new_boost = sum(1 for cik in new_list if cik in ELITE_MANAGERS)
    consensus_boost = 3 if mgr_count >= 6 else (2 if mgr_count >= 5 else 0)
    coord_boost = 1 if (len(up_list) + len(new_list)) >= mgr_count * 0.5 else 0
    signal_score = net_buyers + elite_new_boost + consensus_boost + coord_boost
    return {
        "ticker": ticker, "mgrs": mgr_count, "pct_change": pct_change,
        "net_buyers": net_buyers, "signal_score": signal_score,
        "new": [MANAGER_NAMES.get(cik, cik) for cik in new_list],
        "increased": [MANAGER_NAMES.get(cik, cik) for cik in up_list],
        "decreased": [MANAGER_NAMES.get(cik, cik) for cik in down_list],
    }

def generate_report(data, output_file=None):
    coverage = analyze_coverage(data)
    signals = []
    for ticker, info in data.items():
        score = calculate_signal_score(ticker, info)
        if score:
            signals.append(score)
    signals.sort(key=lambda x: (-x["signal_score"], -x["mgrs"], -x["pct_change"]))
    report = []
    report.append("=" * 110)
    report.append("WAKE ROBIN INSTITUTIONAL SIGNAL REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 110)
    report.append("")
    report.append("COVERAGE SUMMARY:")
    report.append(f"  Total Tickers: {coverage['total_tickers']}")
    report.append(f"  Avg Managers/Ticker: {coverage['avg_managers']}")
    report.append("")
    for key, val in coverage["distribution"].items():
        report.append(f"  {key} managers: {val['count']:3} tickers ({val['pct']:5.1f}%)")
    report.append("")
    report.append("TOP 25 INSTITUTIONAL SIGNALS:")
    report.append("-" * 110)
    report.append(f"{'Ticker':<7} {'Score':<6} {'Mgrs':<5} {'Q/Q':<7} {'Net':<4} {'Activity Summary'}")
    report.append("-" * 110)
    for r in signals[:25]:
        activity = []
        if r["new"]:
            new_str = ", ".join(r["new"][:3])
            if len(r["new"]) > 3: new_str += f" +{len(r['new'])-3}"
            activity.append(f"NEW({len(r['new'])}): {new_str}")
        if r["increased"]:
            up_str = ", ".join(r["increased"][:3])
            if len(r["increased"]) > 3: up_str += f" +{len(r['increased'])-3}"
            activity.append(f"↑({len(r['increased'])}): {up_str}")
        if r["decreased"]:
            activity.append(f"↓({len(r['decreased'])})")
        act_str = " | ".join(activity) if activity else "Stable"
        report.append(f"{r['ticker']:<7} {r['signal_score']:<6} {r['mgrs']:<5} {r['pct_change']:>+6.0f}% {r['net_buyers']:>3}  {act_str}")
    report.append("")
    report.append("=" * 110)
    report.append("SCORING METHODOLOGY:")
    report.append("  Signal Score = Net Buyers + Elite New Bonus + Consensus Bonus + Coordination Bonus")
    report.append("  Net Buyers = (Increased + New) - Decreased")
    report.append("  Elite managers: Baker, RA, Perceptive, OrbiMed, Tang, RTW, Farallon, Redmile, Deerfield")
    report.append("=" * 110)
    output = "\n".join(report)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✅ Report saved to: {output_file}")
    print(output)
    return signals

if __name__ == "__main__":
    data = load_data()
    signals = generate_report(data, "INSTITUTIONAL_SIGNAL_REPORT.txt")

