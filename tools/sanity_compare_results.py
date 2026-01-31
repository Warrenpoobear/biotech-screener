#!/usr/bin/env python3
"""Sanity check script to compare old vs new screening results."""
import json
import sys
from collections import Counter
from statistics import mean, median

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_rows(obj):
    """Extract ranked securities from our results format."""
    # Our format: module_5_composite.ranked_securities
    if isinstance(obj, dict):
        if "module_5_composite" in obj:
            return obj["module_5_composite"].get("ranked_securities", [])
        if "ranked_securities" in obj:
            return obj["ranked_securities"]
        if "results" in obj:
            return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unrecognized results JSON shape")

def tie_stats(scores, top_n=120):
    """Count exact ties among top N scores."""
    top = scores[:top_n]
    c = Counter(round(s, 2) for s in top)  # Round to 2dp for tie detection
    ties = sorted((v, s) for s, v in c.items() if v > 1)
    max_tie = max(c.values()) if c else 0
    return max_tie, len(ties), ties[-5:]

def rank_map(rows):
    """Build ticker -> (rank, score, row) mapping."""
    out = {}
    for r in rows:
        t = r.get("ticker") or r.get("symbol")
        rk = r.get("composite_rank") or r.get("rank")
        sc = r.get("composite_score") or r.get("score")
        if t is None or rk is None or sc is None:
            continue
        out[t] = (int(rk), float(sc), r)
    return out

def pct(x, n):
    return 0.0 if n == 0 else 100.0 * x / n

def main(old_path, new_path):
    old = rank_map(get_rows(load(old_path)))
    new = rank_map(get_rows(load(new_path)))

    common = sorted(set(old) & set(new))
    n = len(common)

    # Rank deltas (+ means improved/moved up)
    deltas = [old[t][0] - new[t][0] for t in common]
    absd = [abs(d) for d in deltas]

    moved_20 = sum(1 for d in absd if d >= 20)
    moved_50 = sum(1 for d in absd if d >= 50)

    # Score ties among top bucket
    old_top = sorted((old[t][1] for t in common), reverse=True)
    new_top = sorted((new[t][1] for t in common), reverse=True)
    old_max_tie, old_tie_groups, old_tail = tie_stats(old_top)
    new_max_tie, new_tie_groups, new_tail = tie_stats(new_top)

    # Momentum ceiling check
    def extract_mom(rows_map):
        vals = []
        for t in common:
            r = rows_map[t][2]
            mom = r.get("momentum_signal", {})
            v = mom.get("momentum_score")
            if v is not None:
                vals.append(float(v))
        vals.sort(reverse=True)
        return vals

    old_mom = extract_mom(old)
    new_mom = extract_mom(new)

    def ceiling_hits(vals, target, eps=0.1):
        return sum(1 for v in vals if abs(v - target) < eps)

    # Smart money comparison
    def extract_sm(rows_map):
        vals = []
        for t in common:
            r = rows_map[t][2]
            sm = r.get("smart_money_signal", {})
            v = sm.get("score") or sm.get("smart_money_score")
            if v is not None:
                vals.append(float(v))
        vals.sort(reverse=True)
        return vals

    old_sm = extract_sm(old)
    new_sm = extract_sm(new)

    print("=== SANITY COMPARE ===")
    print(f"Common tickers: {n}")
    print()
    print("Rank churn:")
    print(f"  >=20 rank moves: {moved_20} ({pct(moved_20,n):.1f}%)")
    print(f"  >=50 rank moves: {moved_50} ({pct(moved_50,n):.1f}%)")
    print(f"  median |Δrank|: {median(absd):.1f}")
    print(f"  mean   |Δrank|: {mean(absd):.1f}")
    print()
    print("Top-score tie diagnostics (top 120 by score):")
    print(f"  OLD max tie size: {old_max_tie} (tie groups: {old_tie_groups})")
    print(f"  NEW max tie size: {new_max_tie} (tie groups: {new_tie_groups})")
    print()
    if old_mom and new_mom:
        print("Momentum tie diagnostics:")
        print(f"  OLD hits @92.75: {ceiling_hits(old_mom, 92.75)}")
        print(f"  NEW hits @92.75: {ceiling_hits(new_mom, 92.75)}")
        print(f"  NEW hits @96.55: {ceiling_hits(new_mom, 96.55)}")
        print(f"  OLD top5 momentum: {[round(x,2) for x in old_mom[:5]]}")
        print(f"  NEW top5 momentum: {[round(x,2) for x in new_mom[:5]]}")
    print()
    if old_sm and new_sm:
        print("Smart money diagnostics:")
        print(f"  OLD max: {max(old_sm):.1f}, mean: {mean(old_sm):.1f}")
        print(f"  NEW max: {max(new_sm):.1f}, mean: {mean(new_sm):.1f}")
        print(f"  OLD top5 SM: {[round(x,1) for x in old_sm[:5]]}")
        print(f"  NEW top5 SM: {[round(x,1) for x in new_sm[:5]]}")
    print()

    # Top movers
    movers = [(t, old[t][0], new[t][0], old[t][0] - new[t][0]) for t in common]
    movers.sort(key=lambda x: -abs(x[3]))

    print("Top 10 rank movers:")
    print(f"  {'Ticker':<8} {'Old':>5} {'New':>5} {'Delta':>6}")
    print("  " + "-"*30)
    for t, old_r, new_r, delta in movers[:10]:
        direction = "↑" if delta > 0 else "↓"
        print(f"  {t:<8} {old_r:>5} {new_r:>5} {delta:>+6} {direction}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/sanity_compare_results.py OLD.json NEW.json")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
