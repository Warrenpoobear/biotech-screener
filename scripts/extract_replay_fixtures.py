#!/usr/bin/env python3
"""
Extract compact replay fixtures from real archives for regression testing.

One-time utility: loads archives via load_archive_data(), trims each rec dict
to the ~12 fields compute_decision_fields() actually reads, and writes compact
JSON per date to tests/fixtures/phase2_replays/.

Also prints calibration metrics (tier distribution, coverage, dispersion) and
top-10 A+B tickers per date to help select golden tickers and tune thresholds.

Usage:
    python scripts/extract_replay_fixtures.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, compute_decision_fields
from run_decision_ruleset_sweep import ARCHIVE_DIR, load_archive_data

DATES = ["2025-04-30", "2025-07-31", "2025-10-31"]
RULESET_PATH = PROJECT_ROOT / "production_data" / "decision_rulesets" / "v2_phase2_default.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "phase2_replays"

# Fields compute_decision_fields actually reads from rec
KEEP_FIELDS = {
    "severity",
    "confidence_overall",
    "fundamental_red_flag",
    "fundamental_red_flag_reasons",
    "catalyst_decay",
    "smart_money_signal",
    "coinvest",
    "defensive_features",
    "score_breakdown",
    "momentum_signal",
    "attn_flags",
    "flags",
}


def trim_rec(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields the decision engine reads; anonymise holder lists."""
    trimmed: Dict[str, Any] = {}
    for key in KEEP_FIELDS:
        if key in rec:
            trimmed[key] = rec[key]

    # Holder lists: engine only calls len(), replace with placeholder
    sms = trimmed.get("smart_money_signal")
    if isinstance(sms, dict):
        for list_key in ("holders_increasing", "holders_decreasing", "tier1_holders_list"):
            if list_key in sms and isinstance(sms[list_key], list):
                sms[list_key] = ["_"] * len(sms[list_key])

    return trimmed


def extract_fixture(date_str: str) -> Dict[str, Any]:
    """Load archive, trim recs, return fixture dict."""
    tar_path = ARCHIVE_DIR / f"{date_str}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    ad = load_archive_data(tar_path, date_str)

    trimmed_recs = {t: trim_rec(ad.recs[t]) for t in ad.tickers if t in ad.recs}

    # Serialize optionalities with None → null
    opts: Dict[str, Optional[float]] = {}
    for t in ad.tickers:
        opts[t] = ad.optionalities.get(t)

    return {
        "date": date_str,
        "tickers": ad.tickers,
        "recs": trimmed_recs,
        "archetypes": {t: ad.archetypes.get(t, "") for t in ad.tickers},
        "optionalities": opts,
    }


def print_calibration(fixture: Dict[str, Any], ruleset: DecisionRuleset) -> None:
    """Run engine on all tickers, print calibration metrics."""
    date_str = fixture["date"]
    tickers = fixture["tickers"]
    recs = fixture["recs"]
    archetypes = fixture["archetypes"]
    opts = fixture["optionalities"]

    results: List[Dict[str, Any]] = []
    for t in tickers:
        if t not in recs:
            continue
        fields = compute_decision_fields(
            rec=recs[t],
            archetype=archetypes.get(t, ""),
            optionality_pct_dev=opts.get(t),
            ruleset=ruleset,
        )
        fields["ticker"] = t
        results.append(fields)

    print(f"\n{'='*60}")
    print(f"  {date_str}  ({len(results)} tickers)")
    print(f"{'='*60}")

    # Tier distribution (dev-stage only)
    dev_results = [r for r in results if r.get("tier_dev") != ""]
    tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in dev_results:
        t = r.get("tier_dev", "")
        if t in tier_counts:
            tier_counts[t] += 1
    n_dev = len(dev_results) or 1
    print(f"\n  Tier distribution (N_dev={len(dev_results)}):")
    for tier, count in tier_counts.items():
        pct = count / n_dev * 100
        print(f"    {tier}: {count:3d} ({pct:5.1f}%)")

    # Eligibility
    eligible = [r for r in results if r.get("eligible") == "1"]
    dev_eligible = [r for r in dev_results if r.get("eligible") == "1"]
    print(f"\n  Eligibility: {len(eligible)}/{len(results)} "
          f"({len(eligible)/len(results)*100:.1f}%)")
    print(f"  Dev eligible: {len(dev_eligible)}/{len(dev_results)} "
          f"({len(dev_eligible)/n_dev*100:.1f}%)")

    # Coverage: catalyst, sponsor, momentum
    def coverage(field: str, empty_vals: set) -> float:
        filled = sum(1 for r in dev_results if r.get(field, "") not in empty_vals)
        return filled / n_dev * 100 if n_dev else 0

    cat_cov = coverage("catalyst_mode", {"missing", ""})
    spon_cov = coverage("sponsor_tier1_count", {""})
    mom_cov = coverage("mom_state", {""})
    print(f"\n  Coverage (dev-stage):")
    print(f"    Catalyst:    {cat_cov:5.1f}%")
    print(f"    Sponsor:     {spon_cov:5.1f}%")
    print(f"    Momentum:    {mom_cov:5.1f}%")

    # Optionality dispersion
    opt_vals = [opts.get(t) for t in tickers
                if opts.get(t) is not None and archetypes.get(t) == "drug_developer"]
    if len(opt_vals) >= 2:
        opt_std = statistics.stdev(opt_vals)
        opt_mean = statistics.mean(opt_vals)
        print(f"\n  Optionality (dev): mean={opt_mean:.3f}, std={opt_std:.3f}, N={len(opt_vals)}")
    else:
        print(f"\n  Optionality (dev): insufficient data ({len(opt_vals)} values)")

    # Size band distribution
    band_counts = {"L": 0, "M": 0, "S": 0, "XS": 0}
    for r in results:
        b = r.get("size_band", "")
        if b in band_counts:
            band_counts[b] += 1
    print(f"\n  Size bands:")
    for band, count in band_counts.items():
        print(f"    {band:2s}: {count:3d}")

    # Top 10 A+B tickers (sorted by tier then band)
    tier_order = {"A": 0, "B": 1}
    band_order = {"L": 0, "M": 1, "S": 2, "XS": 3}
    ab_tickers = [r for r in dev_results if r.get("tier_dev") in ("A", "B")]
    ab_tickers.sort(key=lambda r: (
        tier_order.get(r["tier_dev"], 9),
        band_order.get(r["size_band"], 9),
        r["ticker"],
    ))
    print(f"\n  Top A+B tickers (first 15):")
    for r in ab_tickers[:15]:
        print(f"    {r['ticker']:6s}  tier={r['tier_dev']}  band={r['size_band']}  "
              f"reason={r.get('tier_reason','')}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruleset = DecisionRuleset.from_json(str(RULESET_PATH))
    print(f"Ruleset: {ruleset.ruleset_id}")
    print(f"a_floor={ruleset.tier_a_optionality_floor}, "
          f"b_floor={ruleset.tier_b_optionality_floor}")

    for date_str in DATES:
        print(f"\nExtracting {date_str}...")
        fixture = extract_fixture(date_str)
        print(f"  {len(fixture['tickers'])} tickers, "
              f"{len(fixture['recs'])} recs")

        # Write fixture
        out_path = OUTPUT_DIR / f"{date_str}.json"
        with open(out_path, "w") as f:
            json.dump(fixture, f, indent=None, separators=(",", ":"),
                      default=str)
        size_kb = out_path.stat().st_size / 1024
        print(f"  Written: {out_path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")

        # Print calibration metrics
        print_calibration(fixture, ruleset)


if __name__ == "__main__":
    main()
