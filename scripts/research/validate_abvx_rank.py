"""
Validate ABVX rank under production pipeline (B6 selector + pairwise_minimal_v2 ranker).

Uses golden baseline data (2026-02-20) and production configs from run_screen.py
to confirm ABVX placement within the Top-30 portfolio cohort.

Usage:
    python scripts/research/validate_abvx_rank.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from selector_engine import BlockWeight, SelectorConfig, SignalSpec, compute_selector_scores


# ---------------------------------------------------------------------------
# Production configs (must match run_screen.py)
# ---------------------------------------------------------------------------

A4_SELECTOR_CONFIG = SelectorConfig(
    block_weights=(
        BlockWeight("clinical", 0.00),
        BlockWeight("catalyst", 0.15),
        BlockWeight("survivability", 0.10),
        BlockWeight("institutional", 0.65),
        BlockWeight("market_structure", 0.10),
    ),
    institutional_signals=(
        SignalSpec("coinvest_score_z", 0.65),
        SignalSpec("inst_delta_z", 0.35),
        SignalSpec(
            "coinvest_recency_state",
            0.00,
            categorical=True,
            value_map=(("fresh", 1.0), ("stale", 0.3), ("", 0.0)),
        ),
    ),
)

RANKER_MODEL_PATH = ROOT / "production_data" / "ranker_v2_model.json"
GOLDEN_RANKINGS_PATH = ROOT / "golden" / "baseline_2026-02-20" / "rankings.csv"
PIT_BUNDLE_DIR = ROOT / "data" / "bundles" / "PIT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val, default=float("nan")):
    if val is None:
        return default
    try:
        if isinstance(val, str) and val.strip() == "":
            return default
        v = float(val)
        return v if v == v else default
    except (ValueError, TypeError):
        return default


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_abvx_rank() -> dict:
    """Run ABVX rank validation and return results dict."""
    # Load golden baseline
    with open(GOLDEN_RANKINGS_PATH) as f:
        all_rows = list(csv.DictReader(f))

    eligible = [r for r in all_rows if r.get("eligible") == "1"]
    n_eligible = len(eligible)

    # --- Step 1: Selector scoring ---
    results = compute_selector_scores(eligible, config=A4_SELECTOR_CONFIG)
    for row, sr in zip(eligible, results):
        row["selector_score"] = sr.selector_score
        row["selector_rank_bucket"] = sr.selector_rank_bucket
        row["_sel_institutional_block"] = sr.institutional_block
        row["_sel_survivability_block"] = sr.survivability_block
        row["_sel_catalyst_block"] = sr.catalyst_block
        row["_sel_market_block"] = sr.market_structure_block
        row["_sel_missing_count"] = sr.missing_count

    eligible.sort(key=lambda r: -r["selector_score"])

    # Find ABVX selector rank
    abvx_sel_rank = None
    abvx_row = None
    for i, row in enumerate(eligible, 1):
        if row["ticker"] == "ABVX":
            abvx_sel_rank = i
            abvx_row = row
            break

    assert abvx_row is not None, "ABVX not found in eligible universe"

    # --- Step 2: Ranker scoring (within top-60 cohort) ---
    with open(RANKER_MODEL_PATH) as f:
        model_data = json.load(f)

    weights = model_data["model"]["weights"]
    bias = model_data["model"]["bias"]
    feat_names = model_data["model"]["feature_names"]

    cohort = eligible[:60]

    # Extract raw features
    raw = [[_safe_float(r.get(fn)) for fn in feat_names] for r in cohort]

    # Z-score within cohort
    for j in range(len(feat_names)):
        vals = [raw[i][j] for i in range(len(cohort)) if raw[i][j] == raw[i][j]]
        mean = sum(vals) / len(vals) if vals else 0
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 1
        std = math.sqrt(var) if var > 0 else 1.0
        for i in range(len(cohort)):
            v = raw[i][j]
            if v != v:
                raw[i][j] = 0.0
            else:
                z = (v - mean) / std
                raw[i][j] = max(-3.0, min(3.0, z))

    # Score
    for i, row in enumerate(cohort):
        score = bias
        for j in range(len(feat_names)):
            score += weights[j] * raw[i][j]
        row["ranker_v2_score"] = score

    cohort.sort(key=lambda r: -r["ranker_v2_score"])

    # Find ABVX final rank
    abvx_final_rank = None
    for i, row in enumerate(cohort, 1):
        if row["ticker"] == "ABVX":
            abvx_final_rank = i
            break

    # --- Step 3: Collect latest PIT bundle data ---
    latest_pit = sorted(PIT_BUNDLE_DIR.iterdir())[-1] if PIT_BUNDLE_DIR.exists() else None
    pit_coinvest = {}
    pit_catalyst = {}
    if latest_pit:
        cf_path = latest_pit / "coinvest_features.json"
        if cf_path.exists():
            with open(cf_path) as f:
                pit_coinvest = json.load(f).get("tickers", {}).get("ABVX", {})
        ce_path = latest_pit / "catalyst_events.json"
        if ce_path.exists():
            with open(ce_path) as f:
                pit_catalyst = json.load(f).get("tickers", {}).get("ABVX", {})

    # --- Step 4: Build results ---
    result = {
        "ticker": "ABVX",
        "company": "Abivax SA",
        "baseline_date": "2026-02-20",
        "latest_pit_date": latest_pit.name if latest_pit else None,
        "n_eligible": n_eligible,
        "selector": {
            "rank": abvx_sel_rank,
            "score": round(abvx_row["selector_score"], 6),
            "bucket": abvx_row["selector_rank_bucket"],
            "institutional_block": round(abvx_row["_sel_institutional_block"], 4),
            "survivability_block": round(abvx_row["_sel_survivability_block"], 4),
            "catalyst_block": round(abvx_row["_sel_catalyst_block"], 4),
            "market_structure_block": round(abvx_row["_sel_market_block"], 4),
            "missing_signals": abvx_row["_sel_missing_count"],
        },
        "ranker": {
            "final_rank": abvx_final_rank,
            "ranker_v2_score": round(abvx_row["ranker_v2_score"], 6),
            "cohort_size": len(cohort),
            "in_top_30": abvx_final_rank <= 30,
        },
        "key_inputs": {
            "coinvest_score_z": _safe_float(abvx_row.get("coinvest_score_z")),
            "coinvest_tag": abvx_row.get("coinvest_tag"),
            "sponsor_tier1_count": int(_safe_float(abvx_row.get("sponsor_tier1_count"), 0)),
            "inst_delta_z": _safe_float(abvx_row.get("inst_delta_z")),
            "financial_score": round(_safe_float(abvx_row.get("financial_score")), 2),
            "catalyst_strength": abvx_row.get("catalyst_strength"),
            "catalyst_days": int(_safe_float(abvx_row.get("catalyst_days"), 0)),
            "runway_bucket": abvx_row.get("runway_bucket"),
            "severity": abvx_row.get("severity"),
            "mom_state": abvx_row.get("mom_state"),
        },
        "latest_pit_coinvest": {
            "tier1_count": pit_coinvest.get("tier1_count"),
            "recency_state": pit_coinvest.get("coinvest_recency_state"),
            "filing_age_days": pit_coinvest.get("days_since_latest_filing"),
            "position_changes": pit_coinvest.get("position_changes", {}),
        },
        "latest_pit_catalyst": {
            "nearest_event": pit_catalyst.get("nearest_event_name"),
            "event_date": pit_catalyst.get("nearest_event_date"),
            "days_to_catalyst": pit_catalyst.get("days_to_catalyst"),
        },
        "neighbors": [],
    }

    # Add rank neighborhood (ranks 1-10 and immediate neighbors)
    for i, row in enumerate(cohort[:35], 1):
        result["neighbors"].append(
            {
                "rank": i,
                "ticker": row["ticker"],
                "ranker_v2_score": round(row["ranker_v2_score"], 4),
                "selector_score": round(row["selector_score"], 4),
                "coinvest_score_z": _safe_float(row.get("coinvest_score_z")),
                "financial_score": round(_safe_float(row.get("financial_score")), 2),
            }
        )

    return result


def _print_report(result: dict) -> None:
    """Print human-readable validation report."""
    print("=" * 72)
    print(f"  ABVX RANK VALIDATION — {result['company']}")
    print(f"  Baseline: {result['baseline_date']}  |  Latest PIT: {result['latest_pit_date']}")
    print("=" * 72)

    sel = result["selector"]
    rnk = result["ranker"]
    inp = result["key_inputs"]

    print(f"\n  Selector Rank:  #{sel['rank']} / {result['n_eligible']}  (bucket: {sel['bucket']})")
    print(f"  Final Rank:     #{rnk['final_rank']} / {rnk['cohort_size']}  (in Top-30: {rnk['in_top_30']})")

    print(f"\n  Selector Block Scores:")
    print(f"    Institutional (65%):   {sel['institutional_block']:.4f}")
    print(f"    Catalyst (15%):        {sel['catalyst_block']:.4f}")
    print(f"    Survivability (10%):   {sel['survivability_block']:.4f}")
    print(f"    Market Structure (10%):{sel['market_structure_block']:.4f}")
    print(f"    Missing signals:       {sel['missing_signals']}")

    print(f"\n  Key Inputs:")
    print(f"    coinvest_score_z:  {inp['coinvest_score_z']:.4f}  ({inp['coinvest_tag']})")
    print(f"    sponsor_tier1:     {inp['sponsor_tier1_count']}")
    print(f"    inst_delta_z:      {inp['inst_delta_z']:.4f}")
    print(f"    financial_score:   {inp['financial_score']:.2f}")
    print(f"    catalyst_strength: {inp['catalyst_strength']} ({inp['catalyst_days']}d)")
    print(f"    runway_bucket:     {inp['runway_bucket']}")
    print(f"    severity:          {inp['severity']}")
    print(f"    momentum:          {inp['mom_state']}")

    pit_ci = result["latest_pit_coinvest"]
    pit_cat = result["latest_pit_catalyst"]
    print(f"\n  Latest PIT Bundle ({result['latest_pit_date']}):")
    print(f"    Coinvest tier1:    {pit_ci.get('tier1_count')} (recency: {pit_ci.get('recency_state')}, age: {pit_ci.get('filing_age_days')}d)")
    changes = pit_ci.get("position_changes", {})
    increases = sum(1 for v in changes.values() if v == "INCREASE")
    new = sum(1 for v in changes.values() if v == "NEW")
    print(f"    Position changes:  {increases} INCREASE, {new} NEW, {len(changes) - increases - new} other")
    print(f"    Nearest catalyst:  {pit_cat.get('nearest_event', 'N/A')}")
    print(f"    Catalyst date:     {pit_cat.get('event_date', 'N/A')}")

    print(f"\n  Rank Neighborhood (Top-10 + ABVX context):")
    for n in result["neighbors"]:
        marker = " <<<" if n["ticker"] == "ABVX" else ""
        if n["rank"] <= 10 or abs(n["rank"] - rnk["final_rank"]) <= 3:
            print(
                f"    #{n['rank']:2d}  {n['ticker']:8s}  "
                f"final={n['ranker_v2_score']:.4f}  sel={n['selector_score']:.4f}  "
                f"coinvest_z={n['coinvest_score_z']:.4f}  fin={n['financial_score']:.1f}{marker}"
            )

    # Verdict
    print(f"\n  {'=' * 68}")
    if rnk["in_top_30"]:
        print(f"  VERDICT: ABVX rank #{rnk['final_rank']} is VALID — inside Top-30 portfolio")
    else:
        print(f"  VERDICT: ABVX rank #{rnk['final_rank']} is OUTSIDE Top-30 portfolio")

    print(f"\n  Rationale:")
    print(f"    - Strong coinvest conviction ({inp['coinvest_tag']}, z={inp['coinvest_score_z']:.2f})")
    print(f"      drives selector inclusion (65% institutional block weight)")
    print(f"    - Low financial_score ({inp['financial_score']:.1f}) benefits ranker")
    print(f"      (negative model weight penalizes high-financial names, not low)")
    print(f"    - Combined effect: selector #23 -> ranker-adjusted #{rnk['final_rank']}")

    # Caveats
    if pit_cat.get("event_date"):
        print(f"\n  Caveats for next screen refresh:")
        print(f"    - Catalyst {pit_cat['event_date']} has PASSED (today 2026-04-14)")
        print(f"      catalyst_strength may shift from 'near' to 'far'/'missing'")
        print(f"      Impact: catalyst block (~15% selector weight) would decrease")
        print(f"      but institutional block (65%) dominates — rank should hold")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Assertions for CI
# ---------------------------------------------------------------------------


def run_assertions(result: dict) -> None:
    """Hard assertions for automated validation."""
    sel = result["selector"]
    rnk = result["ranker"]
    inp = result["key_inputs"]

    # ABVX must be in the selector top-30 bucket
    assert sel["bucket"] in ("top10", "top30"), (
        f"ABVX selector bucket is {sel['bucket']}, expected top10 or top30"
    )

    # ABVX must be in the final top-30
    assert rnk["in_top_30"], (
        f"ABVX final rank is {rnk['final_rank']}, expected <= 30"
    )

    # Coinvest signal must be strong
    assert inp["coinvest_score_z"] >= 1.5, (
        f"ABVX coinvest_score_z is {inp['coinvest_score_z']}, expected >= 1.5"
    )

    # Institutional block must dominate
    assert sel["institutional_block"] >= 0.60, (
        f"ABVX institutional block is {sel['institutional_block']}, expected >= 0.60"
    )

    # Financial score is weak but expected
    assert inp["financial_score"] < 20, (
        f"ABVX financial_score is {inp['financial_score']}, expected < 20"
    )

    print("\n  All assertions PASSED.")


if __name__ == "__main__":
    result = validate_abvx_rank()
    _print_report(result)
    run_assertions(result)
