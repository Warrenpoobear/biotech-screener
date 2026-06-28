"""
manager_13f_validation_2026.py

Extended validation for the Wake Robin DEM biotech screener 13F manager attribution analysis.
Runs 4 additional checks on existing artifacts and appends results to the markdown report.

Classification: RESEARCH_DIAGNOSTIC / MANAGER_ATTRIBUTION / SHADOW_REWEIGHTING_ONLY
                NO_MODEL_CHANGE / NO_RANKER_CHANGE / NO_SELECTOR_CHANGE
                NO_SIZING_CHANGE / NO_TRADING_CHANGE
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_DIR = os.path.join(REPO, "artifacts", "research", "manager_13f_performance_2026")

WINDOW_RETURNS_CSV = os.path.join(ARTIFACT_DIR, "manager_window_returns.csv")
HOLDINGS_ATTR_CSV = os.path.join(ARTIFACT_DIR, "manager_holdings_attribution.csv")
SUMMARY_CSV = os.path.join(ARTIFACT_DIR, "manager_summary.csv")
REWEIGHTING_CSV = os.path.join(ARTIFACT_DIR, "manager_reweighting_shadow.csv")
MD_REPORT = os.path.join(ARTIFACT_DIR, "MANAGER_13F_PERFORMANCE.md")

OUT_WALKFORWARD = os.path.join(ARTIFACT_DIR, "manager_validation_walkforward.csv")
OUT_CONCENTRATION = os.path.join(ARTIFACT_DIR, "manager_concentration_analysis.csv")
OUT_DECOMPOSITION = os.path.join(ARTIFACT_DIR, "manager_signal_decomposition.csv")
OUT_SHADOW_TIERS = os.path.join(ARTIFACT_DIR, "manager_shadow_tiers.csv")

QUARTERS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
Q_LABELS = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"]
EVAL_WINDOWS = [2, 3, 4]  # 0-indexed into QUARTERS; eval window 2 = 2025-Q3, etc.

today_str = "2026-06-28"

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
print("Loading artifacts...")
wr = pd.read_csv(WINDOW_RETURNS_CSV)
ha = pd.read_csv(HOLDINGS_ATTR_CSV)
summ = pd.read_csv(SUMMARY_CSV)
rw = pd.read_csv(REWEIGHTING_CSV)

# Filter to 63d window only for wr
wr63 = wr[wr["window_days"] == 63].copy()

# Quarter → label maps
q_to_label = dict(zip(QUARTERS, Q_LABELS))

# Build per-manager per-quarter pivot (ew_excess)
pivot_ew = wr63.pivot_table(index="manager", columns="quarter_end", values="ew_excess", aggfunc="mean")
pivot_new = wr63.pivot_table(index="manager", columns="quarter_end", values="new_pos_excess", aggfunc="mean")
pivot_xbi = wr63.pivot_table(index="manager", columns="quarter_end", values="xbi_return", aggfunc="mean")
pivot_top10 = wr63.pivot_table(index="manager", columns="quarter_end", values="top10_excess", aggfunc="mean")

all_managers = pivot_ew.index.tolist()
n_mgr = len(all_managers)

print(f"  Loaded {n_mgr} managers, {len(wr63)} rows (63d), {len(ha)} holdings rows")

# ---------------------------------------------------------------------------
# CHECK 1: Walk-forward reweighting
# ---------------------------------------------------------------------------
print("\nCheck 1: Walk-forward reweighting...")

# Flat weight for each method = 1/n_managers present in each eval window
walkforward_rows = []

for eval_idx in EVAL_WINDOWS:
    eval_qtr = QUARTERS[eval_idx]
    eval_label = Q_LABELS[eval_idx]
    prior_qtrs = QUARTERS[:eval_idx]  # training windows

    # Managers present in the eval quarter
    eval_managers = wr63[wr63["quarter_end"] == eval_qtr]["manager"].tolist()
    n_eval = len(eval_managers)
    flat_w = 1.0 / n_eval if n_eval > 0 else 0.0

    # Eval returns: ew_excess per manager in eval quarter
    eval_ew = wr63[wr63["quarter_end"] == eval_qtr].set_index("manager")["ew_excess"]

    # Prior mean 63d excess per manager (across training windows)
    prior_ew = wr63[wr63["quarter_end"].isin(prior_qtrs)].groupby("manager")["ew_excess"].mean()
    prior_new = wr63[wr63["quarter_end"].isin(prior_qtrs)].groupby("manager")["new_pos_excess"].mean()

    # Method A: flat equal weight
    method_a_ret = eval_ew.reindex(eval_managers).mean()

    # Method B: quality-weighted (prior mean 63d excess), clip [0.5x, 2x] flat, normalize
    def quality_weights(prior_series, eval_mgrs, flat_w):
        w = prior_series.reindex(eval_mgrs).fillna(0.0)
        # shift so minimum quality signal can still get some weight
        # Use raw prior mean directly; positives get higher weight, negatives lower
        # Map to [0.5, 2.0] of flat_w proportionally
        w_min, w_max = w.min(), w.max()
        if w_max == w_min:
            return pd.Series(flat_w, index=eval_mgrs)
        # Normalize to [0.5*flat, 2*flat]
        w_norm = 0.5 * flat_w + (w - w_min) / (w_max - w_min) * 1.5 * flat_w
        # Clip
        w_norm = w_norm.clip(lower=0.5 * flat_w, upper=2 * flat_w)
        w_norm = w_norm / w_norm.sum()
        return w_norm

    wb = quality_weights(prior_ew, eval_managers, flat_w)
    method_b_ret = (eval_ew.reindex(eval_managers) * wb).sum()

    # Method C: top-half only (positive prior mean 63d excess, equal weight among them)
    pos_managers = [m for m in eval_managers if prior_ew.get(m, 0.0) > 0]
    if len(pos_managers) > 0:
        method_c_ret = eval_ew.reindex(pos_managers).mean()
    else:
        method_c_ret = eval_ew.reindex(eval_managers).mean()  # fallback to flat
    n_pos = len(pos_managers)

    # Method D: walk-forward flow weights (new_pos_excess from prior windows)
    wd = quality_weights(prior_new, eval_managers, flat_w)
    method_d_ret = (eval_ew.reindex(eval_managers) * wd).sum()

    walkforward_rows.append(
        {
            "eval_quarter": eval_label,
            "n_eval_managers": n_eval,
            "n_prior_windows": eval_idx,
            "n_pos_quality_managers": n_pos,
            "method_a_flat_ew_excess": round(method_a_ret, 6),
            "method_b_quality_weighted_excess": round(method_b_ret, 6),
            "method_c_top_half_excess": round(method_c_ret, 6),
            "method_d_flow_weighted_excess": round(method_d_ret, 6),
        }
    )

wf_df = pd.DataFrame(walkforward_rows)

# Note which windows have complete forward returns (not all NaN)
wf_df["has_data"] = wf_df["method_a_flat_ew_excess"].notna()

# Summary means: exclude windows where forward returns are still incomplete (NaN)
mean_a = wf_df.loc[wf_df["has_data"], "method_a_flat_ew_excess"].mean()
mean_b = wf_df.loc[wf_df["has_data"], "method_b_quality_weighted_excess"].mean()
mean_c = wf_df.loc[wf_df["has_data"], "method_c_top_half_excess"].mean()
mean_d = wf_df.loc[wf_df["has_data"], "method_d_flow_weighted_excess"].mean()
n_complete_windows = wf_df["has_data"].sum()

wf_df.to_csv(OUT_WALKFORWARD, index=False)
print(f"  Walk-forward CSV saved: {OUT_WALKFORWARD}")
print(f"  Method A mean: {mean_a:.4f}  B: {mean_b:.4f}  C: {mean_c:.4f}  D: {mean_d:.4f}")

# ---------------------------------------------------------------------------
# CHECK 2: One-name / one-quarter concentration
# ---------------------------------------------------------------------------
print("\nCheck 2: Concentration analysis...")

# Holdings attribution: compute each holding's EW contribution per manager per quarter
# contribution = excess_63d / n_holdings_that_quarter
ha2 = ha.copy()

# Number of holdings per manager per quarter
n_hold = ha2.groupby(["manager", "quarter_end"])["ticker"].transform("count")
ha2["ew_contribution"] = ha2["excess_63d"] / n_hold

# Per manager: aggregate across all quarters
conc_rows = []
for manager, grp in ha2.groupby("manager"):
    contribs = grp["ew_contribution"]
    pos_contribs = contribs[contribs > 0]
    total_pos = pos_contribs.sum()

    n_total_holdings = len(grp)

    if n_total_holdings < 3:
        conc_rows.append(
            {
                "manager": manager,
                "n_holdings_total": n_total_holdings,
                "n_quarters": grp["quarter_end"].nunique(),
                "top1_contribution_pct": np.nan,
                "top3_contribution_pct": np.nan,
                "top5_contribution_pct": np.nan,
                "one_quarter_pct": np.nan,
                "concentration_class": "INSUFFICIENT_DATA",
            }
        )
        continue

    # Sort all holdings by contribution descending
    sorted_contribs = contribs.sort_values(ascending=False).values

    if total_pos <= 0:
        top1_pct = np.nan
        top3_pct = np.nan
        top5_pct = np.nan
    else:
        top1_pct = max(sorted_contribs[0], 0) / total_pos
        top3_pct = sum(max(v, 0) for v in sorted_contribs[:3]) / total_pos
        top5_pct = sum(max(v, 0) for v in sorted_contribs[:5]) / total_pos

    # Quarter concentration: per-quarter sum of ew_contributions
    q_contribs = grp.groupby("quarter_end")["ew_contribution"].sum()
    pos_q = q_contribs[q_contribs > 0]
    if len(pos_q) == 0:
        one_quarter_pct = np.nan
    else:
        one_quarter_pct = pos_q.max() / pos_q.sum()

    # Classification
    if pd.isna(top1_pct) or pd.isna(one_quarter_pct):
        cls = "INSUFFICIENT_DATA"
    elif top1_pct > 0.40:
        cls = "ONE_NAME_DRIVEN"
    elif (one_quarter_pct > 0.75) if not pd.isna(one_quarter_pct) else False:
        cls = "ONE_QUARTER_DRIVEN"
    elif (top3_pct < 0.50) and ((one_quarter_pct < 0.60) if not pd.isna(one_quarter_pct) else True):
        cls = "BROAD_MANAGER_SIGNAL"
    else:
        cls = "CONCENTRATED_BUT_CONSISTENT"

    conc_rows.append(
        {
            "manager": manager,
            "n_holdings_total": n_total_holdings,
            "n_quarters": grp["quarter_end"].nunique(),
            "top1_contribution_pct": round(top1_pct, 4) if not pd.isna(top1_pct) else np.nan,
            "top3_contribution_pct": round(top3_pct, 4) if not pd.isna(top3_pct) else np.nan,
            "top5_contribution_pct": round(top5_pct, 4) if not pd.isna(top5_pct) else np.nan,
            "one_quarter_pct": round(one_quarter_pct, 4) if not pd.isna(one_quarter_pct) else np.nan,
            "concentration_class": cls,
        }
    )

conc_df = pd.DataFrame(conc_rows)
conc_df.to_csv(OUT_CONCENTRATION, index=False)
print(f"  Concentration CSV saved: {OUT_CONCENTRATION}")
print(f"  Class counts:\n{conc_df['concentration_class'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# CHECK 3: Signal decomposition
# ---------------------------------------------------------------------------
print("\nCheck 3: Signal decomposition...")

# Per-manager: mean excess per sleeve across all 5 quarters (63d)
decomp_rows = []
for manager, grp in wr63.groupby("manager"):
    ew_xs = grp["ew_excess"].mean()
    new_xs = grp["new_pos_excess"].mean()
    top10_xs = grp["top10_excess"].mean()
    # No increased_pos column in source data — skip that sleeve
    xbi_ret = grp["xbi_return"].mean()  # XBI mean across windows (roughly 0 excess by def)

    # Best use determination
    sleeves = {
        "ownership": ew_xs,
        "flow/new_positions": new_xs,
        "concentration": top10_xs,
    }
    max_sleeve = max(sleeves, key=sleeves.get)
    max_val = sleeves[max_sleeve]
    min_val = min(sleeves.values())
    spread = max_val - min_val

    if all(abs(v) < 0.01 for v in sleeves.values()):
        best_use = "no_clear_signal"
    elif spread < 0.02:
        best_use = "mixed"
    elif new_xs > ew_xs + 0.03:
        best_use = "flow/new_positions"
    elif top10_xs > ew_xs + 0.02 and top10_xs == max_val:
        best_use = "concentration"
    else:
        best_use = "ownership"

    decomp_rows.append(
        {
            "manager": manager,
            "existing_holdings_xs": round(ew_xs, 4),
            "new_positions_xs": round(new_xs, 4),
            "top10_xs": round(top10_xs, 4),
            "increased_positions_xs": np.nan,  # not in source data
            "n_windows": len(grp),
            "best_use": best_use,
        }
    )

decomp_df = pd.DataFrame(decomp_rows).sort_values("existing_holdings_xs", ascending=False)
decomp_df.to_csv(OUT_DECOMPOSITION, index=False)
print(f"  Decomposition CSV saved: {OUT_DECOMPOSITION}")
print(f"  Best_use distribution:\n{decomp_df['best_use'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# CHECK 4: Robustness re-classification and shadow tiers
# ---------------------------------------------------------------------------
print("\nCheck 4: Robustness re-classification...")

# Merge summary with concentration classification
summ2 = summ.copy()
conc_lookup = conc_df.set_index("manager")[
    ["concentration_class", "top1_contribution_pct", "top3_contribution_pct", "one_quarter_pct"]
]
summ2 = summ2.join(conc_lookup, on="name")


def classify_manager(row):
    n = row.get("n_filing_windows", 0)
    mean_xs = row.get("mean_63d_excess", 0.0)
    hit = row.get("hit_rate_63d", 0.0)
    boot = row.get("bootstrap_pct_63d", 0.0)
    avg_hold = row.get("avg_holdings", 0)
    coverage = row.get("coverage_score", 0.0)
    top1 = row.get("top1_contribution_pct", 1.0)
    cls_conc = row.get("concentration_class", "INSUFFICIENT_DATA")
    one_q = row.get("one_quarter_pct", 1.0)

    # Handle NaN
    if pd.isna(top1):
        top1 = 1.0
    if pd.isna(one_q):
        one_q = 1.0

    is_one_quarter = cls_conc == "ONE_QUARTER_DRIVEN"
    is_insufficient = n < 3 or avg_hold < 5 or coverage < 0.4

    if is_insufficient:
        return "INSUFFICIENT_DATA"

    # UPWEIGHT: ALL gates must pass
    upweight = n >= 3 and mean_xs > 0 and hit >= 0.55 and boot >= 0.65 and top1 < 0.50 and not is_one_quarter
    if upweight:
        return "UPWEIGHT_CANDIDATE_SHADOW"

    # DOWNWEIGHT
    downweight = n >= 3 and (mean_xs < 0 or (mean_xs < 0.02 and hit <= 0.40)) and boot <= 0.35
    if downweight:
        return "DOWNWEIGHT_CANDIDATE_SHADOW"

    return "KEEP_CURRENT_WEIGHT"


summ2["robust_classification"] = summ2.apply(classify_manager, axis=1)


# Assign tiers
def assign_tier(row):
    cls = row["robust_classification"]
    conc_cls = row.get("concentration_class", "INSUFFICIENT_DATA")
    if cls == "UPWEIGHT_CANDIDATE_SHADOW":
        if conc_cls == "BROAD_MANAGER_SIGNAL":
            return "Tier A"
        else:  # ONE_NAME_DRIVEN, ONE_QUARTER_DRIVEN, or CONCENTRATED_BUT_CONSISTENT
            return "Tier B"
    elif cls == "DOWNWEIGHT_CANDIDATE_SHADOW":
        return "Tier D"
    elif cls == "KEEP_CURRENT_WEIGHT":
        return "Tier C"
    else:
        return "Tier E"


summ2["shadow_tier"] = summ2.apply(assign_tier, axis=1)

# Build output
tier_cols = [
    "cik",
    "name",
    "classification",
    "robust_classification",
    "shadow_tier",
    "n_filing_windows",
    "avg_holdings",
    "coverage_score",
    "mean_63d_excess",
    "hit_rate_63d",
    "bootstrap_pct_63d",
    "concentration_class",
    "top1_contribution_pct",
    "top3_contribution_pct",
    "one_quarter_pct",
]
tier_df = summ2[tier_cols].copy()
tier_df.to_csv(OUT_SHADOW_TIERS, index=False)
print(f"  Shadow tiers CSV saved: {OUT_SHADOW_TIERS}")
print(f"  Tier distribution:\n{tier_df['shadow_tier'].value_counts().sort_index().to_string()}")

# ---------------------------------------------------------------------------
# Identify top 15 for signal decomposition table
# ---------------------------------------------------------------------------
# Top 6 upweight + RA/Baker/OrbiMed/Logos/Sofinnova + 2 downweight + 2 borderline KEEP
upweight_names = tier_df[tier_df["robust_classification"] == "UPWEIGHT_CANDIDATE_SHADOW"]["name"].tolist()
downweight_names = tier_df[tier_df["robust_classification"] == "DOWNWEIGHT_CANDIDATE_SHADOW"]["name"].tolist()
keep_names = tier_df[tier_df["robust_classification"] == "KEEP_CURRENT_WEIGHT"]["name"].tolist()

# Notable managers
notable = [
    "RA Capital Management",
    "Baker Bros Advisors",
    "Orbimed Advisors",
    "Logos Global Management",
    "Sofinnova Investments",
]

top6_upweight = [
    n
    for n in (
        tier_df[tier_df["robust_classification"] == "UPWEIGHT_CANDIDATE_SHADOW"]
        .sort_values("mean_63d_excess", ascending=False)["name"]
        .tolist()
    )
][:6]

# Borderline KEEP: lowest-quality KEEP (closest to thresholds)
keep_df = tier_df[tier_df["robust_classification"] == "KEEP_CURRENT_WEIGHT"].copy()
keep_df["dist_to_up"] = keep_df["mean_63d_excess"] * keep_df["hit_rate_63d"]
borderline_keep = keep_df.sort_values("dist_to_up", ascending=False)["name"].tolist()[:2]

top15_candidates = list(dict.fromkeys(top6_upweight + notable + downweight_names + borderline_keep))[:15]

decomp15 = decomp_df[decomp_df["manager"].isin(top15_candidates)].copy()
# Add classification
decomp15 = decomp15.join(tier_df.set_index("name")[["shadow_tier", "robust_classification"]], on="manager")

# ---------------------------------------------------------------------------
# Append to MANAGER_13F_PERFORMANCE.md
# ---------------------------------------------------------------------------
print("\nAppending to markdown report...")


def fmt_pct(v, mult=100):
    if pd.isna(v):
        return "n/a"
    return f"{v*mult:+.1f}%"


def fmt_pct_plain(v, mult=100):
    if pd.isna(v):
        return "n/a"
    return f"{v*mult:.1f}%"


# Walk-forward table
wf_md_rows = []
for _, r in wf_df.iterrows():
    wf_md_rows.append(
        f"| {r['eval_quarter']} | {r['n_prior_windows']} prior Qs | "
        f"{fmt_pct(r['method_a_flat_ew_excess'])} | "
        f"{fmt_pct(r['method_b_quality_weighted_excess'])} | "
        f"{fmt_pct(r['method_c_top_half_excess'])} | "
        f"{fmt_pct(r['method_d_flow_weighted_excess'])} |"
    )

wf_beats_flat = mean_b > mean_a
wf_margin_b = (mean_b - mean_a) * 100
wf_margin_c = (mean_c - mean_a) * 100

# Concentration summary
broad = conc_df[conc_df["concentration_class"] == "BROAD_MANAGER_SIGNAL"]["manager"].tolist()
one_name = conc_df[conc_df["concentration_class"] == "ONE_NAME_DRIVEN"]["manager"].tolist()
one_qtr = conc_df[conc_df["concentration_class"] == "ONE_QUARTER_DRIVEN"]["manager"].tolist()
conc_but = conc_df[conc_df["concentration_class"] == "CONCENTRATED_BUT_CONSISTENT"]["manager"].tolist()

# Signal decomposition table (top 15)
decomp_md_rows = []
for _, r in decomp15.sort_values("existing_holdings_xs", ascending=False).iterrows():
    tier = r.get("shadow_tier", "n/a")
    decomp_md_rows.append(
        f"| {r['manager']} | {fmt_pct(r['existing_holdings_xs'])} | "
        f"{fmt_pct(r['new_positions_xs'])} | n/a | "
        f"{r['best_use']} | {tier} |"
    )

# Revised shadow tier table
tier_md_rows = []
for tier_label in ["Tier A", "Tier B", "Tier C", "Tier D", "Tier E"]:
    names_in_tier = tier_df[tier_df["shadow_tier"] == tier_label]["name"].tolist()
    for name in names_in_tier:
        row = tier_df[tier_df["name"] == name].iloc[0]
        conc_cls = row.get("concentration_class", "n/a")
        tier_md_rows.append(
            f"| {name} | {tier_label} | {row['robust_classification']} | "
            f"{conc_cls} | {fmt_pct(row['mean_63d_excess'])} | "
            f"{fmt_pct_plain(row['hit_rate_63d'])} | {fmt_pct_plain(row['bootstrap_pct_63d'])} |"
        )

# Build the new section text
new_section = f"""

## Extended Validation ({today_str})

*Classification: RESEARCH_DIAGNOSTIC / MANAGER_ATTRIBUTION / SHADOW_REWEIGHTING_ONLY / NO_MODEL_CHANGE*

---

### Check 1: Walk-Forward Reweighting (3 out-of-sample windows)

Protocol: Each evaluation window uses only prior-window data to set manager weights.
No same-window optimization.

| Eval Window | Training | Method A (flat EW) | Method B (quality-weighted) | Method C (top-half only) | Method D (flow-weighted) |
|-------------|----------|-------------------|----------------------------|--------------------------|--------------------------|
{chr(10).join(wf_md_rows)}
| **Mean** | — | **{fmt_pct(mean_a)}** | **{fmt_pct(mean_b)}** | **{fmt_pct(mean_c)}** | **{fmt_pct(mean_d)}** |

**Walk-forward beats flat (Method B vs A): {"Yes" if wf_beats_flat else "No"} ({wf_margin_b:+.1f}pp margin)**
Method C (top-half) vs flat: {wf_margin_c:+.1f}pp
*Note: {n_complete_windows}/3 evaluation windows have complete 63-day forward returns as of {today_str}. 2026-Q1 forward window (63d from ~May 2026 signal date) is incomplete — excluded from means.*

Interpretation: Walk-forward quality weighting {"outperforms" if wf_beats_flat else "does not outperform"} the flat equal-weight baseline across {n_complete_windows} complete out-of-sample windows.
{"This suggests manager signal strength is partially predictable from prior performance." if wf_beats_flat else "The flat baseline remains the conservative default; manager rankings have limited predictive persistence across adjacent quarters."}

---

### Check 2: One-Name / One-Quarter Concentration Analysis

| Class | Count | Managers |
|-------|-------|---------|
| BROAD_MANAGER_SIGNAL | {len(broad)} | {", ".join(broad) or "none"} |
| ONE_NAME_DRIVEN | {len(one_name)} | {", ".join(one_name) or "none"} |
| ONE_QUARTER_DRIVEN | {len(one_qtr)} | {", ".join(one_qtr) or "none"} |
| CONCENTRATED_BUT_CONSISTENT | {len(conc_but)} | {", ".join(conc_but) or "none"} |
| INSUFFICIENT_DATA | {len(conc_df[conc_df["concentration_class"] == "INSUFFICIENT_DATA"])} | {", ".join(conc_df[conc_df["concentration_class"] == "INSUFFICIENT_DATA"]["manager"].tolist()) or "none"} |

Key finding: {"Broad manager signals are common — majority of upweight candidates have signal spread across holdings and quarters." if len(broad) > len(one_name)+len(one_qtr) else "Concentration is prevalent — many managers' alpha is driven by one name or one window, reducing signal reliability for systematic reweighting."}

---

### Check 3: Signal Decomposition (Top 15 Managers)

| Manager | Existing Holdings XS | New Positions XS | Increased Positions XS | Best Use | Tier |
|---------|---------------------|-----------------|----------------------|----------|------|
{chr(10).join(decomp_md_rows)}

*Increased Positions sleeve not available in source data (no increased_pos_return column).*

Best-use highlights:
- **Ownership signal**: {", ".join([r["manager"] for _, r in decomp15[decomp15["best_use"] == "ownership"].iterrows()]) or "none"}
- **Flow/new positions**: {", ".join([r["manager"] for _, r in decomp15[decomp15["best_use"] == "flow/new_positions"].iterrows()]) or "none"}
- **Concentration (top10)**: {", ".join([r["manager"] for _, r in decomp15[decomp15["best_use"] == "concentration"].iterrows()]) or "none"}
- **No clear signal**: {", ".join([r["manager"] for _, r in decomp15[decomp15["best_use"] == "no_clear_signal"].iterrows()]) or "none"}

---

### Check 4: Robustness Re-Classification and Shadow Tiers

Gates applied strictly (all 5 upweight gates must pass simultaneously):
- UPWEIGHT: n_windows≥3, mean_63d_excess>0, hit_rate≥55%, bootstrap≥65%, top1_conc<50%, NOT one_quarter_driven
- DOWNWEIGHT: n_windows≥3, mean_63d_excess<0 OR (mean<2pp AND hit≤40%), bootstrap≤35%
- Tier A = UPWEIGHT + BROAD_MANAGER_SIGNAL
- Tier B = UPWEIGHT + concentrated (ONE_NAME or ONE_QUARTER or CONCENTRATED_BUT_CONSISTENT)
- Tier C = KEEP_CURRENT_WEIGHT (passes n≥3 but misses upweight/downweight gates)
- Tier D = DOWNWEIGHT_CANDIDATE_SHADOW
- Tier E = INSUFFICIENT_DATA

| Manager | Tier | Classification | Concentration | Mean XS | Hit Rate | Bootstrap |
|---------|------|---------------|---------------|---------|----------|-----------|
{chr(10).join(tier_md_rows)}

**Summary:**
- Tier A (robust upweight): {", ".join(tier_df[tier_df["shadow_tier"] == "Tier A"]["name"].tolist()) or "none"}
- Tier B (concentrated/upweight): {", ".join(tier_df[tier_df["shadow_tier"] == "Tier B"]["name"].tolist()) or "none"}
- Tier C (keep current): {len(tier_df[tier_df["shadow_tier"] == "Tier C"])} managers
- Tier D (downweight): {", ".join(tier_df[tier_df["shadow_tier"] == "Tier D"]["name"].tolist()) or "none"}
- Tier E (insufficient data): {", ".join(tier_df[tier_df["shadow_tier"] == "Tier E"]["name"].tolist()) or "none"}

---

### Revised Governance Conclusion

- **Production change: False**
- NO_MODEL_CHANGE / NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE
- All outputs are shadow diagnostics only. No tier assignment changes production weights.
- Walk-forward validation uses 3 out-of-sample windows — minimum viable test; at least 8 total windows required before any production promotion.
- Tier A managers are candidates for future shadow reweighting monitors once the 8-window gate is met.
- Tier B managers require additional verification that their alpha is not single-name driven before any shadow promotion.

*Extended validation generated: {today_str}*
"""

# Append to markdown
with open(MD_REPORT, "a") as f:
    f.write(new_section)

print(f"  Appended extended validation section to {MD_REPORT}")

# ---------------------------------------------------------------------------
# Summary printout for final response
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WALK-FORWARD VERDICT")
print("=" * 70)
for _, r in wf_df.iterrows():
    print(
        f"  {r['eval_quarter']}: A={fmt_pct(r['method_a_flat_ew_excess'])}  "
        f"B={fmt_pct(r['method_b_quality_weighted_excess'])}  "
        f"C={fmt_pct(r['method_c_top_half_excess'])}  "
        f"D={fmt_pct(r['method_d_flow_weighted_excess'])}"
    )
print(f"  Mean: A={fmt_pct(mean_a)}  B={fmt_pct(mean_b)}  C={fmt_pct(mean_c)}  D={fmt_pct(mean_d)}")
print(f"  Walk-forward beats flat: {'Yes' if wf_beats_flat else 'No'} ({wf_margin_b:+.1f}pp)")

print("\nCONCENTRATION FINDINGS")
print("=" * 70)
print(f"  BROAD_MANAGER_SIGNAL ({len(broad)}): {', '.join(broad)}")
print(f"  ONE_NAME_DRIVEN ({len(one_name)}): {', '.join(one_name)}")
print(f"  ONE_QUARTER_DRIVEN ({len(one_qtr)}): {', '.join(one_qtr)}")
print(f"  CONCENTRATED_BUT_CONSISTENT ({len(conc_but)}): {', '.join(conc_but[:5])}{'...' if len(conc_but) > 5 else ''}")

print("\nSIGNAL DECOMPOSITION HIGHLIGHTS")
print("=" * 70)
ownership = decomp_df[decomp_df["best_use"] == "ownership"]["manager"].tolist()
flow = decomp_df[decomp_df["best_use"] == "flow/new_positions"]["manager"].tolist()
conc_best = decomp_df[decomp_df["best_use"] == "concentration"]["manager"].tolist()
no_sig = decomp_df[decomp_df["best_use"] == "no_clear_signal"]["manager"].tolist()
mixed_sig = decomp_df[decomp_df["best_use"] == "mixed"]["manager"].tolist()
print(f"  Ownership: {', '.join(ownership[:8])}{'...' if len(ownership) > 8 else ''}")
print(f"  Flow/new positions: {', '.join(flow[:8])}{'...' if len(flow) > 8 else ''}")
print(f"  Concentration (top10): {', '.join(conc_best[:8])}{'...' if len(conc_best) > 8 else ''}")
print(f"  No clear signal: {', '.join(no_sig[:5])}{'...' if len(no_sig) > 5 else ''}")
print(f"  Mixed: {', '.join(mixed_sig[:5])}{'...' if len(mixed_sig) > 5 else ''}")

print("\nREVISED SHADOW TIERS")
print("=" * 70)
for tier_label in ["Tier A", "Tier B", "Tier C", "Tier D", "Tier E"]:
    names = tier_df[tier_df["shadow_tier"] == tier_label]["name"].tolist()
    print(f"  {tier_label}: {', '.join(names) if names else 'none'}")

print("\n" + "=" * 70)
print("OUTPUTS WRITTEN")
print("=" * 70)
for path in [OUT_WALKFORWARD, OUT_CONCENTRATION, OUT_DECOMPOSITION, OUT_SHADOW_TIERS]:
    print(f"  {path}")
print(f"  {MD_REPORT}  (appended)")

print("\nProduction change: False")
print("Governance: NO_MODEL_CHANGE / NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE")
