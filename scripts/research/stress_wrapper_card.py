"""
Stress Wrapper Card — Weekly top-level monitoring card
VALIDATION_INFRASTRUCTURE / STRESS_WRAPPER_SHADOW / NO_MODEL_CHANGE / NO_SELECTOR_CHANGE

Supersedes weekly_failure_mode_card.py.  Adds:
  - Investability status header
  - Regime context with interpretation (BEAR/STRESS/NEUTRAL/MILD_RALLY/STRONG_RALLY)
  - Conditional risk wrapper activation and basket comparison
  - Full repeat-offender table (tenure, contribution history, replacement)

Usage:
    python3 scripts/research/stress_wrapper_card.py [--snap-date YYYY-MM-DD]
"""

import argparse
import glob
import json
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "artifacts/failure_mode_guards"
OUTDIR.mkdir(parents=True, exist_ok=True)
SCRIPTS = REPO / "scripts/research"

PRICE_FILE = REPO / "production_data/price_history_split_adj.csv"

# Regime XBI trailing 20d thresholds (pp)
REGIME_BEARISH_THRESHOLD = -3.0
REGIME_MILD_RALLY_THRESHOLD = 3.0
REGIME_STRONG_RALLY_THRESHOLD = 8.0


def run_script(script: str, args: list = None) -> tuple[int, str]:
    cmd = [sys.executable, str(SCRIPTS / script)] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return result.returncode, result.stdout + result.stderr


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def find_latest_snap(target: str) -> str:
    dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    eligible = [Path(d).name for d in dirs if len(Path(d).name) == 10 and Path(d).name <= target]
    return eligible[-1] if eligible else None


def compute_xbi_trailing(snap_date: str) -> float | None:
    """Trailing 20-business-day XBI return at snap_date (PIT-safe prior context)."""
    if not PRICE_FILE.exists():
        return None
    prices = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if "XBI" not in prices.columns:
        return None
    xbi = prices["XBI"].dropna()
    prior = xbi.loc[:snap_date]
    if len(prior) < 21:
        return None
    end_px = prior.iloc[-1]
    start_px = prior.iloc[-21]
    if start_px <= 0:
        return None
    return round((end_px - start_px) / start_px * 100, 2)


def classify_regime(xbi_trailing_pct: float | None) -> tuple[str, str]:
    """Returns (regime_label, interpretation_text)."""
    if xbi_trailing_pct is None:
        return "UNKNOWN", "XBI trailing return unavailable."
    if xbi_trailing_pct < REGIME_BEARISH_THRESHOLD:
        return (
            "BEAR/STRESS",
            "Model expected to outperform — quality/survivability dispersion is highest in bearish sectors.",
        )
    if xbi_trailing_pct < REGIME_MILD_RALLY_THRESHOLD:
        return "NEUTRAL", "Model baseline performance expected — no regime amplifier active."
    if xbi_trailing_pct < REGIME_STRONG_RALLY_THRESHOLD:
        return (
            "MILD_RALLY",
            "Model may lag momentum/junk rotation. Mild-rally lag is a KNOWN PATTERN, not model failure.",
        )
    return (
        "STRONG_RALLY",
        "Watch beta/tail participation — strong XBI move may overwhelm name-level selection.",
    )


def compute_tenure(current_top30_tickers: list, as_of_date: str, n_lookback: int = 12) -> dict:
    """Count how many of the last n_lookback snapshots each ticker appeared in Top-30."""
    snap_dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    eligible = [Path(d).name for d in snap_dirs if len(Path(d).name) == 10 and Path(d).name <= as_of_date]
    recent = eligible[-n_lookback:] if len(eligible) >= n_lookback else eligible
    tenure = {t: 0 for t in current_top30_tickers}
    for snap in recent:
        path = REPO / f"data/snapshots/{snap}/rankings.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False, usecols=["ticker", "actionable_rank"])
        in_top30 = set(df[df["actionable_rank"] <= 30]["ticker"].astype(str).tolist())
        for t in current_top30_tickers:
            if t in in_top30:
                tenure[t] += 1
    return tenure


def load_attribution_history() -> pd.DataFrame:
    """Load name-level attribution from autopsy seed + rolling ledger."""
    frames = []
    seed_csv = REPO / "artifacts/autopsy/ytd_top30_failure_postmortem_2026/name_attribution.csv"
    if seed_csv.exists():
        frames.append(pd.read_csv(seed_csv))
    ledger = OUTDIR / "repeat_offender_ledger.jsonl"
    if ledger.exists():
        rows = []
        with open(ledger) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["window", "ticker"], keep="last")


def build_offender_table(snap_date: str, df_snap: pd.DataFrame) -> list[dict]:
    """
    Build full repeat-offender table for all Top-30 names with attribution history.
    Includes tenure, cumulative XS contribution, negative windows, EES, catalyst bucket,
    and replacement candidate.
    """
    top30 = df_snap[df_snap["actionable_rank"] <= 30].copy()
    bench = df_snap[(df_snap["actionable_rank"] > 30) & (df_snap["actionable_rank"] <= 60)].sort_values(
        "actionable_rank"
    )
    if top30.empty:
        return []

    hist = load_attribution_history()
    actual_snap = find_latest_snap(snap_date) or snap_date
    tenure = compute_tenure(top30["ticker"].tolist(), actual_snap)

    table = []
    all_windows = sorted(hist["window"].unique(), reverse=True)[:12] if not hist.empty else []

    for _, row in top30.iterrows():
        ticker = row["ticker"]
        rank = int(row["actionable_rank"])
        ees = row.get("ees_v3_gate")
        cat = row.get("catalyst_bucket", "?")

        neg_windows = 0
        bottom5_windows = 0
        cum_xs = 0.0

        if not hist.empty and "contrib_to_xs" in hist.columns:
            trows = hist[hist["ticker"] == ticker]
            if not trows.empty:
                neg_windows = int((trows["contrib_to_xs"] < 0).sum())
                cum_xs = round(float(trows["contrib_to_xs"].sum()), 3)

            for w in all_windows:
                wdf = hist[hist["window"] == w].dropna(subset=["contrib_to_xs"])
                if wdf.empty:
                    continue
                if ticker in wdf.nsmallest(5, "contrib_to_xs")["ticker"].tolist():
                    bottom5_windows += 1

        # Replacement: first clean bench name
        replacement = None
        for _, br in bench.iterrows():
            ees_b = br.get("ees_v3_gate")
            if str(ees_b).lower() not in ("false", "0"):
                replacement = {
                    "ticker": br["ticker"],
                    "rank": int(br["actionable_rank"]),
                    "ees": str(ees_b),
                    "cat": br.get("catalyst_bucket", "?"),
                }
                break

        table.append(
            {
                "ticker": ticker,
                "rank": rank,
                "weeks_in_top30": tenure.get(ticker, 0),
                "neg_contribution_windows": neg_windows,
                "bottom5_windows": bottom5_windows,
                "cumulative_xs_pp": cum_xs,
                "ees_gate": str(ees),
                "catalyst_bucket": cat,
                "flagged": bottom5_windows >= 2,
                "replacement": replacement,
            }
        )

    table.sort(key=lambda x: (-x["bottom5_windows"], x["cumulative_xs_pp"]))
    return table


def run(snap_date: str) -> None:
    print(f"Stress Wrapper Card — {snap_date}")
    print("=" * 72)

    # ── 1. Run all sub-monitors ──────────────────────────────────────────────
    print("\n[1/5] repeat-offender monitor...")
    run_script("repeat_offender_monitor.py", ["--snap-date", snap_date])

    print("[2/5] EES-guarded shadow...")
    run_script("ees_guarded_shadow.py", ["--snap-date", snap_date])
    run_script("ees_guarded_shadow.py", ["--fill-returns"])

    print("[3/5] rank-depth shadow...")
    run_script("rank_depth_replacement_shadow.py", ["--snap-date", snap_date])
    run_script("rank_depth_replacement_shadow.py", ["--fill-returns"])

    print("[4/5] rolling drawdown trigger...")
    run_script("rolling_drawdown_trigger.py", ["--as-of", snap_date])

    print("[5/5] conditional risk wrapper...")
    run_script("conditional_risk_wrapper.py", ["--snap-date", snap_date])
    run_script("conditional_risk_wrapper.py", ["--fill-returns"])

    # ── 2. Load data ─────────────────────────────────────────────────────────
    trigger_data = load_json(OUTDIR / f"drawdown_trigger_{snap_date}.json")
    load_json(OUTDIR / f"repeat_offender_{snap_date}.json")

    # Load wrapper capture for this date
    wrapper_rec = {}
    if (OUTDIR / "stress_wrapper_captures.jsonl").exists():
        with open(OUTDIR / "stress_wrapper_captures.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("snap_date") == snap_date:
                        wrapper_rec = rec
                        break

    ees_df = pd.DataFrame()
    if (OUTDIR / "ees_guarded_returns.jsonl").exists():
        ees_df = pd.read_json(OUTDIR / "ees_guarded_returns.jsonl", lines=True)

    rd_df = pd.DataFrame()
    if (OUTDIR / "rank_depth_returns.jsonl").exists():
        rd_df = pd.read_json(OUTDIR / "rank_depth_returns.jsonl", lines=True)

    sw_df = pd.DataFrame()
    if (OUTDIR / "stress_wrapper_returns.jsonl").exists():
        sw_df = pd.read_json(OUTDIR / "stress_wrapper_returns.jsonl", lines=True)

    # Load snapshot for regime + bucket analysis
    actual_snap = find_latest_snap(snap_date) or snap_date
    df_snap = pd.DataFrame()
    snap_path = REPO / f"data/snapshots/{actual_snap}/rankings.csv"
    if snap_path.exists():
        df_snap = pd.read_csv(snap_path, low_memory=False)
        df_snap["ticker"] = df_snap["ticker"].astype(str)

    # ── 3. Regime context ────────────────────────────────────────────────────
    xbi_trail = compute_xbi_trailing(snap_date)
    regime_label, regime_interp = classify_regime(xbi_trail)
    xbi_str = f"{xbi_trail:>+.1f}%" if xbi_trail is not None else "n/a"

    # ── 4. Wrapper activation ─────────────────────────────────────────────────
    wrapper_active = wrapper_rec.get("risk_wrapper_active", False)
    active_triggers = wrapper_rec.get("active_triggers", [])
    conditions = wrapper_rec.get("conditions", {})

    # ── 5. Repeat-offender detail table ──────────────────────────────────────
    offender_table = build_offender_table(snap_date, df_snap) if not df_snap.empty else []

    # ── 6. Current catalyst snapshot ─────────────────────────────────────────
    cat_summary = {}
    ees_false_n = 0
    if not df_snap.empty:
        top30 = df_snap[df_snap["actionable_rank"] <= 30]
        if "catalyst_bucket" in top30.columns:
            cat_summary = top30["catalyst_bucket"].value_counts().to_dict()
        if "ees_v3_gate" in top30.columns:
            ees_false_n = int(top30["ees_v3_gate"].eq(False).sum())

    # ── 7. Build card ─────────────────────────────────────────────────────────
    roll_xs = trigger_data.get("rolling_xs_pp", 0.0)
    trigger_status = trigger_data.get("status", "UNKNOWN")

    lines = [
        f"# Stress Wrapper Card — {snap_date}",
        "",
        "**Classification:** VALIDATION_INFRASTRUCTURE / STRESS_WRAPPER_SHADOW / NO_MODEL_CHANGE",
        "",
        "---",
        "",
        "## Investability Status",
        "",
        "```",
        "PILOT_INVESTABLE_WITH_SHADOW_GUARDS",
        "```",
        "",
        "Live pilot may proceed. Promotion question: does raw Top-30 plus monitored",
        "stress guards produce better net forward outcomes without killing convexity?",
        "",
        "---",
        "",
        "## Regime Context",
        "",
        f"XBI trailing 20d return: **{xbi_str}**  →  regime: **{regime_label}**",
        "",
        f"> {regime_interp}",
        "",
        "| regime | expected behavior |",
        "|--------|------------------|",
        "| BEAR/STRESS (<−3%) | Model outperforms — quality dispersion high |",
        "| NEUTRAL (−3% to +3%) | Baseline alpha expected |",
        "| MILD_RALLY (3–8%) | May lag momentum/junk — KNOWN PATTERN, not failure |",
        "| STRONG_RALLY (>8%) | Watch beta/tail — XBI move may overwhelm selection |",
        "",
        "---",
        "",
        "## Conditional Risk Wrapper",
        "",
    ]

    wrapper_badge = "**ACTIVE**" if wrapper_active else "inactive"
    lines.append(f"Status: {wrapper_badge}  |  Drawdown: {roll_xs:>+.2f}pp ({trigger_status})")
    lines.append("")
    lines.append("| condition | threshold | current | triggered |")
    lines.append("|-----------|-----------|---------|-----------|")
    lines.append(
        f"| (A) rolling_4w_xs | ≤ −5.00pp | {conditions.get('rolling_4w_xs_pp', 'n/a'):>+.2f}pp | "
        f"{'YES' if 'cond_A' in active_triggers else 'no'} |"
    )
    lines.append(
        f"| (B) repeat_offender_count | ≥ 2 | {conditions.get('repeat_offender_count', 0)} | "
        f"{'YES' if 'cond_B' in active_triggers else 'no'} |"
    )
    lines.append(
        f"| (C) ees_false_top30 | ≥ 5 | {conditions.get('ees_false_count', 0)} | "
        f"{'YES' if 'cond_C' in active_triggers else 'no'} |"
    )
    lines.append(
        f"| (D) build+less_binary | ≥ 55% | {conditions.get('build_less_binary_pct', 0):.1f}% | "
        f"{'YES' if 'cond_D' in active_triggers else 'no'} |"
    )
    lines.append("")

    if wrapper_active:
        lines += [
            "### Stress Basket Comparison (current window, returns pending)",
            "",
            "When returns are available, the table below will fill. Compare risk_guarded vs raw_top30.",
            "",
        ]

    # Stress wrapper historical summary (active windows only)
    if not sw_df.empty:
        active_sw = sw_df[sw_df["wrapper_active"]] if "wrapper_active" in sw_df.columns else pd.DataFrame()
        passive_sw = sw_df[~sw_df["wrapper_active"]] if "wrapper_active" in sw_df.columns else pd.DataFrame()
        lines += [
            "### Stress Wrapper Returns History",
            "",
            f"Completed windows: **{len(sw_df)}**  |  Active: **{len(active_sw)}**  |  Passive: **{len(passive_sw)}**",
            "",
            "| basket | all avg_xs | active avg_xs | passive avg_xs |",
            "|--------|-----------|--------------|----------------|",
        ]
        xs_cols = [c for c in sw_df.columns if c.startswith("xs_")]
        for c in xs_cols:
            bname = c[3:]
            all_avg = sw_df[c].mean() if not sw_df.empty else float("nan")
            act_avg = active_sw[c].mean() if not active_sw.empty else float("nan")
            pas_avg = passive_sw[c].mean() if not passive_sw.empty else float("nan")
            lines.append(
                f"| {bname} | {all_avg:>+.2f}pp | "
                f"{'n/a' if active_sw.empty else f'{act_avg:>+.2f}pp'} | "
                f"{'n/a' if passive_sw.empty else f'{pas_avg:>+.2f}pp'} |"
            )
        lines.append("")

        if "xs_raw_top30" in sw_df.columns and "xs_risk_guarded" in sw_df.columns and not active_sw.empty:
            delta = active_sw["xs_risk_guarded"].mean() - active_sw["xs_raw_top30"].mean()
            lines.append(f"risk_guarded vs raw in ACTIVE windows: **{delta:>+.2f}pp**")
            lines.append("")
    else:
        lines += ["No completed stress wrapper windows yet.", ""]

    # ── Repeat-offender section ──────────────────────────────────────────────
    flagged = [r for r in offender_table if r["flagged"]]
    watch = [r for r in offender_table if not r["flagged"] and r["bottom5_windows"] >= 1]

    lines += [
        "---",
        "",
        "## Repeat-Offender Table",
        "",
        f"Flagged (≥2 bottom-5 windows): **{len(flagged)}**  |  Watch: **{len(watch)}**",
        "",
        "| ticker | rank | weeks_top30 | neg_windows | btm5_windows | cum_xs | ees | cat_bucket | replacement |",
        "|--------|------|-------------|-------------|--------------|--------|-----|------------|-------------|",
    ]

    for row in flagged + watch:
        repl = row["replacement"]
        repl_str = f"{repl['ticker']} (r{repl['rank']})" if repl else "—"
        flag_str = " **F**" if row["flagged"] else ""
        lines.append(
            f"| {row['ticker']}{flag_str} | {row['rank']} | {row['weeks_in_top30']} | "
            f"{row['neg_contribution_windows']} | {row['bottom5_windows']} | "
            f"{row['cumulative_xs_pp']:>+.3f}pp | {row['ees_gate']} | "
            f"{row['catalyst_bucket']} | {repl_str} |"
        )

    if not flagged and not watch:
        lines += ["No repeat-offender history available.", ""]
    else:
        lines.append("")

    # ── EES summary ──────────────────────────────────────────────────────────
    lines += ["---", "", "## EES-Guarded Shadow", ""]
    if not ees_df.empty:
        n = len(ees_df)
        avg_delta = ees_df["delta_pp"].mean()
        pct_pos = (ees_df["delta_pp"] > 0).mean() * 100

        # By window quality
        val_path = REPO / "artifacts/autopsy/top30_ytd_validation/2026_ytd_top30_validation.json"
        if val_path.exists():
            val = json.loads(val_path.read_text())
            val_rows = val.get("all_snapshots") or val.get("weekly_snapshots", [])
            val_df = pd.DataFrame(val_rows)
            val_df["snap_date"] = val_df["snap_date"].astype(str)
            val_df["xs_5d_pct"] = val_df["xs_5d"] * 100
            merged = val_df[["snap_date", "xs_5d_pct"]].merge(
                ees_df[["snap_date", "delta_pp"]], on="snap_date", how="inner"
            )
            merged["tier"] = merged["xs_5d_pct"].apply(
                lambda x: "success" if x > 1.0 else ("failure" if x < -1.0 else "neutral")
            )
            lines.append("EES delta by window quality (EES is insurance, not unconditional alpha):")
            lines.append("")
            lines.append("| quality | n | avg_delta |")
            lines.append("|---------|---|-----------|")
            for tier in ["success", "neutral", "failure"]:
                grp = merged[merged["tier"] == tier]
                if grp.empty:
                    continue
                lines.append(f"| {tier} | {len(grp)} | {grp['delta_pp'].mean():>+.2f}pp |")
            lines.append("")

        lines += [
            f"Overall: n={n}  avg_delta={avg_delta:>+.2f}pp  pct_positive={pct_pos:.0f}%",
            f"EES-False in current Top-30: **{ees_false_n}** (flag if ≥5)",
            "",
            "Promotion gate: 20w, +delta in failure windows, does NOT reduce full-period net return.",
            "",
        ]
    else:
        lines += ["No completed EES windows yet.", ""]

    # ── Rank-depth summary ───────────────────────────────────────────────────
    lines += ["---", "", "## Rank-Depth Bench", ""]
    if not rd_df.empty:
        bench_delta = rd_df["xs_rank31_60"].mean() - rd_df["xs_top30"].mean() if "xs_rank31_60" in rd_df.columns else 0
        verdict = (
            "PERSISTENT — review rank cutoff"
            if bench_delta > 0.5
            else "FAILURE-SPECIFIC (expected — bench is a stress replacement pool)"
        )
        lines += [
            f"Completed windows: **{len(rd_df)}**",
            f"Bench vs Top-30 avg delta: **{bench_delta:>+.2f}pp** → {verdict}",
            "",
        ]
    else:
        lines += ["No completed rank-depth windows yet.", ""]

    # ── Catalyst concentration ────────────────────────────────────────────────
    lines += ["---", "", "## Current Catalyst Concentration", ""]
    if cat_summary:
        lines.append("| bucket | n | flag |")
        lines.append("|--------|---|------|")
        for b, n in sorted(cat_summary.items(), key=lambda x: -x[1]):
            flag = "ELEVATED" if b in ("build_window", "less_binary") and n >= 10 else ""
            lines.append(f"| {b} | {n} | {flag} |")
        lines.append("")

    # ── Promotion gate status ─────────────────────────────────────────────────
    sw_active_n = 0
    if not sw_df.empty and "wrapper_active" in sw_df.columns:
        sw_active_n = int(sw_df["wrapper_active"].sum())

    lines += [
        "---",
        "",
        "## Promotion Gate Status",
        "",
        "| gate | metric | windows | threshold | status |",
        "|------|--------|---------|-----------|--------|",
        f"| EES guardrail | delta in failure windows | {len(ees_df)} | 20w | {'GATE_MET (count)' if len(ees_df) >= 20 else 'ACCUMULATING'} |",
        f"| Rank-depth replacement | bench_delta | {len(rd_df)} | 20w | {'GATE_MET (count)' if len(rd_df) >= 20 else 'ACCUMULATING'} |",
        f"| Stress wrapper | active-window delta | {sw_active_n} | 20 active windows | {'ACCUMULATING'} |",
        f"| Repeat-offender | serial underperformance | {len(flagged)} flagged | persistence evidence | ACCUMULATING |",
        "",
        "No production selector/ranker/sizing change authorized until gates are met and operator reviews.",
        "",
        f"*Generated: {snap_date}*",
    ]

    card_text = "\n".join(lines)

    # Write outputs
    md_path = OUTDIR / f"STRESS_WRAPPER_CARD_{snap_date}.md"
    with open(md_path, "w") as f:
        f.write(card_text)

    json_path = OUTDIR / f"stress_wrapper_card_{snap_date}.json"
    summary = {
        "as_of": snap_date,
        "regime": regime_label,
        "xbi_trailing_20d_pct": xbi_trail,
        "risk_wrapper_active": wrapper_active,
        "active_triggers": active_triggers,
        "drawdown_status": trigger_status,
        "rolling_xs_pp": roll_xs,
        "n_flagged_offenders": len(flagged),
        "n_watch_offenders": len(watch),
        "ees_windows": len(ees_df),
        "ees_avg_delta": round(float(ees_df["delta_pp"].mean()), 3) if not ees_df.empty else None,
        "ees_false_in_top30": ees_false_n,
        "rd_windows": len(rd_df),
        "sw_total_windows": len(sw_df),
        "sw_active_windows": sw_active_n,
        "investability": "PILOT_INVESTABLE_WITH_SHADOW_GUARDS",
        "classification": "VALIDATION_INFRASTRUCTURE/STRESS_WRAPPER_SHADOW/NO_MODEL_CHANGE",
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"Stress Wrapper Card: {md_path}")
    print(f"JSON summary:        {json_path}")
    print()
    print(f"  Regime:           {regime_label} (XBI {xbi_str})")
    print(f"  Risk wrapper:     {'ACTIVE — ' + str(active_triggers) if wrapper_active else 'inactive'}")
    print(f"  Drawdown:         {roll_xs:>+.2f}pp ({trigger_status})")
    print(f"  Repeat offenders: {len(flagged)} flagged, {len(watch)} watch")
    print(f"  EES windows:      {len(ees_df)} / 20")
    print(f"  SW active wins:   {sw_active_n} / 20")
    print("  Investability:    PILOT_INVESTABLE_WITH_SHADOW_GUARDS")


def main():
    parser = argparse.ArgumentParser(description="Stress wrapper weekly card")
    parser.add_argument("--snap-date", default=str(date.today()))
    args = parser.parse_args()
    run(args.snap_date)


if __name__ == "__main__":
    main()
