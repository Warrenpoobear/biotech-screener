"""
Weekly Failure Mode Card
VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

Combines all four shadow monitors into a single weekly output card.
Meant to be run once per week (e.g. Monday morning) after each snapshot.

Usage:
    python3 scripts/research/weekly_failure_mode_card.py [--snap-date YYYY-MM-DD]

Output:
    artifacts/failure_mode_guards/FAILURE_MODE_CARD_YYYY-MM-DD.md
    artifacts/failure_mode_guards/failure_mode_card_YYYY-MM-DD.json
"""

import argparse
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


def run_script(script: str, args: list = None) -> tuple[int, str]:
    cmd = [sys.executable, str(SCRIPTS / script)] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return result.returncode, result.stdout + result.stderr


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def run(snap_date: str) -> None:
    print(f"Weekly Failure Mode Card — {snap_date}")
    print("=" * 64)

    # ── 1. Repeat-offender monitor ──────────────────────────────────────────
    print("\n[1/4] Running repeat-offender monitor...")
    rc, out = run_script("repeat_offender_monitor.py", ["--snap-date", snap_date])
    print(out[:1000])
    ro_data = load_json(OUTDIR / f"repeat_offender_{snap_date}.json")

    # ── 2. EES-guarded shadow — log baskets ─────────────────────────────────
    print("\n[2/4] Logging EES-guarded baskets...")
    rc, out = run_script("ees_guarded_shadow.py", ["--snap-date", snap_date])
    print(out[:800])

    # Fill returns for any ready windows
    rc, out = run_script("ees_guarded_shadow.py", ["--fill-returns"])
    print(out[:800])

    # ── 3. Rank-depth replacement shadow ────────────────────────────────────
    print("\n[3/4] Logging rank-depth baskets...")
    rc, out = run_script("rank_depth_replacement_shadow.py", ["--snap-date", snap_date])
    print(out[:800])

    rc, out = run_script("rank_depth_replacement_shadow.py", ["--fill-returns"])
    print(out[:800])

    # ── 4. Rolling drawdown trigger ─────────────────────────────────────────
    print("\n[4/4] Checking rolling drawdown trigger...")
    rc, out = run_script("rolling_drawdown_trigger.py", ["--as-of", snap_date])
    print(out[:1200])
    trigger_data = load_json(OUTDIR / f"drawdown_trigger_{snap_date}.json")

    # ── Load returns data for summary ───────────────────────────────────────
    ees_returns = pd.DataFrame()
    ees_log = OUTDIR / "ees_guarded_returns.jsonl"
    if ees_log.exists():
        ees_returns = pd.read_json(ees_log, lines=True)

    rd_returns = pd.DataFrame()
    rd_log = OUTDIR / "rank_depth_returns.jsonl"
    if rd_log.exists():
        rd_returns = pd.read_json(rd_log, lines=True)

    # ── Compute catalyst bucket snapshot ────────────────────────────────────
    import glob

    snap_dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    eligible = [Path(d).name for d in snap_dirs if len(Path(d).name) == 10 and Path(d).name <= snap_date]
    cat_bucket_summary = {}
    if eligible:
        snap = eligible[-1]
        df = pd.read_csv(REPO / f"data/snapshots/{snap}/rankings.csv", low_memory=False)
        top30 = df[df["actionable_rank"] <= 30]
        if "catalyst_bucket" in top30.columns:
            cat_counts = top30["catalyst_bucket"].value_counts()
            cat_bucket_summary = cat_counts.to_dict()
        ees_false_n = 0
        if "ees_v3_gate" in top30.columns:
            ees_false_n = int(top30["ees_v3_gate"].eq(False).sum())

    # ── Build the card ───────────────────────────────────────────────────────
    flagged = ro_data.get("flagged", [])
    watch = ro_data.get("watch", [])
    roll_xs = trigger_data.get("rolling_xs_pp")
    trigger_st = trigger_data.get("status", "UNKNOWN")

    lines = [
        f"# Failure Mode Card — {snap_date}",
        "",
        "**Classification:** VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE",
        "",
        "---",
        "",
        "## Rolling Drawdown Status",
        "",
        f"4-window rolling XS: **{roll_xs:>+.2f}pp** (threshold: −5.00pp)  [{trigger_st}]",
        "",
    ]

    if trigger_st == "TRIGGERED":
        lines += [
            "> ⚠️  DRAWDOWN TRIGGER FIRED — Operator review required before rebalance expansion.",
            "",
        ]

    if trigger_data.get("windows"):
        lines.append("| window | xs_pp |")
        lines.append("|--------|-------|")
        for w in trigger_data["windows"]:
            flag = " **←**" if w["xs_pp"] <= -1.5 else ""
            lines.append(f"| {w['snap_date']} → {w['fwd_date']} | {w['xs_pp']:>+.2f}pp{flag} |")
        lines.append("")

    lines += [
        "## Repeat-Offender Flags",
        "",
        f"Flagged (≥2 recent bottom-5 windows): **{len(flagged)}**  |  Watch (1 window): **{len(watch)}**",
        "",
    ]

    if flagged:
        lines.append("| ticker | rank | bad_windows | total_contrib | ees | replacement |")
        lines.append("|--------|------|-------------|---------------|-----|-------------|")
        for f in flagged:
            repl = f.get("replacement_ticker", "—")
            lines.append(
                f"| {f['ticker']} | {f['current_rank']} | {f['n_recent_bottom_windows']} | "
                f"{f['total_contrib_in_bad_windows']:>+.3f}pp | {f['ees_gate']} | {repl} |"
            )
        lines.append("")
    else:
        lines += ["No names currently flagged.", ""]

    # EES-guarded summary
    lines += ["## EES-Guarded Shadow", ""]
    if not ees_returns.empty:
        n = len(ees_returns)
        avg_delta = ees_returns["delta_pp"].mean()
        pct_pos = (ees_returns["delta_pp"] > 0).mean() * 100
        lines += [
            f"Completed windows: **{n}** / 20 (promotion gate)",
            f"  - avg raw_xs: {ees_returns['raw_xs'].mean():>+.2f}pp",
            f"  - avg guarded_xs: {ees_returns['guarded_xs'].mean():>+.2f}pp",
            f"  - avg delta: **{avg_delta:>+.2f}pp**",
            f"  - pct delta > 0: {pct_pos:.0f}%",
            "",
            "Promotion gate: 20 windows, +delta, ≥60% positive, no degradation in normal weeks.",
            "",
        ]
    else:
        lines += ["No completed EES-guarded windows yet.", ""]

    # Rank-depth summary
    lines += ["## Rank-Depth Replacement Shadow", ""]
    if not rd_returns.empty:
        n = len(rd_returns)
        xs_cols = [c for c in rd_returns.columns if c.startswith("xs_")]
        lines.append(f"Completed windows: **{n}**")
        lines.append("")
        lines.append("| basket | avg_xs | pct_positive |")
        lines.append("|--------|--------|-------------|")
        for c in xs_cols:
            bname = c[3:]
            avg = rd_returns[c].mean()
            pct = (rd_returns[c] > 0).mean() * 100
            lines.append(f"| {bname} | {avg:>+.2f}pp | {pct:.0f}% |")
        lines.append("")

        if "xs_top30" in rd_returns.columns and "xs_rank31_60" in rd_returns.columns:
            bench_delta = rd_returns["xs_rank31_60"].mean() - rd_returns["xs_top30"].mean()
            verdict = "PERSISTENT — review rank cutoff" if bench_delta > 0.5 else "FAILURE-SPECIFIC (expected)"
            lines += [f"Bench vs Top-30 avg delta: {bench_delta:>+.2f}pp → **{verdict}**", ""]
    else:
        lines += ["No completed rank-depth windows yet.", ""]

    # Current catalyst bucket concentration
    lines += ["## Current Catalyst Concentration", ""]
    if cat_bucket_summary:
        lines.append("| bucket | n_names |")
        lines.append("|--------|---------|")
        for b, n in sorted(cat_bucket_summary.items(), key=lambda x: -x[1]):
            flag = " ⚠️" if b in ("build_window", "less_binary") and n >= 10 else ""
            lines.append(f"| {b} | {n}{flag} |")
        lines.append("")
        if "ees_false_n" in dir():
            lines.append(f"EES-False names in Top-30: **{ees_false_n}** (flag if ≥5)")
        lines.append("")

    lines += [
        "---",
        "",
        "## Promotion Gate Status",
        "",
        "| gate | evidence_needed | windows_completed | status |",
        "|------|----------------|-------------------|--------|",
        f"| EES guardrail | 20w +delta ≥60% | {len(ees_returns)} | {'GATE_MET' if len(ees_returns) >= 20 else 'ACCUMULATING'} |",
        "| Repeat-offender warning | 10w serial underperformance | — | ACCUMULATING |",
        f"| Rank-depth replacement | 20w +delta | {len(rd_returns)} | {'GATE_MET' if len(rd_returns) >= 20 else 'ACCUMULATING'} |",
        "",
        "No production change authorized until gates are met and operator reviews.",
        "",
        f"*Generated: {snap_date} — VALIDATION_INFRASTRUCTURE / NO_MODEL_CHANGE*",
    ]

    card_text = "\n".join(lines)

    # Write markdown card
    md_path = OUTDIR / f"FAILURE_MODE_CARD_{snap_date}.md"
    with open(md_path, "w") as f:
        f.write(card_text)

    # Write JSON summary
    json_path = OUTDIR / f"failure_mode_card_{snap_date}.json"
    summary = {
        "as_of": snap_date,
        "drawdown_trigger": trigger_st,
        "rolling_xs_pp": roll_xs,
        "n_flagged_offenders": len(flagged),
        "n_watch_offenders": len(watch),
        "ees_windows_completed": len(ees_returns),
        "ees_avg_delta": round(float(ees_returns["delta_pp"].mean()), 3) if not ees_returns.empty else None,
        "rd_windows_completed": len(rd_returns),
        "cat_bucket_snapshot": cat_bucket_summary,
        "ees_false_in_top30": ees_false_n if cat_bucket_summary else None,
        "classification": "VALIDATION_INFRASTRUCTURE/NO_MODEL_CHANGE",
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*64}")
    print(f"Card written: {md_path}")
    print(f"JSON:         {json_path}")
    print("\nStatus summary:")
    print(f"  Drawdown trigger:   {trigger_st} ({roll_xs:>+.2f}pp)")
    print(f"  Repeat offenders:   {len(flagged)} flagged, {len(watch)} watch")
    print(f"  EES shadow windows: {len(ees_returns)} / 20")
    print(f"  Rank-depth windows: {len(rd_returns)} / 20")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-date", default=str(date.today()))
    args = parser.parse_args()
    run(args.snap_date)


if __name__ == "__main__":
    main()
