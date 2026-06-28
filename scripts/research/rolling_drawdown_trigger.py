"""
Rolling 4-Window XS Drawdown Trigger
VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

Checks 4-week rolling excess return. If <= -5.00pp:
  - emits a TRIGGER alert
  - prints repeat-offender summary
  - prints EES-guarded counterfactual from the most recent filled window
  - requires operator review before rebalance expansion

Usage:
    python3 scripts/research/rolling_drawdown_trigger.py [--as-of YYYY-MM-DD]
"""

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "artifacts/failure_mode_guards"
VAL_JSON = REPO / "artifacts/autopsy/top30_ytd_validation/2026_ytd_top30_validation.json"

DRAWDOWN_THRESHOLD = -5.00  # pp


def load_validation_windows() -> pd.DataFrame:
    if not VAL_JSON.exists():
        return pd.DataFrame()
    with open(VAL_JSON) as f:
        data = json.load(f)
    rows = data.get("all_snapshots") or data.get("weekly_snapshots", [])
    df = pd.DataFrame(rows)
    if "xs_5d" in df.columns:
        df["xs_pp"] = df["xs_5d"] * 100
    return df


def rolling_4w_xs(df: pd.DataFrame, as_of: str) -> dict:
    """Compute rolling 4-window XS sum for the 4 most recent completed windows."""
    completed = df[df["fwd_date"] <= as_of].sort_values("snap_date")
    if len(completed) < 4:
        return {"n_windows": len(completed), "rolling_xs": None, "windows": []}

    last4 = completed.tail(4)
    roll_xs = last4["xs_pp"].sum()
    windows = last4[["snap_date", "fwd_date", "xs_pp"]].to_dict("records")
    return {
        "n_windows": 4,
        "rolling_xs": round(roll_xs, 2),
        "windows": windows,
    }


def print_repeat_offender_summary(as_of: str) -> None:
    # Find most recent repeat-offender output file
    files = sorted(OUTDIR.glob("repeat_offender_*.json"), reverse=True)
    if not files:
        print("  No repeat-offender output found — run repeat_offender_monitor.py first.")
        return
    with open(files[0]) as f:
        ro = json.load(f)
    flagged = ro.get("flagged", [])
    print(f"\n  REPEAT OFFENDERS (from {files[0].name}):")
    if flagged:
        for f in flagged:
            repl = f.get("replacement_ticker", "N/A")
            print(
                f"    {f['ticker']:<8} rank={f['current_rank']:>2}  "
                f"bad_windows={f['n_recent_bottom_windows']}  "
                f"contrib={f['total_contrib_in_bad_windows']:>+6.3f}pp  "
                f"ees={f['ees_gate']}  → replace with {repl}"
            )
    else:
        print("    None currently flagged.")


def print_ees_guarded_summary() -> None:
    returns_log = OUTDIR / "ees_guarded_returns.jsonl"
    if not returns_log.exists():
        print("  No EES-guarded returns yet — run ees_guarded_shadow.py --fill-returns first.")
        return

    df = pd.read_json(returns_log, lines=True)
    if df.empty:
        return

    recent = df.tail(4)
    print(f"\n  EES-GUARDED SHADOW (last {len(recent)} completed windows):")
    for _, row in recent.iterrows():
        print(
            f"    {row['snap_date']}: raw_xs={row['raw_xs']:>+5.2f}pp  "
            f"guarded_xs={row['guarded_xs']:>+5.2f}pp  "
            f"delta={row['delta_pp']:>+5.2f}pp"
        )
    print(f"\n  All-time avg delta: {df['delta_pp'].mean():>+5.2f}pp  " f"({len(df)} windows)")


def run(as_of: str) -> None:
    print(f"Rolling Drawdown Trigger — as_of={as_of}")
    print("=" * 64)

    df = load_validation_windows()
    if df.empty:
        print("ERROR: Could not load validation windows")
        return

    result = rolling_4w_xs(df, as_of)

    if result["rolling_xs"] is None:
        print(f"Insufficient completed windows ({result['n_windows']}) — need 4.")
        return

    roll_xs = result["rolling_xs"]
    status = "TRIGGERED" if roll_xs <= DRAWDOWN_THRESHOLD else "OK"

    print(f"\nROLLING 4-WINDOW XS: {roll_xs:>+6.2f}pp  (threshold: {DRAWDOWN_THRESHOLD:>+.2f}pp)  [{status}]")
    print("\nRecent 4 windows:")
    for w in result["windows"]:
        flag = " ←" if w["xs_pp"] <= -1.5 else ""
        print(f"  {w['snap_date']} → {w['fwd_date']}: xs={w['xs_pp']:>+5.2f}pp{flag}")

    if status == "TRIGGERED":
        print(f"\n{'!'*64}")
        print("DRAWDOWN TRIGGER FIRED — OPERATOR REVIEW REQUIRED BEFORE REBALANCE EXPANSION")
        print(f"{'!'*64}")
        print_repeat_offender_summary(as_of)
        print_ees_guarded_summary()
        print("\nActions:")
        print("  1. Run: python3 scripts/research/repeat_offender_monitor.py")
        print("  2. Run: python3 scripts/research/ees_guarded_shadow.py --fill-returns")
        print("  3. Run: python3 scripts/research/rank_depth_replacement_shadow.py --summary")
        print("  4. Run: /biotech-autopsy (if new failure windows to analyze)")
        print("  5. Require operator authorization before any rebalance expansion")

    # Write trigger record
    record = {
        "as_of": as_of,
        "rolling_xs_pp": roll_xs,
        "threshold_pp": DRAWDOWN_THRESHOLD,
        "status": status,
        "windows": result["windows"],
        "classification": "VALIDATION_INFRASTRUCTURE/NO_MODEL_CHANGE",
    }
    out = OUTDIR / f"drawdown_trigger_{as_of}.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nRecord: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=str(date.today()))
    args = parser.parse_args()
    run(args.as_of)


if __name__ == "__main__":
    main()
