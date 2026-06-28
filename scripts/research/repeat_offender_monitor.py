"""
Repeat-Offender Persistence Monitor
VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

Surfaces names currently in Top-30 that have been bottom contributors
in >=2 recent completed forward-return windows. Track only — no veto.

Usage:
    python3 scripts/research/repeat_offender_monitor.py [--snap-date YYYY-MM-DD]
"""

import argparse
import glob
import json
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "artifacts/failure_mode_guards"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Historical attribution seed (from YTD post-mortem)
SEED_CSV = REPO / "artifacts/autopsy/ytd_top30_failure_postmortem_2026/name_attribution.csv"
# Rolling ledger — appended each run
LEDGER = OUTDIR / "repeat_offender_ledger.jsonl"


def find_latest_snap(target: str) -> str:
    dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    eligible = [Path(d).name for d in dirs if len(Path(d).name) == 10 and Path(d).name <= target]
    return eligible[-1] if eligible else None


def load_attribution_history() -> pd.DataFrame:
    """Load historical per-name attribution from seed + rolling ledger."""
    frames = []

    if SEED_CSV.exists():
        df = pd.read_csv(SEED_CSV)
        df["source"] = "seed_ytd_postmortem"
        frames.append(df)

    if LEDGER.exists():
        rows = []
        with open(LEDGER) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Deduplicate — prefer ledger over seed for same (window, ticker)
    df = df.drop_duplicates(subset=["window", "ticker"], keep="last")
    return df


def bottom_contributor_windows(df: pd.DataFrame, ticker: str, n_recent: int = 10) -> list:
    """
    Return list of windows where ticker was in the bottom 5 contributors
    (by contrib_to_xs) across the most recent n_recent completed windows.
    """
    if df.empty or "contrib_to_xs" not in df.columns:
        return []

    # Get all windows in recency order
    all_windows = sorted(df["window"].unique(), reverse=True)[:n_recent]
    bad_windows = []

    for w in all_windows:
        wdf = df[df["window"] == w].dropna(subset=["contrib_to_xs"])
        if wdf.empty:
            continue
        bottom5 = wdf.nsmallest(5, "contrib_to_xs")["ticker"].tolist()
        if ticker in bottom5:
            bad_windows.append(w)

    return bad_windows


def load_current_snapshot(snap_date: str) -> pd.DataFrame:
    path = REPO / f"data/snapshots/{snap_date}/rankings.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["ticker"] = df["ticker"].astype(str)
    return df


def find_replacement_from_bench(ticker: str, df_snap: pd.DataFrame) -> dict:
    """Find next-best 31-60 candidate not itself flagged."""
    bench = df_snap[(df_snap["actionable_rank"] > 30) & (df_snap["actionable_rank"] <= 60)].sort_values(
        "actionable_rank"
    )
    if bench.empty:
        return {}
    # First bench name with ees_v3_gate != False (prefer clean replacements)
    for _, row in bench.iterrows():
        ees = row.get("ees_v3_gate")
        if str(ees).lower() not in ("false", "0"):
            return {
                "replacement_ticker": row["ticker"],
                "replacement_rank": int(row["actionable_rank"]),
                "replacement_ees": str(ees),
                "replacement_catalyst_bucket": row.get("catalyst_bucket", "?"),
            }
    # Fallback: first bench name
    row = bench.iloc[0]
    return {
        "replacement_ticker": row["ticker"],
        "replacement_rank": int(row["actionable_rank"]),
        "replacement_ees": str(row.get("ees_v3_gate", "?")),
        "replacement_catalyst_bucket": row.get("catalyst_bucket", "?"),
    }


def run(snap_date: str) -> None:
    actual_snap = find_latest_snap(snap_date)
    if not actual_snap:
        print(f"ERROR: No snapshot found at or before {snap_date}")
        return

    print(f"Repeat-Offender Monitor — {snap_date} (snapshot: {actual_snap})")
    print("=" * 64)

    hist = load_attribution_history()
    df_snap = load_current_snapshot(actual_snap)

    if df_snap.empty:
        print("ERROR: Could not load current snapshot rankings")
        return

    top30 = df_snap[df_snap["actionable_rank"] <= 30].copy()
    print(f"Current Top-30: {len(top30)} names  |  Attribution history: {len(hist)} rows")

    # Evaluate each Top-30 name
    flags = []
    for _, row in top30.iterrows():
        ticker = row["ticker"]
        bad_wins = bottom_contributor_windows(hist, ticker, n_recent=12)
        n_bad = len(bad_wins)

        if n_bad == 0:
            continue

        total_contrib = 0.0
        if not hist.empty and "contrib_to_xs" in hist.columns:
            mask = (hist["ticker"] == ticker) & hist["window"].isin(bad_wins)
            total_contrib = hist.loc[mask, "contrib_to_xs"].sum()

        ees = row.get("ees_v3_gate")
        cat = row.get("catalyst_bucket", "?")
        rank = int(row["actionable_rank"])

        flag = {
            "as_of_date": snap_date,
            "snap_used": actual_snap,
            "ticker": ticker,
            "current_rank": rank,
            "n_recent_bottom_windows": n_bad,
            "total_contrib_in_bad_windows": round(total_contrib, 3),
            "recent_bad_windows": bad_wins,
            "ees_gate": str(ees),
            "catalyst_bucket": cat,
            "tier": str(row.get("tier_any", "?")),
            "flagged": n_bad >= 2,
        }
        flag.update(find_replacement_from_bench(ticker, df_snap))
        flags.append(flag)

    # Sort by severity
    flags.sort(key=lambda x: (-(x["n_recent_bottom_windows"]), x["total_contrib_in_bad_windows"]))

    flagged = [f for f in flags if f["flagged"]]
    watch = [f for f in flags if not f["flagged"]]

    print(f"\nFLAGGED (>=2 recent bottom-5 windows): {len(flagged)}")
    for f in flagged:
        print(
            f"  {f['ticker']:<8} rank={f['current_rank']:>2}  "
            f"bad_windows={f['n_recent_bottom_windows']}  "
            f"total_contrib={f['total_contrib_in_bad_windows']:>+6.3f}pp  "
            f"ees={f['ees_gate']}  cat={f['catalyst_bucket']}"
        )
        repl = f.get("replacement_ticker")
        if repl:
            print(
                f"           → replacement candidate: {repl} "
                f"(rank {f.get('replacement_rank')}, ees={f.get('replacement_ees')}, "
                f"cat={f.get('replacement_catalyst_bucket')})"
            )

    print(f"\nWATCH (1 recent bottom-5 window): {len(watch)}")
    for f in watch:
        print(
            f"  {f['ticker']:<8} rank={f['current_rank']:>2}  "
            f"bad_windows={f['n_recent_bottom_windows']}  "
            f"total_contrib={f['total_contrib_in_bad_windows']:>+6.3f}pp  "
            f"ees={f['ees_gate']}  cat={f['catalyst_bucket']}"
        )

    # Write output card
    out_path = OUTDIR / f"repeat_offender_{snap_date}.json"
    output = {
        "as_of_date": snap_date,
        "snap_used": actual_snap,
        "n_top30": len(top30),
        "n_flagged": len(flagged),
        "n_watch": len(watch),
        "flagged": flagged,
        "watch": watch,
        "classification": "VALIDATION_INFRASTRUCTURE/NO_MODEL_CHANGE",
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nOutput: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Repeat-offender persistence monitor")
    parser.add_argument(
        "--snap-date",
        default=str(date.today()),
        help="Snapshot date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    run(args.snap_date)


if __name__ == "__main__":
    main()
