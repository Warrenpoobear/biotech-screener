"""
DEM Forward Regime Shadow Monitor
VALIDATION_INFRASTRUCTURE / DEM_FORWARD_REGIME_SHADOW_MONITOR / NO_MODEL_CHANGE

Reads forward-validation captures + fills, tags each window as RALLY / NON_RALLY,
and computes:
  1. Regime-conditional XS: does the model beat XBI outside narrow rally periods?
  2. Rank monotonicity: top-10 / top-20 / top-30 sub-basket XS vs XBI
  3. Concentration: % of basket XS explained by top-3 contributors
  4. EES warning: how many current Top-30 names have ees_v3_gate=False

Investment gate (model investability verdict, 2026-06-26):
  Repeatable excess return outside narrow biotech rally cluster.
  Minimum sample: 20 completed windows across regime types.

Usage:
    python3 scripts/research/dem_regime_forward_monitor.py [--as-of YYYY-MM-DD]
    python3 scripts/research/dem_regime_forward_monitor.py --summary

Output:
    artifacts/forward_validation/dem_regime_monitor_{date}.json
    artifacts/forward_validation/DEM_REGIME_MONITOR.md  (rolling summary, overwritten)
"""

import argparse
import glob
import json
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
FV_DIR = REPO / "artifacts/forward_validation"
FV_DIR.mkdir(parents=True, exist_ok=True)

CAPTURES_LEDGER = FV_DIR / "captures.jsonl"
FILLS_LEDGER = FV_DIR / "fills.jsonl"
PRICE_FILE = REPO / "production_data/price_history_split_adj.csv"

# Rally = XBI 20d return (forward from window start) >= +5.0%
RALLY_THRESHOLD_PCT = 5.0
HORIZON_5D = 5
HORIZON_20D = 20

# Concentration flag threshold
CONCENTRATION_WARN_PCT = 60.0  # top-3 names explain >60% of absolute XS contribution


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def find_snap(target: str) -> str | None:
    dirs = sorted(glob.glob(str(REPO / "data/snapshots/????-??-??")))
    eligible = [Path(d).name for d in dirs if len(Path(d).name) == 10 and Path(d).name <= target]
    return eligible[-1] if eligible else None


def load_snap(snap_date: str) -> pd.DataFrame:
    p = REPO / f"data/snapshots/{snap_date}/rankings.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df["ticker"] = df["ticker"].astype(str)
    return df


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def xbi_trailing_20d_return(prices: pd.DataFrame, as_of: str, n: int = 20) -> float | None:
    """Trailing 20-trading-day XBI return ending on as_of date.

    Uses prior context only (what you knew before the forward window started).
    RALLY = XBI trailed >= +5% over last 20 trading days.
    """
    if "XBI" not in prices.columns:
        return None
    xbi = prices["XBI"].dropna()
    prior = xbi.loc[:as_of]
    if len(prior) < n + 1:
        return None
    end_px = prior.iloc[-1]
    start_px = prior.iloc[-(n + 1)]
    if start_px <= 0:
        return None
    return (end_px - start_px) / start_px * 100


def basket_xs_subsets(
    tickers_ranked: list[dict], start: str, end: str, prices: pd.DataFrame
) -> dict[str, float | None]:
    """Compute EW XS for top-10, top-20, top-30 sub-baskets."""
    if "XBI" not in prices.columns:
        return {}
    xbi = prices["XBI"].dropna()
    xbi_s = xbi.loc[:start].iloc[-1] if not xbi.loc[:start].empty else None
    xbi_e = xbi.loc[:end].iloc[-1] if not xbi.loc[:end].empty else None
    if xbi_s is None or xbi_e is None:
        return {}
    xbi_ret = (xbi_e - xbi_s) / xbi_s

    def subbasket_xs(tickers: list[str]) -> float | None:
        rets = []
        for t in tickers:
            if t not in prices.columns:
                continue
            p = prices[t].dropna()
            s = p.loc[:start].iloc[-1] if not p.loc[:start].empty else None
            e = p.loc[:end].iloc[-1] if not p.loc[:end].empty else None
            if s and e and s > 0:
                rets.append((e - s) / s)
        return (np.mean(rets) - xbi_ret) * 100 if rets else None

    sorted_tickers = sorted(tickers_ranked, key=lambda x: x.get("rank", 999))
    t30 = [x["ticker"] for x in sorted_tickers[:30]]
    t20 = t30[:20]
    t10 = t30[:10]

    return {
        "top10": subbasket_xs(t10),
        "top20": subbasket_xs(t20),
        "top30": subbasket_xs(t30),
    }


def concentration_score(tickers_ranked: list[dict], start: str, end: str, prices: pd.DataFrame) -> dict:
    """Compute % of basket absolute XS attributable to top-3 contributors."""
    if "XBI" not in prices.columns:
        return {}
    xbi = prices["XBI"].dropna()
    xbi_s = xbi.loc[:start].iloc[-1] if not xbi.loc[:start].empty else None
    xbi_e = xbi.loc[:end].iloc[-1] if not xbi.loc[:end].empty else None
    if xbi_s is None or xbi_e is None:
        return {}
    xbi_ret = (xbi_e - xbi_s) / xbi_s

    n = min(30, len(tickers_ranked))
    contribs = []
    for item in sorted(tickers_ranked, key=lambda x: x.get("rank", 999))[:30]:
        t = item["ticker"]
        if t not in prices.columns:
            continue
        p = prices[t].dropna()
        s = p.loc[:start].iloc[-1] if not p.loc[:start].empty else None
        e = p.loc[:end].iloc[-1] if not p.loc[:end].empty else None
        if s and e and s > 0:
            xs = ((e - s) / s - xbi_ret) / n * 100
            contribs.append({"ticker": t, "contrib_xs": round(xs, 3)})

    if not contribs:
        return {}

    contribs.sort(key=lambda x: abs(x["contrib_xs"]), reverse=True)
    total_abs = sum(abs(c["contrib_xs"]) for c in contribs)
    top3_abs = sum(abs(c["contrib_xs"]) for c in contribs[:3])
    top3_pct = round(top3_abs / total_abs * 100, 1) if total_abs > 0 else 0.0

    return {
        "top3_names": [c["ticker"] for c in contribs[:3]],
        "top3_abs_pct": top3_pct,
        "concentrated": top3_pct >= CONCENTRATION_WARN_PCT,
        "n_contributors": len(contribs),
    }


def ees_warning_for_snap(snap_date: str) -> dict:
    actual = find_snap(snap_date)
    if not actual:
        return {}
    df = load_snap(actual)
    if df.empty or "ees_v3_gate" not in df.columns:
        return {"has_ees": False}
    top30 = df[df["actionable_rank"] <= 30]
    n_false = int(top30["ees_v3_gate"].eq(False).sum())
    return {
        "has_ees": True,
        "snap_used": actual,
        "n_ees_false_in_top30": n_false,
        "ees_warn": n_false >= 5,
    }


def build_window_records(prices: pd.DataFrame) -> list[dict]:
    """
    Join captures + fills, compute regime/monotonicity/concentration per window.
    """
    caps = {c["date"]: c for c in load_ledger(CAPTURES_LEDGER)}
    fills = load_ledger(FILLS_LEDGER)
    records = []

    for fill in fills:
        start = fill.get("capture_date") or fill.get("effective_start_date")
        if not start:
            continue
        xs_5d = fill.get("xs_5d")
        if xs_5d is None:
            continue

        end_5d = fill.get("end_date_5d")
        cap = caps.get(start)
        top30_ranked = cap.get("top30", []) if cap else []

        # Regime tag: XBI trailing 20d return at snapshot date (prior context only)
        xbi_20d = xbi_trailing_20d_return(prices, start, 20)
        regime = (
            "RALLY"
            if xbi_20d is not None and xbi_20d >= RALLY_THRESHOLD_PCT
            else "NON_RALLY" if xbi_20d is not None else "UNKNOWN"
        )

        # Monotonicity
        mono = {}
        if end_5d and top30_ranked:
            mono = basket_xs_subsets(top30_ranked, start, end_5d, prices)

        # Concentration
        conc = {}
        if end_5d and top30_ranked:
            conc = concentration_score(top30_ranked, start, end_5d, prices)

        records.append(
            {
                "snap_date": start,
                "end_5d": end_5d,
                "xs_5d_pct": round(xs_5d * 100, 3),
                "xbi_5d_pct": round((fill.get("xbi_5d") or 0) * 100, 3),
                "regime": regime,
                "xbi_20d_pct": round(xbi_20d, 2) if xbi_20d is not None else None,
                "monotonicity": {k: round(v, 3) if v is not None else None for k, v in mono.items()},
                "concentration": conc,
                "n_top30_captured": len(top30_ranked),
                "xs_20d_pct": round((fill.get("xs_20d") or 0) * 100, 3) if fill.get("xs_20d") else None,
            }
        )

    return sorted(records, key=lambda x: x["snap_date"])


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}

    xs = [r["xs_5d_pct"] for r in records]
    n = len(xs)
    avg_xs = round(np.mean(xs), 3)
    pct_pos = round((np.array(xs) > 0).mean() * 100, 1)

    rally = [r for r in records if r["regime"] == "RALLY"]
    non_rally = [r for r in records if r["regime"] == "NON_RALLY"]

    def grp_stats(grp):
        if not grp:
            return None
        v = [r["xs_5d_pct"] for r in grp]
        return {"n": len(v), "avg_xs": round(np.mean(v), 3), "pct_pos": round((np.array(v) > 0).mean() * 100, 1)}

    # Monotonicity: top-10 vs top-20 vs top-30
    mono_top10 = [r["monotonicity"].get("top10") for r in records if r["monotonicity"].get("top10") is not None]
    mono_top20 = [r["monotonicity"].get("top20") for r in records if r["monotonicity"].get("top20") is not None]
    mono_top30 = [r["monotonicity"].get("top30") for r in records if r["monotonicity"].get("top30") is not None]

    mono_avg = {}
    if mono_top10:
        mono_avg["top10"] = round(np.mean(mono_top10), 3)
    if mono_top20:
        mono_avg["top20"] = round(np.mean(mono_top20), 3)
    if mono_top30:
        mono_avg["top30"] = round(np.mean(mono_top30), 3)

    # Concentration: how often is top-3 > 60%?
    conc_windows = [r for r in records if r.get("concentration")]
    conc_pct_flagged = (
        round(sum(1 for r in conc_windows if r["concentration"].get("concentrated")) / len(conc_windows) * 100, 1)
        if conc_windows
        else None
    )

    return {
        "n": n,
        "gate_progress": f"{n}/20",
        "avg_xs_5d_pct": avg_xs,
        "pct_positive": pct_pos,
        "rally": grp_stats(rally),
        "non_rally": grp_stats(non_rally),
        "monotonicity_avg": mono_avg,
        "concentration_pct_flagged": conc_pct_flagged,
        "gate_met": n >= 20 and (grp_stats(non_rally) or {}).get("avg_xs", 0) > 0,
    }


def run(as_of: str) -> None:
    print(f"DEM Regime Forward Monitor — {as_of}")
    print("=" * 64)

    prices = load_prices()
    records = build_window_records(prices)
    summary = summarize(records)
    ees_warn = ees_warning_for_snap(as_of)

    # ── Print regime-conditional table ──────────────────────────────────────
    print(f"\nWindows with filled 5d returns: {summary['n']} / 20 (gate)")
    print(
        f"Overall avg XS: {summary.get('avg_xs_5d_pct', 0):>+.2f}pp  "
        f"({summary.get('pct_positive', 0):.0f}% positive)"
    )

    r = summary.get("rally")
    nr = summary.get("non_rally")
    if r or nr:
        print("\nRegime-conditional breakdown:")
        if r:
            print(f"  RALLY     (n={r['n']:>2}): avg_xs={r['avg_xs']:>+.2f}pp  {r['pct_pos']:.0f}% positive")
        if nr:
            print(f"  NON_RALLY (n={nr['n']:>2}): avg_xs={nr['avg_xs']:>+.2f}pp  {nr['pct_pos']:.0f}% positive")

    # ── Monotonicity ─────────────────────────────────────────────────────────
    mono = summary.get("monotonicity_avg", {})
    if mono:
        print("\nRank monotonicity (avg XS vs XBI):")
        for tier in ["top10", "top20", "top30"]:
            v = mono.get(tier)
            if v is not None:
                print(f"  {tier}: {v:>+.2f}pp")
        # Flag inversion
        t10 = mono.get("top10")
        t30 = mono.get("top30")
        if t10 is not None and t30 is not None and t10 < t30 - 0.5:
            print("  ⚠ MONOTONICITY INVERSION: Top-10 underperforms Top-30 by >0.5pp — review ranker ordering")

    # ── Concentration ────────────────────────────────────────────────────────
    cp = summary.get("concentration_pct_flagged")
    if cp is not None:
        flag = " ⚠" if cp >= 40 else ""
        print(f"\nConcentration: {cp:.0f}% of windows have top-3 names >{CONCENTRATION_WARN_PCT:.0f}% of abs XS{flag}")

    # ── EES warning ──────────────────────────────────────────────────────────
    if ees_warn.get("has_ees"):
        n_ees = ees_warn["n_ees_false_in_top30"]
        warn_flag = " ⚠" if ees_warn["ees_warn"] else ""
        print(f"\nCurrent EES-False in Top-30: {n_ees}{warn_flag} (flag if ≥5)")

    # ── Per-window table ──────────────────────────────────────────────────────
    if records:
        print(
            f"\n{'snap_date':<12} {'end_5d':<12} {'regime':<10} {'xs_5d':>7} {'xbi_trail':>10} {'top10':>7} {'top30':>7} {'conc%':>6}"
        )
        print("-" * 76)
        for r in records:
            m = r.get("monotonicity", {})
            t10 = f"{m.get('top10', 0):>+.1f}" if m.get("top10") is not None else "  N/A"
            t30 = f"{m.get('top30', 0):>+.1f}" if m.get("top30") is not None else "  N/A"
            c = r.get("concentration", {})
            cp_str = f"{c.get('top3_abs_pct', 0):.0f}%" if c.get("top3_abs_pct") is not None else "  N/A"
            xbi20 = f"{r['xbi_20d_pct']:>+.1f}" if r.get("xbi_20d_pct") is not None else "  N/A"
            print(
                f"{r['snap_date']:<12} {r['end_5d'] or '?':<12} {r['regime']:<10} "
                f"{r['xs_5d_pct']:>+6.2f}pp {xbi20:>7}% {t10:>7} {t30:>7} {cp_str:>6}"
            )

    # ── Gate status ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    gate_met = summary.get("gate_met", False)
    nr_avg = (summary.get("non_rally") or {}).get("avg_xs")
    print(f"Gate status: {'✓ MET' if gate_met else 'ACCUMULATING'}")
    print(
        f"  Windows: {summary['n']}/20  |  Non-rally avg XS: {nr_avg:>+.2f}pp"
        if nr_avg is not None
        else f"  Windows: {summary['n']}/20  |  No non-rally windows yet"
    )
    print("  Gate: ≥20 windows, non-rally avg XS > 0, no concentration warnings")

    # ── Write output ─────────────────────────────────────────────────────────
    out_json = FV_DIR / f"dem_regime_monitor_{as_of}.json"
    payload = {
        "as_of": as_of,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "summary": summary,
        "ees_warning": ees_warn,
        "windows": records,
        "classification": "VALIDATION_INFRASTRUCTURE/DEM_FORWARD_REGIME_SHADOW_MONITOR/NO_MODEL_CHANGE",
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # ── Write rolling summary card ────────────────────────────────────────────
    _write_summary_card(payload)
    print(f"\nJSON:    {out_json}")
    print(f"Summary: {FV_DIR / 'DEM_REGIME_MONITOR.md'}")


def _write_summary_card(payload: dict) -> None:
    as_of = payload["as_of"]
    s = payload["summary"]
    ees = payload["ees_warning"]
    records = payload["windows"]

    r = s.get("rally")
    nr = s.get("non_rally")
    mono = s.get("monotonicity_avg", {})

    lines = [
        f"# DEM Forward Regime Monitor — {as_of}",
        "",
        "**Classification:** VALIDATION_INFRASTRUCTURE / DEM_FORWARD_REGIME_SHADOW_MONITOR / NO_MODEL_CHANGE",
        "",
        "**Investment gate:** Repeatable excess return outside narrow biotech rally cluster.",
        f"**Progress:** {s['n']} / 20 windows completed",
        "",
        "---",
        "",
        "## Regime-Conditional XS",
        "",
        f"Overall: avg={s.get('avg_xs_5d_pct', 0):>+.2f}pp, {s.get('pct_positive', 0):.0f}% positive",
        "",
        "| regime | n | avg_xs_5d | pct_positive |",
        "|--------|---|-----------|--------------|",
    ]

    if r:
        lines.append(f"| RALLY | {r['n']} | {r['avg_xs']:>+.2f}pp | {r['pct_pos']:.0f}% |")
    if nr:
        lines.append(f"| NON_RALLY | {nr['n']} | {nr['avg_xs']:>+.2f}pp | {nr['pct_pos']:.0f}% |")
    if not r and not nr:
        lines.append("| — | 0 | — | — |")

    lines += [
        "",
        "## Rank Monotonicity",
        "",
        "| basket | avg_xs |",
        "|--------|--------|",
    ]
    for tier in ["top10", "top20", "top30"]:
        v = mono.get(tier)
        if v is not None:
            lines.append(f"| {tier} | {v:>+.2f}pp |")

    if mono.get("top10") is not None and mono.get("top30") is not None:
        inversion = mono["top10"] < mono["top30"] - 0.5
        lines += [
            "",
            f"Monotonicity verdict: **{'⚠ INVERTED — top-10 underperforms top-30' if inversion else 'OK'}**",
        ]

    conc = s.get("concentration_pct_flagged")
    if conc is not None:
        lines += [
            "",
            "## Concentration",
            "",
            f"Windows with top-3 names >{CONCENTRATION_WARN_PCT:.0f}% of abs XS: **{conc:.0f}%**",
            "*(Flag if >40% of windows are concentrated)*",
        ]

    if ees.get("has_ees"):
        lines += [
            "",
            "## EES Warning (current snapshot)",
            "",
            f"EES-False names in Top-30: **{ees['n_ees_false_in_top30']}**  ({'⚠ WARN' if ees['ees_warn'] else 'OK'})",
        ]

    lines += [
        "",
        "## Window Log",
        "",
        "| date | end | regime | xs_5d | xbi_20d | top10 | top30 | conc% |",
        "|------|-----|--------|-------|---------|-------|-------|-------|",
    ]
    for rec in records:
        m = rec.get("monotonicity", {})
        t10 = f"{m.get('top10', 0):>+.1f}" if m.get("top10") is not None else "—"
        t30 = f"{m.get('top30', 0):>+.1f}" if m.get("top30") is not None else "—"
        c = rec.get("concentration", {})
        cp_str = f"{c.get('top3_abs_pct', 0):.0f}%" if c.get("top3_abs_pct") is not None else "—"
        xbi20 = f"{rec['xbi_20d_pct']:>+.1f}%" if rec.get("xbi_20d_pct") is not None else "—"
        lines.append(
            f"| {rec['snap_date']} | {rec['end_5d'] or '?'} | {rec['regime']} | "
            f"{rec['xs_5d_pct']:>+.2f}pp | {xbi20} | {t10} | {t30} | {cp_str} |"
        )

    nr_xs_str = f"{nr['avg_xs']:>+.2f}pp" if nr else "no windows"
    mono_ok = mono.get("top10", 0) >= mono.get("top30", -999)
    conc_ok = (conc or 0) < 40
    lines += [
        "",
        "---",
        "",
        "## Gate Status",
        "",
        "| criterion | required | status |",
        "|-----------|----------|--------|",
        f"| Windows completed | ≥20 | {s['n']}/20 |",
        f"| Non-rally avg XS | > 0pp | {nr_xs_str} |",
        f"| Monotonicity | Top-10 ≥ Top-30 | {'OK' if mono_ok else 'WARN'} |",
        f"| Concentration | <40% windows flagged | {'OK' if conc_ok else 'WARN'} |",
        "",
        "No capital scale-up authorized until gate is met and operator reviews.",
        "",
        f"*Generated: {as_of} — VALIDATION_INFRASTRUCTURE / NO_MODEL_CHANGE*",
    ]

    out_md = FV_DIR / "DEM_REGIME_MONITOR.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=str(date.today()))
    parser.add_argument("--summary", action="store_true", help="Print summary only (no snapshot lookup)")
    args = parser.parse_args()

    if args.summary:
        prices = load_prices()
        records = build_window_records(prices)
        summary = summarize(records)
        print(json.dumps(summary, indent=2, default=str))
        return

    run(args.as_of)


if __name__ == "__main__":
    main()
