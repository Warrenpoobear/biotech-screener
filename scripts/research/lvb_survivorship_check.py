"""LVB survivorship check: add back drug/biotech names missing from the snapshot set.

Research only (Tier 2). The snapshot history (e.g. data/snapshots_pit_v9) holds only
names in today's universe, so names acquired or delisted before then never enter
the pool. This script asks Sharadar which SIC 2834/2835/2836 common stocks traded
at each snapshot date, adds the ones missing from that date's rankings.csv to the
LVB pool (as eligible, never Top-30), and reruns the LVB backtest.

Pools compared (all on Sharadar closeadj, total return; names Sharadar lacks fall
back to the production split-adjusted file):
  base          snapshot rankings only (survivor universe)
  +delisted     plus missing names that are delisted today (the survivorship gap)
  +all_missing  plus every missing drug/biotech name (survivorship + coverage gap)

Add-back filter: Sharadar DAILY market cap ($ millions) on the snapshot date at or above
--min-mcap-mm (default: the smallest eligible market cap in that day's rankings).
After a name's last Sharadar close, its return uses that last close (acquisitions
end at the deal price; a collapse after the last print would be missed).

Needs SHARADAR_API_KEY. Writes vendor-data caches under --cache-dir (keep out of git).

Usage:
  python scripts/research/lvb_survivorship_check.py --cache-dir <local dir> --out <local>/lvb_surv.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "research"))

from lvb_sharadar_prices import _load_key  # noqa: E402
from lvb_universe_backtest import run_variant, stats  # noqa: E402

from common.sharadar_client import SharadarClient  # noqa: E402
from tools.low_risk_sleeve_shadow import load_prices  # noqa: E402

SICS = {"2834", "2835", "2836"}
MIN_PRICE = 5.0  # common/data_quality.py
MIN_DOLLAR_VOLUME = 500_000.0  # common/data_quality.py
DD_GATE = -0.40  # ruleset drawdown_gate
DD_REL_GATE = -0.25  # ruleset drawdown_rel_xbi_gate
CATS = {
    "Domestic Common Stock",
    "Domestic Common Stock Primary Class",
    "ADR Common Stock",
    "ADR Common Stock Primary Class",
    "Canadian Common Stock",
}


def _cached(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fn()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", default="data/snapshots_pit_v9")
    ap.add_argument("--split-prices", default="production_data/price_history_split_adj.csv")
    ap.add_argument("--sharadar-prices", required=True, help="closeadj CSV from lvb_sharadar_prices.py")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--min-mcap-mm", type=float, default=None)
    ap.add_argument("--start", default="2020-03-31", help="first holding date reported")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    snap = (REPO / args.snapshot_dir).resolve()
    dates = sorted(p.name for p in snap.iterdir() if p.name[:2] == "20" and (p / "rankings.csv").exists())
    rankings = {}
    for d in dates:
        with open(snap / d / "rankings.csv", encoding="utf-8", newline="") as fh:
            rankings[d] = list(csv.DictReader(fh))

    _load_key()
    client = SharadarClient()

    # 1. Who traded at each snapshot date (SEP, no ticker filter = whole market that day).
    def _trading():
        out = {}
        for d in dates:
            out[d] = sorted({r["ticker"] for r in client.get_table("SEP", filters={"date": d}, fields=["ticker"])})
        return out

    trading = _cached(cache / "sep_tickers_by_date.json", _trading)
    all_tickers = sorted(set().union(*map(set, trading.values())))

    # 2. Ticker metadata: SIC, category, delisted flag.
    in_snap = {d: {r["ticker"] for r in rankings[d]} for d in dates}
    snap_tickers = set().union(*in_snap.values())
    meta_rows = _cached(
        cache / "tickers_meta_v2.json",
        lambda: client.fetch(
            "TICKERS",
            tickers=sorted(set(all_tickers) | snap_tickers),
            fields=["table", "permaticker", "ticker", "name", "category", "isdelisted", "siccode", "lastpricedate"],
        ),
    )
    meta = {}
    for m in meta_rows:
        if m.get("table") in ("SEP", "stocks"):
            meta.setdefault(m["ticker"], m)
    bio = {t for t, m in meta.items() if m.get("siccode") in SICS and m.get("category") in CATS}

    # Renamed tickers: Sharadar keeps a company's older history under a suffixed ticker
    # (SUPN1 = Supernus, already in the universe as SUPN). Same permaticker or same
    # company name as a snapshot name = the same company, not a missing one.
    def _norm(name):
        return " ".join((name or "").upper().replace(",", " ").replace(".", " ").split())

    snap_perma = {str(meta[t].get("permaticker")) for t in snap_tickers if t in meta}
    snap_names = {_norm(meta[t].get("name")) for t in snap_tickers if t in meta}

    def _renamed(t):
        # Suffixed ticker whose base is a snapshot name of the same company, e.g. SUPN1
        # ("SUPERNUS PHARMACEUTICALS INC") vs SUPN ("SUPERNUS INC", new holding company and
        # new permaticker after the 2026-11 reorganisation).
        base = t.rstrip("0123456789")
        if base == t or base not in snap_tickers or base not in meta:
            return False
        a, b = _norm(meta[t].get("name")).split(), _norm(meta[base].get("name")).split()
        return bool(a and b and a[0] == b[0])

    same_company = {
        t
        for t in bio
        if str(meta[t].get("permaticker")) in snap_perma or _norm(meta[t].get("name")) in snap_names or _renamed(t)
    }

    # 3. Missing names per date and their point-in-time market cap.
    missing = {d: sorted((set(trading.get(d, [])) & bio) - in_snap[d] - same_company) for d in dates}
    cand = sorted(set().union(*map(set, missing.values())))

    def _mcaps():
        out = defaultdict(dict)
        for d in dates:
            if not missing[d]:
                continue
            for r in client.fetch("DAILY", tickers=missing[d], filters={"date": d}, fields=["ticker", "marketcap"]):
                if r.get("marketcap") not in (None, ""):
                    out[d][r["ticker"]] = float(r["marketcap"])
        return out

    mcap = _cached(cache / "daily_mcap_missing.json", _mcaps)

    # 4. Prices: Sharadar closeadj for snapshot names + candidates; fallback to split file.
    cand_csv = cache / "sharadar_closeadj_candidates.csv"
    if not cand_csv.exists():
        rows = client.fetch(
            "SEP", tickers=cand, fields=["ticker", "date", "closeadj"], date_range=("2019-06-01", "2026-12-31")
        )
        with open(cand_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["ticker", "date", "close"])
            for r in sorted(rows, key=lambda r: (r["ticker"], str(r["date"]))):
                if r.get("closeadj") not in (None, ""):
                    w.writerow([r["ticker"], str(r["date"])[:10], r["closeadj"]])
    # Raw (unadjusted) close and volume for the price and liquidity rules.
    raw_csv = cache / "sharadar_raw_candidates.csv"
    if not raw_csv.exists():
        rows = client.fetch(
            "SEP", tickers=cand, fields=["ticker", "date", "close", "volume"], date_range=("2019-06-01", "2026-12-31")
        )
        with open(raw_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["ticker", "date", "close", "volume"])
            for r in sorted(rows, key=lambda r: (r["ticker"], str(r["date"]))):
                if r.get("close") not in (None, ""):
                    w.writerow([r["ticker"], str(r["date"])[:10], r["close"], r.get("volume") or 0])
    raw = defaultdict(list)
    with open(raw_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            raw[r["ticker"]].append((r["date"], float(r["close"]), float(r["volume"] or 0)))

    prices = load_prices(Path(args.sharadar_prices))
    for t, s in load_prices(cand_csv).items():
        prices.setdefault(t, s)
    split = load_prices(REPO / args.split_prices)
    fallback = []
    first_needed = "2019-08-01"  # 120 trading days before the first snapshot
    for t in snap_tickers:
        sh = prices.get(t, {})
        if t in split and (not sh or (min(sh) > first_needed and min(split[t]) < min(sh))):
            prices[t] = split[t]
            fallback.append(t)

    # Production-like eligibility for an added name on date d (see decision_engine L0 gates and
    # common/data_quality.py): price >= $5, 20-day median dollar volume >= $500k, and not
    # (252-day drawdown < -40% AND drawdown minus XBI's < -25%). Fundamental red flags and
    # missing financials cannot be replicated here.
    def _drawdown(series, d):
        pts = [series[k] for k in sorted(k for k in series if k <= d)[-252:]]
        return pts[-1] / max(pts) - 1 if pts else None

    def eligible(t, d):
        hist = [x for x in raw.get(t, []) if x[0] <= d][-20:]
        if not hist or hist[-1][1] < MIN_PRICE:
            return "price"
        dv = sorted(c * v for _, c, v in hist)
        if dv[len(dv) // 2] < MIN_DOLLAR_VOLUME:
            return "liquidity"
        dd = _drawdown(prices.get(t, {}), d)
        xdd = _drawdown(prices.get("XBI", {}), d)
        if dd is not None and xdd is not None and dd < DD_GATE and dd - xdd < DD_REL_GATE:
            return "deep_drawdown"
        return None

    # 5. Build augmented rankings and rerun.
    def min_eligible_mcap(d):
        vals = []
        for r in rankings[d]:
            try:
                if r.get("eligible") in ("1", "1.0", "True", "true"):
                    vals.append(float(r["market_cap_mm"]))
            except (TypeError, ValueError):
                pass
        return min(vals) if vals else 0.0

    rejected = defaultdict(lambda: defaultdict(int))

    def augmented(pool):
        out_dir = cache / f"rankings_{pool}"
        added = {}
        for d in dates:
            rows = list(rankings[d])
            floor = args.min_mcap_mm if args.min_mcap_mm is not None else min_eligible_mcap(d)
            adds = []
            for t in missing[d]:
                if pool == "delisted" and meta[t].get("isdelisted") != "Y":
                    continue
                mc = mcap.get(d, {}).get(t)
                if mc is None or mc < floor:  # Sharadar DAILY marketcap is in $ millions
                    rejected[pool]["market_cap"] += 1
                    continue
                why = eligible(t, d)
                if why:
                    rejected[pool][why] += 1
                    continue
                adds.append(t)
                rows.append({"ticker": t, "eligible": "1", "actionable_rank": ""})
            added[d] = adds
            (out_dir / d).mkdir(parents=True, exist_ok=True)
            with open(out_dir / d / "rankings.csv", "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["ticker", "eligible", "actionable_rank"], extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        return out_dir, added

    results, adds_by_pool = {}, {}
    for pool in ("base", "delisted", "all_missing"):
        if pool == "base":
            sdir, added = snap, {d: [] for d in dates}
        else:
            sdir, added = augmented(pool)
        adds_by_pool[pool] = added
        for v in ("full", "ex_top30"):
            rows = [r for r in run_variant(v, dates, sdir, prices) if r["d0"] >= args.start]
            s = stats([r["net"] for r in rows], [r["xbi"] for r in rows])
            added_held = [len(set(r["names"]) & set(added.get(r["d0"], []))) for r in rows]
            s["avg_added_names_held"] = sum(added_held) / len(rows)
            s["avg_added_to_pool"] = sum(len(added.get(r["d0"], [])) for r in rows) / len(rows)
            results[f"{pool}:{v}"] = {"summary": s, "monthly": rows}

    held_counts = defaultdict(int)
    for r in results["all_missing:full"]["monthly"]:
        for t in set(r["names"]) & set(adds_by_pool["all_missing"].get(r["d0"], [])):
            held_counts[t] += 1
    summary = {k: v["summary"] for k, v in results.items()}
    summary["most_held_added_names"] = sorted(
        (
            [t, n, meta[t].get("name"), meta[t].get("isdelisted"), meta[t].get("lastpricedate")]
            for t, n in held_counts.items()
        ),
        key=lambda x: -x[1],
    )[:25]
    summary["fallback_to_split_prices"] = sorted(fallback)
    summary["n_candidates"] = len(cand)
    summary["same_company_excluded"] = sorted(same_company - snap_tickers)
    summary["addback_rejections"] = {p: dict(v) for p, v in rejected.items()}
    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"summary": summary, "monthly": {k: v["monthly"] for k, v in results.items()}}, indent=1, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
