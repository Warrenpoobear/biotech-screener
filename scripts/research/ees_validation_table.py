#!/usr/bin/env python3
"""Daily EES validation table — Section 2 of the validation diagnostics spec.

For each (snapshot, universe) pair, emit one row capturing:
  - sample sizes (total, in-universe, resolved, quarantined)
  - field coverage for the 4 expectation-model inputs + ees_v3_score
  - distribution stats on priced_move_pct and realized 5d move
  - Spearman ICs: EES, pmv, EES⊥pmv (the headline incremental metric)
  - top-third / bottom-third excess-return spread
  - directional Brier score (sign(EES) vs sign(excess_5d))

Universes: A (broad, ees_v3_score non-null), A_eligible (production gate),
B (event-aligned, next_catalyst_date within ≤7 calendar days). All metrics
are computed on the quarantine-clean subset (priced_move_pct < 500).

Per [policy_alpha_freeze] / Spec 064: this is diagnostic-only.
The headline metric for promotion is `spearman_ees_resid_vs_excess_5d`
in Universe B. Verdict review queued 2026-05-22 (see
`ees_v3_incremental_ic_first_read_2026_04_30` memory).

Usage:
    python -m scripts.research.ees_validation_table
    python -m scripts.research.ees_validation_table --output /path/out.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAP_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_PANEL = DEFAULT_SNAP_ROOT / "_forward_returns_panel.csv"
DEFAULT_OUTPUT = DEFAULT_SNAP_ROOT / "_ees_validation_table.csv"

EES_VALID_FROM = "2026-04-14"

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.research.ees_validation_filters import (  # noqa: E402
    DEFAULT_CATALYST_WINDOW_CDAYS,
    in_universe_a,
    in_universe_b,
    is_quarantined,
)


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _is_present(v) -> bool:
    return _safe_float(v) is not None


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 5:
        return None
    # Tied-constant guard: when all xs (or ys) are equal, competitive
    # ranking below would produce ranks 1..n in stable-sort (original)
    # order — yielding a numerically meaningless rank correlation against
    # the other variable's order. Detect degeneracy BEFORE ranking.
    # See `incomplete_production_run_fallback_2026_05_01` memo for the
    # outage scenario that motivates this guard.
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for i, idx in enumerate(rx):
        rank_x[idx] = i + 1
    for i, idx in enumerate(ry):
        rank_y[idx] = i + 1
    mx = sum(rank_x) / n
    my = sum(rank_y) / n
    num = sum((rank_x[i] - mx) * (rank_y[i] - my) for i in range(n))
    dx = math.sqrt(sum((rank_x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((rank_y[i] - my) ** 2 for i in range(n)))
    if dx * dy == 0:
        return None
    return num / (dx * dy)


def _t_stat(rho: Optional[float], n: int) -> Optional[float]:
    if rho is None or abs(rho) >= 1 or n <= 2:
        return None
    return rho * math.sqrt(n - 2) / math.sqrt(1 - rho * rho)


def _residualize_linear(ys: List[float], xs: List[float]) -> List[float]:
    n = len(xs)
    if n < 2:
        return list(ys)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx == 0:
        return list(ys)
    b = sxy / sxx
    a = my - b * mx
    return [ys[i] - (a + b * xs[i]) for i in range(n)]


def _percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[i]


def _load_panel(panel_path: Path) -> Dict[tuple, dict]:
    out = {}
    with panel_path.open(newline="") as f:
        for r in csv.DictReader(f):
            out[(r["snap_date"], r["ticker"])] = r
    return out


def _load_rankings(snap_root: Path, snap_date: str) -> List[dict]:
    p = snap_root / snap_date / "rankings.csv"
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _filter_universe(
    rows: List[dict],
    universe: str,
    snap_date: date,
    catalyst_window_cdays: int = DEFAULT_CATALYST_WINDOW_CDAYS,
) -> List[dict]:
    if universe == "A":
        return [r for r in rows if in_universe_a(r)]
    if universe == "A_eligible":
        return [r for r in rows if in_universe_a(r) and str(r.get("ees_eligible", "")).lower() == "true"]
    if universe == "B":
        return [r for r in rows if in_universe_a(r) and in_universe_b(r, snap_date, catalyst_window_cdays)]
    raise ValueError(f"Unknown universe: {universe}")


def _row_for(
    snap_date: str,
    universe: str,
    rankings: List[dict],
    panel: Dict[tuple, dict],
    catalyst_window_cdays: int = DEFAULT_CATALYST_WINDOW_CDAYS,
) -> dict:
    sd = date.fromisoformat(snap_date)
    n_total = len(rankings)
    in_uni = _filter_universe(rankings, universe, sd, catalyst_window_cdays)
    n_uni = len(in_uni)
    n_quar = sum(1 for r in in_uni if is_quarantined(r))

    clean = [r for r in in_uni if not is_quarantined(r)]
    n_clean = len(clean)

    # Coverage on the universe (not just clean) — coverage is a data
    # integrity statement, not a signal statement.
    def cov(field: str) -> Optional[float]:
        if not n_uni:
            return None
        return sum(1 for r in in_uni if _is_present(r.get(field))) / n_uni

    pmv_cov = cov("priced_move_pct")
    si_cov = cov("short_interest_pct")
    mc_cov = cov("market_cap_mm")
    cp_cov = cov("close_price")
    ees_cov = cov("ees_v3_score")

    # Distribution of pmv on clean subset
    pmvs = [_safe_float(r.get("priced_move_pct")) for r in clean]
    pmvs = [v for v in pmvs if v is not None]
    median_pmv = statistics.median(pmvs) if pmvs else None
    p90_pmv = _percentile(pmvs, 0.90)

    # Resolved subset (with forward returns)
    resolved = []
    for r in clean:
        pr = panel.get((snap_date, r["ticker"]))
        if pr is None:
            continue
        if pr.get("forward_complete") != "true":
            continue
        resolved.append({**r, "_panel": pr})
    n_resolved = len(resolved)

    abs_5d = [_safe_float(r["_panel"].get("actual_abs_move_5d")) for r in resolved]
    abs_5d = [v for v in abs_5d if v is not None]
    median_abs_5d = statistics.median(abs_5d) if abs_5d else None

    ees_vals = [_safe_float(r.get("ees_v3_score")) for r in resolved]
    ee_vals = [_safe_float(r.get("expectation_error_score")) for r in resolved]
    pmv_vals = [_safe_float(r.get("priced_move_pct")) for r in resolved]
    ex_vals = [_safe_float(r["_panel"].get("excess_return_5d")) for r in resolved]

    # Triple-complete subset for IC computation
    triples = [
        (e, p, ex)
        for e, p, ex in zip(ees_vals, pmv_vals, ex_vals)
        if e is not None and p is not None and ex is not None
    ]
    n_ic = len(triples)
    mean_ee = statistics.mean(v for v in ee_vals if v is not None) if any(v is not None for v in ee_vals) else None

    if n_ic >= 5:
        ees = [t[0] for t in triples]
        pmv = [t[1] for t in triples]
        exr = [t[2] for t in triples]
        ees_resid = _residualize_linear(ees, pmv)
        rho_ees = _spearman(ees, exr)
        rho_pmv = _spearman(pmv, exr)
        rho_resid = _spearman(ees_resid, exr)
        t_ees = _t_stat(rho_ees, n_ic)
        t_pmv = _t_stat(rho_pmv, n_ic)
        t_resid = _t_stat(rho_resid, n_ic)

        # Top-third minus bottom-third by EES
        ranked = sorted(triples, key=lambda t: t[0])
        cut = max(2, n_ic // 3)
        bot = ranked[:cut]
        top = ranked[-cut:]
        bot_ex = sum(t[2] for t in bot) / len(bot)
        top_ex = sum(t[2] for t in top) / len(top)
        spread = top_ex - bot_ex

        # Brier directional: sign(EES) vs sign(ex). Treat EES>0 as predicting positive.
        # Brier score = mean( (predicted_prob - actual)^2 ); use sigmoid-like collapse:
        # If EES>0 → predict P(positive)=1, else 0; actual = 1 if ex>=0 else 0.
        # That's hit-rate-equivalent. Use directional accuracy as the gate.
        n_correct = sum(1 for e, _, ex in triples if (e >= 0 and ex >= 0) or (e < 0 and ex < 0))
        hit_rate = n_correct / n_ic
        brier = sum(((1.0 if e >= 0 else 0.0) - (1.0 if ex >= 0 else 0.0)) ** 2 for e, _, ex in triples) / n_ic
    else:
        rho_ees = rho_pmv = rho_resid = None
        t_ees = t_pmv = t_resid = None
        top_ex = bot_ex = spread = None
        hit_rate = brier = None

    return {
        "snap_date": snap_date,
        "universe": universe,
        "n_total": n_total,
        "n_universe": n_uni,
        "n_quarantined": n_quar,
        "n_clean": n_clean,
        "n_resolved": n_resolved,
        "n_ic": n_ic,
        "priced_move_cov": _r4(pmv_cov),
        "short_interest_cov": _r4(si_cov),
        "market_cap_cov": _r4(mc_cov),
        "close_price_cov": _r4(cp_cov),
        "ees_score_cov": _r4(ees_cov),
        "median_pmv": _r4(median_pmv),
        "p90_pmv": _r4(p90_pmv),
        "median_abs_realized_5d": _r4(median_abs_5d),
        "mean_expectation_error": _r4(mean_ee),
        "top_third_excess_5d": _r4(top_ex),
        "bottom_third_excess_5d": _r4(bot_ex),
        "spread_5d": _r4(spread),
        "spearman_ees_vs_ex5d": _r3(rho_ees),
        "tstat_ees": _r2(t_ees),
        "spearman_pmv_vs_ex5d": _r3(rho_pmv),
        "tstat_pmv": _r2(t_pmv),
        "spearman_ees_resid_vs_ex5d": _r3(rho_resid),
        "tstat_ees_resid": _r2(t_resid),
        "directional_hit_rate": _r4(hit_rate),
        "brier_directional": _r4(brier),
    }


def _r2(v):
    return round(v, 2) if v is not None else None


def _r3(v):
    return round(v, 3) if v is not None else None


def _r4(v):
    return round(v, 4) if v is not None else None


def build_table(
    snap_root: Path = DEFAULT_SNAP_ROOT,
    panel_path: Path = DEFAULT_PANEL,
    since: str = EES_VALID_FROM,
    universes: tuple = ("A", "A_eligible", "B"),
    catalyst_window_cdays: int = DEFAULT_CATALYST_WINDOW_CDAYS,
) -> List[dict]:
    panel = _load_panel(panel_path)
    snapshots = sorted({k[0] for k in panel.keys()})
    snapshots = [s for s in snapshots if s >= since]
    logger.info(f"Building validation table for {len(snapshots)} snapshots, " f"{len(universes)} universes")
    rows = []
    for snap in snapshots:
        rankings = _load_rankings(snap_root, snap)
        if not rankings:
            continue
        for u in universes:
            rows.append(_row_for(snap, u, rankings, panel, catalyst_window_cdays))
    return rows


def write_table(rows: List[dict], output: Path = DEFAULT_OUTPUT) -> None:
    if not rows:
        logger.warning("No rows to write.")
        return
    cols = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {output}")


def print_summary(rows: List[dict]) -> None:
    """Print a tight per-universe rolling summary to console."""
    by_uni = defaultdict(list)
    for r in rows:
        by_uni[r["universe"]].append(r)

    print("\n=== Rolling-aggregate summary (across all resolved snapshots) ===")
    print(
        f"{'universe':<14} {'snaps':>5} {'n_ic':>6} {'EES_IC':>8} {'pmv_IC':>8} "
        f"{'EES⊥pmv_IC':>11} {'spread':>8} {'hit_rt':>7}"
    )
    print("-" * 80)
    for u in ("A", "A_eligible", "B"):
        urows = by_uni.get(u, [])
        # Only count snapshots where IC was computable
        with_ic = [r for r in urows if r["n_ic"] >= 5]
        if not with_ic:
            print(f"{u:<14} {len(urows):>5} {'-':>6} {'-':>8} {'-':>8} {'-':>11} {'-':>8} {'-':>7}")
            continue
        snaps = len(with_ic)
        # Total resolved triples
        n_ic_tot = sum(r["n_ic"] for r in with_ic)

        # Avg ICs (simple mean across snapshots — matches per-snapshot focus per [policy_freeze])
        def avg(field):
            vs = [r[field] for r in with_ic if r[field] is not None]
            return sum(vs) / len(vs) if vs else None

        ees_ic = avg("spearman_ees_vs_ex5d")
        pmv_ic = avg("spearman_pmv_vs_ex5d")
        resid_ic = avg("spearman_ees_resid_vs_ex5d")
        spread = avg("spread_5d")
        hit = avg("directional_hit_rate")

        def fmt(x, n=3):
            return f"{x:+.{n}f}" if x is not None else "  -  "

        print(
            f"{u:<14} {snaps:>5} {n_ic_tot:>6} "
            f"{fmt(ees_ic):>8} {fmt(pmv_ic):>8} {fmt(resid_ic):>11} "
            f"{(f'{spread*100:+.2f}pp' if spread is not None else '-'):>8} "
            f"{(f'{hit:.3f}' if hit is not None else '-'):>7}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--snap-root", type=Path, default=DEFAULT_SNAP_ROOT)
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--since", default=EES_VALID_FROM)
    p.add_argument(
        "--catalyst-window",
        type=int,
        default=DEFAULT_CATALYST_WINDOW_CDAYS,
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    rows = build_table(
        snap_root=args.snap_root,
        panel_path=args.panel,
        since=args.since,
        catalyst_window_cdays=args.catalyst_window,
    )
    write_table(rows, output=args.output)
    if not args.quiet:
        print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
