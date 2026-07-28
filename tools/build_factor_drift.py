#!/usr/bin/env python3
"""Two-baseline, thresholded factor-drift monitor.

Every cycle, compares the current portfolio against:
  1. The eligible universe (unintended tilt detection)
  2. The trailing 20-cycle history of the portfolio itself (model drift)

Tracks:
  - Factor tilts: de_beta_xbi_60d, de_vol_60d, de_drawdown, de_rsi_14d
  - Momentum-state mix: % tailwind / neutral / headwind
  - Universe composition: Jaccard, rank correlation, HHI, eligible size
  - Signal-distribution drift: mean/std for key z-scored signals

Output:
    artifacts/factor_drift/YYYY-MM-DD_factor_drift.json

Usage:
    python tools/build_factor_drift.py --as-of-date 2026-04-05
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("factor_drift")

SCHEMA_VERSION = "factor_drift.v1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FACTOR_COLS = ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]

SIGNAL_COLS = [
    "financial_score",
    "coinvest_score_z",
    "inst_delta_z",
    "selector_score",
    "clinical_score_v2_z",
]

MOM_STATES = ["tailwind", "neutral", "headwind"]

TRAILING_WINDOW = 20  # cycles

# Thresholds — yellow / red
YELLOW_FACTOR_TILT = 0.10  # 10% absolute tilt vs universe
RED_FACTOR_TILT = 0.20

YELLOW_JACCARD = 0.80
RED_JACCARD = 0.70

YELLOW_HHI_EXCESS = 0.15  # 15% above trailing average
RED_HHI_EXCESS = 0.25

YELLOW_MOM_SHIFT_PP = 10.0  # 10 percentage points
RED_MOM_SHIFT_PP = 20.0  # not in original spec but useful

YELLOW_SIGNAL_DRIFT_SD = 1.0  # 1 std dev vs trailing 20-cycle
RED_SIGNAL_DRIFT_SD = 1.5

RED_UNIVERSE_SIZE_CHANGE = 0.20  # 20% vs trailing median


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sf(val: Any) -> float:
    """Safe float conversion — returns NaN for missing/invalid."""
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return math.nan


def _is_eligible(row: Dict[str, str]) -> bool:
    """Robust eligibility check — handles '1', '1.0', 'true', 'yes'."""
    v = str(row.get("eligible", "")).strip().lower()
    return v in ("1", "1.0", "true", "yes")


def _load_rankings(date: str, snapshots_dir: Path) -> Optional[List[Dict[str, str]]]:
    """Load rankings.csv for a date, return None if missing."""
    path = snapshots_dir / date / "rankings.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _mean(vals: List[float]) -> Optional[float]:
    finite = [v for v in vals if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def _std(vals: List[float]) -> Optional[float]:
    finite = [v for v in vals if math.isfinite(v)]
    if len(finite) < 2:
        return None
    m = sum(finite) / len(finite)
    var = sum((v - m) ** 2 for v in finite) / (len(finite) - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Factor tilt computation
# ---------------------------------------------------------------------------


def compute_factor_tilts(
    port_rows: List[Dict[str, str]],
    universe_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Compute portfolio mean vs universe mean for each factor.

    Returns {factor: {port_mean, univ_mean, tilt}} where tilt is the
    relative difference (port - univ) / |univ|.
    """
    result = {}
    for col in FACTOR_COLS:
        port_vals = [_sf(r.get(col)) for r in port_rows]
        univ_vals = [_sf(r.get(col)) for r in universe_rows]
        p_mean = _mean(port_vals)
        u_mean = _mean(univ_vals)

        if p_mean is not None and u_mean is not None and u_mean != 0:
            tilt = (p_mean - u_mean) / abs(u_mean)
        else:
            tilt = None

        result[col] = {
            "port_mean": round(p_mean, 4) if p_mean is not None else None,
            "univ_mean": round(u_mean, 4) if u_mean is not None else None,
            "tilt": round(tilt, 4) if tilt is not None else None,
        }
    return result


# ---------------------------------------------------------------------------
# Momentum mix
# ---------------------------------------------------------------------------


def compute_momentum_mix(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Return percentage in each momentum state."""
    n = len(rows) or 1
    counts = {s: 0 for s in MOM_STATES}
    for r in rows:
        ms = r.get("mom_state", "")
        if ms in counts:
            counts[ms] += 1
    return {s: round(100 * counts[s] / n, 1) for s in MOM_STATES}


# ---------------------------------------------------------------------------
# Signal moments
# ---------------------------------------------------------------------------


def compute_signal_moments(
    port_rows: List[Dict[str, str]],
    universe_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Mean and std for key z-scored signals, for both portfolio and universe."""
    result = {}
    for col in SIGNAL_COLS:
        pv = [_sf(r.get(col)) for r in port_rows]
        uv = [_sf(r.get(col)) for r in universe_rows]
        result[col] = {
            "port_mean": round(v, 4) if (v := _mean(pv)) is not None else None,
            "port_std": round(v, 4) if (v := _std(pv)) is not None else None,
            "univ_mean": round(v, 4) if (v := _mean(uv)) is not None else None,
            "univ_std": round(v, 4) if (v := _std(uv)) is not None else None,
        }
    return result


# ---------------------------------------------------------------------------
# Universe composition metrics
# ---------------------------------------------------------------------------


def compute_jaccard(set_a: set, set_b: set) -> Optional[float]:
    if not set_a or not set_b:
        return None
    union = set_a | set_b
    return round(len(set_a & set_b) / len(union), 4) if union else None


def compute_rank_correlation(curr_ranks: Dict[str, int], prior_ranks: Dict[str, int]) -> Optional[float]:
    common = sorted(set(curr_ranks) & set(prior_ranks))
    if len(common) < 5:
        return None
    x = [curr_ranks[t] for t in common]
    y = [prior_ranks[t] for t in common]
    n = len(x)
    d_sq = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
    rho = 1 - 6 * d_sq / (n * (n * n - 1)) if n > 1 else None
    return round(rho, 4) if rho is not None else None


def compute_hhi(weights: List[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    shares = [w / total * 100 for w in weights]
    return round(sum(s * s for s in shares), 1)


def _rank_map(rows: List[Dict[str, str]], col: str = "actionable_rank") -> Dict[str, int]:
    result = {}
    for r in rows:
        try:
            rank = int(float(r.get(col, "")))
            ticker = r.get("ticker", "")
            if ticker:
                result[ticker] = rank
        except (ValueError, TypeError):
            pass
    return result


# ---------------------------------------------------------------------------
# Trailing history loader
# ---------------------------------------------------------------------------


def load_trailing_history(
    as_of_date: str,
    drift_dir: Path,
    n: int = TRAILING_WINDOW,
) -> List[Dict[str, Any]]:
    """Load the most recent *n* factor_drift artifacts before as_of_date."""
    if not drift_dir.exists():
        return []
    import re

    pat = re.compile(r"^(\d{4}-\d{2}-\d{2})_factor_drift\.json$")
    candidates = []
    for f in drift_dir.iterdir():
        m = pat.match(f.name)
        if m and m.group(1) < as_of_date:
            candidates.append((m.group(1), f))
    candidates.sort(key=lambda x: x[0], reverse=True)

    history = []
    for _, path in candidates[:n]:
        data = _load_json(path)
        if data:
            history.append(data)
    history.reverse()  # oldest first
    return history


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def _check_factor_alerts(
    tilts: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Factor tilt vs eligible universe alerts."""
    alerts = []
    for col, info in tilts.items():
        tilt = info.get("tilt")
        if tilt is None:
            continue
        abs_tilt = abs(tilt)
        if abs_tilt > RED_FACTOR_TILT:
            alerts.append(
                {
                    "level": "RED",
                    "code": "FACTOR_TILT",
                    "detail": f"{col} tilt={tilt:+.1%} vs universe (>{RED_FACTOR_TILT:.0%})",
                }
            )
        elif abs_tilt > YELLOW_FACTOR_TILT:
            alerts.append(
                {
                    "level": "YELLOW",
                    "code": "FACTOR_TILT",
                    "detail": f"{col} tilt={tilt:+.1%} vs universe (>{YELLOW_FACTOR_TILT:.0%})",
                }
            )
    return alerts


def _check_jaccard_alert(jaccard: Optional[float]) -> List[Dict[str, str]]:
    if jaccard is None:
        return []
    if jaccard < RED_JACCARD:
        return [{"level": "RED", "code": "JACCARD_LOW", "detail": f"Jaccard={jaccard:.2f} (<{RED_JACCARD})"}]
    if jaccard < YELLOW_JACCARD:
        return [{"level": "YELLOW", "code": "JACCARD_LOW", "detail": f"Jaccard={jaccard:.2f} (<{YELLOW_JACCARD})"}]
    return []


def _check_hhi_alert(hhi: float, trailing: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """HHI vs trailing 20-cycle average."""
    trailing_hhis = [h.get("hhi", 0) for h in trailing if h.get("hhi") is not None]
    if not trailing_hhis:
        return []
    avg = sum(trailing_hhis) / len(trailing_hhis)
    if avg <= 0:
        return []
    excess = (hhi - avg) / avg
    if excess > RED_HHI_EXCESS:
        return [
            {"level": "RED", "code": "HHI_DRIFT", "detail": f"HHI={hhi:.0f} vs trail avg={avg:.0f} (+{excess:.0%})"}
        ]
    if excess > YELLOW_HHI_EXCESS:
        return [
            {"level": "YELLOW", "code": "HHI_DRIFT", "detail": f"HHI={hhi:.0f} vs trail avg={avg:.0f} (+{excess:.0%})"}
        ]
    return []


def _history_portfolio_mix(entry: Dict[str, Any]) -> Dict[str, float]:
    """Portfolio momentum mix from one history entry.

    build_factor_drift writes ``momentum_mix`` as ``{"portfolio": ..., "universe": ...}``.
    Reading that nested mapping as a flat ``{state: pct}`` yields 0 for every state,
    which reports the entire current mix as a shift. Entries written before the
    nesting was introduced stored the portfolio mix flat, so both are accepted.
    """
    mix = entry.get("momentum_mix") or {}
    nested = mix.get("portfolio")
    if isinstance(nested, dict):
        return nested
    return mix


def _check_momentum_alert(current_mix: Dict[str, float], trailing: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Momentum-state mix shift vs trailing 20-cycle average."""
    if not trailing:
        return []
    trailing_mixes = [mix for mix in (_history_portfolio_mix(h) for h in trailing) if mix]
    if not trailing_mixes:
        return []

    alerts = []
    for state in MOM_STATES:
        trail_vals = [m.get(state, 0) for m in trailing_mixes]
        trail_avg = sum(trail_vals) / len(trail_vals)
        shift = abs(current_mix.get(state, 0) - trail_avg)
        if shift > YELLOW_MOM_SHIFT_PP:
            alerts.append(
                {
                    "level": "YELLOW",
                    "code": "MOMENTUM_SHIFT",
                    "detail": f"{state}: {current_mix.get(state, 0):.0f}% vs trail avg {trail_avg:.0f}% (shift={shift:.0f}pp)",
                }
            )
    return alerts


def _check_signal_drift_alerts(
    moments: Dict[str, Dict[str, Any]], trailing: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Signal mean drift vs trailing 20-cycle history."""
    if not trailing:
        return []

    alerts = []
    for col in SIGNAL_COLS:
        current_mean = moments.get(col, {}).get("port_mean")
        if current_mean is None:
            continue

        trail_means = []
        for h in trailing:
            sm = h.get("signal_moments", {}).get(col, {})
            pm = sm.get("port_mean")
            if pm is not None:
                trail_means.append(pm)

        if len(trail_means) < 3:
            continue

        t_mean = sum(trail_means) / len(trail_means)
        t_std = _std(trail_means)
        if t_std is None or t_std < 1e-9:
            continue

        z = abs(current_mean - t_mean) / t_std
        if z > RED_SIGNAL_DRIFT_SD:
            alerts.append(
                {
                    "level": "RED",
                    "code": "SIGNAL_DRIFT",
                    "detail": f"{col} mean={current_mean:.3f} vs trail {t_mean:.3f} (z={z:.2f})",
                }
            )
        elif z > YELLOW_SIGNAL_DRIFT_SD:
            alerts.append(
                {
                    "level": "YELLOW",
                    "code": "SIGNAL_DRIFT",
                    "detail": f"{col} mean={current_mean:.3f} vs trail {t_mean:.3f} (z={z:.2f})",
                }
            )
    return alerts


def _check_universe_size_alert(current_size: int, trailing: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Universe size change vs trailing median."""
    sizes = [h.get("universe_size", 0) for h in trailing if h.get("universe_size")]
    if not sizes:
        return []
    sizes_sorted = sorted(sizes)
    median = sizes_sorted[len(sizes_sorted) // 2]
    if median <= 0:
        return []
    change = abs(current_size - median) / median
    if change > RED_UNIVERSE_SIZE_CHANGE:
        return [
            {
                "level": "RED",
                "code": "UNIVERSE_SIZE",
                "detail": f"Eligible={current_size} vs trail median={median} (change={change:.0%})",
            }
        ]
    return []


def generate_alerts(
    tilts: Dict[str, Dict[str, Any]],
    jaccard: Optional[float],
    hhi: float,
    momentum_mix: Dict[str, float],
    signal_moments: Dict[str, Dict[str, Any]],
    universe_size: int,
    trailing: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Generate all alerts from current metrics + trailing history."""
    alerts = []
    alerts.extend(_check_factor_alerts(tilts))
    alerts.extend(_check_jaccard_alert(jaccard))
    alerts.extend(_check_hhi_alert(hhi, trailing))
    alerts.extend(_check_momentum_alert(momentum_mix, trailing))
    alerts.extend(_check_signal_drift_alerts(signal_moments, trailing))
    alerts.extend(_check_universe_size_alert(universe_size, trailing))
    return alerts


# ---------------------------------------------------------------------------
# Trailing-baseline comparison
# ---------------------------------------------------------------------------


def compute_trailing_comparison(
    tilts: Dict[str, Dict[str, Any]],
    trailing: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compare current factor tilts against trailing 20-cycle portfolio means."""
    if not trailing:
        return {}

    result = {}
    for col in FACTOR_COLS:
        trail_port_means = []
        for h in trailing:
            pvu = h.get("portfolio_vs_universe", {}).get(col, {})
            pm = pvu.get("port_mean")
            if pm is not None:
                trail_port_means.append(pm)

        if not trail_port_means:
            continue

        current_port_mean = tilts.get(col, {}).get("port_mean")
        trail_avg = sum(trail_port_means) / len(trail_port_means)
        trail_sd = _std(trail_port_means)

        drift_z = None
        if current_port_mean is not None and trail_sd is not None and trail_sd > 1e-9:
            drift_z = (current_port_mean - trail_avg) / trail_sd

        result[col] = {
            "trail_mean": round(trail_avg, 4),
            "trail_std": round(trail_sd, 4) if trail_sd is not None else None,
            "drift_z": round(drift_z, 2) if drift_z is not None else None,
        }
    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_factor_drift(
    as_of_date: str,
    *,
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    top_k: int = 30,
) -> Dict[str, Any]:
    """Build the factor drift artifact for one cycle."""
    rankings = _load_rankings(as_of_date, snapshots_dir)
    if not rankings:
        return {"error": f"no rankings.csv for {as_of_date}"}

    # Split into portfolio (top-K eligible) and eligible universe
    eligible = [r for r in rankings if _is_eligible(r)]
    ranked = _rank_map(rankings)
    eligible_ranked = sorted(
        [r for r in eligible if r.get("ticker", "") in ranked],
        key=lambda r: ranked.get(r.get("ticker", ""), 999999),
    )
    port_rows = eligible_ranked[:top_k]
    port_tickers = {r.get("ticker", "") for r in port_rows}

    # Weights — use target_weight_pct or equal weight
    weights = []
    for r in port_rows:
        w = _sf(r.get("target_weight_pct"))
        weights.append(w if math.isfinite(w) and w > 0 else 100.0 / max(len(port_rows), 1))

    # --- Baseline 1: portfolio vs eligible universe ---
    factor_tilts = compute_factor_tilts(port_rows, eligible)
    port_mom_mix = compute_momentum_mix(port_rows)
    univ_mom_mix = compute_momentum_mix(eligible)
    signal_moments = compute_signal_moments(port_rows, eligible)

    # --- Universe composition ---
    # Find prior snapshot for Jaccard and rank corr
    prior_date = _find_prior_date(as_of_date, snapshots_dir)
    jaccard_prev = None
    rank_corr_prev = None
    if prior_date:
        prior_rankings = _load_rankings(prior_date, snapshots_dir)
        if prior_rankings:
            prior_eligible = [r for r in prior_rankings if _is_eligible(r)]
            prior_ranked = _rank_map(prior_rankings)
            prior_port = sorted(
                [r for r in prior_eligible if r.get("ticker", "") in prior_ranked],
                key=lambda r: prior_ranked.get(r.get("ticker", ""), 999999),
            )[:top_k]
            prior_tickers = {r.get("ticker", "") for r in prior_port}
            jaccard_prev = compute_jaccard(port_tickers, prior_tickers)
            rank_corr_prev = compute_rank_correlation(ranked, prior_ranked)

    hhi = compute_hhi(weights)
    universe_size = len(eligible)

    # --- Baseline 2: trailing 20-cycle history ---
    drift_dir = artifacts_dir / "factor_drift"
    trailing = load_trailing_history(as_of_date, drift_dir, TRAILING_WINDOW)
    trailing_comparison = compute_trailing_comparison(factor_tilts, trailing)

    # --- Alerts ---
    alerts = generate_alerts(
        factor_tilts,
        jaccard_prev,
        hhi,
        port_mom_mix,
        signal_moments,
        universe_size,
        trailing,
    )

    # Classify attention level
    n_red = sum(1 for a in alerts if a["level"] == "RED")
    n_yellow = sum(1 for a in alerts if a["level"] == "YELLOW")
    if n_red > 0:
        attention = "RED"
    elif n_yellow > 0:
        attention = "YELLOW"
    else:
        attention = "GREEN"

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attention": attention,
        "n_portfolio": len(port_rows),
        "universe_size": universe_size,
        "prior_date": prior_date,
        "portfolio_vs_universe": factor_tilts,
        "portfolio_vs_trailing_20": trailing_comparison,
        "momentum_mix": {
            "portfolio": port_mom_mix,
            "universe": univ_mom_mix,
        },
        "signal_moments": signal_moments,
        "jaccard_prev": jaccard_prev,
        "rank_corr_prev": rank_corr_prev,
        "hhi": hhi,
        "alerts": alerts,
    }

    # Write artifact
    drift_dir.mkdir(parents=True, exist_ok=True)
    out_path = drift_dir / f"{as_of_date}_factor_drift.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", out_path)

    result["_path"] = str(out_path)
    return result


def _find_prior_date(date: str, snapshots_dir: Path) -> Optional[str]:
    """Find most recent snapshot date before the given date."""
    if not snapshots_dir.exists():
        return None
    candidates = []
    for d in snapshots_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if len(name) == 10 and name < date and (d / "rankings.csv").exists():
            candidates.append(name)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Two-baseline factor drift monitor")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args(argv)

    result = build_factor_drift(
        args.as_of_date,
        snapshots_dir=args.snapshots_dir,
        artifacts_dir=args.artifacts_dir,
        top_k=args.top_k,
    )

    if "error" in result:
        logger.error(result["error"])
        return 1

    n_alerts = len(result.get("alerts", []))
    logger.info(
        "Factor drift: %s (%d alerts), universe=%d, Jaccard=%s, HHI=%.0f",
        result["attention"],
        n_alerts,
        result["universe_size"],
        result.get("jaccard_prev", "?"),
        result["hhi"],
    )
    for a in result["alerts"]:
        logger.info("  [%s] %s: %s", a["level"], a["code"], a["detail"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
