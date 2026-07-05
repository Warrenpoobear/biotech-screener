#!/usr/bin/env python3
"""Data Integrity Audit — invariant checks + price-derived cross-validation.

Usage:
    python tools/data_integrity_audit.py --snapshot-dir data/snapshots/2026-02-19 \
        --price-history data/price_history.csv --as-of-date 2026-02-19

Outputs (in output/data_integrity/):
    invariants_report.csv       — every invariant violation
    price_recompute_diff.csv    — full-universe price-derived cross-validation
    catalyst_diff_sample.csv    — catalyst spot-check (25 tickers)
    root_cause_summary.md       — root cause classification + recommendations
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from common.corporate_actions import CorporateActionRegistry

# When invoked as a script (the production pipeline runs it as a direct
# subprocess), sys.path[0] is tools/, so repo-internal imports like
# common.corporate_actions are not resolvable. Put the repo root on the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DRAWDOWN_WINDOW = 252
RSI_PERIOD = 14
MIN_BARS_DD = 126
MIN_BARS_RSI = RSI_PERIOD + 1
# Must match run_screen.py MIN_OVERLAP_BARS
MIN_OVERLAP_BARS = 30

TOLERANCES = {
    "drawdown": 0.02,
    "drawdown_xbi": 0.02,
    "drawdown_rel_xbi": 0.03,
    "beta_xbi_60d": 0.20,
    "rsi_14d": 3.0,
    "alpha_60d": 0.03,
}

SANITY_RANGES = {
    "de_drawdown": (-1.0, 0.5),
    "de_drawdown_xbi": (-1.0, 0.5),
    "de_drawdown_rel_xbi": (-1.0, 1.0),
    "de_beta_xbi_60d": (-5.0, 5.0),
    "de_rsi_14d": (0.0, 100.0),
    "de_alpha_60d": (-5.0, 5.0),  # 60d excess return — small-cap biotech routinely > ±2
    "clinical_optionality_pct_dev": (0.0, 1.0),
    "clinical_rank_pct_dev": (0.0, 1.0),
    "commercial_quality_pct": (0.0, 1.0),
    "alpha_cohort_pct": (0.0, 1.0),
    "score_rank_pct": (0.0, 1.0),
    "missingness_penalty": (0.0, 3.0),
}

# Violation severity: determines exit code behaviour.
#   critical → exit 1 (FAIL)
#   warn     → exit 2 (WARN) — structural inconsistency needing human review
#   info     → exit 0 (OK)   — logged but not actionable
VIOLATION_SEVERITY: Dict[str, str] = {
    # Critical — data model broken
    "eligible_reasons_mismatch": "critical",
    "ineligible_has_rank": "critical",
    # Warn — structural inconsistency
    "catalyst_window_no_days": "warn",
    "catalyst_window_negative_days": "warn",
    "specific_days_no_days": "warn",
    "specific_days_non_integer": "warn",
    "specific_days_invalid": "warn",
    "penalty_no_components": "warn",
    "tier_no_reason": "warn",
    "deep_dd_no_value": "warn",
    "rsi_flag_no_value": "warn",
    "beta_flag_no_value": "warn",
    "unknown_missing_component": "warn",
    "universe_missing": "warn",
}
# Anything starting with "range_" defaults to "info" (see _violation_severity())


def _violation_severity(rule: str) -> str:
    """Return severity for a violation rule name."""
    if rule in VIOLATION_SEVERITY:
        return VIOLATION_SEVERITY[rule]
    if rule.startswith("range_"):
        return "info"
    return "warn"  # unknown rules default to warn


# ---------------------------------------------------------------------------
# Part B — Invariant Checks
# ---------------------------------------------------------------------------
def check_invariants(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Run invariant checks on rankings DataFrame. Returns list of violations."""
    violations = []

    def _add(ticker, rule, details):
        violations.append({"ticker": ticker, "rule": rule, "details": details})

    for _, row in df.iterrows():
        t = row.get("ticker", "?")
        eligible = str(row.get("eligible", "")).strip()
        inelig = str(row.get("ineligible_reasons", "")).strip()
        act_rank = str(row.get("actionable_rank", "")).strip()
        cat_iw = str(row.get("catalyst_in_window", "")).strip().lower()
        cat_days = row.get("catalyst_days")
        cat_mode = str(row.get("catalyst_mode", "")).strip()
        mp = row.get("missingness_penalty")
        mc = str(row.get("missing_components", "")).strip()
        tier_any = str(row.get("tier_any", "")).strip()
        tier_reason = str(row.get("tier_any_reason", "")).strip()
        risk = str(row.get("risk_flags", "")).strip()

        # 1. eligible=1 → ineligible_reasons empty
        if eligible == "1" and inelig not in ("", "nan"):
            _add(t, "eligible_reasons_mismatch", f"eligible=1 but ineligible_reasons='{inelig}'")

        # 2. eligible≠1 → actionable_rank empty
        if eligible != "1" and act_rank not in ("", "nan"):
            _add(t, "ineligible_has_rank", f"eligible={eligible} but actionable_rank={act_rank}")

        # 3. catalyst_in_window=True → catalyst_days present and >=0
        if cat_iw in ("true", "1"):
            if pd.isna(cat_days) or str(cat_days).strip() in ("", "nan"):
                _add(t, "catalyst_window_no_days", "catalyst_in_window=True but catalyst_days missing")
            elif float(cat_days) < 0:
                _add(t, "catalyst_window_negative_days", f"catalyst_in_window=True but catalyst_days={cat_days}")

        # 4. catalyst_mode=specific_days → catalyst_days integer-like
        if cat_mode == "specific_days":
            if pd.isna(cat_days) or str(cat_days).strip() in ("", "nan"):
                _add(t, "specific_days_no_days", "catalyst_mode=specific_days but catalyst_days missing")
            else:
                try:
                    d = float(cat_days)
                    if d != int(d):
                        _add(
                            t,
                            "specific_days_non_integer",
                            f"catalyst_mode=specific_days but catalyst_days={cat_days} (non-integer)",
                        )
                except (ValueError, TypeError):
                    _add(t, "specific_days_invalid", f"catalyst_mode=specific_days but catalyst_days='{cat_days}'")

        # 5. missingness_penalty > 0 → missing_components non-empty
        try:
            mp_val = float(mp) if pd.notna(mp) else 0.0
        except (ValueError, TypeError):
            mp_val = 0.0
        if mp_val > 0 and mc in ("", "nan"):
            _add(t, "penalty_no_components", f"missingness_penalty={mp_val} but missing_components empty")

        # 5b. missing_components values must be in known set
        if mc not in ("", "nan"):
            _KNOWN_MISSING_COMPONENTS = {"catalyst", "sponsor", "drawdown"}
            for comp in mc.split("|"):
                comp = comp.strip()
                if comp and comp not in _KNOWN_MISSING_COMPONENTS:
                    _add(
                        t,
                        "unknown_missing_component",
                        f"missing_components contains '{comp}', "
                        f"expected one of {sorted(_KNOWN_MISSING_COMPONENTS)}",
                    )

        # 6. tier_any non-empty → tier_any_reason non-empty
        if tier_any not in ("", "nan") and tier_reason in ("", "nan"):
            _add(t, "tier_no_reason", f"tier_any='{tier_any}' but tier_any_reason empty")

        # 7. risk_flags contains deep_drawdown → de_drawdown present
        if "deep_drawdown" in risk and "deep_drawdown_rel" not in risk.split("deep_drawdown")[0][-4:]:
            dd = row.get("de_drawdown")
            if pd.isna(dd) or str(dd).strip() in ("", "nan"):
                _add(t, "deep_dd_no_value", "risk_flags has deep_drawdown but de_drawdown missing")

        # 8. risk_flags contains overbought_rsi → de_rsi_14d present
        if "overbought_rsi" in risk:
            rsi = row.get("de_rsi_14d")
            if pd.isna(rsi) or str(rsi).strip() in ("", "nan"):
                _add(t, "rsi_flag_no_value", "risk_flags has overbought_rsi but de_rsi_14d missing")

        # 9. risk_flags contains high_beta → de_beta_xbi_60d present
        if "high_beta" in risk:
            beta = row.get("de_beta_xbi_60d")
            if pd.isna(beta) or str(beta).strip() in ("", "nan"):
                _add(t, "beta_flag_no_value", "risk_flags has high_beta but de_beta_xbi_60d missing")

        # --- Sanity range checks ---
        for col, (lo, hi) in SANITY_RANGES.items():
            val = row.get(col)
            if pd.isna(val) or str(val).strip() in ("", "nan"):
                continue
            try:
                v = float(val)
                if v < lo or v > hi:
                    _add(t, f"range_{col}", f"{col}={v:.6f} outside [{lo}, {hi}]")
            except (ValueError, TypeError):
                pass

    return violations


def check_universe_coverage(
    df: pd.DataFrame,
    universe_path: Path,
) -> List[Dict[str, str]]:
    """Verify all universe.json tickers are present in rankings DataFrame.

    Returns list of violations (one per missing ticker).
    """
    violations = []
    if not universe_path.exists():
        return violations

    with open(universe_path) as f:
        universe = json.load(f)

    uni_tickers: set[str] = set()
    for entry in universe:
        t = entry.get("ticker") if isinstance(entry, dict) else str(entry)
        if t and not t.startswith("_"):
            uni_tickers.add(t)

    rankings_tickers = set(df["ticker"].tolist()) if "ticker" in df.columns else set()
    missing = sorted(uni_tickers - rankings_tickers)
    for t in missing:
        violations.append(
            {
                "ticker": t,
                "rule": "universe_missing",
                "details": f"ticker '{t}' in universe.json but absent from rankings.csv",
            }
        )
    return violations


# ---------------------------------------------------------------------------
# Part C1 — Price-derived recomputation
# ---------------------------------------------------------------------------
def _compute_drawdown_from_prices(
    prices: pd.DataFrame,
    ticker: str,
    as_of: str,
    window: int = DRAWDOWN_WINDOW,
) -> Optional[float]:
    """Compute drawdown from price history CSV data."""
    sub = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)].copy()
    sub = sub[sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if len(sub) < MIN_BARS_DD:
        return None
    recent = sub.tail(window)
    peak = recent["close"].max()
    current = recent["close"].iloc[-1]
    if peak <= 0:
        return None
    return round((current / peak) - 1.0, 6)


def _compute_rsi_from_prices(
    prices: pd.DataFrame,
    ticker: str,
    as_of: str,
    period: int = RSI_PERIOD,
) -> Optional[float]:
    """Compute Wilder-smoothed RSI from price history CSV data."""
    sub = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)].copy()
    sub = sub[sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if len(sub) < period + 1:
        return None
    closes = sub["close"].values
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    # Seed with first `period` deltas
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    # Wilder smoothing
    for d in deltas[period:]:
        g = max(d, 0)
        loss = max(-d, 0)
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)


def _compute_beta_from_prices(
    prices: pd.DataFrame,
    ticker: str,
    as_of: str,
    window: int = 60,
) -> Optional[float]:
    """Compute 60-day beta vs XBI from price history.

    Mirrors run_screen.py _hydrate_beta_rsi() logic:
    - MIN_OVERLAP_BARS + 1 minimum aligned bars (not window + 1)
    - Deduplicate on date (keep last) to match dict-based XBI lookup
    - Take last min(window+1, len) aligned bars
    """
    t_sub = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)].copy()
    t_sub = t_sub[t_sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    x_sub = prices[(prices["ticker"] == "XBI") & (prices["date"] <= as_of)].copy()
    x_sub = x_sub[x_sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if len(t_sub) < 2 or len(x_sub) < 2:
        return None
    # Align on dates FIRST, then take the most recent window
    t_df = t_sub[["date", "close"]].rename(columns={"close": "t_close"})
    x_df = x_sub[["date", "close"]].rename(columns={"close": "x_close"})
    merged = t_df.merge(x_df, on="date", how="inner").sort_values("date")
    if len(merged) < MIN_OVERLAP_BARS + 1:
        return None
    merged = merged.tail(min(window + 1, len(merged)))
    t_rets = merged["t_close"].pct_change().dropna().values
    x_rets = merged["x_close"].pct_change().dropna().values
    if len(t_rets) < 10:
        return None
    # Beta = cov(t, x) / var(x)
    x_mean = sum(x_rets) / len(x_rets)
    t_mean = sum(t_rets) / len(t_rets)
    cov = sum((t_rets[i] - t_mean) * (x_rets[i] - x_mean) for i in range(len(t_rets))) / len(t_rets)
    var_x = sum((x_rets[i] - x_mean) ** 2 for i in range(len(x_rets))) / len(x_rets)
    if var_x == 0:
        return None
    return round(cov / var_x, 4)


def _compute_alpha_from_prices(
    prices: pd.DataFrame,
    ticker: str,
    as_of: str,
    window: int = 60,
    beta: Optional[float] = None,
) -> Optional[float]:
    """Compute 60-day alpha (excess return vs XBI) from price history.

    Uses date-aligned returns (inner-join by date, same as production code)
    to ensure ticker and XBI windows match exactly.
    Deduplicates on date (keep last) to match production dict-based lookup.
    """
    t_sub = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)].copy()
    t_sub = t_sub[t_sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    x_sub = prices[(prices["ticker"] == "XBI") & (prices["date"] <= as_of)].copy()
    x_sub = x_sub[x_sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if len(t_sub) < 2 or len(x_sub) < 2:
        return None
    # Align on dates FIRST, then take the most recent window
    t_df = t_sub[["date", "close"]].rename(columns={"close": "t_close"})
    x_df = x_sub[["date", "close"]].rename(columns={"close": "x_close"})
    merged = t_df.merge(x_df, on="date", how="inner").sort_values("date")
    if len(merged) < 2:
        return None
    merged = merged.tail(min(window + 1, len(merged)))
    t_ret = (merged["t_close"].iloc[-1] / merged["t_close"].iloc[0]) - 1.0
    x_ret = (merged["x_close"].iloc[-1] / merged["x_close"].iloc[0]) - 1.0
    b = beta if beta is not None else 1.0
    alpha = t_ret - b * x_ret
    return round(alpha, 6)


def _split_adjust_prices(
    prices: pd.DataFrame,
    as_of: str,
    registry: Optional["CorporateActionRegistry"] = None,
) -> pd.DataFrame:
    """Split-adjust raw closes to be comparable to ``as_of``.

    The production price-derived features (drawdown, RSI, beta, alpha) are
    computed from split-adjusted prices. This audit historically recomputed
    from the *raw* ``price_history.csv``, so any ticker with a split in the
    trailing window (e.g. MLTX's 10:1 forward split on 2025-09-29) produced a
    phantom drawdown mismatch — comparing a post-split price to a pre-split
    high. Applying the same ``corporate_actions.json`` split factors the
    features use makes the recompute a valid cross-check.

    Uses ``cumulative_split_factor`` per (ticker, date): the multiplier that
    scales a price on that date to be comparable to ``as_of``. Only tickers
    with a split on/before ``as_of`` are touched; everything else is returned
    unchanged. Fail-open: an empty/None registry is a no-op.
    """
    from common.corporate_actions import cumulative_split_factor, get_splits_only, load_actions

    reg = registry if registry is not None else load_actions()
    if not reg.actions:
        return prices

    split_tickers = {t for t in prices["ticker"].unique() if get_splits_only(t, reg, as_of=as_of)}
    if not split_tickers:
        return prices

    adjusted = prices.copy()
    mask = adjusted["ticker"].isin(split_tickers)
    adjusted.loc[mask, "close"] = adjusted.loc[mask].apply(
        lambda r: r["close"] * cumulative_split_factor(r["ticker"], r["date"], as_of, reg),
        axis=1,
    )
    return adjusted


def recompute_price_fields(
    rankings: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: str,
) -> List[Dict[str, Any]]:
    """Recompute all price-derived fields and compare against stored values."""
    results = []
    tickers = rankings["ticker"].tolist()

    # Pre-compute XBI drawdown once
    xbi_dd = _compute_drawdown_from_prices(prices, "XBI", as_of)

    for ticker in tickers:
        row = rankings[rankings["ticker"] == ticker].iloc[0]
        entry = {"ticker": ticker}

        # Drawdown
        recomp_dd = _compute_drawdown_from_prices(prices, ticker, as_of)
        stored_dd = _safe_float(row.get("de_drawdown"))
        entry["dd_stored"] = stored_dd
        entry["dd_recomputed"] = recomp_dd
        entry["dd_diff"] = (
            abs((stored_dd or 0) - (recomp_dd or 0)) if stored_dd is not None and recomp_dd is not None else None
        )
        entry["dd_verdict"] = _verdict(stored_dd, recomp_dd, TOLERANCES["drawdown"])
        entry["dd_source"] = "price_csv" if recomp_dd is not None else "no_data"

        # Drawdown XBI
        stored_dd_xbi = _safe_float(row.get("de_drawdown_xbi"))
        entry["dd_xbi_stored"] = stored_dd_xbi
        entry["dd_xbi_recomputed"] = xbi_dd
        entry["dd_xbi_diff"] = (
            abs((stored_dd_xbi or 0) - (xbi_dd or 0)) if stored_dd_xbi is not None and xbi_dd is not None else None
        )
        entry["dd_xbi_verdict"] = _verdict(stored_dd_xbi, xbi_dd, TOLERANCES["drawdown_xbi"])

        # Drawdown relative
        stored_dd_rel = _safe_float(row.get("de_drawdown_rel_xbi"))
        recomp_dd_rel = round(recomp_dd - xbi_dd, 6) if recomp_dd is not None and xbi_dd is not None else None
        entry["dd_rel_stored"] = stored_dd_rel
        entry["dd_rel_recomputed"] = recomp_dd_rel
        entry["dd_rel_diff"] = (
            abs((stored_dd_rel or 0) - (recomp_dd_rel or 0))
            if stored_dd_rel is not None and recomp_dd_rel is not None
            else None
        )
        entry["dd_rel_verdict"] = _verdict(stored_dd_rel, recomp_dd_rel, TOLERANCES["drawdown_rel_xbi"])

        # RSI
        recomp_rsi = _compute_rsi_from_prices(prices, ticker, as_of)
        stored_rsi = _safe_float(row.get("de_rsi_14d"))
        entry["rsi_stored"] = stored_rsi
        entry["rsi_recomputed"] = recomp_rsi
        entry["rsi_diff"] = (
            abs((stored_rsi or 0) - (recomp_rsi or 0)) if stored_rsi is not None and recomp_rsi is not None else None
        )
        entry["rsi_verdict"] = _verdict(stored_rsi, recomp_rsi, TOLERANCES["rsi_14d"])

        # Beta — check missing_reason first (xbi_stale etc. are explained, not FAIL)
        beta_missing_reason = str(row.get("de_beta_xbi_60d_missing_reason", "")).strip()
        if beta_missing_reason in ("nan", "None"):
            beta_missing_reason = ""
        recomp_beta = _compute_beta_from_prices(prices, ticker, as_of)
        stored_beta = _safe_float(row.get("de_beta_xbi_60d"))
        entry["beta_stored"] = stored_beta
        entry["beta_recomputed"] = recomp_beta
        entry["beta_missing_reason"] = beta_missing_reason
        entry["beta_diff"] = (
            abs((stored_beta or 0) - (recomp_beta or 0))
            if stored_beta is not None and recomp_beta is not None
            else None
        )
        if beta_missing_reason:
            entry["beta_verdict"] = f"EXPLAINED:{beta_missing_reason}"
        else:
            entry["beta_verdict"] = _verdict(stored_beta, recomp_beta, TOLERANCES["beta_xbi_60d"])

        # Alpha — check missing_reason first
        alpha_missing_reason = str(row.get("de_alpha_60d_missing_reason", "")).strip()
        if alpha_missing_reason in ("nan", "None"):
            alpha_missing_reason = ""
        recomp_alpha = _compute_alpha_from_prices(prices, ticker, as_of, beta=recomp_beta)
        stored_alpha = _safe_float(row.get("de_alpha_60d"))
        entry["alpha_stored"] = stored_alpha
        entry["alpha_recomputed"] = recomp_alpha
        entry["alpha_missing_reason"] = alpha_missing_reason
        entry["alpha_diff"] = (
            abs((stored_alpha or 0) - (recomp_alpha or 0))
            if stored_alpha is not None and recomp_alpha is not None
            else None
        )
        if alpha_missing_reason:
            entry["alpha_verdict"] = f"EXPLAINED:{alpha_missing_reason}"
        else:
            entry["alpha_verdict"] = _verdict(stored_alpha, recomp_alpha, TOLERANCES["alpha_60d"])

        results.append(entry)

    return results


def _safe_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        s = str(val).strip()
        if s in ("", "nan", "None"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _verdict(stored, recomputed, tol) -> str:
    if stored is None and recomputed is None:
        return "BOTH_MISSING"
    if stored is None:
        return "STORED_MISSING"
    if recomputed is None:
        return "RECOMP_MISSING"
    diff = abs(stored - recomputed)
    if diff <= tol:
        return "OK"
    return "FAIL"


# ---------------------------------------------------------------------------
# Part C2 — Catalyst spot-check
# ---------------------------------------------------------------------------
def catalyst_spot_check(
    rankings: pd.DataFrame,
    snapshot_dir: Path,
    n_sample: int = 25,
) -> List[Dict[str, str]]:
    """Spot-check catalyst_days/mode/in_window for a sample of tickers."""
    results = []

    # Try to load catalyst source mix for cross-reference (loaded for future use)
    csm_path = snapshot_dir / "catalyst_source_mix.json"
    if csm_path.exists():
        with open(csm_path) as f:
            _ = json.load(f)

    # Sample across modes and sources
    modes = rankings["catalyst_mode"].unique()
    sample_tickers = []
    for mode in modes:
        mode_df = rankings[rankings["catalyst_mode"] == mode]
        n_take = max(1, min(5, len(mode_df)))
        sample_tickers.extend(mode_df["ticker"].head(n_take).tolist())
    # Top up from remaining
    remaining = [t for t in rankings["ticker"] if t not in sample_tickers]
    need = max(0, n_sample - len(sample_tickers))
    sample_tickers.extend(remaining[:need])
    sample_tickers = sample_tickers[:n_sample]

    for ticker in sample_tickers:
        row = rankings[rankings["ticker"] == ticker].iloc[0]
        entry = {
            "ticker": ticker,
            "stored_mode": str(row.get("catalyst_mode", "")),
            "stored_days": str(row.get("catalyst_days", "")),
            "stored_de_days": str(row.get("de_catalyst_days", "")),
            "stored_de_mode": str(row.get("de_catalyst_mode", "")),
            "stored_in_window": str(row.get("catalyst_in_window", "")),
            "stored_de_in_window": str(row.get("de_catalyst_in_window", "")),
            "stored_source": str(row.get("catalyst_source", "")),
            "stored_event_type": str(row.get("catalyst_event_type", "")),
        }

        # Cross-check: catalyst_days vs de_catalyst_days should match
        cd = str(row.get("catalyst_days", "")).strip()
        de_cd = str(row.get("de_catalyst_days", "")).strip()
        if cd != de_cd and cd not in ("", "nan") and de_cd not in ("", "nan"):
            entry["days_match"] = "MISMATCH"
            entry["days_detail"] = f"catalyst_days={cd} vs de_catalyst_days={de_cd}"
        else:
            entry["days_match"] = "OK"
            entry["days_detail"] = ""

        # Cross-check: mode vs days consistency
        mode = str(row.get("catalyst_mode", "")).strip()
        if mode == "specific_days":
            try:
                d = float(cd) if cd not in ("", "nan") else None
                if d is not None and d < 0:
                    entry["mode_consistency"] = "FAIL"
                    entry["mode_detail"] = f"specific_days but days={d} (negative)"
                elif d is None:
                    entry["mode_consistency"] = "FAIL"
                    entry["mode_detail"] = "specific_days but days missing"
                else:
                    entry["mode_consistency"] = "OK"
                    entry["mode_detail"] = ""
            except (ValueError, TypeError):
                entry["mode_consistency"] = "FAIL"
                entry["mode_detail"] = f"specific_days but days invalid: {cd}"
        elif mode in ("no_upcoming", "missing"):
            if cd not in ("", "nan", "None") and pd.notna(row.get("catalyst_days")):
                entry["mode_consistency"] = "WARN"
                entry["mode_detail"] = f"mode={mode} but catalyst_days={cd} present"
            else:
                entry["mode_consistency"] = "OK"
                entry["mode_detail"] = ""
        else:
            entry["mode_consistency"] = "OK"
            entry["mode_detail"] = ""

        # Cross-check: in_window vs days
        iw = str(row.get("de_catalyst_in_window", "")).strip().lower()
        if iw in ("true", "1"):
            try:
                d = float(de_cd) if de_cd not in ("", "nan") else None
                if d is not None and d > 180:
                    entry["window_consistency"] = "WARN"
                    entry["window_detail"] = f"in_window=True but days={d} > 180"
                else:
                    entry["window_consistency"] = "OK"
                    entry["window_detail"] = ""
            except (ValueError, TypeError):
                entry["window_consistency"] = "FAIL"
                entry["window_detail"] = f"in_window=True but days invalid: {de_cd}"
        else:
            entry["window_consistency"] = "OK"
            entry["window_detail"] = ""

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Part D — Root cause summary
# ---------------------------------------------------------------------------
def classify_root_causes(price_diffs: List[Dict]) -> Dict[str, Any]:
    """Classify FAIL rows by root cause pattern.

    EXPLAINED verdicts (from explicit missing_reason) are tracked separately.
    """
    causes = defaultdict(list)
    for row in price_diffs:
        ticker = row["ticker"]
        for field_prefix in ["dd", "rsi", "beta", "alpha"]:
            verdict_key = f"{field_prefix}_verdict"
            verdict = str(row.get(verdict_key, ""))
            # EXPLAINED verdicts are tracked but not mixed with FAIL
            if verdict.startswith("EXPLAINED:"):
                reason = verdict.split(":", 1)[1]
                causes["explained_missing_reason"].append(
                    {
                        "ticker": ticker,
                        "field": field_prefix,
                        "reason": reason,
                    }
                )
                continue
            if verdict != "FAIL":
                continue
            stored = row.get(f"{field_prefix}_stored")
            recomp = row.get(f"{field_prefix}_recomputed")

            # Classify
            if stored is not None and recomp is not None:
                causes["stale_but_present"].append(
                    {
                        "ticker": ticker,
                        "field": field_prefix,
                        "stored": stored,
                        "recomputed": recomp,
                        "diff": abs(stored - recomp),
                    }
                )
            elif stored is None and recomp is not None:
                causes["missing_in_snapshot"].append(
                    {
                        "ticker": ticker,
                        "field": field_prefix,
                    }
                )
            elif stored is not None and recomp is None:
                causes["no_price_data"].append(
                    {
                        "ticker": ticker,
                        "field": field_prefix,
                    }
                )
    return dict(causes)


def write_root_cause_summary(
    causes: Dict[str, Any],
    invariant_violations: List[Dict],
    price_diffs: List[Dict],
    catalyst_diffs: List[Dict],
    out_path: Path,
):
    """Write root_cause_summary.md."""
    lines = ["# Data Integrity Audit — Root Cause Summary\n"]
    lines.append(f"Generated: {datetime.now(tz=None).isoformat()}Z\n")

    # --- Price recompute summary ---
    lines.append("## 1. Price-Derived Cross-Validation\n")
    for field in ["dd", "rsi", "beta", "alpha", "dd_xbi", "dd_rel"]:
        vk = f"{field}_verdict"
        total = sum(1 for r in price_diffs if r.get(vk) is not None)
        ok = sum(1 for r in price_diffs if r.get(vk) == "OK")
        fail = sum(1 for r in price_diffs if r.get(vk) == "FAIL")
        explained = sum(1 for r in price_diffs if str(r.get(vk, "")).startswith("EXPLAINED:"))
        missing = total - ok - fail - explained
        label = {
            "dd": "Drawdown (252d)",
            "rsi": "RSI (14d)",
            "beta": "Beta XBI (60d)",
            "alpha": "Alpha (60d)",
            "dd_xbi": "Drawdown XBI",
            "dd_rel": "Drawdown Relative",
        }.get(field, field)
        parts = [f"{ok} OK", f"{fail} FAIL"]
        if explained:
            parts.append(f"{explained} EXPLAINED")
        parts.append(f"{missing} missing/skip (of {total})")
        lines.append(f"- **{label}**: {', '.join(parts)}")
    lines.append("")

    # --- Root cause classification ---
    lines.append("## 2. Root Cause Classification\n")
    if not causes:
        lines.append("No FAIL rows found — all price-derived fields within tolerance.\n")
    else:
        for cause, items in sorted(causes.items(), key=lambda x: -len(x[1])):
            nice = {
                "stale_but_present": "Stale-but-present (hydration only filled missing)",
                "missing_in_snapshot": "Missing in snapshot but recomputable",
                "no_price_data": "No price data to recompute",
            }.get(cause, cause)
            lines.append(f"### {nice}: {len(items)} cases\n")
            # Top offenders
            by_field = defaultdict(list)
            for item in items:
                by_field[item["field"]].append(item)
            for fld, fld_items in sorted(by_field.items()):
                tickers = [i["ticker"] for i in fld_items[:10]]
                lines.append(f"- **{fld}** ({len(fld_items)}): {', '.join(tickers)}")
                if fld_items and "diff" in fld_items[0]:
                    worst = sorted(fld_items, key=lambda x: -x.get("diff", 0))[:3]
                    for w in worst:
                        lines.append(
                            f"  - {w['ticker']}: stored={w['stored']:.4f}, recomp={w['recomputed']:.4f}, diff={w['diff']:.4f}"
                        )
            lines.append("")

    # --- Invariant violations ---
    lines.append("## 3. Invariant Violations\n")
    if not invariant_violations:
        lines.append("No invariant violations found.\n")
    else:
        by_rule = defaultdict(list)
        for v in invariant_violations:
            by_rule[v["rule"]].append(v)
        for rule, items in sorted(by_rule.items(), key=lambda x: -len(x[1])):
            lines.append(f"- **{rule}** ({len(items)}): {', '.join(i['ticker'] for i in items[:10])}")
            if len(items) > 10:
                lines.append(f"  ... and {len(items) - 10} more")
        lines.append("")

    # --- Catalyst spot-check ---
    lines.append("## 4. Catalyst Spot-Check\n")
    n_cat = len(catalyst_diffs)
    n_days_mm = sum(1 for r in catalyst_diffs if r.get("days_match") == "MISMATCH")
    n_mode_fail = sum(1 for r in catalyst_diffs if r.get("mode_consistency") == "FAIL")
    n_win_fail = sum(1 for r in catalyst_diffs if r.get("window_consistency") in ("FAIL", "WARN"))
    lines.append(f"- Sampled {n_cat} tickers")
    lines.append(f"- Days mismatch: {n_days_mm}")
    lines.append(f"- Mode consistency failures: {n_mode_fail}")
    lines.append(f"- Window consistency issues: {n_win_fail}")
    if n_days_mm + n_mode_fail + n_win_fail == 0:
        lines.append("- **All spot-checks passed.**")
    else:
        lines.append("")
        for r in catalyst_diffs:
            issues = []
            if r.get("days_match") == "MISMATCH":
                issues.append(f"days: {r['days_detail']}")
            if r.get("mode_consistency") == "FAIL":
                issues.append(f"mode: {r['mode_detail']}")
            if r.get("window_consistency") in ("FAIL", "WARN"):
                issues.append(f"window: {r['window_detail']}")
            if issues:
                lines.append(f"- **{r['ticker']}**: {'; '.join(issues)}")
    lines.append("")

    # --- Recommendations ---
    lines.append("## 5. Recommendations\n")
    stale_count = len(causes.get("stale_but_present", []))
    stale_fields = set(i["field"] for i in causes.get("stale_but_present", []))

    if stale_count > 0:
        lines.append(
            f"### CRITICAL: {stale_count} stale-but-present values found in fields: {', '.join(sorted(stale_fields))}\n"
        )
        if "beta" in stale_fields:
            lines.append("- **`beta_xbi_60d`**: Change `_hydrate_beta_rsi()` to OVERWRITE existing pipeline values")
            lines.append("  (same fix as drawdown: recompute from price_history.csv for all tickers)")
        if "rsi" in stale_fields:
            lines.append("- **`rsi_14d`**: Change `_hydrate_beta_rsi()` to OVERWRITE existing pipeline values")
        if "alpha" in stale_fields:
            lines.append(
                "- **`alpha_60d`**: Add inline recomputation from price_history.csv in save_validation_snapshot()"
            )
            lines.append("  Formula: `alpha_60d = return_60d - beta * xbi_return_60d`")
        lines.append("")

    lines.append("### Source Tagging (recommended for all price-derived fields)")
    lines.append("- Add `de_drawdown_source`, `de_beta_source`, `de_rsi_source`, `de_alpha_source` columns")
    lines.append("- Values: 'price_csv', 'cache', 'pipeline_fallback', 'missing'")
    lines.append("")

    lines.append("### Regression Tests")
    lines.append("- Add test for 'stale-but-present' beta overwrite (mirrors test_stale_drawdown_overwrite)")
    lines.append("- Add test for 'stale-but-present' RSI overwrite")
    lines.append("- Add invariant assertion in CI (fast mode: load rankings.csv, run check_invariants)")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Data Integrity Audit")
    parser.add_argument("--snapshot-dir", required=True, help="Path to snapshot directory")
    parser.add_argument("--price-history", required=True, help="Path to price_history.csv")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="output/data_integrity", help="Output directory")
    parser.add_argument("--skip-prices", action="store_true", help="Skip price recomputation (fast mode)")
    parser.add_argument("--universe", default=None, help="Path to universe.json (for coverage check)")
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rankings
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        print(f"ERROR: {rankings_path} not found")
        sys.exit(1)
    df = pd.read_csv(rankings_path)
    print(f"Loaded {len(df)} rows from {rankings_path}")

    # --- Part B: Invariants ---
    print("\n=== Part B: Invariant Checks ===")
    violations = check_invariants(df)

    # Universe coverage check (optional)
    universe_path = Path(args.universe) if args.universe else (snap_dir.parent / "universe.json")
    if universe_path.exists():
        uni_violations = check_universe_coverage(df, universe_path)
        if uni_violations:
            violations.extend(uni_violations)
            print(f"  Universe coverage: {len(uni_violations)} missing tickers")
        else:
            print("  Universe coverage: all tickers present")
    else:
        print(f"  Universe coverage: skipped ({universe_path} not found)")

    inv_path = out_dir / "invariants_report.csv"
    if violations:
        pd.DataFrame(violations).to_csv(inv_path, index=False)
        print(f"  {len(violations)} violations → {inv_path}")
        by_rule = defaultdict(int)
        for v in violations:
            by_rule[v["rule"]] += 1
        for rule, cnt in sorted(by_rule.items(), key=lambda x: -x[1]):
            print(f"    {rule}: {cnt}")
    else:
        pd.DataFrame(columns=["ticker", "rule", "details"]).to_csv(inv_path, index=False)
        print("  No violations found.")

    # --- Part C1: Price recomputation ---
    price_diffs = []
    if not args.skip_prices:
        print("\n=== Part C1: Price-Derived Recomputation ===")
        price_path = Path(args.price_history)
        if not price_path.exists():
            print(f"  WARNING: {price_path} not found, skipping price recomputation")
        else:
            prices = pd.read_csv(price_path, dtype={"ticker": str})
            prices["date"] = prices["date"].astype(str)
            print(f"  Loaded {len(prices)} price rows, {prices['ticker'].nunique()} tickers")
            # Cross-validate against split-adjusted prices — the same basis the
            # production features use. Without this, split tickers (e.g. MLTX's
            # 10:1 forward split) false-flag as stale drawdown mismatches.
            adjusted = _split_adjust_prices(prices, args.as_of_date)
            if adjusted is not prices:
                print("  Applied corporate-action split adjustment before recompute")
            price_diffs = recompute_price_fields(df, adjusted, args.as_of_date)
            diff_path = out_dir / "price_recompute_diff.csv"
            pd.DataFrame(price_diffs).to_csv(diff_path, index=False)

            # Summary
            for field in ["dd", "rsi", "beta", "alpha", "dd_xbi", "dd_rel"]:
                vk = f"{field}_verdict"
                ok = sum(1 for r in price_diffs if r.get(vk) == "OK")
                fail = sum(1 for r in price_diffs if r.get(vk) == "FAIL")
                total = sum(1 for r in price_diffs if r.get(vk) is not None)
                label = {
                    "dd": "drawdown",
                    "rsi": "rsi_14d",
                    "beta": "beta_60d",
                    "alpha": "alpha_60d",
                    "dd_xbi": "dd_xbi",
                    "dd_rel": "dd_rel",
                }.get(field, field)
                status = "PASS" if fail == 0 else "FAIL"
                print(f"  [{status}] {label}: {ok}/{total} OK, {fail} FAIL")
                if fail > 0:
                    fails = [r for r in price_diffs if r.get(vk) == "FAIL"]
                    for f_row in fails[:5]:
                        print(
                            f"         {f_row['ticker']}: stored={f_row.get(f'{field}_stored')}, recomp={f_row.get(f'{field}_recomputed')}"
                        )
    else:
        print("\n=== Part C1: Skipped (--skip-prices) ===")

    # --- Part C2: Catalyst spot-check ---
    print("\n=== Part C2: Catalyst Spot-Check ===")
    catalyst_diffs = catalyst_spot_check(df, snap_dir)
    cat_path = out_dir / "catalyst_diff_sample.csv"
    pd.DataFrame(catalyst_diffs).to_csv(cat_path, index=False)
    n_issues = sum(
        1
        for r in catalyst_diffs
        if r.get("days_match") == "MISMATCH"
        or r.get("mode_consistency") == "FAIL"
        or r.get("window_consistency") in ("FAIL", "WARN")
    )
    print(f"  Sampled {len(catalyst_diffs)} tickers, {n_issues} issues")

    # --- Part D: Root causes ---
    print("\n=== Part D: Root Cause Summary ===")
    causes = classify_root_causes(price_diffs)
    summary_path = out_dir / "root_cause_summary.md"
    write_root_cause_summary(causes, violations, price_diffs, catalyst_diffs, summary_path)
    total_stale = len(causes.get("stale_but_present", []))
    print(f"  Stale-but-present: {total_stale}")
    print(f"  Written to {summary_path}")

    # --- Classify violations by severity ---
    by_severity: Dict[str, List[Dict[str, str]]] = {
        "critical": [],
        "warn": [],
        "info": [],
    }
    for v in violations:
        sev = _violation_severity(v["rule"])
        by_severity[sev].append(v)

    # Write machine-readable summary (consumed by run_daily_production.py)
    summary = {
        "total": len(violations),
        "critical": len(by_severity["critical"]),
        "warn": len(by_severity["warn"]),
        "info": len(by_severity["info"]),
        "by_rule": {},
    }
    for v in violations:
        key = v["rule"]
        summary["by_rule"][key] = summary["by_rule"].get(key, 0) + 1
    summary_path_json = out_dir / "invariants_summary.json"
    with open(summary_path_json, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Exit code (4-tier semantic) ---
    #   0 = OK (info-only)
    #   1 = Critical invariant violation (data model broken) → FAIL
    #   2 = Structural warning (catalyst/tier inconsistency) → WARN
    #   3 = Price recompute mismatch (stale but explained)  → WARN
    has_price_fails = any(r.get(f"{f}_verdict") == "FAIL" for r in price_diffs for f in ["dd", "rsi", "beta", "alpha"])
    has_critical = len(by_severity["critical"]) > 0
    has_warn = len(by_severity["warn"]) > 0

    if has_critical:
        verdict = "FAIL"
        exit_code = 1  # data model broken → hard FAIL
    elif has_price_fails:
        verdict = "STALE_MISMATCH"
        exit_code = 3  # stale recompute mismatch → WARN
    elif has_warn:
        verdict = "WARN"
        exit_code = 2  # structural inconsistency → WARN
    else:
        # info-only violations (e.g. range outliers) → OK
        verdict = "OK"
        exit_code = 0

    n_info = len(by_severity["info"])
    info_suffix = f" ({n_info} info-level)" if n_info > 0 and verdict == "OK" else ""
    print(f"\n{'='*60}")
    print(f"AUDIT RESULT: {verdict}{info_suffix}")
    print(f"{'='*60}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
