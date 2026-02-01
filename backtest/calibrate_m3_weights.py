#!/usr/bin/env python3
"""
Milestone 3: Module 5 weight calibration from backtest panel.

Learns component weights via pooled ridge regression (or FMB betas),
enforces sign priors from diagnostics, L1-normalizes, and writes JSON
for optional consumption by Module 5.

Usage:
    python3 backtest/calibrate_m3_weights.py \
        --panel output/backtest_m1_full/panel_weekly.csv.gz \
        --return-col fwd_21d \
        --out data/module5_weights_m3.json

Dependencies: pandas, numpy only (no scipy/sklearn/statsmodels).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.fmb import (
    fama_macbeth,
    zscore_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_m3")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES = [
    "clinical_normalized",
    "financial_normalized",
    "momentum_normalized",
    "valuation_normalized",
]

# Sign priors from diagnostics: expected sign of weight.
# None = no constraint (allow either sign).
_SIGN_PRIORS: Dict[str, Optional[int]] = {
    "valuation_normalized": -1,
    "momentum_normalized": -1,
    "financial_normalized": -1,
    "clinical_normalized": None,  # unstable, no prior
    "catalyst_normalized": None,
    "uncertainty_penalty": None,
    "composite_score": None,
}


# ---------------------------------------------------------------------------
# Ridge regression (pooled, cross-sectionally z-scored)
# ---------------------------------------------------------------------------


def pooled_ridge(
    panel: pd.DataFrame,
    return_col: str,
    feature_cols: List[str],
    date_col: str = "rebalance_date",
    ridge_lambda: float = 10.0,
    min_stocks_per_week: int = 100,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Pooled ridge regression on cross-sectionally z-scored features.

    For each date, z-scores features cross-sectionally, then stacks all
    dates and solves: w = (X'X + λI)^{-1} X'y via least-squares.

    Args:
        panel: DataFrame with date_col, feature_cols, return_col
        return_col: forward return column name
        feature_cols: list of feature column names
        date_col: date column name
        ridge_lambda: regularization strength (higher = more shrinkage)
        min_stocks_per_week: minimum cross-section size to include a week

    Returns:
        (weights_dict, metadata_dict)
    """
    feature_cols = sorted(feature_cols)
    df = panel[[date_col, "ticker"] + feature_cols + [return_col]].copy()
    df = df.dropna(subset=[return_col])

    # Z-score features cross-sectionally per date
    df = zscore_features(df, feature_cols, date_col)

    # Filter dates with enough observations and non-constant features
    dates = sorted(df[date_col].unique())
    usable_dates = []
    dropped_dates = []
    for dt in dates:
        mask = df[date_col] == dt
        n = mask.sum()
        if n < min_stocks_per_week:
            dropped_dates.append((dt, f"too_few_stocks_{n}"))
            continue
        # Check for constant features (std ~ 0 after z-scoring = all NaN → filled 0)
        slice_df = df.loc[mask, feature_cols]
        non_constant = [
            c for c in feature_cols
            if slice_df[c].std(ddof=1) > 1e-12
        ]
        if len(non_constant) < len(feature_cols):
            dropped_cols = sorted(set(feature_cols) - set(non_constant))
            dropped_dates.append((dt, f"constant_features:{','.join(dropped_cols)}"))
            continue
        usable_dates.append(dt)

    if not usable_dates:
        logger.warning("No usable dates for ridge regression")
        return {c: 0.0 for c in feature_cols}, {"n_usable_dates": 0}

    # Stack usable dates
    usable_mask = df[date_col].isin(usable_dates)
    X = df.loc[usable_mask, feature_cols].values.astype(np.float64)
    y = df.loc[usable_mask, return_col].values.astype(np.float64)

    # Drop any remaining NaN rows
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    n_obs, k = X.shape
    if n_obs < k + 2:
        logger.warning(f"Insufficient observations ({n_obs}) for {k} features")
        return {c: 0.0 for c in feature_cols}, {"n_usable_dates": len(usable_dates)}

    # Ridge: (X'X + λI)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    ridge_mat = XtX + ridge_lambda * np.eye(k)

    try:
        weights_vec, _, _, _ = np.linalg.lstsq(ridge_mat, Xty, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("Ridge solve failed, returning zero weights")
        return {c: 0.0 for c in feature_cols}, {"n_usable_dates": len(usable_dates)}

    weights = {col: float(weights_vec[i]) for i, col in enumerate(feature_cols)}

    metadata = {
        "n_usable_dates": len(usable_dates),
        "n_dropped_dates": len(dropped_dates),
        "n_observations": int(n_obs),
        "avg_n_stocks": round(n_obs / max(len(usable_dates), 1), 1),
        "dropped_dates_sample": dropped_dates[:5],
    }

    return weights, metadata


# ---------------------------------------------------------------------------
# FMB-based weight estimation
# ---------------------------------------------------------------------------


def fmb_weights(
    panel: pd.DataFrame,
    return_col: str,
    feature_cols: List[str],
    date_col: str = "rebalance_date",
    nw_lags: int = 4,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Estimate weights as FMB mean betas (already implemented in fmb.py).

    Returns:
        (weights_dict, metadata_dict)
    """
    summary, _ = fama_macbeth(
        panel,
        return_col=return_col,
        feature_cols=feature_cols,
        date_col=date_col,
        standardize=True,
        nw_lags=nw_lags,
        prune_collinear=True,
    )

    weights = {}
    meta_rows = []
    for _, row in summary.iterrows():
        feat = row["feature"]
        weights[feat] = float(row["mean_beta"])
        meta_rows.append({
            "feature": feat,
            "mean_beta": row["mean_beta"],
            "t_stat": row["t_stat"],
            "n_valid_weeks": int(row["n_valid_weeks"]),
        })

    # Features that were in feature_cols but not in summary (fully dropped)
    for c in feature_cols:
        if c not in weights:
            weights[c] = 0.0

    metadata = {
        "n_features_estimated": len(meta_rows),
        "fmb_details": meta_rows,
    }

    return weights, metadata


# ---------------------------------------------------------------------------
# Sign priors + normalization
# ---------------------------------------------------------------------------


def apply_sign_priors(
    weights: Dict[str, float],
    sign_priors: Dict[str, Optional[int]],
    disable_negative: bool = False,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Enforce sign priors. If a weight violates its prior, set to 0.

    Args:
        weights: raw estimated weights
        sign_priors: {feature: expected_sign} where sign is -1, +1, or None
        disable_negative: if True, clamp all negative weights to 0

    Returns:
        (adjusted_weights, violations_log)
    """
    adjusted = {}
    violations = {}

    for feat, w in sorted(weights.items()):
        if disable_negative and w < 0:
            adjusted[feat] = 0.0
            violations[feat] = f"negative_disabled (was {w:.6f})"
            continue

        prior = sign_priors.get(feat)
        if prior is not None:
            if prior < 0 and w > 0:
                adjusted[feat] = 0.0
                violations[feat] = f"sign_prior_negative_violated (was +{w:.6f})"
            elif prior > 0 and w < 0:
                adjusted[feat] = 0.0
                violations[feat] = f"sign_prior_positive_violated (was {w:.6f})"
            else:
                adjusted[feat] = w
        else:
            adjusted[feat] = w

    return adjusted, violations


def normalize_weights(
    weights: Dict[str, float],
    max_abs_weight: float = 0.60,
) -> Dict[str, float]:
    """
    L1-normalize weights to sum(|w|) = 1.0, with per-weight cap.

    Iteratively caps and re-normalizes until all weights are within bounds.
    """
    result = {f: w for f, w in sorted(weights.items())}

    for _ in range(20):  # convergence is fast
        total_abs = sum(abs(w) for w in result.values())
        if total_abs < 1e-15:
            return {f: 0.0 for f in sorted(result.keys())}

        # L1 normalize
        result = {f: w / total_abs for f, w in result.items()}

        # Check cap
        any_capped = False
        for f in result:
            if abs(result[f]) > max_abs_weight:
                result[f] = max_abs_weight if result[f] > 0 else -max_abs_weight
                any_capped = True

        if not any_capped:
            break

    # Final rounding
    return {f: round(w, 8) for f, w in sorted(result.items())}


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _git_hash() -> Optional[str]:
    """Get current git commit hash, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def write_weights_json(
    path: Path,
    weights: Dict[str, float],
    metadata: Dict[str, Any],
) -> None:
    """Write calibrated weights JSON with stable formatting."""
    content = json.dumps(
        {**metadata, "feature_weights": weights},
        sort_keys=True,
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def write_calibration_report(
    path: Path,
    raw_weights: Dict[str, float],
    final_weights: Dict[str, float],
    violations: Dict[str, str],
    dropped: Dict[str, str],
) -> None:
    """Write per-feature calibration report CSV."""
    rows = []
    all_features = sorted(set(raw_weights) | set(final_weights))
    for feat in all_features:
        rows.append({
            "feature": feat,
            "raw_weight": round(raw_weights.get(feat, 0.0), 8),
            "final_weight": round(final_weights.get(feat, 0.0), 8),
            "sign_violation": violations.get(feat, ""),
            "dropped_reason": dropped.get(feat, ""),
        })
    df = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="M3: Calibrate Module 5 weights from backtest panel",
    )
    p.add_argument("--panel", required=True, help="Path to M1 panel (csv.gz)")
    p.add_argument("--out", default="data/module5_weights_m3.json",
                   help="Output weights JSON path")
    p.add_argument("--outdir", default=None,
                   help="Directory for report CSVs (default: same as --out parent)")
    p.add_argument("--return-col", default="fwd_21d",
                   help="Forward return column to calibrate on")
    p.add_argument("--features", nargs="+", default=None,
                   help="Feature columns (default: 4 core normalized)")
    p.add_argument("--method", choices=["ridge", "fmb"], default="ridge",
                   help="Estimation method (default: ridge)")
    p.add_argument("--ridge-lambda", type=float, default=10.0,
                   help="Ridge regularization parameter (default: 10.0)")
    p.add_argument("--min-stocks-per-week", type=int, default=100,
                   help="Minimum cross-section size per week (default: 100)")
    p.add_argument("--max-abs-weight", type=float, default=0.60,
                   help="Maximum absolute weight per feature (default: 0.60)")
    p.add_argument("--nw-lags", type=int, default=4,
                   help="Newey-West lags for FMB method (default: 4)")
    p.add_argument("--m3-disable-negative", action="store_true",
                   help="Clamp all negative weights to 0 (long-only mode)")
    p.add_argument("--date-col", default="rebalance_date")
    return p.parse_args(argv)


def calibrate(
    panel: pd.DataFrame,
    return_col: str,
    feature_cols: List[str],
    method: str = "ridge",
    ridge_lambda: float = 10.0,
    min_stocks_per_week: int = 100,
    nw_lags: int = 4,
    max_abs_weight: float = 0.60,
    disable_negative: bool = False,
    date_col: str = "rebalance_date",
    sign_priors: Optional[Dict[str, Optional[int]]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Full calibration pipeline: estimate → sign priors → normalize.

    Returns:
        (final_weights, full_metadata)
    """
    if sign_priors is None:
        sign_priors = _SIGN_PRIORS

    feature_cols = sorted(feature_cols)
    present = [c for c in feature_cols if c in panel.columns]
    missing = sorted(set(feature_cols) - set(present))

    # Check coverage: drop features that are constant in >80% of weeks
    dates = sorted(panel[date_col].unique())
    dropped_features: Dict[str, str] = {}
    for feat in list(present):
        constant_weeks = 0
        for dt in dates:
            vals = panel.loc[panel[date_col] == dt, feat].dropna()
            if len(vals) < 2 or vals.std(ddof=1) < 1e-12:
                constant_weeks += 1
        pct_constant = constant_weeks / max(len(dates), 1)
        if pct_constant > 0.80:
            dropped_features[feat] = f"constant_{pct_constant:.0%}_of_weeks"
            present.remove(feat)
    for feat in missing:
        dropped_features[feat] = "not_in_panel"

    if not present:
        logger.warning("No usable features after coverage filter")
        final = {c: 0.0 for c in feature_cols}
        meta = {"method": method, "error": "no_usable_features"}
        return final, meta

    logger.info(f"  Usable features: {present}")
    if dropped_features:
        logger.info(f"  Dropped features: {dropped_features}")

    # Estimate raw weights
    if method == "ridge":
        raw_weights, est_meta = pooled_ridge(
            panel, return_col, present, date_col,
            ridge_lambda=ridge_lambda,
            min_stocks_per_week=min_stocks_per_week,
        )
    elif method == "fmb":
        raw_weights, est_meta = fmb_weights(
            panel, return_col, present, date_col,
            nw_lags=nw_lags,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply sign priors
    adjusted, violations = apply_sign_priors(
        raw_weights, sign_priors, disable_negative=disable_negative,
    )

    # Add zero entries for dropped features
    for feat in dropped_features:
        adjusted[feat] = 0.0

    # Normalize
    final = normalize_weights(adjusted, max_abs_weight=max_abs_weight)

    # Build metadata
    date_range = (dates[0], dates[-1]) if dates else ("", "")
    metadata = {
        "as_of_date": date_range[1],
        "date_range_start": date_range[0],
        "date_range_end": date_range[1],
        "return_col": return_col,
        "method": method,
        "ridge_lambda": ridge_lambda if method == "ridge" else None,
        "nw_lags": nw_lags if method == "fmb" else None,
        "n_weeks": len(dates),
        "features_requested": sorted(feature_cols),
        "features_used": sorted(present),
        "dropped_features": dropped_features,
        "sign_violations": violations,
        "max_abs_weight": max_abs_weight,
        "disable_negative": disable_negative,
        "estimation_details": est_meta,
        "raw_weights": {k: round(v, 8) for k, v in sorted(raw_weights.items())},
        "git_hash": _git_hash(),
    }

    return final, metadata


def main(argv=None):
    args = parse_args(argv)

    logger.info(f"Loading panel from {args.panel}...")
    if args.panel.endswith(".csv.gz"):
        panel = pd.read_csv(args.panel, compression="gzip")
    else:
        panel = pd.read_csv(args.panel)
    panel[args.date_col] = panel[args.date_col].astype(str)
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel = panel.sort_values([args.date_col, "ticker"]).reset_index(drop=True)

    dates = sorted(panel[args.date_col].unique())
    logger.info(
        f"  Panel: {len(panel)} rows, {len(dates)} dates, "
        f"{panel['ticker'].nunique()} tickers"
    )

    feature_cols = args.features or _DEFAULT_FEATURES
    logger.info(f"  Features: {feature_cols}")
    logger.info(f"  Return col: {args.return_col}")
    logger.info(f"  Method: {args.method}, lambda={args.ridge_lambda}")

    final_weights, metadata = calibrate(
        panel=panel,
        return_col=args.return_col,
        feature_cols=feature_cols,
        method=args.method,
        ridge_lambda=args.ridge_lambda,
        min_stocks_per_week=args.min_stocks_per_week,
        nw_lags=args.nw_lags,
        max_abs_weight=args.max_abs_weight,
        disable_negative=args.m3_disable_negative,
        date_col=args.date_col,
    )

    # Content hash for integrity
    weights_json = json.dumps(final_weights, sort_keys=True)
    metadata["content_hash"] = hashlib.sha256(
        weights_json.encode()
    ).hexdigest()[:16]

    # Write outputs
    out_path = Path(args.out)
    write_weights_json(out_path, final_weights, metadata)
    logger.info(f"  Wrote weights: {out_path}")

    report_dir = Path(args.outdir) if args.outdir else out_path.parent
    report_path = report_dir / "calibration_report.csv"
    write_calibration_report(
        report_path,
        raw_weights=metadata.get("raw_weights", {}),
        final_weights=final_weights,
        violations=metadata.get("sign_violations", {}),
        dropped=metadata.get("dropped_features", {}),
    )
    logger.info(f"  Wrote report: {report_path}")

    # Print summary
    logger.info("  Calibrated weights:")
    for feat in sorted(final_weights.keys()):
        w = final_weights[feat]
        raw = metadata.get("raw_weights", {}).get(feat, 0.0)
        sign_note = ""
        if feat in metadata.get("sign_violations", {}):
            sign_note = f" [ZEROED: {metadata['sign_violations'][feat]}]"
        elif feat in metadata.get("dropped_features", {}):
            sign_note = f" [DROPPED: {metadata['dropped_features'][feat]}]"
        logger.info(f"    {feat:30s}  raw={raw:+.6f}  final={w:+.8f}{sign_note}")

    l1 = sum(abs(w) for w in final_weights.values())
    logger.info(f"  L1 norm: {l1:.6f}")


if __name__ == "__main__":
    main()
