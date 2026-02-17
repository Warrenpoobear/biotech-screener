#!/usr/bin/env python3
"""Top Returners Signal Recall Study — diagnostic script.

Take the top-returning biotech tickers over multiple horizons, then ask
whether the model's signals at t0 would have flagged them ex-ante.
Validates predictive power of clinical + catalyst signals on realized returns.

Multi-horizon recall (3m/6m/12m/24m) aligns evaluation windows to what
clinical/catalyst signals are actually designed to predict.

Usage:
    python3 scripts/diag_top_returners_recall.py \
        --archive data/archives/2024-01-31.tar.gz \
        --start-date 2024-01-31 \
        --end-date 2026-01-27 \
        --price-csv production_data/price_history.csv \
        --output-dir data/diag \
        [--no-fetch]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import tarfile
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGE_TO_PHASE = {"early": "phase 1", "mid": "phase 2", "late": "phase 3"}
PRICE_CACHE_NAME = "price_cache_2y.csv"
PRICE_COLS = ["date", "ticker", "close"]

# Multi-horizon: label → trading days from t0 (None = full window to end_date)
RETURN_HORIZONS = {"3m": 63, "6m": 126, "12m": 252, "24m": None}

# Clinical alpha weights (must match run_screen.py _CE_WEIGHTS)
_CE_WEIGHTS = {"phase": 0.35, "proximity": 0.35, "quality": 0.30}

_PHASE_NUM = {
    "approved": 1.0, "phase 3": 0.83, "phase 2/3": 0.67,
    "phase 2": 0.50, "phase 1/2": 0.33, "phase 1": 0.17,
    "preclinical": 0.08,
}

_CLINICAL_PHASES = frozenset({
    "PHASE1", "PHASE2", "PHASE3",
    "PHASE 1", "PHASE 2", "PHASE 3",
    "PHASE1/PHASE2", "PHASE 1/PHASE 2",
    "PHASE2/PHASE3", "PHASE 2/PHASE 3",
    "EARLY_PHASE1",
})

# Stage multipliers (must match DE ruleset defaults)
_STAGE_MULTS = {"early": 0.0, "mid": 1.0, "late": 1.5}

# All flags evaluated in the study
FLAG_NAMES = [
    "model_top20", "tier_ab",
    "clinical_z_high", "clinical_top20",
    "clinical_score_z_top20",
    "clinical_score_z_tier_top20", "clinical_cz_eff_top20",
    "cs_z_top20_ab", "cz_eff_top20_ab",
    "clinical_near_readout", "clinical_high_and_near", "clinical_late_high",
    "catalyst_near",
    "any_signal",
]

FLAG_DEFS = {
    "model_top20": "composite_rank in top 20% at t0",
    "tier_ab": "tier_dev in {A, B} at t0",
    "clinical_z_high": "clinical_alpha_z >= +1.0 (approx PIT)",
    "clinical_top20": "clinical_alpha_z in top 20% (approx PIT)",
    "clinical_score_z_top20": "zscore(clinical_score) top 20% within cohort (PIT-safe)",
    "clinical_score_z_tier_top20": "tier-local zscore(clinical_score) top 20% within cohort (PIT-safe)",
    "clinical_cz_eff_top20": "cz_eff (stage-gated, pos-only, clamped) top 20% within cohort (PIT-safe)",
    "cs_z_top20_ab": "clinical_score_z top 20% restricted to tier A+B (DE scope)",
    "cz_eff_top20_ab": "cz_eff top 20% restricted to tier A+B (DE scope)",
    "clinical_near_readout": "readout_days <= 180 + coverage (approx PIT)",
    "clinical_high_and_near": "clin_z >= 1.0 AND readout <= 180",
    "clinical_late_high": "stage mid/late AND clin_score_z top 20%",
    "catalyst_near": "catalyst specific/blended, days <= 180 (PIT-safe)",
    "any_signal": "any of the above (excl any_signal)",
}

_FLAG_ABBREV = {
    "model_top20": "rank",
    "tier_ab": "tier",
    "clinical_z_high": "cz_hi",
    "clinical_top20": "cz20",
    "clinical_score_z_top20": "cs20",
    "clinical_score_z_tier_top20": "czt20",
    "clinical_cz_eff_top20": "eff20",
    "cs_z_top20_ab": "csAB",
    "cz_eff_top20_ab": "effAB",
    "clinical_near_readout": "rd",
    "clinical_high_and_near": "cz+rd",
    "clinical_late_high": "late",
    "catalyst_near": "cat",
    "any_signal": "any",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        val = float(v)
        return val if val == val else None  # NaN guard
    except (ValueError, TypeError):
        return None


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.2f}%"


def _fmt_f2(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def _parse_trial_date(s) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _clinical_proximity(days: Optional[int]) -> float:
    if days is None:
        return 0.0
    if days <= 0:
        return 1.0
    return math.exp(-days / 90)


def _zscore(values: List[float]) -> List[float]:
    """Cross-sectional z-score with p5/p95 winsorization."""
    n = len(values)
    if n == 0:
        return []
    if n >= 20:
        sv = sorted(values)
        p5 = sv[max(0, int(n * 0.05))]
        p95 = sv[min(n - 1, int(n * 0.95))]
    else:
        p5, p95 = min(values), max(values)
    winsorized = [max(p5, min(p95, v)) for v in values]
    mean_val = sum(winsorized) / len(winsorized)
    var_val = sum((v - mean_val) ** 2 for v in winsorized) / len(winsorized)
    std_val = var_val ** 0.5
    if std_val > 0:
        return [(v - mean_val) / std_val for v in winsorized]
    return [0.0] * n


# ---------------------------------------------------------------------------
# Step 1 — Load t0 signals from archive
# ---------------------------------------------------------------------------
def load_archive_rankings(archive_path: Path) -> List[dict]:
    """Extract rankings.csv from tar.gz archive -> list of dicts."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                    if "rankings" in m.name and m.name.endswith(".csv")]
        if not members:
            print("  ERROR: No rankings.csv found in archive", file=sys.stderr)
            return []
        f = tar.extractfile(members[0])
        if f is None:
            return []
        text = f.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    return rows


# ---------------------------------------------------------------------------
# Step 2 — Recompute clinical signals at t0
# ---------------------------------------------------------------------------
def compute_clinical_readout_days(
    trial_records: List[dict],
    as_of_date: str,
) -> Dict[str, Optional[int]]:
    """Mirrors run_screen.py _compute_clinical_readout_days."""
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    by_ticker: Dict[str, list] = {}
    for trial in trial_records:
        tk = trial.get("ticker", "")
        if tk:
            by_ticker.setdefault(tk.upper(), []).append(trial)

    result: Dict[str, Optional[int]] = {}
    for ticker, trials in by_ticker.items():
        best_days: Optional[int] = None
        for t in trials:
            if (t.get("study_type") or "").upper() != "INTERVENTIONAL":
                continue
            phase = str(t.get("phase", "")).upper()
            if phase not in _CLINICAL_PHASES:
                continue
            fp = _parse_trial_date(t.get("first_posted"))
            lup = _parse_trial_date(t.get("last_update_posted"))
            pit_date = fp or lup
            if pit_date is None or pit_date > as_of:
                continue

            pcd = _parse_trial_date(t.get("primary_completion_date"))
            cd = _parse_trial_date(t.get("completion_date"))
            rfp = _parse_trial_date(t.get("results_first_posted"))

            if pcd is not None:
                if pcd > as_of:
                    days = (pcd - as_of).days
                elif rfp is None:
                    days = 0
                else:
                    continue
            elif cd is not None and cd > as_of:
                days = (cd - as_of).days
            else:
                continue

            if best_days is None or days < best_days:
                best_days = days

        if best_days is not None:
            result[ticker] = best_days
    return result


def compute_clinical_alpha_z(
    csv_rows: List[dict],
    readout_days: Dict[str, Optional[int]],
) -> None:
    """Compute clinical_alpha_z (approximate PIT) in-place.

    Uses stage_bucket proxy for lead_phase + today's trial records
    filtered by first_posted. PCD/completion dates may have shifted
    since t0 — label as approximate PIT.
    """
    eligible_indices: List[int] = []
    raw_scores: List[float] = []

    for i, row in enumerate(csv_rows):
        ticker = row.get("ticker", "").upper()
        clinical_score = _safe_float(row.get("clinical_score"))
        if clinical_score is None:
            continue

        stage_bucket = (row.get("stage_bucket") or "").lower().strip()
        lead_phase = STAGE_TO_PHASE.get(stage_bucket, "")
        phase_num = _PHASE_NUM.get(lead_phase, 0.0)

        rd = readout_days.get(ticker)
        proximity = _clinical_proximity(rd)
        quality = clinical_score / 100.0

        raw = (_CE_WEIGHTS["phase"] * phase_num
               + _CE_WEIGHTS["proximity"] * proximity
               + _CE_WEIGHTS["quality"] * quality)

        eligible_indices.append(i)
        raw_scores.append(raw)

        csv_rows[i]["clinical_readout_days"] = rd if rd is not None else ""
        csv_rows[i]["clinical_coverage_flag"] = 1 if rd is not None else 0

    if not raw_scores:
        return

    zs = _zscore(raw_scores)
    for j, idx in enumerate(eligible_indices):
        csv_rows[idx]["clinical_alpha_z"] = round(zs[j], 4)

    print(f"  clinical_alpha_z: {len(eligible_indices)} tickers", file=sys.stderr)


def compute_clinical_score_z(csv_rows: List[dict]) -> None:
    """Compute clinical_score_z (fully PIT-safe) in-place.

    Per-cohort z-score of the archive's clinical_score column.
    Cohorts: drug_developer, commercial_biotech, commercial_pharma.
    No trial records dependency — maximally PIT-faithful.
    """
    _CLINICAL_COHORTS = ("drug_developer", "commercial_biotech", "commercial_pharma")
    total_filled = 0

    for cohort_arch in _CLINICAL_COHORTS:
        eligible_indices: List[int] = []
        scores: List[float] = []

        for i, row in enumerate(csv_rows):
            if (row.get("archetype") or "") != cohort_arch:
                continue
            cs = _safe_float(row.get("clinical_score"))
            if cs is not None:
                eligible_indices.append(i)
                scores.append(cs)

        if not scores:
            continue

        zs = _zscore(scores)
        for j, idx in enumerate(eligible_indices):
            csv_rows[idx]["clinical_score_z"] = round(zs[j], 4)

        total_filled += len(eligible_indices)
        print(f"  clinical_score_z ({cohort_arch}): {len(eligible_indices)} tickers, "
              f"mean={statistics.mean(scores):.1f}", file=sys.stderr)

    print(f"  clinical_score_z (PIT-safe): {total_filled} total tickers", file=sys.stderr)


def compute_clinical_score_z_tier(csv_rows: List[dict]) -> None:
    """Compute clinical_score_z_tier (tier-local, ddof=0, PIT-safe) in-place.

    Drug developers: z-scored within (tier_dev × drug_developer) groups.
    Commercial: z-scored within (tier_commercial × archetype) groups.
    If tier_commercial not available (older archives), use clinical_score_z
    as fallback (archetype-level z, no tier sub-grouping).
    Population std (ddof=0) for consistency with run_screen.py.
    """
    from collections import defaultdict

    def _fill_tier_groups(pairs_by_tier: Dict[str, List[Tuple[int, float]]]) -> int:
        """Z-score within each tier group. Returns count filled."""
        filled = 0
        for tier, pairs in pairs_by_tier.items():
            if len(pairs) < 2:
                for idx, _ in pairs:
                    csv_rows[idx]["clinical_score_z_tier"] = 0.0
                filled += len(pairs)
                continue
            scores = [s for _, s in pairs]
            mean_val = sum(scores) / len(scores)
            var_val = sum((s - mean_val) ** 2 for s in scores) / len(scores)  # ddof=0
            std_val = var_val ** 0.5
            for idx, s in pairs:
                if std_val > 0:
                    csv_rows[idx]["clinical_score_z_tier"] = round(
                        (s - mean_val) / std_val, 4
                    )
                else:
                    csv_rows[idx]["clinical_score_z_tier"] = 0.0
            filled += len(pairs)
        return filled

    # --- Drug developers: group by tier_dev ---
    dev_indices: List[int] = []
    for i, row in enumerate(csv_rows):
        cs = _safe_float(row.get("clinical_score"))
        if cs is not None and (row.get("archetype") or "") == "drug_developer":
            dev_indices.append(i)

    tier_groups: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i in dev_indices:
        tier = csv_rows[i].get("tier_dev", "")
        if tier:
            tier_groups[tier].append((i, float(csv_rows[i]["clinical_score"])))

    n_dev = _fill_tier_groups(tier_groups)
    print(f"  clinical_score_z_tier (drug_developer, ddof=0): {n_dev} tickers, "
          f"{len(tier_groups)} tiers", file=sys.stderr)

    # --- Commercial cohorts: group by tier_commercial ---
    _COMM_ARCHETYPES = ("commercial_biotech", "commercial_pharma")
    has_tier_comm = any(row.get("tier_commercial") for row in csv_rows)

    n_comm = 0
    for arch in _COMM_ARCHETYPES:
        comm_indices = [i for i, row in enumerate(csv_rows)
                        if (row.get("archetype") or "") == arch
                        and _safe_float(row.get("clinical_score")) is not None]
        if not comm_indices:
            continue

        if has_tier_comm:
            comm_tier_groups: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
            for i in comm_indices:
                tier = csv_rows[i].get("tier_commercial", "")
                if not tier:
                    csv_rows[i]["clinical_score_z_tier"] = 0.0
                    n_comm += 1
                    continue
                comm_tier_groups[tier].append((i, float(csv_rows[i]["clinical_score"])))
            n_comm += _fill_tier_groups(comm_tier_groups)
        else:
            # Fallback: use clinical_score_z (archetype-level z) as proxy
            for i in comm_indices:
                cz = _safe_float(csv_rows[i].get("clinical_score_z"))
                csv_rows[i]["clinical_score_z_tier"] = round(cz, 4) if cz is not None else 0.0
                n_comm += 1

    print(f"  clinical_score_z_tier (commercial, ddof=0): {n_comm} tickers"
          f"{' (tier_commercial fallback to archetype z)' if not has_tier_comm else ''}",
          file=sys.stderr)


def compute_cz_eff(csv_rows: List[dict]) -> None:
    """Compute cz_eff (stage-gated, pos-only, clamped) in-place.

    Matches DE ruleset defaults:
      stage_mults = {early: 0.0, mid: 1.0, late: 1.5}
      cz_eff = clamp(max(0, cz_tier), 0, 2) * stage_mult
    """
    n_filled = 0
    for row in csv_rows:
        cz_tier = _safe_float(row.get("clinical_score_z_tier"))
        if cz_tier is None:
            continue
        stage = (row.get("stage_bucket") or "").lower().strip()
        stage_mult = _STAGE_MULTS.get(stage, 0.0)
        cz_pos = max(0.0, cz_tier)
        cz_clamped = min(2.0, cz_pos)
        row["cz_eff"] = round(cz_clamped * stage_mult, 4)
        n_filled += 1

    # Summary by stage
    stage_counts: Dict[str, int] = {}
    stage_nonzero: Dict[str, int] = {}
    for row in csv_rows:
        v = _safe_float(row.get("cz_eff"))
        if v is None:
            continue
        stage = (row.get("stage_bucket") or "").lower().strip()
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if v > 0:
            stage_nonzero[stage] = stage_nonzero.get(stage, 0) + 1

    print(f"  cz_eff (stage-gated, pos-only, clamped ±2): {n_filled} tickers",
          file=sys.stderr)
    for s in sorted(stage_counts):
        nz = stage_nonzero.get(s, 0)
        print(f"    {s or '(empty)'}: {stage_counts[s]} total, {nz} nonzero",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 3 — Fetch prices + compute multi-horizon returns
# ---------------------------------------------------------------------------
def load_prices(price_csv: Path, cache_csv: Optional[Path]) -> Dict[Tuple[str, str], float]:
    """Return (date_str, ticker) -> close price dict from CSV(s)."""
    prices: Dict[Tuple[str, str], float] = {}
    for path in [price_csv, cache_csv]:
        if path is None or not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row["date"][:10]
                t = row["ticker"]
                c = row.get("close")
                if c is None or c == "":
                    continue
                try:
                    val = float(c)
                except (ValueError, TypeError):
                    continue
                if val != val:
                    continue
                prices[(d, t)] = val
    return prices


def _trading_dates_sorted(prices: Dict[Tuple[str, str], float]) -> List[str]:
    return sorted({k[0] for k in prices})


def _nearest_date(trading_dates: List[str], target: str,
                  direction: str = "on_or_after") -> Optional[str]:
    if direction == "on_or_after":
        for d in trading_dates:
            if d >= target:
                return d
    else:
        for d in reversed(trading_dates):
            if d <= target:
                return d
    return None


def _forward_date(trading_dates: List[str], base: str, offset: int) -> Optional[str]:
    """Return trading date at +offset bars from base, or None."""
    try:
        idx = trading_dates.index(base)
    except ValueError:
        return None
    target = idx + offset
    if 0 <= target < len(trading_dates):
        return trading_dates[target]
    return None


def identify_missing_tickers(
    tickers: set,
    prices: Dict[Tuple[str, str], float],
    trading_dates: List[str],
    t0_date: str,
    t1_date: str,
) -> set:
    """Return tickers that need price data for return computation."""
    t0 = _nearest_date(trading_dates, t0_date, "on_or_after")
    t1 = _nearest_date(trading_dates, t1_date, "on_or_before")
    if not t0 or not t1:
        return tickers
    needed: set = set()
    for t in tickers:
        if (t0, t) not in prices or (t1, t) not in prices:
            needed.add(t)
    return needed


def fetch_missing_prices(
    tickers: set,
    cache_csv: Path,
    start_date: str,
    end_date: str,
) -> Dict[Tuple[str, str], float]:
    """Fetch prices via yfinance and cache to CSV."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        print("  yfinance not installed — skipping fetch", file=sys.stderr)
        return {}

    if not tickers:
        return {}

    print(f"  Fetching prices for {len(tickers)} tickers via yfinance...",
          file=sys.stderr)

    new_prices: Dict[Tuple[str, str], float] = {}
    ticker_list = sorted(tickers)
    batch_size = 50
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i: i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} tickers",
              file=sys.stderr)
        for ticker in batch:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(start=start_date, end=end_date,
                                  auto_adjust=False)
                for idx, row in hist.iterrows():
                    close = row.get("Close")
                    if close is None or close != close:
                        continue
                    d = idx.strftime("%Y-%m-%d")
                    new_prices[(d, ticker)] = float(close)
            except Exception as exc:
                print(f"    {ticker}: error {exc}", file=sys.stderr)

    if new_prices:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        existing_keys: set = set()
        if cache_csv.exists():
            with cache_csv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing_keys.add((row["date"], row["ticker"]))

        rows_to_write = [
            {"date": k[0], "ticker": k[1], "close": f"{v:.4f}"}
            for k, v in sorted(new_prices.items())
            if k not in existing_keys
        ]
        if rows_to_write:
            write_header = not cache_csv.exists()
            with cache_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=PRICE_COLS)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows_to_write)
        print(f"  Cached {len(rows_to_write)} new price rows to {cache_csv}",
              file=sys.stderr)

    return new_prices


def compute_multi_horizon_returns(
    rows: List[dict],
    prices: Dict[Tuple[str, str], float],
    trading_dates: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[List[dict], str, str, Dict[str, str]]:
    """Compute returns at 3m/6m/12m/24m horizons.

    Returns (rows, actual_t0, actual_t1, horizon_end_dates).
    """
    t0 = _nearest_date(trading_dates, start_date, "on_or_after")
    t1 = _nearest_date(trading_dates, end_date, "on_or_before")
    if not t0 or not t1:
        print("  ERROR: Could not find trading dates", file=sys.stderr)
        return rows, start_date, end_date, {}

    # Resolve end date for each horizon
    horizon_ends: Dict[str, str] = {}
    for label, bars in RETURN_HORIZONS.items():
        if bars is None:
            horizon_ends[label] = t1
        else:
            hd = _forward_date(trading_dates, t0, bars)
            if hd:
                horizon_ends[label] = hd
            else:
                horizon_ends[label] = t1  # clip to available data

    print(f"  t0={t0}", file=sys.stderr)
    for label, end in horizon_ends.items():
        print(f"    {label}: t0+{RETURN_HORIZONS[label] or 'max'} -> {end}",
              file=sys.stderr)

    # Compute returns for each ticker at each horizon
    for row in rows:
        ticker = row["ticker"]
        p0 = prices.get((t0, ticker))
        row["price_t0"] = p0

        for label, end in horizon_ends.items():
            p_end = prices.get((end, ticker))
            if p0 and p_end and p0 > 0:
                row[f"ret_{label}"] = p_end / p0 - 1
            else:
                row[f"ret_{label}"] = None

        # total_return = 24m for backward compat
        row["total_return"] = row.get("ret_24m")
        row["price_t1"] = prices.get((t1, ticker))

    # Coverage summary
    for label in RETURN_HORIZONS:
        n = sum(1 for r in rows if r.get(f"ret_{label}") is not None)
        print(f"  {label} coverage: {n}/{len(rows)}", file=sys.stderr)

    return rows, t0, t1, horizon_ends


# ---------------------------------------------------------------------------
# Step 4 — Select top returners (per horizon)
# ---------------------------------------------------------------------------
def select_top_returners_by_horizon(
    rows: List[dict],
    top_k: int,
    top_pct: float,
) -> Dict[str, Tuple[set, set]]:
    """Return {horizon: (top_k_set, top_pct_set)} for each horizon."""
    result: Dict[str, Tuple[set, set]] = {}
    for label in RETURN_HORIZONS:
        col = f"ret_{label}"
        with_ret = [r for r in rows if r.get(col) is not None]
        with_ret.sort(key=lambda x: x[col], reverse=True)
        k = min(top_k, len(with_ret))
        top_k_tickers = {r["ticker"] for r in with_ret[:k]}
        n_pct = max(1, int(len(with_ret) * top_pct))
        top_pct_tickers = {r["ticker"] for r in with_ret[:n_pct]}
        result[label] = (top_k_tickers, top_pct_tickers)
    return result


# ---------------------------------------------------------------------------
# Step 5 — Flag definitions + recall/lift computation
# ---------------------------------------------------------------------------
def compute_flags(rows: List[dict]) -> None:
    """Compute all flag columns on each row (in-place)."""
    # --- Thresholds ---
    ranks = [v for v in (_safe_float(r.get("composite_rank")) for r in rows)
             if v is not None]
    rank_top20 = sorted(ranks)[: max(1, int(len(ranks) * 0.20))]
    rank_threshold = max(rank_top20) if rank_top20 else 0

    clin_zs = sorted(
        [v for v in (_safe_float(r.get("clinical_alpha_z")) for r in rows)
         if v is not None],
        reverse=True,
    )
    n_top20_clin = max(1, int(len(clin_zs) * 0.20))
    clin_top20_thr = clin_zs[n_top20_clin - 1] if clin_zs else 0

    # Per-cohort top-20% thresholds (prevents mixed-distribution distortion)
    _SIGNAL_COHORTS = ("drug_developer", "commercial_biotech", "commercial_pharma")

    def _cohort_top20_thr(col: str) -> Dict[str, float]:
        """Return {archetype: threshold} for top-20% within each cohort."""
        thrs: Dict[str, float] = {}
        for arch in _SIGNAL_COHORTS:
            vals = sorted(
                [v for r in rows
                 if (r.get("archetype") or "") == arch
                 for v in [_safe_float(r.get(col))]
                 if v is not None],
                reverse=True,
            )
            n20 = max(1, int(len(vals) * 0.20))
            thrs[arch] = vals[n20 - 1] if vals else 0
        return thrs

    cscore_top20_by_cohort = _cohort_top20_thr("clinical_score_z")
    czt_top20_by_cohort = _cohort_top20_thr("clinical_score_z_tier")
    czeff_top20_by_cohort = _cohort_top20_thr("cz_eff")

    # A+B tier thresholds (DE scope: where clinical sort actually operates)
    ab_rows = [r for r in rows if (r.get("tier_dev") or "").upper() in ("A", "B")]
    ab_cs_zs = sorted(
        [v for v in (_safe_float(r.get("clinical_score_z")) for r in ab_rows)
         if v is not None],
        reverse=True,
    )
    n_top20_cs_ab = max(1, int(len(ab_cs_zs) * 0.20))
    cs_top20_ab_thr = ab_cs_zs[n_top20_cs_ab - 1] if ab_cs_zs else 0

    ab_czeff = sorted(
        [v for v in (_safe_float(r.get("cz_eff")) for r in ab_rows)
         if v is not None],
        reverse=True,
    )
    n_top20_czeff_ab = max(1, int(len(ab_czeff) * 0.20))
    czeff_top20_ab_thr = ab_czeff[n_top20_czeff_ab - 1] if ab_czeff else 0

    for row in rows:
        comp_rank = _safe_float(row.get("composite_rank"))
        tier = (row.get("tier_dev") or "").upper()
        arch = row.get("archetype") or ""
        clin_z = _safe_float(row.get("clinical_alpha_z"))
        cs_z = _safe_float(row.get("clinical_score_z"))
        cz_tier = _safe_float(row.get("clinical_score_z_tier"))
        cz_eff = _safe_float(row.get("cz_eff"))
        cat_mode = (row.get("catalyst_mode")
                    or row.get("de_catalyst_mode") or "").lower()
        cat_days = _safe_float(row.get("catalyst_days"))
        rd = row.get("clinical_readout_days")
        rd_val = _safe_float(rd) if rd != "" else None
        cov = row.get("clinical_coverage_flag")
        stage = (row.get("stage_bucket") or "").lower().strip()

        # Per-cohort threshold lookup (non-matching archetypes get 0 → never flagged)
        _cs_thr = cscore_top20_by_cohort.get(arch, 999)
        _czt_thr = czt_top20_by_cohort.get(arch, 999)
        _czeff_thr = czeff_top20_by_cohort.get(arch, 999)

        # Original flags
        row["flag_model_top20"] = 1 if (
            comp_rank is not None and comp_rank <= rank_threshold
        ) else 0
        row["flag_tier_ab"] = 1 if tier in ("A", "B") else 0
        row["flag_clinical_z_high"] = 1 if (
            clin_z is not None and clin_z >= 1.0
        ) else 0
        row["flag_clinical_top20"] = 1 if (
            clin_z is not None and clin_z >= clin_top20_thr
        ) else 0
        row["flag_catalyst_near"] = 1 if (
            cat_mode in ("specific_days", "blended_window")
            and cat_days is not None and cat_days <= 180
        ) else 0

        # PIT-safe clinical_score_z flag (per-cohort top 20%)
        row["flag_clinical_score_z_top20"] = 1 if (
            cs_z is not None and cs_z >= _cs_thr
        ) else 0

        # Tier-local z-score flag (per-cohort top 20%, PIT-safe)
        row["flag_clinical_score_z_tier_top20"] = 1 if (
            cz_tier is not None and cz_tier >= _czt_thr
        ) else 0

        # cz_eff flag — the DE sort signal (per-cohort top 20%, PIT-safe, stage-gated)
        row["flag_clinical_cz_eff_top20"] = 1 if (
            cz_eff is not None and cz_eff >= _czeff_thr and cz_eff > 0
        ) else 0

        # A+B tier flags (DE scope: where sort signal actually operates)
        row["flag_cs_z_top20_ab"] = 1 if (
            tier in ("A", "B")
            and cs_z is not None and cs_z >= cs_top20_ab_thr
        ) else 0
        row["flag_cz_eff_top20_ab"] = 1 if (
            tier in ("A", "B")
            and cz_eff is not None and cz_eff >= czeff_top20_ab_thr
            and cz_eff > 0
        ) else 0

        # Near readout (approx PIT)
        row["flag_clinical_near_readout"] = 1 if (
            rd_val is not None and rd_val <= 180 and cov == 1
        ) else 0

        # High z AND near readout
        row["flag_clinical_high_and_near"] = 1 if (
            row["flag_clinical_z_high"] == 1
            and row["flag_clinical_near_readout"] == 1
        ) else 0

        # Late-stage + high clinical score z (per-cohort threshold)
        row["flag_clinical_late_high"] = 1 if (
            stage in ("mid", "late")
            and cs_z is not None and cs_z >= _cs_thr
        ) else 0

        # Any signal (union of non-meta flags)
        row["flag_any_signal"] = 1 if any(
            row.get(f"flag_{f}") == 1
            for f in FLAG_NAMES if f != "any_signal"
        ) else 0


def compute_recall_lift(
    rows: List[dict],
    top_k_set: set,
    top_pct_set: set,
    top_k: int,
    ret_col: str = "total_return",
) -> List[dict]:
    """Compute recall, lift, precision, and random baseline for each flag."""
    with_ret = [r for r in rows if r.get(ret_col) is not None]
    n_universe = len(with_ret)
    if not with_ret:
        return []

    results = []
    for flag in FLAG_NAMES:
        col = f"flag_{flag}"
        flagged = [r for r in with_ret if r.get(col) == 1]
        unflagged = [r for r in with_ret if r.get(col) != 1]

        n_flagged = len(flagged)
        flagged_tickers = {r["ticker"] for r in flagged}

        # Random baseline: expected recall if top-K were random draws
        random_baseline = n_flagged / n_universe if n_universe else 0

        recall_k = (len(flagged_tickers & top_k_set) / len(top_k_set)
                    if top_k_set else None)
        recall_pct = (len(flagged_tickers & top_pct_set) / len(top_pct_set)
                      if top_pct_set else None)
        precision_k = (len(flagged_tickers & top_k_set) / n_flagged
                       if n_flagged else None)

        rets_flagged = [r[ret_col] for r in flagged]
        rets_unflagged = [r[ret_col] for r in unflagged]
        mean_flagged = statistics.mean(rets_flagged) if rets_flagged else None
        mean_unflagged = (statistics.mean(rets_unflagged)
                          if rets_unflagged else None)
        lift = (mean_flagged / mean_unflagged
                if mean_flagged is not None
                and mean_unflagged and mean_unflagged > 0
                else None)

        results.append({
            "flag": flag,
            "definition": FLAG_DEFS.get(flag, ""),
            "n_flagged": n_flagged,
            "random_baseline": random_baseline,
            "recall_k": recall_k,
            "recall_pct": recall_pct,
            "precision_k": precision_k,
            "mean_return_flagged": mean_flagged,
            "mean_return_unflagged": mean_unflagged,
            "lift": lift,
        })
    return results


def compute_recall_by_horizon(
    rows: List[dict],
    top_sets: Dict[str, Tuple[set, set]],
    top_k: int,
) -> Dict[str, List[dict]]:
    """Compute recall/lift for each flag at each horizon."""
    results: Dict[str, List[dict]] = {}
    for label in RETURN_HORIZONS:
        top_k_set, top_pct_set = top_sets[label]
        ret_col = f"ret_{label}"
        results[label] = compute_recall_lift(
            rows, top_k_set, top_pct_set, top_k, ret_col
        )
    return results


# ---------------------------------------------------------------------------
# Step 6 — Output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "ticker", "ret_3m", "ret_6m", "ret_12m", "ret_24m",
    "price_t0", "price_t1",
    "archetype", "stage_bucket", "composite_rank", "score_rank_pct",
    "tier_dev", "clinical_score", "clinical_score_z",
    "clinical_score_z_tier", "cz_eff",
    "clinical_alpha_z", "clinical_readout_days", "clinical_coverage_flag",
    "catalyst_days", "catalyst_mode",
] + [f"flag_{f}" for f in FLAG_NAMES]


def _fmt_csv_val(col: str, v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if "ret" in col or "return" in col or "price" in col:
            return f"{v:.6f}"
        if "alpha_z" in col or "score_z" in col:
            return f"{v:.4f}"
        return f"{v:.2f}"
    return str(v)


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        [r for r in rows if r.get("ret_24m") is not None],
        key=lambda x: x["ret_24m"],
        reverse=True,
    )
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow({c: _fmt_csv_val(c, r.get(c)) for c in CSV_COLUMNS})
    print(f"  CSV: {output_path} ({len(sorted_rows)} rows)", file=sys.stderr)


def write_report(
    rows: List[dict],
    recall_24m: List[dict],
    recall_by_horizon: Dict[str, List[dict]],
    t0: str,
    t1: str,
    horizon_ends: Dict[str, str],
    top_k: int,
    top_pct: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_ret = sorted(
        [r for r in rows if r.get("ret_24m") is not None],
        key=lambda x: x["ret_24m"],
        reverse=True,
    )
    n_total = len(rows)
    n_survivors = len(with_ret)

    L: List[str] = []  # output lines

    L.append("# Top Returners Signal Recall Study")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append("")

    # --- 1. Study parameters ---
    L.append("## 1. Study Parameters")
    L.append(f"- **t0 archive**: {t0}")
    L.append(f"- **Universe at t0**: {n_total} tickers")
    L.append(f"- **Survivors (prices at both ends)**: {n_survivors}")
    L.append(f"- **Top K**: {top_k} | **Top pct**: {top_pct * 100:.0f}%")
    L.append("")
    L.append("**Horizons**:")
    L.append("| Label | End Date | Trading Days |")
    L.append("|-------|----------|-------------|")
    for label in RETURN_HORIZONS:
        bars = RETURN_HORIZONS[label]
        L.append(f"| {label} | {horizon_ends.get(label, '?')} | "
                 f"{bars if bars else 'max'} |")
    L.append("")

    L.append("**PIT status of signals**:")
    L.append("| Signal | PIT Status | Notes |")
    L.append("|--------|-----------|-------|")
    L.append("| composite_rank, tier_dev, catalyst_days/mode | **PIT-safe** | "
             "From archive at t0 |")
    L.append("| clinical_score, clinical_score_z | **PIT-safe** | "
             "Pure z-score of archive clinical_score |")
    L.append("| clinical_score_z_tier, cz_eff | **PIT-safe** | "
             "Tier-local z (ddof=0) + stage-gated/pos-only/clamped transform; "
             "matches DE sort signal |")
    L.append("| clinical_alpha_z, readout_days | **Approximate PIT** | "
             "Uses today's trial records filtered by first_posted <= t0; "
             "PCD/completion dates may have shifted post-t0 |")
    L.append("")
    L.append("> **Survivorship bias**: delisted/acquired tickers excluded. "
             "In biotech, many top returns come from M&A — results may "
             "understate signal quality for acquired names.")
    L.append("")

    # --- 2. Return distribution by horizon ---
    L.append("## 2. Return Distribution by Horizon")
    L.append("| Stat | 3m | 6m | 12m | 24m |")
    L.append("|------|----|----|-----|-----|")
    for stat_name, stat_fn in [
        ("N", lambda vals: str(len(vals))),
        ("Mean", lambda vals: _fmt_pct(statistics.mean(vals)) if vals else "—"),
        ("Median", lambda vals: _fmt_pct(statistics.median(vals)) if vals else "—"),
        ("p10", lambda vals: _fmt_pct(sorted(vals)[int(len(vals) * 0.10)]) if len(vals) > 10 else "—"),
        ("p90", lambda vals: _fmt_pct(sorted(vals)[min(len(vals) - 1, int(len(vals) * 0.90))]) if len(vals) > 10 else "—"),
    ]:
        cells = []
        for label in RETURN_HORIZONS:
            col = f"ret_{label}"
            vals = [r[col] for r in rows if r.get(col) is not None]
            cells.append(stat_fn(vals))
        L.append(f"| {stat_name} | {' | '.join(cells)} |")
    L.append("")

    # --- 3. HEADLINE: Recall/Lift by horizon (the key table) ---
    L.append("## 3. Recall / Lift by Horizon (key table)")
    L.append("")
    L.append(f"Recall@{top_k} for each flag at each horizon. "
             f"**Random baseline** = flagged/universe (expected recall "
             f"if top-{top_k} were random draws).")
    L.append("")
    L.append("> **Primary horizons: 6m / 12m** (where clinical/catalyst signals "
             "are designed to predict). **24m is informational only** "
             "(diluted by noise, M&A, regime shifts). Lift < 1 at 24m does "
             "not invalidate signal — check 6m/12m for actionable evidence.")
    L.append("")

    # Show per-horizon universe size (baseline denominator varies)
    horizon_n = {}
    for label in RETURN_HORIZONS:
        ret_col = f"ret_{label}"
        horizon_n[label] = sum(1 for r in rows if r.get(ret_col) is not None)
    L.append(f"Universe per horizon: "
             + ", ".join(f"{h}={horizon_n[h]}" for h in RETURN_HORIZONS))
    L.append("")

    # Build a compact horizon × flag table with per-horizon baseline
    L.append(f"| Flag | N | "
             + " | ".join(f"BL {h} | R@{top_k} {h} | Lift {h}"
                          for h in RETURN_HORIZONS)
             + " |")
    L.append("|------|----|"
             + "|".join("--------|-----------|-------"
                        for _ in RETURN_HORIZONS)
             + "|")
    for fi, flag in enumerate(FLAG_NAMES):
        r24 = recall_by_horizon["24m"][fi]
        nf = r24["n_flagged"]
        cells = []
        for label in RETURN_HORIZONS:
            rh = recall_by_horizon[label][fi]
            baseline = _fmt_pct(rh["random_baseline"])
            recall = _fmt_pct(rh["recall_k"])
            lift = _fmt_f2(rh["lift"])
            cells.append(f"{baseline} | {recall} | {lift}")
        L.append(f"| {flag} | {nf} | {' | '.join(cells)} |")
    L.append("")

    # --- 4. Detailed recall table (24m, with precision) ---
    L.append("## 4. Detailed Recall — 24m Horizon (informational)")
    L.append("")
    L.append("> 24m horizon is noisy and diluted. Lift < 1.0 is expected for "
             "signals designed around 6-12 month catalysts. Use this table "
             "for completeness, not for signal validation.")
    L.append(
        f"| Flag | Definition | N | Baseline | Recall @{top_k} | "
        f"Recall @{top_pct * 100:.0f}% | Lift | Precision @{top_k} |"
    )
    L.append("|------|-----------|---|----------|------------|"
             "-------------|------|--------------|")
    for rr in recall_24m:
        flag = rr["flag"]
        defn = rr["definition"]
        nf = rr["n_flagged"]
        bl = _fmt_pct(rr["random_baseline"])
        rk = _fmt_pct(rr["recall_k"])
        rp = _fmt_pct(rr["recall_pct"])
        lift = _fmt_f2(rr["lift"])
        pk = _fmt_pct(rr["precision_k"])
        L.append(f"| {flag} | {defn} | {nf} | {bl} | {rk} | {rp} | "
                 f"{lift} | {pk} |")
    L.append("")

    # --- 5. Signal decomposition for top 30 ---
    L.append("## 5. Top 30 Returners (24m) — Signal Decomposition")
    L.append(
        "| # | Ticker | 24m | 12m | 6m | 3m | "
        "Arch | Rank | Tier | ClinZ | CS_Z | CZT | czEff | Rd | CatD | CatMode | Flags |"
    )
    L.append(
        "|---|--------|-----|-----|----|----|"
        "-----|------|------|-------|------|-----|-------|----|----|---------|-------|"
    )
    for i, r in enumerate(with_ret[:30], 1):
        r24 = _fmt_pct(r.get("ret_24m"))
        r12 = _fmt_pct(r.get("ret_12m"))
        r6 = _fmt_pct(r.get("ret_6m"))
        r3 = _fmt_pct(r.get("ret_3m"))
        arch = (r.get("archetype") or "")[:12]
        comp = _fmt_f2(_safe_float(r.get("composite_rank")))
        tier = r.get("tier_dev") or ""
        clin_z = _fmt_f2(_safe_float(r.get("clinical_alpha_z")))
        cs_z = _fmt_f2(_safe_float(r.get("clinical_score_z")))
        czt = _fmt_f2(_safe_float(r.get("clinical_score_z_tier")))
        czeff = _fmt_f2(_safe_float(r.get("cz_eff")))
        rd = r.get("clinical_readout_days")
        rd_str = str(rd) if rd != "" and rd is not None else "—"
        cat_d = r.get("catalyst_days") or "—"
        cat_m = (r.get("catalyst_mode")
                 or r.get("de_catalyst_mode") or "")[:12]
        flags = []
        for f in FLAG_NAMES:
            if f == "any_signal":
                continue
            if r.get(f"flag_{f}") == 1:
                flags.append(_FLAG_ABBREV.get(f, f))
        flag_str = ",".join(flags) if flags else "none"
        L.append(
            f"| {i} | {r['ticker']} | {r24} | {r12} | {r6} | {r3} | "
            f"{arch} | {comp} | {tier} | {clin_z} | {cs_z} | "
            f"{czt} | {czeff} | {rd_str} | {cat_d} | {cat_m} | {flag_str} |"
        )
    L.append("")

    # --- 6. Signal means: top vs rest ---
    L.append("## 6. Signal Means — Top Returners vs Rest")
    top_set_rows = with_ret[:top_k]
    rest_rows = with_ret[top_k:]

    signal_cols = [
        ("clinical_alpha_z", "Clinical Alpha Z (approx PIT)"),
        ("clinical_score_z", "Clinical Score Z (PIT-safe)"),
        ("clinical_score_z_tier", "Clinical Score Z Tier-local (PIT-safe)"),
        ("cz_eff", "cz_eff (DE sort signal, PIT-safe)"),
        ("clinical_score", "Clinical Score (raw)"),
        ("catalyst_days", "Catalyst Days"),
        ("composite_rank", "Composite Rank"),
    ]
    L.append(f"| Signal | Top {top_k} Mean | Rest Mean | Delta |")
    L.append("|--------|-----------|-----------|-------|")
    for col, label in signal_cols:
        top_vals = [v for v in (_safe_float(r.get(col)) for r in top_set_rows)
                    if v is not None]
        rest_vals = [v for v in (_safe_float(r.get(col)) for r in rest_rows)
                     if v is not None]
        top_mean = statistics.mean(top_vals) if top_vals else None
        rest_mean = statistics.mean(rest_vals) if rest_vals else None
        delta = ((top_mean - rest_mean)
                 if top_mean is not None and rest_mean is not None else None)
        L.append(f"| {label} | {_fmt_f2(top_mean)} | {_fmt_f2(rest_mean)} "
                 f"| {_fmt_f2(delta)} |")
    L.append("")

    # --- 7. Coverage decomposition for top-K ---
    L.append(f"## 7. Coverage Decomposition — Top {top_k} Returners (24m)")
    L.append("")
    L.append("Why some top returners aren't flagged by clinical signals:")
    L.append("")

    # Categorize top-K returners
    top_k_rows = with_ret[:top_k]
    n_dd = sum(1 for r in top_k_rows
               if (r.get("archetype") or "") == "drug_developer")
    n_comm_bio = sum(1 for r in top_k_rows
                     if (r.get("archetype") or "") == "commercial_biotech")
    n_comm_pharma = sum(1 for r in top_k_rows
                        if (r.get("archetype") or "") == "commercial_pharma")
    n_comm = n_comm_bio + n_comm_pharma
    n_other = len(top_k_rows) - n_dd - n_comm
    n_early = sum(1 for r in top_k_rows
                  if (r.get("stage_bucket") or "").lower() == "early")
    n_no_tier = sum(1 for r in top_k_rows
                    if not (r.get("tier_dev") or "")
                    and not (r.get("tier_commercial") or ""))
    n_tier_a = sum(1 for r in top_k_rows
                   if (r.get("tier_dev") or "").upper() == "A")
    n_tier_b = sum(1 for r in top_k_rows
                   if (r.get("tier_dev") or "").upper() == "B")
    n_tier_cd = sum(1 for r in top_k_rows
                    if (r.get("tier_dev") or "").upper() in ("C", "D"))
    n_comm_with_czeff = sum(1 for r in top_k_rows
                            if (r.get("archetype") or "").startswith("commercial_")
                            and _safe_float(r.get("cz_eff")) is not None
                            and _safe_float(r.get("cz_eff")) > 0)
    n_cat_near = sum(1 for r in top_k_rows
                     if r.get("flag_catalyst_near") == 1)
    n_no_cat = sum(1 for r in top_k_rows
                   if (r.get("catalyst_mode") or r.get("de_catalyst_mode")
                       or "").lower() == "missing")

    L.append(f"| Category | Count | % of Top-{top_k} |")
    L.append("|----------|-------|--------------|")
    for label, n in [
        ("Drug developer (clinical signal scope)", n_dd),
        ("Commercial biotech (clinical signal scope)", n_comm_bio),
        ("Commercial pharma (clinical signal scope)", n_comm_pharma),
        ("Other archetype (no clinical signal)", n_other),
        ("Early stage (cz_eff=0 by design)", n_early),
        ("No tier (excluded from cz_tier)", n_no_tier),
        ("Tier A (dev)", n_tier_a),
        ("Tier B (dev)", n_tier_b),
        ("Tier C/D (dev)", n_tier_cd),
        ("Commercial with nonzero cz_eff", n_comm_with_czeff),
        ("Catalyst near (days<=180)", n_cat_near),
        ("No catalyst (missing)", n_no_cat),
    ]:
        pct = n / len(top_k_rows) * 100 if top_k_rows else 0
        L.append(f"| {label} | {n} | {pct:.0f}% |")
    L.append("")
    L.append(f"> Of {len(top_k_rows)} top returners, {n_dd} are drug developers, "
             f"{n_comm} are commercial (biotech+pharma), and {n_other} are other "
             f"archetypes. {n_early} are early-stage (cz_eff=0 by design). "
             f"{n_comm_with_czeff} commercial tickers have nonzero cz_eff.")
    L.append("")

    # --- 8. Notes ---
    L.append("## 8. Notes")
    missing_t0 = sum(1 for r in rows if r.get("price_t0") is None)
    missing_t1 = sum(1 for r in rows
                     if r.get("price_t1") is None and r.get("price_t0") is not None)
    L.append(f"- Tickers missing t0 price: {missing_t0}")
    L.append(f"- Tickers missing t1 price (had t0): {missing_t1}")
    L.append("- Survivorship bias: delisted/acquired tickers excluded.")
    L.append("- clinical_alpha_z uses stage_bucket as proxy for lead_phase "
             "(not in archive); readout_days from today's trial cache "
             "(PCD may have drifted).")
    L.append("- clinical_score_z is a pure z-score of the archive's "
             "clinical_score — fully PIT-safe.")
    L.append("- clinical_score_z_tier: tier-local z (ddof=0). "
             "Drug developers: within tier_dev. Commercial: within "
             "tier_commercial × archetype (fallback to archetype z "
             "for older archives without tier_commercial).")
    L.append("- cz_eff: stage-gated (early=0, mid=1.0, late=1.5), "
             "positive-only, clamped [0, 2]. The actual DE sort signal.")
    L.append("- Top-20% flags for clinical_score_z, clinical_score_z_tier, and "
             "cz_eff use **per-cohort thresholds** (drug_developer, "
             "commercial_biotech, commercial_pharma computed separately, "
             "then unioned). This prevents mixed-distribution distortion.")
    L.append("")

    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  Report: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top Returners Signal Recall Study"
    )
    parser.add_argument(
        "--archive", type=Path,
        default=Path("data/archives/2024-01-31.tar.gz"),
        help="Archive tar.gz for t0 (default: data/archives/2024-01-31.tar.gz)",
    )
    parser.add_argument(
        "--start-date", default="2024-01-31",
        help="Return window start (default: 2024-01-31)",
    )
    parser.add_argument(
        "--end-date", default="2026-01-27",
        help="Return window end (default: 2026-01-27)",
    )
    parser.add_argument(
        "--trial-records", type=Path, default=None,
        help="Trial records JSON (default: auto-detect latest ctgov cache)",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=Path("production_data/price_history.csv"),
        help="Price history CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/diag"),
        help="Output directory",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-pct", type=float, default=0.10)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip yfinance fetch")
    parser.add_argument("--price-cache", type=Path, default=None)
    args = parser.parse_args()

    cache_csv = args.price_cache or (args.output_dir / PRICE_CACHE_NAME)

    # ------------------------------------------------------------------
    # Step 1: Load t0 signals from archive
    # ------------------------------------------------------------------
    print("Step 1: Loading archive rankings...", file=sys.stderr)
    if not args.archive.exists():
        print(f"  ERROR: Archive not found: {args.archive}", file=sys.stderr)
        sys.exit(1)
    rows = load_archive_rankings(args.archive)
    if not rows:
        print("  ERROR: No rows loaded from archive", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(rows)} tickers from {args.archive.name}", file=sys.stderr)

    archetypes: Dict[str, int] = {}
    for r in rows:
        a = r.get("archetype", "unknown")
        archetypes[a] = archetypes.get(a, 0) + 1
    for a, c in sorted(archetypes.items(), key=lambda x: -x[1]):
        print(f"    {a}: {c}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 2: Compute clinical signals at t0
    # ------------------------------------------------------------------
    print("Step 2: Computing clinical signals...", file=sys.stderr)

    # 2a: PIT-safe clinical_score_z (no trial records needed)
    compute_clinical_score_z(rows)

    # 2a2: Tier-local z-score + cz_eff (PIT-safe, matches DE sort signal)
    compute_clinical_score_z_tier(rows)
    compute_cz_eff(rows)

    # 2b: Approximate-PIT clinical_alpha_z (needs trial records)
    trial_records_path = args.trial_records
    if trial_records_path is None:
        ctgov_dir = Path("cache/ctgov")
        if ctgov_dir.exists():
            candidates = sorted(ctgov_dir.glob("trial_records_*.json"),
                                reverse=True)
            if candidates:
                trial_records_path = candidates[0]
                print(f"  Auto-detected trial records: "
                      f"{trial_records_path.name}", file=sys.stderr)

    if trial_records_path is None or not trial_records_path.exists():
        print("  WARNING: No trial records — clinical_alpha_z will be empty",
              file=sys.stderr)
        trial_records: List[dict] = []
    else:
        with trial_records_path.open(encoding="utf-8") as f:
            all_trials = json.load(f)
        as_of = args.start_date
        trial_records = [
            t for t in all_trials
            if (t.get("first_posted") or "") <= as_of
        ]
        print(f"  Loaded {len(all_trials)} trials, {len(trial_records)} "
              f"PIT-safe (first_posted <= {as_of})", file=sys.stderr)

    readout_days = compute_clinical_readout_days(trial_records, args.start_date)
    tickers_with_rd = sum(1 for v in readout_days.values() if v is not None)
    print(f"  Readout days: {tickers_with_rd} tickers", file=sys.stderr)

    compute_clinical_alpha_z(rows, readout_days)

    # Sanity check z-scores
    for col_name in ["clinical_alpha_z", "clinical_score_z",
                     "clinical_score_z_tier", "cz_eff"]:
        zs = [v for v in (_safe_float(r.get(col_name)) for r in rows)
              if v is not None]
        if zs:
            print(f"  {col_name}: n={len(zs)}, mean={statistics.mean(zs):.4f}, "
                  f"std={statistics.stdev(zs):.4f}, "
                  f"range=[{min(zs):.2f}, {max(zs):.2f}]", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 3: Load prices + compute multi-horizon returns
    # ------------------------------------------------------------------
    print("Step 3: Loading prices...", file=sys.stderr)
    prices = load_prices(args.price_csv, cache_csv)
    trading_dates = _trading_dates_sorted(prices)
    print(f"  Loaded {len(prices)} price points, "
          f"{len(trading_dates)} trading dates "
          f"({trading_dates[0] if trading_dates else '?'} to "
          f"{trading_dates[-1] if trading_dates else '?'})", file=sys.stderr)

    if not args.no_fetch:
        all_tickers = {r["ticker"] for r in rows}
        missing = identify_missing_tickers(
            all_tickers, prices, trading_dates,
            args.start_date, args.end_date,
        )
        if missing:
            print(f"  {len(missing)} tickers need price data", file=sys.stderr)
            new_prices = fetch_missing_prices(
                missing, cache_csv, args.start_date, args.end_date
            )
            prices.update(new_prices)
            trading_dates = _trading_dates_sorted(prices)
        else:
            print("  No missing tickers to fetch", file=sys.stderr)

    rows, actual_t0, actual_t1, horizon_ends = compute_multi_horizon_returns(
        rows, prices, trading_dates, args.start_date, args.end_date
    )

    # ------------------------------------------------------------------
    # Step 4: Select top returners per horizon
    # ------------------------------------------------------------------
    print("Step 4: Selecting top returners...", file=sys.stderr)
    top_sets = select_top_returners_by_horizon(rows, args.top_k, args.top_pct)
    for label in RETURN_HORIZONS:
        tk_set, tp_set = top_sets[label]
        print(f"  {label}: top-{args.top_k}={len(tk_set)}, "
              f"top-{args.top_pct * 100:.0f}%={len(tp_set)}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 5: Compute flags + recall/lift by horizon
    # ------------------------------------------------------------------
    print("Step 5: Computing flags and recall...", file=sys.stderr)
    compute_flags(rows)

    recall_by_horizon = compute_recall_by_horizon(
        rows, top_sets, args.top_k
    )
    recall_24m = recall_by_horizon["24m"]

    # Print horizon comparison for key flags
    print(f"\n  {'Flag':28s} | "
          + " | ".join(f"R@{args.top_k} {h:>3s}" for h in RETURN_HORIZONS)
          + " | Baseline", file=sys.stderr)
    print(f"  {'-' * 28}-+-"
          + "-+-".join("-" * 9 for _ in RETURN_HORIZONS)
          + "-+---------", file=sys.stderr)
    for fi, flag in enumerate(FLAG_NAMES):
        cells = []
        for label in RETURN_HORIZONS:
            rh = recall_by_horizon[label][fi]
            cells.append(_fmt_pct(rh["recall_k"]))
        bl = _fmt_pct(recall_24m[fi]["random_baseline"])
        print(f"  {flag:28s} | "
              + " | ".join(f"{c:>9s}" for c in cells)
              + f" | {bl}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 6: Output
    # ------------------------------------------------------------------
    print("\nStep 6: Writing output...", file=sys.stderr)
    t0_str = actual_t0.replace("-", "")
    t1_str = actual_t1.replace("-", "")
    stem = f"top_returners_{t0_str}_{t1_str}"

    csv_path = args.output_dir / f"{stem}.csv"
    md_path = args.output_dir / f"{stem}.md"

    write_csv(rows, csv_path)
    write_report(rows, recall_24m, recall_by_horizon,
                 actual_t0, actual_t1, horizon_ends,
                 args.top_k, args.top_pct, md_path)

    # Quick summary to stdout
    print(f"\n--- Recall by Horizon (key table) ---")
    print(f"Universe: {len(rows)} tickers, "
          f"{sum(1 for r in rows if r.get('ret_24m') is not None)} survivors")
    print(f"Primary horizons: 6m / 12m | 24m: informational only")
    print(f"{'Flag':28s} | "
          + " | ".join(f"R@{args.top_k} {h:>3s}" for h in RETURN_HORIZONS)
          + " | Baseline | Lift 6m | Lift 12m")
    for fi, flag in enumerate(FLAG_NAMES):
        cells = []
        for label in RETURN_HORIZONS:
            rh = recall_by_horizon[label][fi]
            cells.append(_fmt_pct(rh["recall_k"]))
        bl = _fmt_pct(recall_24m[fi]["random_baseline"])
        lift_6m = _fmt_f2(recall_by_horizon["6m"][fi]["lift"])
        lift_12m = _fmt_f2(recall_by_horizon["12m"][fi]["lift"])
        print(f"  {flag:28s} | "
              + " | ".join(f"{c:>9s}" for c in cells)
              + f" | {bl:>8s} | {lift_6m:>7s} | {lift_12m}")


if __name__ == "__main__":
    main()
