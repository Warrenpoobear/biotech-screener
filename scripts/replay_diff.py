#!/usr/bin/env python3
"""Replay-bundle regression harness: compare two rankings.csv sources.

Computes diff metrics across overlap, rank stability, score stability,
tier/eligibility, and categorical drift.  Emits diff_report.json +
diff_report.md and exits 0/1/2 based on configurable thresholds.

Usage:
    python scripts/replay_diff.py \
        --baseline data/archives/2026-02-07.tar.gz \
        --candidate data/snapshots/2026-02-07/ \
        --output-dir output/ \
        [--thresholds path/to/thresholds.json] \
        [--json-out diff_report.json] \
        [--md-out diff_report.md] \
        [--strict]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from dataclasses import dataclass, fields as dc_fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# sys.path for sibling imports (pattern: compare_rulesets_replay.py:23)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "v1.1.0"

REQUIRED_COLUMNS = frozenset({
    "ticker", "eligible", "tier_dev", "composite_rank", "archetype",
})

# actionable_rank is preferred but optional — falls back to composite_rank
RANK_COLUMN = "actionable_rank"
RANK_FALLBACK = "composite_rank"

GOOD_CATALYST_MODES = frozenset({"specific_days", "blended_window", "far_window"})
BAD_CATALYST_MODES = frozenset({"no_upcoming", "missing"})

# ---------------------------------------------------------------------------
# Helpers (pattern: run_phase2_snapshot_delta.py:207-223)
# ---------------------------------------------------------------------------


def _safe_float(val, default=0.0) -> float:
    try:
        if pd.isna(val) or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        if pd.isna(val) or val == "":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _jaccard_overlap(set_a: set, set_b: set) -> float:
    """Jaccard overlap as percentage (pattern: compare_rulesets_replay.py:165)."""
    if not set_a and not set_b:
        return 100.0
    union = set_a | set_b
    if not union:
        return 100.0
    return round(100 * len(set_a & set_b) / len(union), 1)


def _spearman_rho(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Compute Spearman rank correlation without scipy.

    Uses Pearson-of-ranks formula (handles ties correctly).
    Aligns inputs by index before computation.
    """
    # Align by index to ensure paired values correspond to same ticker
    x_aligned, y_aligned = x.align(y, join="inner")
    valid = x_aligned.notna() & y_aligned.notna()
    x_v = x_aligned[valid].values.astype(float)
    y_v = y_aligned[valid].values.astype(float)
    n = len(x_v)
    if n < 3:
        return None

    # Rank-transform (average ties)
    def _rank(arr):
        temp = arr.argsort().argsort().astype(float) + 1.0
        # Handle ties: average rank for equal values
        for val in set(arr):
            mask = arr == val
            if mask.sum() > 1:
                temp[mask] = temp[mask].mean()
        return temp

    rx = _rank(x_v)
    ry = _rank(y_v)

    # Pearson correlation of ranks (handles ties correctly)
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    num = (rx_m * ry_m).sum()
    denom = math.sqrt((rx_m ** 2).sum() * (ry_m ** 2).sum())
    if denom == 0:
        return None
    return round(float(num / denom), 4)


def _is_eligible(val) -> bool:
    """Check if an eligible field is truthy."""
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in ("1", "1.0", "true", "yes")


def _is_dev_stage(archetype) -> bool:
    """True for drug_developer and platform archetypes."""
    if pd.isna(archetype):
        return False
    s = str(archetype).strip().lower()
    return s.startswith("drug_developer") or s.startswith("platform_")


def _is_commercial(archetype) -> bool:
    if pd.isna(archetype):
        return False
    return str(archetype).strip().lower().startswith("commercial_")


# ---------------------------------------------------------------------------
# Baseline freshness
# ---------------------------------------------------------------------------


def check_baseline_freshness(
    baseline_path: str, max_age_days: int = 14
) -> Dict[str, object]:
    """Check whether the golden baseline is stale.

    Looks for ``baseline_meta.json`` sidecar next to rankings.csv.
    Falls back to file mtime if no sidecar exists.

    Returns a dict with ``age_days``, ``stale``, ``source`` ("meta"|"mtime"|"unknown").
    """
    p = Path(baseline_path)

    # If baseline_path is a tarball, we can't check freshness meaningfully
    if not p.is_dir():
        return {"age_days": None, "stale": False, "source": "unknown"}

    meta_path = p / "baseline_meta.json"
    today = date.today()

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        created_str = meta.get("created_at") or meta.get("as_of_date")
        if created_str:
            created = date.fromisoformat(created_str)
            age = (today - created).days
            return {"age_days": age, "stale": age > max_age_days, "source": "meta"}

    # Fallback: rankings.csv mtime
    csv_path = p / "rankings.csv"
    if csv_path.exists():
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        age = (today - mtime.date()).days
        return {"age_days": age, "stale": age > max_age_days, "source": "mtime"}

    return {"age_days": None, "stale": False, "source": "unknown"}


# ---------------------------------------------------------------------------
# DiffThresholds (pattern: run_phase2_snapshot_delta.py:42-88)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffThresholds:
    """Immutable, versioned configuration for diff regression thresholds.

    Frozen so instances are hashable and produce a deterministic thresholds_id.
    """
    # Overlap (Jaccard)
    fail_top20_overlap_pct: float = 50.0
    warn_top20_overlap_pct: float = 80.0
    fail_top60_overlap_pct: float = 60.0
    warn_top60_overlap_pct: float = 85.0
    warn_top100_overlap_pct: float = 90.0
    # Rank stability
    fail_rank_spearman_rho: float = 0.80
    warn_rank_spearman_rho: float = 0.92
    warn_mean_abs_rank_delta_top60: float = 10.0
    # Score stability
    warn_composite_score_rmse: float = 5.0
    warn_score_rank_pct_mae: float = 0.10
    # Eligibility / Tier
    fail_eligibility_change_count: int = 20
    warn_eligibility_change_count: int = 5
    warn_tier_migration_count: int = 10
    # Catalyst
    warn_catalyst_good_to_bad: int = 5
    warn_catalyst_mode_change_count: int = 15
    # Weight
    warn_weight_l1_top20: float = 30.0

    def _canonical_json(self) -> str:
        """Deterministic JSON representation for hashing."""
        d = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def thresholds_id(self) -> str:
        """8-char hex hash of all parameters for change detection."""
        return hashlib.sha256(self._canonical_json().encode()).hexdigest()[:8]

    def to_json(self, path: str) -> None:
        d = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def from_json(cls, path: str) -> "DiffThresholds":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(**d)


# ---------------------------------------------------------------------------
# DiffResult
# ---------------------------------------------------------------------------


@dataclass
class DiffResult:
    """All computed diff metrics between baseline and candidate rankings."""
    # Metadata
    baseline_source: str
    candidate_source: str
    baseline_n_rows: int
    candidate_n_rows: int
    common_tickers: int
    schema_drift: Dict[str, List[str]]  # {"baseline_only": [...], "candidate_only": [...]}
    rank_column_info: Dict[str, object]  # provenance of rank column used

    # Tier 1 — Overlap
    top20_overlap_pct: float
    top60_overlap_pct: float
    top100_overlap_pct: float
    top20_entrants: List[str]
    top20_exits: List[str]
    top60_entrants: List[str]
    top60_exits: List[str]

    # Tier 2 — Rank stability
    rank_spearman_rho: Optional[float]
    mean_abs_rank_delta_top60: float
    max_abs_rank_delta_top60: int
    biggest_rank_movers: List[Tuple[str, int, int]]  # (ticker, base_rank, cand_rank)

    # Tier 3 — Score stability
    composite_score_rmse: Optional[float]
    score_rank_pct_mae: Optional[float]

    # Tier 4 — Tier / Eligibility
    eligibility_change_count: int
    eligibility_gained: List[str]
    eligibility_lost: List[str]
    tier_migration_count: int
    tier_migration_matrix: Dict[str, int]
    tier_dist_baseline: Dict[str, int]
    tier_dist_candidate: Dict[str, int]

    # Tier 5 — Categorical drift
    catalyst_mode_change_count: int
    catalyst_bad_to_good: List[str]
    catalyst_good_to_bad: List[str]
    weight_l1_top20: Optional[float]

    # Commercial section (all Optional)
    commercial_top20_overlap_pct: Optional[float]
    commercial_tier_migration_count: Optional[int]
    commercial_tier_migration_matrix: Optional[Dict[str, int]]

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        d = {}
        for f in dc_fields(self):
            v = getattr(self, f.name)
            if isinstance(v, list) and v and isinstance(v[0], tuple):
                # biggest_rank_movers: list of tuples → list of dicts
                d[f.name] = [
                    {"ticker": t, "base_rank": br, "cand_rank": cr}
                    for t, br, cr in v
                ]
            else:
                d[f.name] = v
        return d


# ---------------------------------------------------------------------------
# HealthVerdict
# ---------------------------------------------------------------------------


@dataclass
class HealthVerdict:
    status: str         # "OK" | "WARN" | "FAIL"
    fail_reasons: List[str]
    warn_reasons: List[str]
    exit_code: int      # 0 / 2 / 1


# ---------------------------------------------------------------------------
# load_rankings
# ---------------------------------------------------------------------------


def load_rankings(source: str) -> pd.DataFrame:
    """Load rankings.csv from a .tar.gz archive or a snapshot directory.

    Returns a DataFrame with dtype=str, deduplicated on ticker.
    Raises FileNotFoundError or ValueError on problems.
    """
    source_path = Path(source)

    if source_path.suffix == ".gz" and source_path.stem.endswith(".tar"):
        df = _load_from_tarball(source_path)
    elif source_path.is_dir():
        csv_path = source_path / "rankings.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"rankings.csv not found in {source_path}")
        df = pd.read_csv(csv_path, dtype=str)
    else:
        raise FileNotFoundError(
            f"Source must be a .tar.gz file or directory: {source}"
        )

    if df.empty:
        raise ValueError(f"rankings.csv is empty in {source}")

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"rankings.csv in {source} lacks required columns: "
            f"{sorted(missing)}"
        )

    # Synthesize actionable_rank from composite_rank if missing
    has_native_rank = RANK_COLUMN in df.columns
    if not has_native_rank and RANK_FALLBACK in df.columns:
        df[RANK_COLUMN] = df[RANK_FALLBACK]
    df.attrs["has_actionable_rank"] = has_native_rank
    df.attrs["rank_column_used"] = RANK_COLUMN if has_native_rank else RANK_FALLBACK

    df = df.drop_duplicates(subset="ticker", keep="first")
    return df


def _load_from_tarball(tar_path: Path) -> pd.DataFrame:
    """Stream-read rankings.csv from a .tar.gz.

    Safety: uses extractfile() only (never extracts to disk), rejects
    path-traversal names, and fails fast if multiple rankings.csv exist.
    Pattern: run_rank_ic_backtest.py:1116.
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        matches = []
        for member in tar.getmembers():
            # Skip non-files and suspicious path-traversal names
            if not member.isfile():
                continue
            if member.name.startswith("/") or ".." in member.name:
                continue
            basename = member.name.rsplit("/", 1)[-1] if "/" in member.name else member.name
            if basename == "rankings.csv":
                matches.append(member)

        if not matches:
            raise FileNotFoundError(f"rankings.csv not found in {tar_path}")

        if len(matches) > 1:
            names = [m.name for m in matches]
            raise ValueError(
                f"Ambiguous: {len(matches)} rankings.csv found in {tar_path}: "
                f"{names}"
            )

        csv_member = matches[0]
        f = io.TextIOWrapper(tar.extractfile(csv_member), encoding="utf-8")
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"rankings.csv is empty in {tar_path}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------


def _top_n_tickers(df: pd.DataFrame, n: int) -> set:
    """Get top-N tickers by actionable_rank."""
    ranked = df.copy()
    ranked["_arank"] = pd.to_numeric(ranked["actionable_rank"], errors="coerce")
    ranked = ranked.dropna(subset=["_arank"])
    ranked = ranked.sort_values("_arank").head(n)
    return set(ranked["ticker"])


def _tier_distribution(df: pd.DataFrame, col: str) -> Dict[str, int]:
    """Count tickers per tier value."""
    if col not in df.columns:
        return {}
    counts = df[col].fillna("").value_counts()
    return {k: int(v) for k, v in counts.items() if k}


def compute_diff(baseline: pd.DataFrame, candidate: pd.DataFrame) -> DiffResult:
    """Compare two rankings DataFrames and compute all diff metrics."""

    # Rank column provenance
    rank_column_info = {
        "baseline_has_actionable_rank": baseline.attrs.get("has_actionable_rank", True),
        "candidate_has_actionable_rank": candidate.attrs.get("has_actionable_rank", True),
        "baseline_rank_column_used": baseline.attrs.get("rank_column_used", RANK_COLUMN),
        "candidate_rank_column_used": candidate.attrs.get("rank_column_used", RANK_COLUMN),
    }

    # Schema drift
    base_cols = set(baseline.columns)
    cand_cols = set(candidate.columns)
    schema_drift = {
        "baseline_only": sorted(base_cols - cand_cols),
        "candidate_only": sorted(cand_cols - base_cols),
    }

    # Align on common tickers
    common = set(baseline["ticker"]) & set(candidate["ticker"])
    n_common = len(common)

    base = baseline[baseline["ticker"].isin(common)].copy()
    cand = candidate[candidate["ticker"].isin(common)].copy()

    # Index by ticker for aligned access
    base = base.set_index("ticker")
    cand = cand.set_index("ticker")

    # ── Tier 1: Overlap ──────────────────────────────────────────────────
    base_top20 = _top_n_tickers(baseline, 20)
    cand_top20 = _top_n_tickers(candidate, 20)
    base_top60 = _top_n_tickers(baseline, 60)
    cand_top60 = _top_n_tickers(candidate, 60)
    base_top100 = _top_n_tickers(baseline, 100)
    cand_top100 = _top_n_tickers(candidate, 100)

    top20_overlap = _jaccard_overlap(base_top20, cand_top20)
    top60_overlap = _jaccard_overlap(base_top60, cand_top60)
    top100_overlap = _jaccard_overlap(base_top100, cand_top100)

    top20_entrants = sorted(cand_top20 - base_top20)
    top20_exits = sorted(base_top20 - cand_top20)
    top60_entrants = sorted(cand_top60 - base_top60)
    top60_exits = sorted(base_top60 - cand_top60)

    # ── Tier 2: Rank stability ───────────────────────────────────────────
    rank_spearman_rho: Optional[float] = None
    if n_common > 0:
        base_arank = pd.to_numeric(base["actionable_rank"], errors="coerce")
        cand_arank = pd.to_numeric(cand["actionable_rank"], errors="coerce")

        # Eligible dev-stage for Spearman — use Series with same ticker index
        base_elig = pd.Series(
            [_is_eligible(base.loc[t, "eligible"]) for t in base.index],
            index=base.index,
        )
        cand_elig = pd.Series(
            [_is_eligible(cand.loc[t, "eligible"]) for t in cand.index],
            index=cand.index,
        )
        both_eligible = base_elig & cand_elig
        valid_for_spearman = both_eligible & base_arank.notna() & cand_arank.notna()

        if valid_for_spearman.sum() >= 3:
            rank_spearman_rho = _spearman_rho(
                base_arank[valid_for_spearman],
                cand_arank[valid_for_spearman],
            )
    else:
        base_arank = pd.Series(dtype=float)
        cand_arank = pd.Series(dtype=float)

    # Top-60 intersection rank deltas
    top60_common = base_top60 & cand_top60
    abs_deltas_top60 = []
    for t in top60_common:
        if t in base_arank.index and t in cand_arank.index:
            br = _safe_float(base_arank.get(t))
            cr = _safe_float(cand_arank.get(t))
            abs_deltas_top60.append(abs(br - cr))

    mean_abs_delta_top60 = round(
        sum(abs_deltas_top60) / len(abs_deltas_top60), 2
    ) if abs_deltas_top60 else 0.0
    max_abs_delta_top60 = int(max(abs_deltas_top60)) if abs_deltas_top60 else 0

    # Biggest movers (top 15 by |delta|)
    all_deltas = []
    for t in common:
        br = _safe_float(base_arank.get(t))
        cr = _safe_float(cand_arank.get(t))
        if br > 0 and cr > 0:
            all_deltas.append((t, int(br), int(cr), abs(br - cr)))
    all_deltas.sort(key=lambda x: x[3], reverse=True)
    biggest_movers = [(t, br, cr) for t, br, cr, _ in all_deltas[:15]]

    # ── Tier 3: Score stability ──────────────────────────────────────────
    composite_score_rmse: Optional[float] = None
    if "composite_score" in base.columns and "composite_score" in cand.columns:
        base_cs = pd.to_numeric(base["composite_score"], errors="coerce")
        cand_cs = pd.to_numeric(cand["composite_score"], errors="coerce")
        valid = base_cs.notna() & cand_cs.notna()
        if valid.sum() > 0:
            deltas = (base_cs[valid] - cand_cs[valid])
            composite_score_rmse = round(
                math.sqrt((deltas ** 2).mean()), 4
            )

    score_rank_pct_mae: Optional[float] = None
    if "score_rank_pct" in base.columns and "score_rank_pct" in cand.columns:
        base_srp = pd.to_numeric(base["score_rank_pct"], errors="coerce")
        cand_srp = pd.to_numeric(cand["score_rank_pct"], errors="coerce")
        valid = base_srp.notna() & cand_srp.notna()
        if valid.sum() > 0:
            score_rank_pct_mae = round(
                float((base_srp[valid] - cand_srp[valid]).abs().mean()), 4
            )

    # ── Tier 4: Tier / Eligibility ───────────────────────────────────────
    elig_gained = []
    elig_lost = []
    for t in common:
        be = _is_eligible(base.loc[t, "eligible"])
        ce = _is_eligible(cand.loc[t, "eligible"])
        if not be and ce:
            elig_gained.append(t)
        elif be and not ce:
            elig_lost.append(t)

    elig_gained.sort()
    elig_lost.sort()
    eligibility_change_count = len(elig_gained) + len(elig_lost)

    # Tier migration (dev)
    tier_mig_matrix: Dict[str, int] = {}
    tier_mig_count = 0
    for t in common:
        bt = str(base.loc[t, "tier_dev"]).strip() if pd.notna(base.loc[t, "tier_dev"]) else ""
        ct = str(cand.loc[t, "tier_dev"]).strip() if pd.notna(cand.loc[t, "tier_dev"]) else ""
        if bt and ct and bt != ct:
            key = f"{bt}->{ct}"
            tier_mig_matrix[key] = tier_mig_matrix.get(key, 0) + 1
            tier_mig_count += 1

    tier_dist_base = _tier_distribution(baseline, "tier_dev")
    tier_dist_cand = _tier_distribution(candidate, "tier_dev")

    # ── Tier 5: Categorical drift ────────────────────────────────────────
    cat_mode_changes = 0
    cat_bad_to_good = []
    cat_good_to_bad = []
    has_catalyst_mode = "catalyst_mode" in base.columns and "catalyst_mode" in cand.columns

    if has_catalyst_mode:
        for t in common:
            bm = str(base.loc[t, "catalyst_mode"]).strip() if pd.notna(base.loc[t, "catalyst_mode"]) else ""
            cm = str(cand.loc[t, "catalyst_mode"]).strip() if pd.notna(cand.loc[t, "catalyst_mode"]) else ""
            if bm != cm:
                cat_mode_changes += 1
                if bm in BAD_CATALYST_MODES and cm in GOOD_CATALYST_MODES:
                    cat_bad_to_good.append(t)
                elif bm in GOOD_CATALYST_MODES and cm in BAD_CATALYST_MODES:
                    cat_good_to_bad.append(t)
        cat_bad_to_good.sort()
        cat_good_to_bad.sort()

    # Weight L1 (top-20 common)
    weight_l1: Optional[float] = None
    has_weight = "target_weight_pct" in base.columns and "target_weight_pct" in cand.columns
    if has_weight:
        top20_common = base_top20 & cand_top20
        l1_sum = 0.0
        for t in top20_common:
            if t in base.index and t in cand.index:
                bw = _safe_float(base.loc[t, "target_weight_pct"])
                cw = _safe_float(cand.loc[t, "target_weight_pct"])
                l1_sum += abs(bw - cw)
        # Also add weight for entrants/exits (count as full delta)
        for t in top20_entrants:
            if t in cand.index:
                cw = _safe_float(cand.loc[t, "target_weight_pct"])
                l1_sum += cw
        for t in top20_exits:
            if t in base.index:
                bw = _safe_float(base.loc[t, "target_weight_pct"])
                l1_sum += bw
        weight_l1 = round(l1_sum, 2)

    # ── Commercial section ───────────────────────────────────────────────
    comm_top20_overlap: Optional[float] = None
    comm_tier_mig_count: Optional[int] = None
    comm_tier_mig_matrix: Optional[Dict[str, int]] = None

    has_tier_commercial = (
        "tier_commercial" in baseline.columns
        and "tier_commercial" in candidate.columns
    )
    if has_tier_commercial:
        base_comm = baseline[baseline["archetype"].map(_is_commercial)].copy()
        cand_comm = candidate[candidate["archetype"].map(_is_commercial)].copy()

        if not base_comm.empty and not cand_comm.empty:
            bc_top20 = _top_n_tickers(base_comm, 20)
            cc_top20 = _top_n_tickers(cand_comm, 20)
            comm_top20_overlap = _jaccard_overlap(bc_top20, cc_top20)

            # Commercial tier migration
            comm_common = set(base_comm["ticker"]) & set(cand_comm["ticker"])
            bc_idx = base_comm.set_index("ticker")
            cc_idx = cand_comm.set_index("ticker")
            comm_tier_mig_matrix = {}
            comm_tier_mig_count = 0
            for t in comm_common:
                bt = str(bc_idx.loc[t, "tier_commercial"]).strip() if pd.notna(bc_idx.loc[t, "tier_commercial"]) else ""
                ct = str(cc_idx.loc[t, "tier_commercial"]).strip() if pd.notna(cc_idx.loc[t, "tier_commercial"]) else ""
                if bt and ct and bt != ct:
                    key = f"{bt}->{ct}"
                    comm_tier_mig_matrix[key] = comm_tier_mig_matrix.get(key, 0) + 1
                    comm_tier_mig_count += 1

    return DiffResult(
        baseline_source=str(baseline.attrs.get("source", "baseline")),
        candidate_source=str(candidate.attrs.get("source", "candidate")),
        baseline_n_rows=len(baseline),
        candidate_n_rows=len(candidate),
        common_tickers=n_common,
        schema_drift=schema_drift,
        rank_column_info=rank_column_info,
        top20_overlap_pct=top20_overlap,
        top60_overlap_pct=top60_overlap,
        top100_overlap_pct=top100_overlap,
        top20_entrants=top20_entrants,
        top20_exits=top20_exits,
        top60_entrants=top60_entrants,
        top60_exits=top60_exits,
        rank_spearman_rho=rank_spearman_rho,
        mean_abs_rank_delta_top60=mean_abs_delta_top60,
        max_abs_rank_delta_top60=max_abs_delta_top60,
        biggest_rank_movers=biggest_movers,
        composite_score_rmse=composite_score_rmse,
        score_rank_pct_mae=score_rank_pct_mae,
        eligibility_change_count=eligibility_change_count,
        eligibility_gained=elig_gained,
        eligibility_lost=elig_lost,
        tier_migration_count=tier_mig_count,
        tier_migration_matrix=tier_mig_matrix,
        tier_dist_baseline=tier_dist_base,
        tier_dist_candidate=tier_dist_cand,
        catalyst_mode_change_count=cat_mode_changes,
        catalyst_bad_to_good=cat_bad_to_good,
        catalyst_good_to_bad=cat_good_to_bad,
        weight_l1_top20=weight_l1,
        commercial_top20_overlap_pct=comm_top20_overlap,
        commercial_tier_migration_count=comm_tier_mig_count,
        commercial_tier_migration_matrix=comm_tier_mig_matrix,
    )


# ---------------------------------------------------------------------------
# evaluate_health
# ---------------------------------------------------------------------------


def evaluate_health(
    result: DiffResult, thresholds: DiffThresholds
) -> HealthVerdict:
    """Evaluate DiffResult against thresholds. FAIL checked first, then WARN."""
    fails: List[str] = []
    warns: List[str] = []

    # ── Pre-check: meaningless comparison ─────────────────────────────────
    if result.common_tickers == 0:
        return HealthVerdict(
            status="FAIL",
            fail_reasons=["no_common_universe: baseline and candidate share 0 tickers"],
            warn_reasons=[],
            exit_code=1,
        )

    # ── FAIL gates ────────────────────────────────────────────────────────
    if result.top20_overlap_pct < thresholds.fail_top20_overlap_pct:
        fails.append(
            f"top20_overlap {result.top20_overlap_pct:.1f}% "
            f"< {thresholds.fail_top20_overlap_pct:.1f}%"
        )

    if result.top60_overlap_pct < thresholds.fail_top60_overlap_pct:
        fails.append(
            f"top60_overlap {result.top60_overlap_pct:.1f}% "
            f"< {thresholds.fail_top60_overlap_pct:.1f}%"
        )

    if (
        result.rank_spearman_rho is not None
        and result.rank_spearman_rho < thresholds.fail_rank_spearman_rho
    ):
        fails.append(
            f"rank_spearman_rho {result.rank_spearman_rho:.4f} "
            f"< {thresholds.fail_rank_spearman_rho}"
        )

    if result.eligibility_change_count > thresholds.fail_eligibility_change_count:
        fails.append(
            f"eligibility_changes {result.eligibility_change_count} "
            f"> {thresholds.fail_eligibility_change_count}"
        )

    # ── WARN gates ────────────────────────────────────────────────────────
    if result.top20_overlap_pct < thresholds.warn_top20_overlap_pct:
        warns.append(
            f"top20_overlap {result.top20_overlap_pct:.1f}% "
            f"< {thresholds.warn_top20_overlap_pct:.1f}%"
        )

    if result.top60_overlap_pct < thresholds.warn_top60_overlap_pct:
        warns.append(
            f"top60_overlap {result.top60_overlap_pct:.1f}% "
            f"< {thresholds.warn_top60_overlap_pct:.1f}%"
        )

    if result.top100_overlap_pct < thresholds.warn_top100_overlap_pct:
        warns.append(
            f"top100_overlap {result.top100_overlap_pct:.1f}% "
            f"< {thresholds.warn_top100_overlap_pct:.1f}%"
        )

    if (
        result.rank_spearman_rho is not None
        and result.rank_spearman_rho < thresholds.warn_rank_spearman_rho
    ):
        warns.append(
            f"rank_spearman_rho {result.rank_spearman_rho:.4f} "
            f"< {thresholds.warn_rank_spearman_rho}"
        )

    if result.mean_abs_rank_delta_top60 > thresholds.warn_mean_abs_rank_delta_top60:
        warns.append(
            f"mean_abs_rank_delta_top60 {result.mean_abs_rank_delta_top60:.2f} "
            f"> {thresholds.warn_mean_abs_rank_delta_top60}"
        )

    if (
        result.composite_score_rmse is not None
        and result.composite_score_rmse > thresholds.warn_composite_score_rmse
    ):
        warns.append(
            f"composite_score_rmse {result.composite_score_rmse:.4f} "
            f"> {thresholds.warn_composite_score_rmse}"
        )

    if (
        result.score_rank_pct_mae is not None
        and result.score_rank_pct_mae > thresholds.warn_score_rank_pct_mae
    ):
        warns.append(
            f"score_rank_pct_mae {result.score_rank_pct_mae:.4f} "
            f"> {thresholds.warn_score_rank_pct_mae}"
        )

    if result.eligibility_change_count > thresholds.warn_eligibility_change_count:
        warns.append(
            f"eligibility_changes {result.eligibility_change_count} "
            f"> {thresholds.warn_eligibility_change_count}"
        )

    if result.tier_migration_count > thresholds.warn_tier_migration_count:
        warns.append(
            f"tier_migration {result.tier_migration_count} "
            f"> {thresholds.warn_tier_migration_count}"
        )

    if result.catalyst_mode_change_count > thresholds.warn_catalyst_mode_change_count:
        warns.append(
            f"catalyst_mode_changes {result.catalyst_mode_change_count} "
            f"> {thresholds.warn_catalyst_mode_change_count}"
        )

    if len(result.catalyst_good_to_bad) > thresholds.warn_catalyst_good_to_bad:
        warns.append(
            f"catalyst_good_to_bad {len(result.catalyst_good_to_bad)} "
            f"> {thresholds.warn_catalyst_good_to_bad}"
        )

    if (
        result.weight_l1_top20 is not None
        and result.weight_l1_top20 > thresholds.warn_weight_l1_top20
    ):
        warns.append(
            f"weight_l1_top20 {result.weight_l1_top20:.2f} "
            f"> {thresholds.warn_weight_l1_top20}"
        )

    # ── Verdict ───────────────────────────────────────────────────────────
    if fails:
        return HealthVerdict(
            status="FAIL", fail_reasons=fails, warn_reasons=warns, exit_code=1
        )
    if warns:
        return HealthVerdict(
            status="WARN", fail_reasons=[], warn_reasons=warns, exit_code=2
        )
    return HealthVerdict(
        status="OK", fail_reasons=[], warn_reasons=[], exit_code=0
    )


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------


def format_json_report(
    result: DiffResult,
    verdict: HealthVerdict,
    thresholds: DiffThresholds,
    baseline_freshness: Optional[Dict[str, object]] = None,
) -> dict:
    """Build a JSON-serializable report dict."""
    report = {
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_id": thresholds.thresholds_id,
        "verdict": {
            "status": verdict.status,
            "exit_code": verdict.exit_code,
            "fail_reasons": verdict.fail_reasons,
            "warn_reasons": verdict.warn_reasons,
        },
        "metrics": result.to_dict(),
        "thresholds": {
            f.name: getattr(thresholds, f.name) for f in dc_fields(thresholds)
        },
    }
    if baseline_freshness is not None:
        report["baseline_freshness"] = baseline_freshness
    return report


def format_markdown_report(
    result: DiffResult,
    verdict: HealthVerdict,
) -> str:
    """Build a human-readable markdown report."""
    lines: List[str] = []
    status_badge = {"OK": "OK", "WARN": "WARN", "FAIL": "**FAIL**"}

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("# Replay Diff Report")
    lines.append("")
    lines.append(f"**Status**: {status_badge.get(verdict.status, verdict.status)}")
    lines.append(f"**Baseline**: `{result.baseline_source}`")
    lines.append(f"**Candidate**: `{result.candidate_source}`")
    lines.append(
        f"**Universe**: {result.baseline_n_rows} baseline / "
        f"{result.candidate_n_rows} candidate / "
        f"{result.common_tickers} common"
    )

    # Rank column provenance warning
    rci = result.rank_column_info
    if not rci.get("baseline_has_actionable_rank") or not rci.get("candidate_has_actionable_rank"):
        parts = []
        if not rci.get("baseline_has_actionable_rank"):
            parts.append(f"baseline uses `{rci.get('baseline_rank_column_used')}`")
        if not rci.get("candidate_has_actionable_rank"):
            parts.append(f"candidate uses `{rci.get('candidate_rank_column_used')}`")
        lines.append(
            f"**Rank column fallback**: {'; '.join(parts)} "
            f"(actionable_rank not present)"
        )
    lines.append("")

    if verdict.fail_reasons:
        lines.append("## FAIL Reasons")
        for r in verdict.fail_reasons:
            lines.append(f"- {r}")
        lines.append("")

    if verdict.warn_reasons:
        lines.append("## WARN Reasons")
        for r in verdict.warn_reasons:
            lines.append(f"- {r}")
        lines.append("")

    # ── Overlap ───────────────────────────────────────────────────────────
    lines.append("## Overlap (Jaccard)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Top-20 overlap | {result.top20_overlap_pct:.1f}% |")
    lines.append(f"| Top-60 overlap | {result.top60_overlap_pct:.1f}% |")
    lines.append(f"| Top-100 overlap | {result.top100_overlap_pct:.1f}% |")
    lines.append("")

    if result.top20_entrants or result.top20_exits:
        lines.append("**Top-20 changes:**")
        if result.top20_entrants:
            lines.append(f"- Entrants: {', '.join(result.top20_entrants)}")
        if result.top20_exits:
            lines.append(f"- Exits: {', '.join(result.top20_exits)}")
        lines.append("")

    # ── Rank stability ────────────────────────────────────────────────────
    lines.append("## Rank Stability")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    rho_str = f"{result.rank_spearman_rho:.4f}" if result.rank_spearman_rho is not None else "(n/a)"
    lines.append(f"| Spearman rho | {rho_str} |")
    lines.append(f"| Mean |rank delta| top-60 | {result.mean_abs_rank_delta_top60:.2f} |")
    lines.append(f"| Max |rank delta| top-60 | {result.max_abs_rank_delta_top60} |")
    lines.append("")

    # ── Score stability ───────────────────────────────────────────────────
    lines.append("## Score Stability")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    rmse_str = f"{result.composite_score_rmse:.4f}" if result.composite_score_rmse is not None else "(column missing)"
    mae_str = f"{result.score_rank_pct_mae:.4f}" if result.score_rank_pct_mae is not None else "(column missing)"
    lines.append(f"| Composite score RMSE | {rmse_str} |")
    lines.append(f"| Score rank pct MAE | {mae_str} |")
    lines.append("")

    # ── Tier migration ────────────────────────────────────────────────────
    lines.append("## Tier Migration (dev)")
    lines.append("")
    lines.append(f"Total tier changes: {result.tier_migration_count}")
    lines.append("")
    if result.tier_migration_matrix:
        lines.append("| Transition | Count |")
        lines.append("|-----------|-------|")
        for k in sorted(result.tier_migration_matrix.keys()):
            lines.append(f"| {k} | {result.tier_migration_matrix[k]} |")
        lines.append("")

    lines.append("**Distribution (baseline → candidate):**")
    lines.append("")
    all_tiers = sorted(set(list(result.tier_dist_baseline.keys()) + list(result.tier_dist_candidate.keys())))
    if all_tiers:
        lines.append("| Tier | Baseline | Candidate |")
        lines.append("|------|----------|-----------|")
        for tier in all_tiers:
            lines.append(
                f"| {tier} | {result.tier_dist_baseline.get(tier, 0)} | "
                f"{result.tier_dist_candidate.get(tier, 0)} |"
            )
        lines.append("")

    # ── Eligibility ───────────────────────────────────────────────────────
    lines.append("## Eligibility Changes")
    lines.append("")
    lines.append(f"Total: {result.eligibility_change_count}")
    if result.eligibility_gained:
        lines.append(f"- Gained: {', '.join(result.eligibility_gained)}")
    if result.eligibility_lost:
        lines.append(f"- Lost: {', '.join(result.eligibility_lost)}")
    lines.append("")

    # ── Catalyst transitions ──────────────────────────────────────────────
    lines.append("## Catalyst Mode Transitions")
    lines.append("")
    lines.append(f"Total mode changes: {result.catalyst_mode_change_count}")
    if result.catalyst_bad_to_good:
        lines.append(f"- Bad→Good: {', '.join(result.catalyst_bad_to_good)}")
    if result.catalyst_good_to_bad:
        lines.append(f"- Good→Bad: {', '.join(result.catalyst_good_to_bad)}")
    lines.append("")

    # ── Weight drift ──────────────────────────────────────────────────────
    lines.append("## Weight Drift")
    lines.append("")
    if result.weight_l1_top20 is not None:
        lines.append(f"Top-20 weight L1: {result.weight_l1_top20:.2f} pp")
    else:
        lines.append("Top-20 weight L1: (column missing)")
    lines.append("")

    # ── Schema drift ──────────────────────────────────────────────────────
    if result.schema_drift["baseline_only"] or result.schema_drift["candidate_only"]:
        lines.append("## Schema Drift")
        lines.append("")
        if result.schema_drift["baseline_only"]:
            lines.append(
                f"- Baseline only: {', '.join(result.schema_drift['baseline_only'])}"
            )
        if result.schema_drift["candidate_only"]:
            lines.append(
                f"- Candidate only: {', '.join(result.schema_drift['candidate_only'])}"
            )
        lines.append("")

    # ── Biggest movers ────────────────────────────────────────────────────
    if result.biggest_rank_movers:
        lines.append("## Biggest Rank Movers")
        lines.append("")
        lines.append("| Ticker | Baseline Rank | Candidate Rank | Delta |")
        lines.append("|--------|--------------|----------------|-------|")
        for ticker, br, cr in result.biggest_rank_movers:
            lines.append(f"| {ticker} | {br} | {cr} | {cr - br:+d} |")
        lines.append("")

    # ── Commercial section ────────────────────────────────────────────────
    if result.commercial_top20_overlap_pct is not None:
        lines.append("## Commercial (Informational)")
        lines.append("")
        lines.append(f"Top-20 overlap: {result.commercial_top20_overlap_pct:.1f}%")
        if result.commercial_tier_migration_count is not None:
            lines.append(
                f"Tier migration count: {result.commercial_tier_migration_count}"
            )
        if result.commercial_tier_migration_matrix:
            lines.append("")
            lines.append("| Transition | Count |")
            lines.append("|-----------|-------|")
            for k in sorted(result.commercial_tier_migration_matrix.keys()):
                lines.append(
                    f"| {k} | {result.commercial_tier_migration_matrix[k]} |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two rankings.csv sources and compute regression metrics."
    )
    p.add_argument(
        "--baseline", required=True,
        help="Path to baseline .tar.gz or snapshot directory",
    )
    p.add_argument(
        "--candidate", required=True,
        help="Path to candidate .tar.gz or snapshot directory",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: output/)",
    )
    p.add_argument(
        "--thresholds",
        help="Path to DiffThresholds JSON override",
    )
    p.add_argument(
        "--json-out", default="diff_report.json",
        help="JSON output filename (default: diff_report.json)",
    )
    p.add_argument(
        "--md-out", default="diff_report.md",
        help="Markdown output filename (default: diff_report.md)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="WARN also exits 1 (for CI lockdown)",
    )
    p.add_argument(
        "--baseline-max-age-days", type=int, default=14,
        help="Warn if golden baseline is older than N days (default: 14)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Load thresholds
    if args.thresholds:
        thresholds = DiffThresholds.from_json(args.thresholds)
    else:
        thresholds = DiffThresholds()

    print(f"Thresholds ID: {thresholds.thresholds_id}")

    # Baseline freshness
    freshness = check_baseline_freshness(
        args.baseline, args.baseline_max_age_days
    )
    if freshness["age_days"] is not None:
        print(f"Baseline age: {freshness['age_days']} days (source: {freshness['source']})")
        if freshness["stale"]:
            print(f"  WARNING: baseline is stale (>{args.baseline_max_age_days} days)")

    # Load sources
    print(f"Loading baseline: {args.baseline}")
    baseline = load_rankings(args.baseline)
    baseline.attrs["source"] = args.baseline
    print(f"  {len(baseline)} rows")

    print(f"Loading candidate: {args.candidate}")
    candidate = load_rankings(args.candidate)
    candidate.attrs["source"] = args.candidate
    print(f"  {len(candidate)} rows")

    # Compute diff
    result = compute_diff(baseline, candidate)

    # Evaluate health
    verdict = evaluate_health(result, thresholds)

    # Inject baseline_stale warning if applicable
    if freshness["stale"]:
        age = freshness["age_days"]
        verdict.warn_reasons.append(
            f"baseline_stale: golden baseline is {age} days old "
            f"(max {args.baseline_max_age_days})"
        )
        if verdict.status == "OK":
            verdict.status = "WARN"
            verdict.exit_code = 2

    print(f"\nVerdict: {verdict.status} (exit code {verdict.exit_code})")

    if verdict.fail_reasons:
        for r in verdict.fail_reasons:
            print(f"  FAIL: {r}")
    if verdict.warn_reasons:
        for r in verdict.warn_reasons:
            print(f"  WARN: {r}")

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / args.json_out
    report = format_json_report(result, verdict, thresholds, freshness)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"\nJSON report: {json_path}")

    md_path = out_dir / args.md_out
    md_content = format_markdown_report(result, verdict)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)
    print(f"Markdown report: {md_path}")

    exit_code = verdict.exit_code
    if args.strict and exit_code == 2:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
