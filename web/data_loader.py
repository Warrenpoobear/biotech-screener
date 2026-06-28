"""Data loading layer for the Cell Atlas.

Pure-Python (stdlib ``csv``/``json`` only) loaders that read the project's
on-disk data and project each biotech company onto a biological "cell"
metaphor:

    * membrane   -> market status, cap tier
    * nucleus    -> pipeline phase, lead asset, mechanism of action
    * organelles -> financial health, cash runway
    * receptors  -> catalysts (PDUFA dates, trial readouts)
    * signaling  -> trial-network connections

All paths are resolved relative to the repository root (the parent of this
``web`` package) so the app works regardless of the current working
directory, as long as it is launched from somewhere inside the repo.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Absolute path to the biotech_screener project root."""


def _p(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def universe_path() -> Path:
    return _p("data", "universe", "biotech_universe_v1.csv")


def prices_path() -> Path:
    return _p("data", "daily_prices.csv")


def trial_mapping_path() -> Path:
    return _p("data", "trial_mapping.csv")


def snapshots_dir() -> Path:
    """Directory holding composite-score snapshot JSONs.

    Prefers a top-level ``snapshots/`` directory (the documented location);
    falls back to ``output/`` where generated snapshots actually live.
    """
    primary = _p("snapshots")
    if primary.is_dir():
        return primary
    secondary = _p("output")
    if secondary.is_dir():
        return secondary
    return primary  # return default even if missing; callers handle gracefully


def aact_snapshots_dir() -> Path:
    return _p("data", "aact_snapshots")


# ---------------------------------------------------------------------------
# Therapeutic category ("tissue") lookup
# ---------------------------------------------------------------------------

# A curated ticker -> therapeutic-category map covering the known universe
# and the tickers that appear in generated snapshots. Unknown tickers fall
# back to "Diversified".
TISSUE_LOOKUP: Dict[str, str] = {
    "ACAD": "Neurology",
    "ALNY": "Rare Disease",
    "AMGN": "Diversified",
    "ARWR": "Rare Disease",
    "BEAM": "Rare Disease",
    "BIIB": "Neurology",
    "BLUE": "Rare Disease",
    "BMRN": "Rare Disease",
    "BNTX": "Oncology",
    "EDIT": "Rare Disease",
    "EXEL": "Oncology",
    "FOLD": "Rare Disease",
    "GILD": "Virology",
    "HALO": "Enabling Tech",
    "IMVT": "Immunology",
    "INCY": "Oncology",
    "IONS": "Neurology",
    "KRTX": "Neurology",
    "MRNA": "Infectious Disease",
    "PCVX": "Infectious Disease",
    "RARE": "Rare Disease",
    "REGN": "Immunology",
    "SGEN": "Oncology",
    "SRPT": "Rare Disease",
    "VRTX": "Rare Disease",
}


def tissue_for(ticker: str, flags: Optional[List[str]] = None) -> str:
    """Return the therapeutic category ("tissue") for a ticker."""
    return TISSUE_LOOKUP.get(ticker.upper(), "Diversified")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float; returns ``None`` on failure/None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    return int(round(f)) if f is not None else None


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse a YYYY-MM-DD prefix into a ``date``; ``None`` on empty/failure."""
    if not value or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_universe() -> Dict[str, Dict[str, str]]:
    """Load the universe CSV -> ``{ticker: {name, sector, ...}}``."""
    out: Dict[str, Dict[str, str]] = {}
    path = universe_path()
    if not path.exists():
        logger.warning("Universe file not found: %s", path)
        return out
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            out[ticker] = {k: (v or "").strip() for k, v in row.items()}
    return out


def load_latest_prices(as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load the most recent (date, adj_close) per ticker from daily prices.

    When ``as_of`` (YYYY-MM-DD) is provided, only rows dated on or before it are
    considered, so the price is point-in-time aligned with the selected
    snapshot rather than always reflecting the latest available close.

    Returns ``{ticker: {price, date}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    path = prices_path()
    if not path.exists():
        logger.warning("Prices file not found: %s", path)
        return out
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            d = (row.get("date") or "").strip()
            if as_of and d and d > as_of:
                continue  # PIT guard: ignore prices after the as-of date
            price = _to_float(row.get("adj_close"))
            if price is None:
                continue
            existing = out.get(ticker)
            if existing is None or d >= existing["date"]:
                out[ticker] = {"price": price, "date": d}
    return out


def load_trial_mapping() -> Dict[str, List[Dict[str, str]]]:
    """Load trial-to-ticker mapping -> ``{ticker: [trial_row, ...]}``."""
    out: Dict[str, List[Dict[str, str]]] = {}
    path = trial_mapping_path()
    if not path.exists():
        logger.warning("Trial mapping file not found: %s", path)
        return out
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            clean = {k: (v or "").strip() for k, v in row.items()}
            out.setdefault(ticker, []).append(clean)
    return out


def load_snapshot_dates() -> List[str]:
    """Return sorted (ascending) list of snapshot date strings (YYYY-MM-DD)."""
    d = snapshots_dir()
    if not d.is_dir():
        return []
    dates: List[str] = []
    for p in d.glob("snapshot_*.json"):
        # snapshot_YYYY-MM-DD.json
        stem = p.stem  # snapshot_YYYY-MM-DD
        ds = stem.replace("snapshot_", "")
        if len(ds) == 10 and ds[4] == "-":
            dates.append(ds)
    return sorted(dates)


def load_snapshot(snapshot_date: str) -> Dict[str, Any]:
    """Load a single snapshot JSON by date string (YYYY-MM-DD)."""
    d = snapshots_dir()
    # Support both snapshot_YYYY-MM-DD.json and YYYY-MM-DD.json naming
    for name in (f"snapshot_{snapshot_date}.json", f"{snapshot_date}.json"):
        p = d / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def load_latest_snapshot() -> Dict[str, Any]:
    """Load the most recent snapshot (or ``{}`` if none)."""
    dates = load_snapshot_dates()
    if not dates:
        return {}
    return load_snapshot(dates[-1])


def load_snapshot_index() -> Dict[str, Dict[str, Any]]:
    """Map every snapshot date -> snapshot dict."""
    return {d: load_snapshot(d) for d in load_snapshot_dates()}


def _ranked_sec_to_dict(sec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a ``ranked_securities`` entry, coercing string numerics."""
    return {
        "ticker": (sec.get("ticker") or "").upper(),
        "composite_score": _to_float(sec.get("composite_score")),
        "composite_rank": _to_int(sec.get("composite_rank")),
        "clinical_dev_score": _to_float(sec.get("clinical_dev_normalized") or sec.get("clinical_dev_raw")),
        "clinical_dev_raw": _to_float(sec.get("clinical_dev_raw")),
        "financial_score": _to_float(sec.get("financial_normalized") or sec.get("financial_raw")),
        "financial_raw": _to_float(sec.get("financial_raw")),
        "catalyst_score": _to_float(sec.get("catalyst_normalized") or sec.get("catalyst_raw")),
        "catalyst_raw": _to_float(sec.get("catalyst_raw")),
        "market_cap_bucket": sec.get("market_cap_bucket") or "unknown",
        "stage_bucket": sec.get("stage_bucket") or "unknown",
        "severity": sec.get("severity") or "unknown",
        "uncertainty_penalty": _to_float(sec.get("uncertainty_penalty")),
        "missing_subfactor_pct": _to_float(sec.get("missing_subfactor_pct")),
        "rankable": bool(sec.get("rankable", False)),
        "flags": list(sec.get("flags") or []),
    }


def snapshot_scores_for(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten ``ranked_securities`` of a snapshot into ``{ticker: scores}``."""
    out: Dict[str, Dict[str, Any]] = {}
    for sec in snapshot.get("ranked_securities") or []:
        ticker = (sec.get("ticker") or "").upper()
        if ticker:
            out[ticker] = _ranked_sec_to_dict(sec)
    return out


def _latest_aact_snapshot_dir() -> Optional[Path]:
    d = aact_snapshots_dir()
    if not d.is_dir():
        return None
    subdirs = sorted(p.name for p in d.iterdir() if p.is_dir())
    if not subdirs:
        return None
    return d / subdirs[-1]


def load_aact_studies() -> Dict[str, Dict[str, str]]:
    """Load AACT studies from the latest AACT snapshot dir -> ``{nct_id: row}``."""
    sd = _latest_aact_snapshot_dir()
    if sd is None:
        return {}
    studies = sd / "studies.csv"
    if not studies.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with open(studies, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nct = (row.get("nct_id") or "").strip()
            if nct:
                out[nct] = {k: (v or "").strip() for k, v in row.items()}
    return out


def load_aact_sponsors() -> Dict[str, List[str]]:
    """Load AACT sponsors -> ``{nct_id: [sponsor_name, ...]}``."""
    sd = _latest_aact_snapshot_dir()
    if sd is None:
        return {}
    sponsors = sd / "sponsors.csv"
    if not sponsors.exists():
        return {}
    out: Dict[str, List[str]] = {}
    with open(sponsors, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nct = (row.get("nct_id") or "").strip()
            name = (row.get("name") or "").strip()
            if nct and name:
                out.setdefault(nct, []).append(name)
    return out


# ---------------------------------------------------------------------------
# Cell construction (the biological metaphor)
# ---------------------------------------------------------------------------


def _phase_rank(phase: str) -> int:
    """Numeric rank for a trial phase string (higher = later stage)."""
    p = (phase or "").lower()
    if "phase 4" in p or "phase 3" in p:
        return 4
    if "phase 2" in p and "phase 3" in p:
        return 3
    if "phase 2" in p:
        return 2
    if "phase 1" in p and "phase 2" in p:
        return 2
    if "phase 1" in p:
        return 1
    if "early" in p or "phase 0" in p:
        return 0
    return 0


def build_cell(
    ticker: str,
    universe: Dict[str, Dict[str, str]],
    prices: Dict[str, Dict[str, Any]],
    trials: Dict[str, List[Dict[str, str]]],
    scores: Dict[str, Dict[str, Any]],
    aact_studies: Dict[str, Dict[str, str]],
    aact_sponsors: Dict[str, List[str]],
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full cell-atlas record for a single ticker.

    ``as_of`` (YYYY-MM-DD) anchors the "upcoming catalyst" test to a snapshot's
    as-of date for point-in-time correctness; defaults to today when omitted.
    """
    ticker = ticker.upper()
    uni = universe.get(ticker, {})
    price_info = prices.get(ticker, {})
    ticker_trials = trials.get(ticker, [])
    sc = scores.get(ticker, {})

    # ---- Nucleus: pipeline phase / lead asset / MoA ----
    lead_phase: Optional[str] = None
    lead_nct: Optional[str] = None
    lead_status: Optional[str] = None
    lead_pcd: Optional[str] = None
    best_rank = -1
    for trial in ticker_trials:
        nct = trial.get("nct_id", "")
        study = aact_studies.get(nct, {})
        phase = study.get("phase") or trial.get("phase") or "Unknown"
        rank = _phase_rank(phase)
        if rank > best_rank:
            best_rank = rank
            lead_phase = phase
            lead_nct = nct
            lead_status = study.get("overall_status") or trial.get("status") or ""
            lead_pcd = study.get("primary_completion_date") or ""

    # ---- Receptors: catalysts (upcoming trial readouts) ----
    catalysts: List[Dict[str, Any]] = []
    # PIT anchor: a snapshot's as-of date when provided, else today.
    today = _parse_iso_date(as_of) or datetime.now(timezone.utc).date()
    for trial in ticker_trials:
        nct = trial.get("nct_id", "")
        study = aact_studies.get(nct, {})
        pcd = study.get("primary_completion_date", "")
        status = study.get("overall_status", "")
        # A trial is a "receptor"/catalyst if it has an upcoming primary
        # completion date or is still actively recruiting.
        upcoming = False
        if pcd and len(pcd) >= 10:
            try:
                pcd_date = date.fromisoformat(pcd[:10])
                upcoming = pcd_date >= today
            except ValueError:
                upcoming = False
        if upcoming or status.lower() in {"recruiting", "active, not recruiting"}:
            catalysts.append(
                {
                    "nct_id": nct,
                    "phase": study.get("phase", "Unknown"),
                    "status": status,
                    "primary_completion_date": pcd,
                    "type": "readout",
                    "upcoming": upcoming,
                }
            )

    # ---- Signaling: trial connections (shared sponsors) ----
    sponsor_set: set = set()
    for trial in ticker_trials:
        nct = trial.get("nct_id", "")
        own_sponsor = trial.get("sponsor_name_at_map_time", "")
        for sp in aact_sponsors.get(nct, []):
            if own_sponsor and sp.lower() == own_sponsor.lower():
                continue
            sponsor_set.add(sp)

    flags: List[str] = sc.get("flags") or []

    # ---- Market cap tier for membrane ----
    cap_bucket = sc.get("market_cap_bucket") or "unknown"
    cap_tier = cap_bucket  # e.g. mega/large/mid/small/micro/unknown

    # ---- Compose the cell ----
    composite = sc.get("composite_score")
    name = uni.get("name") or ticker
    sector = uni.get("sector") or "Biotechnology"
    tissue = tissue_for(ticker, flags)

    cell = {
        # identity
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "tissue": tissue,
        # scores
        "composite_score": composite,
        "composite_rank": sc.get("composite_rank"),
        "clinical_dev_score": sc.get("clinical_dev_score"),
        "financial_score": sc.get("financial_score"),
        "catalyst_score": sc.get("catalyst_score"),
        # market / price
        "latest_price": price_info.get("price"),
        "latest_price_date": price_info.get("date"),
        "market_cap_bucket": cap_bucket,
        "stage_bucket": sc.get("stage_bucket") or "unknown",
        "severity": sc.get("severity") or "unknown",
        "flags": flags,
        "rankable": sc.get("rankable", False),
        # trial info
        "trial_count": len(ticker_trials),
        "nct_ids": [t.get("nct_id") for t in ticker_trials if t.get("nct_id")],
        # cell-atlas metaphor
        "membrane": {
            "cap_tier": cap_tier,
            "status": "active" if price_info.get("price") else "no_data",
            "stage": sc.get("stage_bucket") or "unknown",
            "severity": sc.get("severity") or "unknown",
        },
        "nucleus": {
            "lead_phase": lead_phase or "No active trial",
            "lead_nct": lead_nct,
            "lead_status": lead_status or "unknown",
            "lead_pcd": lead_pcd,
            "moa": sector,  # placeholder; no MoA column on disk
        },
        "organelles": {
            "financial_health": sc.get("financial_score"),
            "cash_runway": "unknown",
            "uncertainty_penalty": sc.get("uncertainty_penalty"),
            "missing_subfactor_pct": sc.get("missing_subfactor_pct"),
        },
        "receptors": {
            "catalyst_count": len(catalysts),
            "catalysts": catalysts,
        },
        "signaling": {
            "trial_connections": len(sponsor_set),
            "shared_sponsors": sorted(sponsor_set),
        },
    }
    return cell


# ---------------------------------------------------------------------------
# Top-level convenience: build all cells (cached per snapshot)
# ---------------------------------------------------------------------------

_cells_cache: Dict[str, List[Dict[str, Any]]] = {}


def build_all_cells(snapshot_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build cells for every ticker in the universe + snapshots.

    If ``snapshot_date`` is ``None`` the latest snapshot is used. The union of
    universe tickers and tickers present in the chosen snapshot is returned.
    """
    universe = load_universe()
    trials = load_trial_mapping()
    aact_studies = load_aact_studies()
    aact_sponsors = load_aact_sponsors()

    if snapshot_date is None:
        snapshot = load_latest_snapshot()
        snapshot_date = snapshot.get("as_of_date")
    else:
        snapshot = load_snapshot(snapshot_date)

    # Point-in-time anchor: prefer the snapshot's own as_of_date so prices and
    # "upcoming" catalysts align with the selected snapshot rather than today.
    as_of = snapshot.get("as_of_date") or snapshot_date
    prices = load_latest_prices(as_of)
    scores = snapshot_scores_for(snapshot)

    # Union of tickers: universe + any ticker in the snapshot scores.
    tickers = set(universe.keys()) | set(scores.keys())
    # Also include tickers that have trial mappings or prices.
    tickers |= set(trials.keys())

    cache_key = snapshot_date or "latest"
    if cache_key in _cells_cache:
        return _cells_cache[cache_key]

    cells = [
        build_cell(
            t, universe, prices, trials, scores, aact_studies, aact_sponsors,
            as_of=as_of,
        )
        for t in sorted(tickers)
    ]
    _cells_cache[cache_key] = cells
    return cells


def get_cell(ticker: str, snapshot_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return a single cell by ticker (case-insensitive), or ``None``."""
    ticker = ticker.upper()
    for cell in build_all_cells(snapshot_date):
        if cell["ticker"] == ticker:
            return cell
    return None


def clear_cache() -> None:
    """Invalidate the in-memory cell cache (used on reload)."""
    _cells_cache.clear()
