#!/usr/bin/env python3
"""
event_ledger.py - Unified PIT Event Ledger with Leakage Gates

Normalizes all catalyst cache sources (CTGov, SEC 8-K, SEC multi-form,
FDA ADCOM, FDA regulatory, PDUFA) into a single LedgerEntry schema
with deterministic event_id, strict PIT filtering, and window-aware
nearest-catalyst queries.

PIT Safety:
- Every entry requires disclosed_at (when the market could first know)
- Entries with disclosed_at > as_of_date are excluded
- PDUFA entries without curated_disclosed_at are tagged LOW confidence

Usage:
    from event_ledger import build_event_ledger, query_nearest_catalyst, LedgerConfig
    ledger = build_event_ledger(date(2026, 2, 17), LedgerConfig())
    nearest = query_nearest_catalyst(ledger, "ACAD", date(2026, 2, 17))
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Project root (where this script lives)
_PROJECT_ROOT = Path(__file__).parent


# =============================================================================
# LEDGER ENTRY
# =============================================================================

@dataclass(frozen=True)
class LedgerEntry:
    """Single normalized catalyst event with PIT anchor."""
    event_id: str
    ticker: str
    event_type: str
    event_date: Optional[str]       # ISO date (required for DAY precision)
    event_date_end: Optional[str]   # ISO date (for windowed: QUARTER, HALF_YEAR)
    date_precision: str             # DAY, WEEK, MONTH, QUARTER, HALF_YEAR, YEAR, UNKNOWN
    disclosed_at: str               # PIT anchor (required)
    source: str                     # CTGOV, SEC_8K, SEC_MULTI, FDA_ADCOM, FDA_FEDREG, PDUFA_MANUAL
    source_uid: str                 # nct_id, accession, meeting_id, etc.
    confidence: str                 # HIGH, MED, LOW
    extractor_version: str
    event_name: str
    field_changed: str
    value: str                      # raw value (the date/window as extracted)
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "event_type": self.event_type,
            "event_date": self.event_date,
            "event_date_end": self.event_date_end,
            "date_precision": self.date_precision,
            "disclosed_at": self.disclosed_at,
            "source": self.source,
            "source_uid": self.source_uid,
            "confidence": self.confidence,
            "extractor_version": self.extractor_version,
            "event_name": self.event_name,
            "field_changed": self.field_changed,
            "value": self.value,
            "tags": list(self.tags),
        }


def compute_event_id(
    source: str,
    source_uid: str,
    ticker: str,
    event_type: str,
    field_changed: str,
    value: str,
    event_date: Optional[str],
    event_date_end: Optional[str],
    disclosed_at: str,
    extractor_version: str,
) -> str:
    """Deterministic event ID from canonical fields (SHA-1, 16 hex chars)."""
    parts = "|".join([
        source, source_uid, ticker, event_type, field_changed,
        value, event_date or "", event_date_end or "",
        disclosed_at, extractor_version,
    ])
    return hashlib.sha1(parts.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LedgerConfig:
    """Paths and flags for building the event ledger."""
    ctgov_cache_dir: Optional[Path] = None
    sec_cache_dir: Optional[Path] = None
    fda_cache_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    strict_ctgov: bool = False

    def __post_init__(self):
        if self.ctgov_cache_dir is None:
            self.ctgov_cache_dir = _PROJECT_ROOT / "cache" / "ctgov"
        if self.sec_cache_dir is None:
            self.sec_cache_dir = _PROJECT_ROOT / "cache" / "sec" / "8k_catalysts"
        if self.fda_cache_dir is None:
            self.fda_cache_dir = _PROJECT_ROOT / "cache" / "fda"
        if self.data_dir is None:
            self.data_dir = _PROJECT_ROOT / "production_data"


# =============================================================================
# SOURCE LOADERS
# =============================================================================

def _load_ctgov_events(
    as_of_date: date,
    ctgov_cache_dir: Path,
    strict: bool,
    data_dir: Path,
) -> List[LedgerEntry]:
    """Load CTGov trial records and emit LedgerEntry per upcoming PCD/CD."""
    pit_path = ctgov_cache_dir / f"trial_records_{as_of_date.isoformat()}.json"

    if pit_path.exists():
        records = json.loads(pit_path.read_text())
    elif strict:
        raise FileNotFoundError(
            f"CTGov PIT cache missing: {pit_path}. "
            f"Run: python warm_caches.py --sources ctgov --as-of-date {as_of_date}"
        )
    else:
        fallback = data_dir / "trial_records.json"
        if not fallback.exists():
            logger.warning("CTGov: no trial_records.json found at %s", fallback)
            return []
        logger.warning(
            "CTGov PIT cache missing for %s, falling back to unfiltered %s",
            as_of_date, fallback,
        )
        records = json.loads(fallback.read_text())
        cutoff = as_of_date.isoformat()
        records = [r for r in records if (r.get("last_update_posted") or "")[:10] <= cutoff]

    entries: List[LedgerEntry] = []
    for rec in records:
        study_type = (rec.get("study_type") or "").upper()
        if "INTERVENTIONAL" not in study_type:
            continue

        phase = (rec.get("phase") or "").upper()
        if not any(p in phase for p in ("PHASE 1", "PHASE 2", "PHASE 3", "PHASE1", "PHASE2", "PHASE3")):
            continue

        ticker = rec.get("ticker", "")
        nct_id = rec.get("nct_id", "")
        disclosed_at = (rec.get("last_update_posted") or "")[:10]
        if not disclosed_at:
            continue

        # Primary completion date
        pcd = (rec.get("primary_completion_date") or "")[:10]
        if pcd:
            eid = compute_event_id(
                "CTGOV", nct_id, ticker, "CLINICAL_PCD",
                "primary_completion_date", pcd, pcd, None,
                disclosed_at, "ctgov_pit",
            )
            entries.append(LedgerEntry(
                event_id=eid, ticker=ticker, event_type="CLINICAL_PCD",
                event_date=pcd, event_date_end=None,
                date_precision="DAY", disclosed_at=disclosed_at,
                source="CTGOV", source_uid=nct_id, confidence="HIGH",
                extractor_version="ctgov_pit",
                event_name=f"{nct_id} PCD {pcd}",
                field_changed="primary_completion_date", value=pcd,
            ))

        # Completion date fallback
        cd = (rec.get("completion_date") or "")[:10]
        if cd and cd != pcd:
            eid = compute_event_id(
                "CTGOV", nct_id, ticker, "CLINICAL_CD",
                "completion_date", cd, cd, None,
                disclosed_at, "ctgov_pit",
            )
            entries.append(LedgerEntry(
                event_id=eid, ticker=ticker, event_type="CLINICAL_CD",
                event_date=cd, event_date_end=None,
                date_precision="DAY", disclosed_at=disclosed_at,
                source="CTGOV", source_uid=nct_id, confidence="HIGH",
                extractor_version="ctgov_pit",
                event_name=f"{nct_id} CD {cd}",
                field_changed="completion_date", value=cd,
            ))

    return entries


def _load_sec_8k_events(
    as_of_date: date,
    sec_cache_dir: Path,
) -> List[LedgerEntry]:
    """Load SEC 8-K cache events into ledger entries."""
    entries: List[LedgerEntry] = []

    # Try versioned cache path first, then plain
    import glob as glob_mod
    pattern = str(sec_cache_dir / f"8k_catalysts_{as_of_date.isoformat()}_*.json")
    matches = sorted(glob_mod.glob(pattern))
    if matches:
        cache_path = Path(matches[-1])  # latest version
    else:
        cache_path = sec_cache_dir / f"8k_catalysts_{as_of_date.isoformat()}.json"

    if not cache_path.exists():
        return entries

    try:
        events = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("SEC 8-K cache read error: %s", e)
        return entries

    # Extract pattern version from filename
    ext_version = "unknown"
    stem = cache_path.stem
    parts = stem.split("_")
    if len(parts) >= 4:
        ext_version = parts[-1]

    for ev in events:
        ticker = ev.get("ticker", "")
        event_type = ev.get("event_type", "UNKNOWN")
        event_date = ev.get("event_date")
        event_date_end = ev.get("event_date_end")
        date_precision = ev.get("date_precision", "DAY")
        disclosed_at = ev.get("disclosed_at", "")
        confidence = ev.get("confidence", "MED")
        event_name = ev.get("event_name", "")
        tags = tuple(ev.get("tags", []))

        if not ticker or not disclosed_at:
            continue

        source_uid = f"{ticker}_{event_type}_{disclosed_at}"
        eid = compute_event_id(
            "SEC_8K", source_uid, ticker, event_type,
            "event_date", event_date or "", event_date, event_date_end,
            disclosed_at, ext_version,
        )
        entries.append(LedgerEntry(
            event_id=eid, ticker=ticker, event_type=event_type,
            event_date=event_date, event_date_end=event_date_end,
            date_precision=date_precision, disclosed_at=disclosed_at,
            source="SEC_8K", source_uid=source_uid, confidence=confidence,
            extractor_version=ext_version,
            event_name=event_name, field_changed="event_date",
            value=event_date or "", tags=tags,
        ))

    return entries


def _load_sec_multi_form_events(
    as_of_date: date,
    sec_cache_dir: Path,
) -> List[LedgerEntry]:
    """Load SEC multi-form (10-Q, 10-K, 6-K) cache events."""
    entries: List[LedgerEntry] = []

    import glob as glob_mod
    pattern = str(sec_cache_dir / f"sec_filings_{as_of_date.isoformat()}_*.json")
    matches = sorted(glob_mod.glob(pattern))
    if matches:
        cache_path = Path(matches[-1])
    else:
        cache_path = sec_cache_dir / f"sec_filings_{as_of_date.isoformat()}.json"

    if not cache_path.exists():
        return entries

    try:
        events = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("SEC multi-form cache read error: %s", e)
        return entries

    ext_version = "unknown"
    stem = cache_path.stem
    parts = stem.split("_")
    if len(parts) >= 4:
        ext_version = parts[-1]

    for ev in events:
        ticker = ev.get("ticker", "")
        event_type = ev.get("event_type", "UNKNOWN")
        event_date = ev.get("event_date")
        event_date_end = ev.get("event_date_end")
        date_precision = ev.get("date_precision", "DAY")
        disclosed_at = ev.get("disclosed_at", "")
        confidence = ev.get("confidence", "MED")
        event_name = ev.get("event_name", "")
        tags = tuple(ev.get("tags", []))

        if not ticker or not disclosed_at:
            continue

        source_uid = f"{ticker}_{event_type}_{disclosed_at}"
        eid = compute_event_id(
            "SEC_MULTI", source_uid, ticker, event_type,
            "event_date", event_date or "", event_date, event_date_end,
            disclosed_at, ext_version,
        )
        entries.append(LedgerEntry(
            event_id=eid, ticker=ticker, event_type=event_type,
            event_date=event_date, event_date_end=event_date_end,
            date_precision=date_precision, disclosed_at=disclosed_at,
            source="SEC_MULTI", source_uid=source_uid, confidence=confidence,
            extractor_version=ext_version,
            event_name=event_name, field_changed="event_date",
            value=event_date or "", tags=tags,
        ))

    return entries


def _load_fda_adcom_events(
    as_of_date: date,
    fda_cache_dir: Path,
) -> List[LedgerEntry]:
    """Load FDA ADCOM calendar cache events."""
    entries: List[LedgerEntry] = []
    cache_path = fda_cache_dir / f"adcom_calendar_{as_of_date.isoformat()}.json"

    if not cache_path.exists():
        return entries

    try:
        events = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("FDA ADCOM cache read error: %s", e)
        return entries

    for ev in events:
        ticker = ev.get("ticker", "")
        event_date = ev.get("event_date", "")
        disclosed_at = ev.get("disclosed_at", "")
        event_name = ev.get("event_name", "")
        tags = tuple(ev.get("tags", []))

        if not ticker or not disclosed_at:
            continue

        source_uid = f"{ticker}_{event_date}"
        eid = compute_event_id(
            "FDA_ADCOM", source_uid, ticker, "FDA_ADCOM",
            "event_date", event_date, event_date, None,
            disclosed_at, "fda_adcom",
        )
        entries.append(LedgerEntry(
            event_id=eid, ticker=ticker, event_type="FDA_ADCOM",
            event_date=event_date, event_date_end=None,
            date_precision="DAY", disclosed_at=disclosed_at,
            source="FDA_ADCOM", source_uid=source_uid, confidence="HIGH",
            extractor_version="fda_adcom",
            event_name=event_name, field_changed="event_date",
            value=event_date, tags=tags,
        ))

    return entries


def _load_fda_regulatory_events(
    as_of_date: date,
    fda_cache_dir: Path,
) -> List[LedgerEntry]:
    """Load FDA regulatory notice cache events."""
    entries: List[LedgerEntry] = []
    cache_path = fda_cache_dir / f"fda_regulatory_{as_of_date.isoformat()}.json"

    if not cache_path.exists():
        return entries

    try:
        events = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("FDA regulatory cache read error: %s", e)
        return entries

    for ev in events:
        ticker = ev.get("ticker", "")
        event_type = ev.get("event_type", "UNKNOWN")
        event_date = ev.get("event_date", "")
        disclosed_at = ev.get("disclosed_at", "")
        event_name = ev.get("event_name", "")
        tags = tuple(ev.get("tags", []))

        if not ticker or not disclosed_at:
            continue

        source_uid = f"{ticker}_{event_type}_{event_date}"
        eid = compute_event_id(
            "FDA_FEDREG", source_uid, ticker, event_type,
            "event_date", event_date, event_date, None,
            disclosed_at, "fda_fedreg",
        )
        entries.append(LedgerEntry(
            event_id=eid, ticker=ticker, event_type=event_type,
            event_date=event_date, event_date_end=None,
            date_precision="DAY", disclosed_at=disclosed_at,
            source="FDA_FEDREG", source_uid=source_uid, confidence="HIGH",
            extractor_version="fda_fedreg",
            event_name=event_name, field_changed="event_date",
            value=event_date, tags=tags,
        ))

    return entries


def _load_pdufa_events(data_dir: Path) -> List[LedgerEntry]:
    """Load PDUFA dates from production data."""
    entries: List[LedgerEntry] = []
    pdufa_path = data_dir / "pdufa_dates.json"

    if not pdufa_path.exists():
        return entries

    try:
        records = json.loads(pdufa_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("PDUFA dates read error: %s", e)
        return entries

    for rec in records:
        ticker = rec.get("ticker", "")
        pdufa_date = rec.get("pdufa_date", "")
        drug_name = rec.get("drug_name", "")
        indication = rec.get("indication", "")

        if not ticker or not pdufa_date:
            continue

        curated_disclosed_at = rec.get("curated_disclosed_at")
        if curated_disclosed_at:
            disclosed_at = curated_disclosed_at
            confidence = "HIGH"
            tags: Tuple[str, ...] = ()
        else:
            # No curated disclosure date — conservative: LOW confidence + tagged
            disclosed_at = pdufa_date  # fallback to PDUFA date itself
            confidence = "LOW"
            tags = ("missing_disclosed_at",)

        source_uid = f"{ticker}_pdufa"
        event_name = f"PDUFA: {drug_name} ({indication})"
        eid = compute_event_id(
            "PDUFA_MANUAL", source_uid, ticker, "PDUFA",
            "pdufa_date", pdufa_date, pdufa_date, None,
            disclosed_at, "manual",
        )
        entries.append(LedgerEntry(
            event_id=eid, ticker=ticker, event_type="PDUFA",
            event_date=pdufa_date, event_date_end=None,
            date_precision="DAY", disclosed_at=disclosed_at,
            source="PDUFA_MANUAL", source_uid=source_uid,
            confidence=confidence, extractor_version="manual",
            event_name=event_name, field_changed="pdufa_date",
            value=pdufa_date, tags=tags,
        ))

    return entries


# =============================================================================
# BUILD LEDGER
# =============================================================================

def build_event_ledger(
    as_of_date: date,
    config: Optional[LedgerConfig] = None,
) -> List[LedgerEntry]:
    """
    Build unified event ledger from all cache sources.

    PIT filter: only entries with disclosed_at <= as_of_date are included.
    """
    if config is None:
        config = LedgerConfig()

    as_of_str = as_of_date.isoformat()
    all_entries: List[LedgerEntry] = []

    # 1. CTGov trials
    try:
        ctgov = _load_ctgov_events(as_of_date, config.ctgov_cache_dir, config.strict_ctgov, config.data_dir)
        all_entries.extend(ctgov)
        logger.info("Ledger: %d CTGov entries loaded", len(ctgov))
    except FileNotFoundError:
        raise  # strict mode — let it propagate
    except Exception as e:
        logger.warning("Ledger: CTGov load failed: %s", e)

    # 2. SEC 8-K
    try:
        sec_8k = _load_sec_8k_events(as_of_date, config.sec_cache_dir)
        all_entries.extend(sec_8k)
        logger.info("Ledger: %d SEC 8-K entries loaded", len(sec_8k))
    except Exception as e:
        logger.warning("Ledger: SEC 8-K load failed: %s", e)

    # 3. SEC multi-form
    try:
        sec_multi = _load_sec_multi_form_events(as_of_date, config.sec_cache_dir)
        all_entries.extend(sec_multi)
        logger.info("Ledger: %d SEC multi-form entries loaded", len(sec_multi))
    except Exception as e:
        logger.warning("Ledger: SEC multi-form load failed: %s", e)

    # 4. FDA ADCOM
    try:
        fda_adcom = _load_fda_adcom_events(as_of_date, config.fda_cache_dir)
        all_entries.extend(fda_adcom)
        logger.info("Ledger: %d FDA ADCOM entries loaded", len(fda_adcom))
    except Exception as e:
        logger.warning("Ledger: FDA ADCOM load failed: %s", e)

    # 5. FDA regulatory
    try:
        fda_reg = _load_fda_regulatory_events(as_of_date, config.fda_cache_dir)
        all_entries.extend(fda_reg)
        logger.info("Ledger: %d FDA regulatory entries loaded", len(fda_reg))
    except Exception as e:
        logger.warning("Ledger: FDA regulatory load failed: %s", e)

    # 6. PDUFA
    try:
        pdufa = _load_pdufa_events(config.data_dir)
        all_entries.extend(pdufa)
        logger.info("Ledger: %d PDUFA entries loaded", len(pdufa))
    except Exception as e:
        logger.warning("Ledger: PDUFA load failed: %s", e)

    # PIT filter: only include entries disclosed on or before as_of_date
    ledger = [e for e in all_entries if e.disclosed_at <= as_of_str]

    n_excluded = len(all_entries) - len(ledger)
    if n_excluded > 0:
        logger.info("Ledger PIT filter: excluded %d future-disclosed entries", n_excluded)

    logger.info("Ledger: %d total entries after PIT filter", len(ledger))
    return ledger


# =============================================================================
# QUERY: NEAREST CATALYST
# =============================================================================

def query_nearest_catalyst(
    ledger: List[LedgerEntry],
    ticker: str,
    as_of_date: date,
) -> Optional[Dict]:
    """
    Find the nearest upcoming catalyst for a ticker from the ledger.

    Window-aware: for QUARTER/HALF_YEAR events, uses event_date_end
    to determine if the window has passed.

    Returns None if no upcoming catalyst found, or dict with:
        days_to_catalyst, event_id, disclosed_at, source, source_uid,
        event_type, event_name, event_date
    """
    as_of_str = as_of_date.isoformat()
    candidates: List[Tuple[int, LedgerEntry]] = []

    for entry in ledger:
        if entry.ticker != ticker:
            continue
        if entry.confidence == "LOW":
            continue

        if entry.date_precision == "DAY":
            if not entry.event_date or entry.event_date <= as_of_str:
                continue
            days = (date.fromisoformat(entry.event_date) - as_of_date).days
        else:
            # QUARTER/MONTH/HALF_YEAR: use event_date_end
            end = entry.event_date_end or entry.event_date
            if not end or end < as_of_str:
                continue
            days = (date.fromisoformat(end) - as_of_date).days

        candidates.append((days, entry))

    if not candidates:
        return None

    nearest_days, nearest_entry = min(candidates, key=lambda x: x[0])
    return {
        "days_to_catalyst": nearest_days,
        "event_id": nearest_entry.event_id,
        "disclosed_at": nearest_entry.disclosed_at,
        "source": nearest_entry.source,
        "source_uid": nearest_entry.source_uid,
        "event_type": nearest_entry.event_type,
        "event_name": nearest_entry.event_name,
        "event_date": nearest_entry.event_date,
    }


# =============================================================================
# LEAKAGE AUDIT
# =============================================================================

def leakage_audit(
    ledger: List[LedgerEntry],
    as_of_date: date,
    all_raw_entries: Optional[List[LedgerEntry]] = None,
) -> Dict:
    """
    Audit the ledger for PIT leakage and coverage gaps.

    If all_raw_entries is provided (pre-PIT-filter), computes exclusion
    breakdown. Otherwise uses only the filtered ledger for partial stats.
    """
    as_of_str = as_of_date.isoformat()

    n_included = len(ledger)

    # Counts from the filtered ledger
    n_low_confidence = sum(1 for e in ledger if e.confidence == "LOW")
    n_pdufa_missing = sum(
        1 for e in ledger
        if e.source == "PDUFA_MANUAL" and "missing_disclosed_at" in e.tags
    )

    # Stale window: event_date_end (or event_date) < as_of_date
    n_stale_window = 0
    for e in ledger:
        end = e.event_date_end or e.event_date
        if end and end < as_of_str:
            n_stale_window += 1

    result: Dict = {
        "n_included": n_included,
        "n_excluded_low_confidence": n_low_confidence,
        "n_excluded_stale_window": n_stale_window,
        "n_pdufa_missing_disclosed_at": n_pdufa_missing,
    }

    if all_raw_entries is not None:
        n_total_raw = len(all_raw_entries)
        n_excluded_future = sum(1 for e in all_raw_entries if e.disclosed_at > as_of_str)
        excluded_by_source: Dict[str, int] = {}
        for e in all_raw_entries:
            if e.disclosed_at > as_of_str:
                excluded_by_source[e.source] = excluded_by_source.get(e.source, 0) + 1
        result["n_total_raw"] = n_total_raw
        result["n_excluded_future_disclosed"] = n_excluded_future
        result["excluded_by_source"] = excluded_by_source

    return result


# =============================================================================
# WRITE LEDGER
# =============================================================================

def write_ledger_jsonl(ledger: List[LedgerEntry], path: Path) -> None:
    """Write ledger entries as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in ledger:
            f.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    logger.info("Wrote %d ledger entries to %s", len(ledger), path)
