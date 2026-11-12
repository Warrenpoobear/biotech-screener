"""
sec_8k_catalyst_collector.py - SEC 8-K Timing Phrase Extractor

Searches SEC EDGAR for 8-K filings and extracts timing phrases related to
clinical data readouts and FDA milestones.

Two-phase approach:
  1. Search EDGAR full-text search-index for 8-K filings matching timing keywords
     → returns metadata (accession number, CIK, display_names, file_date) but NO text
  2. For filings matching our universe tickers, fetch the actual filing documents
     (main 8-K + press release exhibits) from SEC Archives to extract timing phrases

Source: SEC EDGAR (https://efts.sec.gov/LATEST/search-index + Archives)
Rate limiting: 10 req/sec per SEC EDGAR fair access guidelines
Cache: results cached per as_of_date
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# SEC EDGAR endpoints
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# SEC requires User-Agent header with contact info
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Wake Robin Research contact@wakerobincapital.com")

# Default cache directory — matches Module3Config's default path (relative to project root)
DEFAULT_CACHE_DIR = Path("cache") / "sec" / "8k_catalysts"

# Rate limit: 10 req/sec for SEC EDGAR
_SEC_MIN_INTERVAL = 0.1

# Hard cap on filings to fetch per run (safety against runaway costs)
_MAX_FILINGS_PER_RUN = 500

# Hard cap for multi-form (10-Q, 10-K, 6-K) collection
_MAX_FILINGS_PER_MULTI_FORM_RUN = 300

# Per-ticker cap for multi-form: keep only the N most recent filings per ticker
_MAX_FILINGS_PER_TICKER = 2

# Map filing form type to source label for events
FORM_TO_SOURCE = {
    "8-K": "SEC_8K_FILING",
    "10-Q": "SEC_10Q_FILING",
    "10-K": "SEC_10K_FILING",
    "6-K": "SEC_6K_FILING",
    "6-K/A": "SEC_6K_FILING",  # Amended 6-K → same source label
    "10-Q/A": "SEC_10Q_FILING",  # Amended 10-Q → same source label
    "10-K/A": "SEC_10K_FILING",  # Amended 10-K → same source label
}

# SEC official ticker-to-CIK mapping
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Timing phrase patterns: (regex, event_type, date_precision, meta)
# meta is None or a dict with optional fields:
#   "event_status": "extended" | "resubmission_accepted"  (review-window change marker)
#   "tags_extra":   list[str]  (appended to event["tags"])
TIMING_PATTERNS = [
    # "expects topline data in Q2 2026" / "anticipates top-line results in Q1 2026"
    (
        r"(?:expects?|anticipates?)\s+(?:topline|top-line|interim)\s+(?:data|results?)\s+.*?(Q[1-4]\s*\d{4})",
        "DATA_READOUT",
        "QUARTER",
        None,
    ),
    # "topline results in Q2 2026" (without expects/anticipates prefix)
    (
        r"(?:topline|top-line)\s+(?:data|results?)\s+(?:in|by|during)\s+.*?(Q[1-4]\s*\d{4})",
        "DATA_READOUT",
        "QUARTER",
        None,
    ),
    # ── Review-window CHANGE patterns (event_status="extended") ──
    # Must come BEFORE the generic "PDUFA date of ..." pattern so review-window
    # changes are tagged as extensions rather than as fresh upcoming PDUFA events.
    # "new PDUFA date of June 29, 2026" / "revised PDUFA date is ..." / "new PDUFA goal date of ..."
    (
        r"(?:new|revised)\s+PDUFA\s+(?:date|target\s+action\s+date|goal\s+date|action\s+date)\s+(?:of|is|set\s+for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "extended", "tags_extra": ["review_window_change", "extended"]},
    ),
    # "three-month extension ... June 29, 2026" / "3-month extension of the PDUFA date to ..."
    # Bound the gap to avoid grabbing unrelated dates downstream.
    (
        r"(?:three[- ]month|3[- ]month|six[- ]month|6[- ]month)\s+extension\b[^.]{0,200}?(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "extended", "tags_extra": ["review_window_change", "extended"]},
    ),
    # "review period has been extended ... new action date of ..." / "extended the review period ... June 29, 2026"
    (
        r"(?:review\s+period\s+(?:has\s+been\s+|was\s+)?extended|extended\s+the\s+review\s+period)\b[^.]{0,200}?(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "extended", "tags_extra": ["review_window_change", "extended"]},
    ),
    # Verb-first extension: "FDA extended the PDUFA target action date from March 29 to June 29, 2026"
    # / "FDA has extended the action date to August 22, 2026"
    (
        r"extended\s+the\s+(?:PDUFA\s+)?(?:target\s+)?(?:action\s+)?date\b[^.]{0,200}?(?:to|until)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "extended", "tags_extra": ["review_window_change", "extended"]},
    ),
    # "major amendment ... new PDUFA date of June 29, 2026" — strict to avoid contract/agreement matches
    (
        r"major\s+amendment\s+(?:to\s+(?:the\s+)?(?:application|NDA|BLA|sNDA|sBLA))?\s*.*?PDUFA\s+(?:date|action\s+date|goal\s+date)\s+(?:of|is|set\s+for)?\s*(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "extended", "tags_extra": ["review_window_change", "major_amendment"]},
    ),
    # ── Class 2 resubmission (event_status="resubmission_accepted") ──
    # "accepted as a Class 2 resubmission ... PDUFA date of ..." / "Class 2 resubmission with a six-month review period ... June 5, 2026"
    (
        r"Class\s*2\s+resubmission\b[^.]{0,200}?(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "resubmission_accepted", "tags_extra": ["class_2_resubmission"]},
    ),
    # "six-month review period ... PDUFA date of ..." (Class 2 resubmission statutory timeline).
    # Require explicit PDUFA/action-date wording near the captured date to avoid
    # matching boilerplate phrasing like "six-month review period from the date of resubmission".
    (
        r"six[- ]month\s+review\s+period\b[^.]{0,150}?(?:PDUFA|action\s+date|goal\s+date|target\s+action)\s+(?:date\s+)?(?:of|is|set\s+for)?\s*(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "resubmission_accepted", "tags_extra": ["six_month_review"]},
    ),
    # ── PDUFA extension / revision (existing language, tagged as extended) ──
    # "PDUFA date was extended to July 15, 2026" / "PDUFA date revised to ..."
    (
        r"PDUFA\s+(?:date|action date)\s+(?:was\s+)?(?:extended|revised|moved|pushed)\s+(?:to|until)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        {"event_status": "extended", "tags_extra": ["review_window_change", "extended"]},
    ),
    (
        r"PDUFA\s+(?:date|action date)\s+(?:was\s+)?(?:extended|revised)\s+.*?(Q[1-4]\s*\d{4})",
        "FDA_PDUFA_DATE",
        "QUARTER",
        {"event_status": "extended", "tags_extra": ["review_window_change", "extended"]},
    ),
    # ── PDUFA goal date / FDA goal date (DAY) ──
    (
        r"PDUFA\s+goal\s+date\s+(?:of|is|set\s+for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    (
        r"FDA\s+goal\s+date\s+(?:of|is|set\s+for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    # "target action date of ..." (without FDA prefix; relies on biopharma context guard upstream)
    (
        r"target\s+action\s+date\s+(?:of|is|set\s+for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    # "PDUFA date of March 15, 2026" or "PDUFA action date set for March 15, 2026"
    (
        r"PDUFA\s+(?:date|action date)\s+(?:of|is|set for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    # "expects to report topline results ... second half of 2026"
    # Gaps are bounded to [^.]{0,60} so the anchor and the year stay within the
    # same sentence — an unbounded .*? bridged across sentences and captured a
    # far-away/dateline year (e.g. KYMR "late 2027" text grabbing "…2026").
    (
        r"(?:expects?|anticipates?)\s+[^.]{0,60}?(?:results?|data)\s+[^.]{0,60}?(?:first|second)\s+half\s+(?:of\s+)?(\d{4})",
        "DATA_READOUT",
        "HALF_YEAR",
        None,
    ),
    # "data expected in mid-2026" / "results expected by year-end 2026"
    (
        r"(?:data|results?)\s+expected\s+[^.]{0,60}?(?:mid|year[- ]?end)[- ]?(\d{4})",
        "DATA_READOUT",
        "HALF_YEAR",
        None,
    ),
    # "data expected (to be reported) in late/early 2027" / "results ... end of 2027"
    # HALF_YEAR: "late"/"end of" → second half, "early" → first half (see below).
    (
        r"(?:data|results?)\s+(?:are\s+|is\s+)?(?:expected|anticipated)\s+[^.]{0,60}?\b(?:late|early|end[- ]of)\s+(\d{4})",
        "DATA_READOUT",
        "HALF_YEAR",
        None,
    ),
    # ── FDA action date variants (DAY precision) ──
    # "FDA action date of March 15, 2026" / "FDA target action date of ..."
    (
        r"FDA\s+(?:target\s+)?action\s+date\s+(?:of|is|set for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    # "NDA accepted...PDUFA date of April 3, 2026" / "BLA...PDUFA date of ..."
    (
        r"(?:NDA|BLA|sNDA|sBLA)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?PDUFA\s+date\s+(?:of|is)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    # "Prescription Drug User Fee Act date of March 15, 2026"
    (
        r"Prescription\s+Drug\s+User\s+Fee\s+Act\s+(?:date|goal date)\s+(?:of|is|set for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    # ── FDA decision in quarter (QUARTER precision) ──
    # "regulatory decision expected in Q2 2026" / "FDA decision expected in Q1 2026"
    (
        r"(?:regulatory|FDA)\s+(?:decision|review)\s+(?:is\s+)?(?:expected|anticipated)\s+.*?(Q[1-4]\s*\d{4})",
        "FDA_PDUFA_DATE",
        "QUARTER",
        None,
    ),
    # "FDA approval expected in Q2 2026" / "FDA approval anticipated in Q3 2026"
    (
        r"FDA\s+approval\s+(?:is\s+)?(?:expected|anticipated)\s+.*?(Q[1-4]\s*\d{4})",
        "FDA_PDUFA_DATE",
        "QUARTER",
        None,
    ),
    # ── FDA approval in half-year (HALF_YEAR precision) ──
    # "FDA approval anticipated second half of 2026" / "approval expected first half of 2026"
    (
        r"(?:FDA\s+)?approval\s+(?:is\s+)?(?:expected|anticipated)\s+.*?(?:first|second)\s+half\s+(?:of\s+)?(\d{4})",
        "FDA_PDUFA_DATE",
        "HALF_YEAR",
        None,
    ),
    # ── Advisory Committee / ADCOM (DAY and QUARTER) ──
    # "Advisory Committee meeting on July 17, 2026" / "advisory committee scheduled for ..."
    (
        r"Advisory\s+Committee\s+(?:meeting\s+)?(?:on|scheduled for|set for)\s+(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_ADCOM",
        "DAY",
        None,
    ),
    # "ADCOM meeting expected in Q3 2026" / "ADCOM scheduled for Q2 2026"
    (
        r"ADCOM\s+(?:meeting\s+)?(?:is\s+)?(?:expected|scheduled|anticipated)\s+.*?(Q[1-4]\s*\d{4})",
        "FDA_ADCOM",
        "QUARTER",
        None,
    ),
    # ── NDA/BLA submission and resubmission (QUARTER / DAY) ──
    # "NDA resubmission accepted" / "BLA resubmitted" / "sNDA filed"
    # Typed FDA_SUBMISSION (same REGULATORY family / T1 tier), NOT
    # FDA_PDUFA_DATE: a submission is not an FDA action date. Gaps are bounded
    # to one sentence; the unbounded .*? attached unrelated later dates at DAY
    # precision (STOK "NDA submission in H2 2027" -> PDUFA 2026-10-29, the
    # press-release date; DYN "BLA submission ... Q3 2027" -> 2026-06-30).
    (
        r"(?:NDA|BLA|sNDA|sBLA)\s+(?:re)?(?:submission|submitted|filed|accepted)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(Q[1-4]\s*\d{4})",
        "FDA_SUBMISSION",
        "QUARTER",
        None,
    ),
    (
        r"(?:NDA|BLA|sNDA|sBLA)\s+(?:re)?(?:submission|submitted|filed|accepted)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_SUBMISSION",
        "DAY",
        None,
    ),
    # ── FDA meetings (Type A/B/C, mid-cycle, late-cycle) ──
    # "Type A meeting scheduled for March 2026" / "late-cycle meeting"
    (
        r"Type\s+[ABC]\s+meeting\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(?:scheduled|set|planned)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    (
        r"(?:mid[- ]?cycle|late[- ]?cycle)\s+(?:review|meeting)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(\w+\s+\d{1,2},?\s+\d{4})",
        "FDA_PDUFA_DATE",
        "DAY",
        None,
    ),
    (
        r"(?:mid[- ]?cycle|late[- ]?cycle)\s+(?:review|meeting)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(Q[1-4]\s*\d{4})",
        "FDA_PDUFA_DATE",
        "QUARTER",
        None,
    ),
    # ── Breakthrough / Priority / Accelerated designations with timeline ──
    # "granted Breakthrough Therapy Designation ... expects approval Q2 2026"
    (
        r"(?:Breakthrough|Priority|Accelerated)\s+(?:Therapy\s+)?(?:Designation|Review)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(?:expects?|anticipates?)\s+(?:U\.S\.|Inc\.|Co\.|No\.|[^.]){0,120}?(Q[1-4]\s*\d{4})",
        "FDA_PDUFA_DATE",
        "QUARTER",
        None,
    ),
    # ── Phase 2 data readout (QUARTER) ──
    # "Phase 2 data expected in Q1 2026" / "Phase 2 results anticipated Q3 2026"
    (
        r"Phase\s+2\s+.*?(?:data|results?)\s+(?:expected|anticipated)\s+.*?(Q[1-4]\s*\d{4})",
        "DATA_READOUT",
        "QUARTER",
        None,
    ),
]

# Downside patterns: (regex, event_type, confidence)
# These detect negative catalyst language in 8-K filing text.
DOWNSIDE_PATTERNS = [
    # Complete Response Letter (CRL)
    (
        r"received\s+a\s+Complete\s+Response\s+Letter",
        "FDA_CRL",
        "HIGH",
    ),
    (
        r"CRL\s+from\s+(?:the\s+)?FDA",
        "FDA_CRL",
        "HIGH",
    ),
    # CRL variants
    (
        r"[Cc]omplete\s+[Rr]esponse\s+[Ll]etter\s+(?:was\s+)?received",
        "FDA_CRL",
        "HIGH",
    ),
    (
        r"FDA\s+issued\s+a\s+[Cc]omplete\s+[Rr]esponse\s+[Ll]etter",
        "FDA_CRL",
        "HIGH",
    ),
    # Clinical hold
    (
        r"FDA\s+(?:has\s+)?placed\s+.*?\s+on\s+clinical\s+hold",
        "CLINICAL_HOLD",
        "HIGH",
    ),
    (
        r"clinical\s+hold\s+(?:on|for)\s+",
        "CLINICAL_HOLD",
        "HIGH",
    ),
    # Safety signals
    (
        r"serious\s+adverse\s+event",
        "SAFETY_SIGNAL",
        "MED",
    ),
    (
        r"\bSAE\b",
        "SAFETY_SIGNAL",
        "MED",
    ),
    (
        r"patient\s+death",
        "SAFETY_SIGNAL",
        "HIGH",
    ),
    (
        r"DSMB\s+recommended\s+(?:halt|stop|discontinu)",
        "SAFETY_SIGNAL",
        "HIGH",
    ),
    # FDA Refuse to File
    (
        r"(?:refused?\s+to\s+file|Refuse\s+to\s+File)",
        "FDA_RTF",
        "HIGH",
    ),
    # FDA Warning Letter
    (
        r"FDA\s+warning\s+letter",
        "FDA_WARNING_LETTER",
        "MED",
    ),
]

# Pattern version: hash of all extraction patterns so cache auto-invalidates when patterns change.
_PATTERN_HASH_INPUT = repr(TIMING_PATTERNS) + repr(DOWNSIDE_PATTERNS)
PATTERN_VERSION = hashlib.sha256(_PATTERN_HASH_INPUT.encode()).hexdigest()[:8]

# ---------------------------------------------------------------------------
# Diagnostic counters — reset per collection run, harvested when alerts fire
# ---------------------------------------------------------------------------
_DIAG_COUNTERS: Dict[str, int] = {
    "raw_matches": 0,
    "boilerplate_dropped": 0,
    "biopharma_context_dropped": 0,
    "year_guard_dropped": 0,
    "staleness_dropped": 0,
    "accepted": 0,
}
_DIAG_DROPPED: List[Dict[str, Any]] = []


def _append_dropped(ticker: str, event_type: str, captured: str, reason: str, snippet: str) -> None:
    if len(_DIAG_DROPPED) < 10:
        _DIAG_DROPPED.append(
            {
                "ticker": ticker,
                "event_type": event_type,
                "captured": captured,
                "drop_reason": reason,
                "snippet": snippet[:100],
            }
        )


def reset_extraction_diagnostics() -> None:
    """Reset all diagnostic counters and dropped examples."""
    for k in _DIAG_COUNTERS:
        _DIAG_COUNTERS[k] = 0
    _DIAG_DROPPED.clear()


def get_extraction_diagnostics() -> Dict[str, Any]:
    """Return a snapshot of diagnostic counters and dropped examples."""
    return {"counters": dict(_DIAG_COUNTERS), "dropped_examples": list(_DIAG_DROPPED)}


def _versioned_cache_path(cache_dir: Path, as_of_date: date) -> Path:
    """Cache path that includes pattern version to invalidate on pattern changes."""
    return cache_dir / f"8k_catalysts_{as_of_date.isoformat()}_{PATTERN_VERSION}.json"


# Boilerplate markers: text after these is forward-looking disclaimer, not substance
_BOILERPLATE_MARKERS = [
    "forward-looking statements",
    "safe harbor",
    "Private Securities Litigation Reform Act",
]


def _strip_boilerplate(text: str) -> str:
    """
    Truncate filing text at the first forward-looking statements / safe harbor marker.

    This prevents matching boilerplate disclaimer language that often mentions
    risks, adverse events, etc. as generic risk factors rather than actual events.
    """
    text_lower = text.lower()
    earliest = len(text)
    for marker in _BOILERPLATE_MARKERS:
        idx = text_lower.find(marker.lower())
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def _extract_downside_events(
    text: str,
    ticker: str,
    filing_date: str,
    *,
    form: str = "8-K",
) -> List[Dict[str, Any]]:
    """
    Extract downside events from filing text.

    Applies boilerplate filtering, then matches DOWNSIDE_PATTERNS.
    Caps at 1 event per type per filing to avoid over-counting.

    Args:
        text: Filing text content
        ticker: Associated ticker symbol
        filing_date: ISO date string of the filing
        form: SEC form type ("8-K" or "6-K"). Drives event_name prefix and
            source label via FORM_TO_SOURCE.

    Returns:
        List of event dicts
    """
    clean_text = _strip_boilerplate(text)
    events = []
    seen_types: Set[str] = set()
    source_label = FORM_TO_SOURCE.get(form, "SEC_8K_FILING")

    for pattern, event_type, confidence in DOWNSIDE_PATTERNS:
        if event_type in seen_types:
            continue
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            seen_types.add(event_type)
            events.append(
                {
                    "ticker": ticker,
                    "event_type": event_type,
                    "event_date": filing_date,
                    "event_date_end": None,
                    "date_precision": "DAY",
                    "event_name": f"{form}: {match.group(0)[:80]}",
                    "confidence": confidence,
                    "source": source_label,
                    "disclosed_at": filing_date,
                    "tags": ["sec_8k", "downside"],
                }
            )

    return events


# Regex to extract ticker from EDGAR display_names.
# Handles both single-ticker and comma-separated formats:
#   "AGIOS PHARMACEUTICALS, INC.  (AGIO)  (CIK 0001439222)"
#   "Revolution Medicines, Inc.  (RVMD, RVMDW)  (CIK 0001628171)"
_TICKER_FROM_DISPLAY = re.compile(r"\(([A-Z]{2,5})(?:\)|[,\s])")


def _quarter_to_date_range(quarter_str: str) -> Tuple[str, str]:
    """
    Convert quarter string to date range.

    Args:
        quarter_str: e.g., "Q2 2026" or "Q2 2026"

    Returns:
        (start_date, end_date) as ISO strings
    """
    match = re.match(r"Q([1-4])\s*(\d{4})", quarter_str.strip())
    if not match:
        raise ValueError(f"Invalid quarter string: {quarter_str}")

    q = int(match.group(1))
    year = int(match.group(2))

    quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

    start_month, start_day = quarter_starts[q]
    end_month, end_day = quarter_ends[q]

    return (
        date(year, start_month, start_day).isoformat(),
        date(year, end_month, end_day).isoformat(),
    )


def _half_year_to_date_range(year_str: str, half: str = "first") -> Tuple[str, str]:
    """
    Convert half-year string to date range.

    Args:
        year_str: e.g., "2026"
        half: "first" or "second"

    Returns:
        (start_date, end_date) as ISO strings
    """
    year = int(year_str)
    if half == "first":
        return (
            date(year, 1, 1).isoformat(),
            date(year, 6, 30).isoformat(),
        )
    else:
        return (
            date(year, 7, 1).isoformat(),
            date(year, 12, 31).isoformat(),
        )


def _parse_exact_date(date_str: str) -> Optional[str]:
    """
    Parse an exact date string like "March 15, 2026" to ISO format.

    Returns:
        ISO date string or None if parse fails
    """
    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


_BOILERPLATE_KWS = (
    "foreseeable future",
    "going concern",
    "material weakness",
    "accounting standard",
    "fasb",
    "asu",
    "impairment",
    "adopt",
)
_BIOPHARMA_KWS = (
    "fda",
    "pdufa",
    "nda",
    "bla",
    "crl",
    "phase 1",
    "phase 2",
    "phase 3",
    "topline",
    "readout",
    "primary endpoint",
    "enrollment",
    "interim analysis",
    "clinical",
    "regulatory",
)

# Prior-date capture for review-window changes.
# Looks for "from {date} to ..." / "previously {date}" / "originally {date}"
# / "prior PDUFA date of {date}" in the window before the new-date match.
_PRIOR_DATE_RE = re.compile(
    r"(?:from|previously|originally|prior\s+(?:PDUFA\s+)?(?:date|action\s+date|goal\s+date)\s+(?:of|was))\s+"
    r"(\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


def _extract_prior_date(text: str, match_start: int, match_end: int) -> Optional[str]:
    """
    Scan the 200-char window around the match for an explicit prior PDUFA date.

    Returns ISO date string of the prior date if extractable, else None.
    Skips any candidate that is the same as the new (matched) date.
    """
    window_start = max(0, match_start - 200)
    window_end = min(len(text), match_end + 100)
    window = text[window_start:window_end]
    best: Optional[str] = None
    for m in _PRIOR_DATE_RE.finditer(window):
        iso = _parse_exact_date(m.group(1).strip())
        if iso:
            best = iso  # take the last (closest-to-match) plausible prior date
    return best


def _extract_timing_events(
    text: str,
    ticker: str,
    filing_date: str,
    as_of_date: Optional[date] = None,
    *,
    require_biopharma_context: bool = False,
    block_boilerplate: bool = False,
    form: str = "8-K",
) -> List[Dict[str, Any]]:
    """
    Extract timing events from filing text.

    Args:
        text: Filing text content
        ticker: Associated ticker symbol
        filing_date: ISO date string of the filing
        as_of_date: Current analysis date (used for year guard + staleness filter)
        require_biopharma_context: If True, only accept matches with biopharma
            keywords (FDA, Phase, topline, etc.) in the surrounding 300-char window.
        block_boilerplate: If True, reject matches where the surrounding context
            contains accounting/boilerplate language.
        form: SEC form type ("8-K" or "6-K"). Drives event_name prefix and
            source label via FORM_TO_SOURCE.

    Returns:
        List of event dicts
    """
    events = []
    source_label = FORM_TO_SOURCE.get(form, "SEC_8K_FILING")

    # Year guard: reject extracted dates outside [as_of_year - 1, as_of_year + 3]
    if as_of_date is not None:
        min_year = as_of_date.year - 1
        max_year = as_of_date.year + 3
    else:
        min_year = 0
        max_year = 9999

    # Staleness cutoff: reject events whose end date is >180d before as_of_date
    staleness_cutoff = (as_of_date - timedelta(days=180)).isoformat() if as_of_date else None

    for entry in TIMING_PATTERNS:
        # Tolerate legacy 3-tuples for forward-compat, but the in-tree shape is 4-tuple.
        if len(entry) == 4:
            pattern, event_type, precision, meta = entry
        else:
            pattern, event_type, precision = entry
            meta = None
        for match in re.finditer(pattern, text, re.IGNORECASE):
            _DIAG_COUNTERS["raw_matches"] += 1
            # Context-based relevance filters (±300 chars around match)
            ctx = text[max(0, match.start() - 300) : min(len(text), match.end() + 300)].lower()
            if block_boilerplate and any(k in ctx for k in _BOILERPLATE_KWS):
                _DIAG_COUNTERS["boilerplate_dropped"] += 1
                _append_dropped(ticker, event_type, match.group(1).strip(), "boilerplate", ctx[:100])
                continue
            if require_biopharma_context and not any(k in ctx for k in _BIOPHARMA_KWS):
                _DIAG_COUNTERS["biopharma_context_dropped"] += 1
                _append_dropped(ticker, event_type, match.group(1).strip(), "biopharma_context", ctx[:100])
                continue

            captured = match.group(1).strip()

            if precision == "DAY":
                event_date = _parse_exact_date(captured)
                if not event_date:
                    continue
                event_date_end = None
                confidence = "HIGH"

            elif precision == "QUARTER":
                try:
                    event_date, event_date_end = _quarter_to_date_range(captured)
                except ValueError:
                    continue
                confidence = "MED"

            elif precision == "HALF_YEAR":
                # Determine which half from surrounding context
                context_start = max(0, match.start() - 100)
                context = text[context_start : match.end()].lower()
                _second_markers = ("second", "mid", "year-end", "year end", "late", "end of", "end-of")
                half = "second" if any(m in context for m in _second_markers) else "first"
                try:
                    event_date, event_date_end = _half_year_to_date_range(captured, half)
                except (ValueError, TypeError):
                    continue
                confidence = "LOW"

            else:
                continue

            # Year guard: reject dates outside plausible range
            try:
                event_year = int(event_date[:4])
            except (ValueError, TypeError):
                continue
            if event_year < min_year or event_year > max_year:
                _DIAG_COUNTERS["year_guard_dropped"] += 1
                _append_dropped(
                    ticker, event_type, captured, "year_guard", f"{event_date} outside [{min_year},{max_year}]"
                )
                logger.debug(
                    f"Year guard: skipping {ticker} {event_type} {event_date} "
                    f"(year {event_year} outside [{min_year}, {max_year}])"
                )
                continue

            # Staleness filter: reject events whose end date is >180d before as_of_date
            end_for_staleness = event_date_end or event_date
            if staleness_cutoff and end_for_staleness < staleness_cutoff:
                _DIAG_COUNTERS["staleness_dropped"] += 1
                _append_dropped(ticker, event_type, captured, "staleness", f"{end_for_staleness} < {staleness_cutoff}")
                logger.debug(
                    f"Staleness filter: skipping {ticker} {event_type} {end_for_staleness} "
                    f"(older than {staleness_cutoff})"
                )
                continue

            _DIAG_COUNTERS["accepted"] += 1

            # event_status default: "upcoming" for fresh PDUFA matches; review-window
            # changes are tagged via meta["event_status"] from the pattern definition.
            if meta and meta.get("event_status"):
                event_status = meta["event_status"]
            elif event_type == "FDA_PDUFA_DATE":
                event_status = "upcoming"
            else:
                event_status = None

            tags = ["sec_8k"]
            if meta and meta.get("tags_extra"):
                tags.extend(meta["tags_extra"])

            # prior_date: only populate for review-window changes when an explicit
            # earlier date appears nearby. Never infer.
            prior_date: Optional[str] = None
            if meta and meta.get("event_status") == "extended" and precision == "DAY":
                candidate = _extract_prior_date(text, match.start(), match.end())
                if candidate and candidate != event_date:
                    prior_date = candidate

            event_dict: Dict[str, Any] = {
                "ticker": ticker,
                "event_type": event_type,
                "event_date": event_date,
                "event_date_end": event_date_end,
                "date_precision": precision,
                "event_name": f"{form}: {match.group(0)[:80]}",
                "confidence": confidence,
                "source": source_label,
                "disclosed_at": filing_date,
                "tags": tags,
            }
            if event_status is not None:
                event_dict["event_status"] = event_status
            if prior_date is not None:
                event_dict["prior_date"] = prior_date
            events.append(event_dict)

    return events


def _extract_ticker_from_display_names(display_names: List[str]) -> Optional[str]:
    """
    Extract ticker symbol from EDGAR display_names field.

    EDGAR display_names look like:
      "AGIOS PHARMACEUTICALS, INC.  (AGIO)  (CIK 0001439222)"

    Returns first non-CIK parenthesized match, or None.
    """
    for name in display_names:
        for m in _TICKER_FROM_DISPLAY.finditer(name):
            candidate = m.group(1)
            if candidate != "CIK":
                return candidate
    return None


def _rate_limit():
    """Sleep to respect SEC rate limit."""
    time.sleep(_SEC_MIN_INTERVAL)


def _sec_get(session: Any, url: str, **kwargs) -> Any:
    """GET with backoff on 403/429."""
    _rate_limit()
    resp = session.get(url, timeout=30, **kwargs)
    if resp.status_code in (403, 429):
        wait = int(resp.headers.get("Retry-After", 10))
        logger.warning(f"SEC {resp.status_code} on {url}, backing off {wait}s")
        time.sleep(wait)
        _rate_limit()
        resp = session.get(url, timeout=30, **kwargs)
    return resp


def _fetch_filing_text(
    cik: str,
    adsh: str,
    session: Any,
    *,
    prefer_exhibits_only: bool = False,
) -> str:
    """
    Fetch the actual text of an 8-K filing from SEC EDGAR Archives.

    Uses index.json (structured) to find filing documents instead of
    scraping the directory listing HTML.

    Args:
        cik: CIK number (with leading zeros)
        adsh: Accession number (e.g., "0001439222-26-000018")
        session: requests.Session with User-Agent set

    Returns:
        Concatenated plain text from the filing documents
    """
    adsh_nodash = adsh.replace("-", "")
    cik_stripped = cik.lstrip("0")
    base_url = f"{SEC_ARCHIVES_BASE}/{cik_stripped}/{adsh_nodash}"

    # Use index.json for structured file listing
    try:
        resp = _sec_get(session, f"{base_url}/index.json")
        if resp.status_code != 200:
            logger.debug(f"Filing index.json {resp.status_code}: {base_url}")
            return ""
        items = resp.json().get("directory", {}).get("item", [])
    except Exception as e:
        logger.debug(f"Filing index.json fetch error: {e}")
        return ""

    # Categorize .htm files from the index
    exhibit_files = []
    main_files = []
    for item in items:
        name = item.get("name", "")
        name_lower = name.lower()
        if not name_lower.endswith(".htm"):
            continue
        if "ex99" in name_lower or "ex-99" in name_lower or "pr" in name_lower:
            exhibit_files.append(name)
        elif "index" not in name_lower and "R1" not in name:
            main_files.append(name)

    # Fetch exhibits first (press releases have the timing language), then main doc
    all_text_parts = []
    docs_to_fetch = exhibit_files[:3] if prefer_exhibits_only else (exhibit_files[:3] + main_files[:2])
    _MAX_DOC_BYTES = 2_000_000  # Skip docs > 2 MB (large 10-Ks cause pathological parse)
    skipped_oversize = 0
    fetched_any = False

    for filename in docs_to_fetch:
        doc_url = f"{base_url}/{filename}"
        try:
            doc_resp = _sec_get(session, doc_url)
            if doc_resp.status_code == 200:
                if len(doc_resp.text) > _MAX_DOC_BYTES:
                    logger.debug(f"Skipping oversized doc ({len(doc_resp.text)} bytes): {doc_url}")
                    skipped_oversize += 1
                    continue
                fetched_any = True
                # Strip HTML tags to get plain text
                text = re.sub(r"<[^>]+>", " ", doc_resp.text)
                text = re.sub(r"&nbsp;", " ", text)
                text = re.sub(r"&#\d+;", " ", text)
                text = re.sub(r"&\w+;", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                all_text_parts.append(text)
        except Exception as e:
            logger.debug(f"Doc fetch error {doc_url}: {e}")

    if skipped_oversize:
        logger.info(f"Multi-form: skipped_oversize_docs={skipped_oversize}, fetched_any={fetched_any}")
    return " ".join(all_text_parts)


def collect_8k_timing_events(
    universe: List[Dict[str, Any]],
    as_of_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lookback_days: int = 180,
    forms: Tuple[str, ...] = ("8-K", "6-K"),
) -> List[Dict[str, Any]]:
    """
    Collect timing events from SEC current-report filings for universe tickers.

    Default forms = ("8-K", "6-K"): 8-K covers domestic filers; 6-K covers
    foreign private issuers (TEVA, AZN, BNTX, ARGX, ASND, BNTX, …) which
    otherwise yield zero hits in the active cache.

    Two-phase approach:
      Phase 1: Search EDGAR full-text index for filings of each form with
               timing keywords. Extract ticker from display_names, filter to
               our universe. Track form per accession.
      Phase 2: For matched filings, fetch actual filing text from EDGAR
               Archives and extract timing phrases with date ranges. Records
               are tagged with the originating form via source/event_name.

    Args:
        universe: List of universe entries (must have 'ticker' and optionally 'cik')
        as_of_date: Current analysis date
        cache_dir: Directory for caching results
        lookback_days: How far back to search for filings
        forms: Tuple of form types to query in EDGAR full-text search.

    Returns:
        List of event dicts matching corporate catalyst schema
    """
    reset_extraction_diagnostics()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _versioned_cache_path(cache_dir, as_of_date)

    # Check cache — primary dir first, then fallback to pipeline subdir
    _cache_to_load = cache_path
    if not cache_path.exists():
        _fallback_dir = Path("wake_robin_data_pipeline") / "cache" / "sec" / "8k_catalysts"
        _fallback_path = _versioned_cache_path(_fallback_dir, as_of_date)
        if _fallback_path.exists():
            logger.info(f"SEC 8-K: cache not in {cache_dir}, using fallback {_fallback_path}")
            _cache_to_load = _fallback_path

    if _cache_to_load.exists():
        try:
            with open(_cache_to_load, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(f"Loaded {len(cached)} SEC 8-K catalyst events from cache")
            return cached
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; skipping SEC 8-K collection")
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Build universe ticker set and bidirectional CIK mappings
    ticker_set: Set[str] = set()
    ticker_to_cik: Dict[str, str] = {}
    cik_to_ticker: Dict[str, str] = {}
    for entry in universe:
        ticker = entry.get("ticker", "")
        cik = entry.get("cik", "")
        if ticker:
            ticker_set.add(ticker.upper())
        if ticker and cik:
            cik_stripped = str(cik).lstrip("0")
            ticker_to_cik[ticker.upper()] = cik_stripped
            cik_to_ticker[cik_stripped] = ticker.upper()

    # Supplement with SEC's official company_tickers.json for reliable CIK mapping
    try:
        ct_resp = _sec_get(session, SEC_COMPANY_TICKERS_URL)
        if ct_resp.status_code == 200:
            for _k, entry in ct_resp.json().items():
                t = entry.get("ticker", "").upper()
                c = str(entry.get("cik_str", "")).lstrip("0")
                if t in ticker_set and c:
                    if t not in ticker_to_cik:
                        ticker_to_cik[t] = c
                    if c not in cik_to_ticker:
                        cik_to_ticker[c] = t
            logger.info(f"Loaded SEC company_tickers.json: {len(ticker_to_cik)} CIK mappings for universe")
    except Exception as e:
        logger.warning(f"Could not load company_tickers.json: {e}")

    start_date = (as_of_date - timedelta(days=lookback_days)).isoformat()
    end_date = as_of_date.isoformat()

    # ── Phase 1: Search EDGAR for 8-K filings with timing keywords ──
    # Each query targets a distinct class of catalyst timing language.
    # EDGAR search-index returns max 100 hits per request, so we paginate.
    search_queries = [
        '"topline data" OR "top-line data"',
        '"topline results" OR "top-line results"',
        '"PDUFA date" OR "PDUFA action date"',
        '"data readout" OR "interim data"',
        '"pivotal trial" AND ("first half" OR "second half")',
        '"Phase 3" AND ("expects" OR "anticipates") AND ("results" OR "data")',
        '"FDA action date" OR "FDA target action date"',
        '"Prescription Drug User Fee Act" AND "date"',
        '"FDA approval" AND ("expected" OR "anticipated")',
        '"regulatory decision" AND ("expected" OR "anticipated")',
        '"Advisory Committee" AND ("meeting" OR "scheduled")',
        '"ADCOM" AND ("meeting" OR "scheduled" OR "expected")',
        '"NDA" AND ("resubmission" OR "resubmitted" OR "accepted")',
        '"BLA" AND ("resubmission" OR "resubmitted" OR "accepted")',
        '"PDUFA" AND ("extended" OR "revised" OR "moved")',
        '"new PDUFA date" OR "revised PDUFA date" OR "PDUFA goal date"',
        '"major amendment" AND "PDUFA"',
        '"three-month extension" OR "3-month extension" OR "review period has been extended"',
        '"Class 2 resubmission" OR "six-month review period"',
        '"target action date"',
        '"mid-cycle review" OR "late-cycle meeting"',
        '"Type A meeting" OR "Type B meeting" OR "Type C meeting"',
        '"Phase 2" AND ("data" OR "results") AND ("expected" OR "anticipated")',
    ]

    # Max pages per query to avoid runaway requests
    _MAX_PAGES_PER_QUERY = 10  # 10 × 100 = 1000 hits max per query

    # Collect unique filings keyed by accession number
    # {adsh: {ticker, cik, file_date, form}}
    filings_to_fetch: Dict[str, Dict[str, str]] = {}

    for form in forms:
        for query in search_queries:
            try:
                offset = 0
                query_total = None

                while True:
                    params = {
                        "q": query,
                        "dateRange": "custom",
                        "startdt": start_date,
                        "enddt": end_date,
                        "forms": form,
                        "from": offset,
                    }

                    resp = _sec_get(session, SEC_SEARCH_URL, params=params)

                    if resp.status_code != 200:
                        logger.warning(f"SEC search returned {resp.status_code} for {form} '{query[:40]}'")
                        break

                    results = resp.json()
                    hits = results.get("hits", {}).get("hits", [])

                    if query_total is None:
                        query_total = results.get("hits", {}).get("total", {}).get("value", 0)
                        logger.info(f"EDGAR search [{form}] '{query[:40]}': {query_total} total hits")

                    if not hits:
                        break

                    for hit in hits:
                        source = hit.get("_source", {})
                        display_names = source.get("display_names", [])
                        file_date = source.get("file_date", "")
                        adsh = source.get("adsh", "")
                        ciks = source.get("ciks", [])

                        if not adsh or not file_date:
                            continue

                        # PIT safety: skip filings after as_of_date
                        if file_date > end_date:
                            continue

                        # Extract ticker: try display_names first, then CIK fallback
                        ticker = _extract_ticker_from_display_names(display_names)
                        ticker_upper = ticker.upper() if ticker else ""

                        if not ticker_upper or ticker_upper not in ticker_set:
                            # Fallback: match via CIK from search result
                            ticker_upper = ""
                            for cik_val in ciks:
                                cik_stripped = str(cik_val).lstrip("0")
                                if cik_stripped in cik_to_ticker:
                                    ticker_upper = cik_to_ticker[cik_stripped]
                                    break
                            if not ticker_upper:
                                continue

                        # Use our ticker→CIK mapping (more reliable than search-index ciks)
                        cik = ticker_to_cik.get(ticker_upper) or (ciks[0] if ciks else "")

                        # Deduplicate by accession number; first form to surface wins
                        # (8-K queried before 6-K, so domestic filings take precedence
                        # if the same accession appears under multiple form labels).
                        if adsh not in filings_to_fetch:
                            filings_to_fetch[adsh] = {
                                "ticker": ticker_upper,
                                "cik": cik,
                                "file_date": file_date,
                                "form": form,
                            }

                    offset += len(hits)
                    page = offset // 100
                    if offset >= query_total or page >= _MAX_PAGES_PER_QUERY:
                        break

            except Exception as e:
                logger.warning(f"SEC {form} search error for '{query[:40]}': {e}")

    logger.info(f"Phase 1: {len(filings_to_fetch)} unique filings from universe to fetch")

    # Hard cap: don't fetch more than _MAX_FILINGS_PER_RUN filings.
    # Sort by file_date descending (most recent first) so the cap keeps
    # the freshest filings rather than whichever query ran first.
    if len(filings_to_fetch) > _MAX_FILINGS_PER_RUN:
        logger.warning(f"Capping filing fetches at {_MAX_FILINGS_PER_RUN} " f"(found {len(filings_to_fetch)})")
        sorted_items = sorted(
            filings_to_fetch.items(),
            key=lambda x: x[1]["file_date"],
            reverse=True,
        )
        filings_to_fetch = dict(sorted_items[:_MAX_FILINGS_PER_RUN])

    # ── Phase 2: Fetch filing text and extract timing phrases ──
    all_events = []
    fetch_errors = 0

    for adsh, meta in filings_to_fetch.items():
        ticker = meta["ticker"]
        cik = meta["cik"]
        file_date = meta["file_date"]
        form = meta.get("form", "8-K")

        try:
            text = _fetch_filing_text(cik, adsh, session)
            if not text:
                fetch_errors += 1
                continue

            extracted = _extract_timing_events(text, ticker, file_date, as_of_date, form=form)
            downside = _extract_downside_events(text, ticker, file_date, form=form)
            total_extracted = len(extracted) + len(downside)
            if total_extracted:
                logger.info(
                    f"  {ticker} ({form} {adsh}, {file_date}): "
                    f"{len(extracted)} timing + {len(downside)} downside events"
                )
            all_events.extend(extracted)
            all_events.extend(downside)

        except Exception as e:
            logger.warning(f"Error processing {ticker} {form} {adsh}: {e}")
            fetch_errors += 1

    if fetch_errors:
        logger.info(f"Phase 2: {fetch_errors} filing fetch errors")

    # Deduplicate events by (ticker, event_type, event_date)
    seen = set()
    deduped = []
    for event in all_events:
        key = (event["ticker"], event["event_type"], event["event_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(event)

    logger.info(
        f"Collected {len(deduped)} SEC 8-K timing events " f"({len(all_events) - len(deduped)} duplicates removed)"
    )

    # Cache results
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

    return deduped


def _multi_form_cache_path(cache_dir: Path, as_of_date: date) -> Path:
    """Cache path for multi-form (10-Q, 10-K, 6-K) results."""
    return cache_dir / f"sec_filings_{as_of_date.isoformat()}_{PATTERN_VERSION}.json"


def _adsh_cache_dir(cache_dir: Path) -> Path:
    """Directory for per-filing parsed event caches."""
    return cache_dir / "multi_form_parsed"


def _adsh_cache_path(cache_dir: Path, adsh: str) -> Path:
    """Cache path for parsed events from a single filing (keyed by adsh + PATTERN_VERSION)."""
    safe_adsh = adsh.replace("/", "_").replace("\\", "_")
    return _adsh_cache_dir(cache_dir) / f"{safe_adsh}_{PATTERN_VERSION}.json"


def _load_adsh_cache(cache_dir: Path, adsh: str) -> Optional[List[Dict[str, Any]]]:
    """Load cached parsed events for a single filing. Returns None on miss."""
    path = _adsh_cache_path(cache_dir, adsh)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_adsh_cache(cache_dir: Path, adsh: str, events: List[Dict[str, Any]]) -> None:
    """Save parsed events for a single filing."""
    d = _adsh_cache_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = _adsh_cache_path(cache_dir, adsh)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
    except Exception as e:
        logger.warning(f"adsh cache write error for {adsh}: {e}")


def collect_sec_filing_events(
    universe: List[Dict[str, Any]],
    as_of_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lookback_days: int = 365,
    forms: Tuple[str, ...] = ("10-Q", "10-K", "6-K"),
) -> List[Dict[str, Any]]:
    """
    Collect timing events from SEC 10-Q, 10-K, and 6-K filings.

    Same extraction pipeline as 8-K but searches additional form types.
    Uses a separate cache to avoid invalidating existing 8-K caches.
    The collect_8k_timing_events() function is left unchanged.

    Args:
        universe: List of universe entries (must have 'ticker' and optionally 'cik')
        as_of_date: Current analysis date
        cache_dir: Directory for caching results
        lookback_days: How far back to search for filings
        forms: Tuple of SEC form types to search

    Returns:
        List of event dicts matching corporate catalyst schema
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _multi_form_cache_path(cache_dir, as_of_date)

    # Check cache — primary first, then fallback
    _cache_to_load = cache_path
    if not cache_path.exists():
        _fallback_dir = Path("wake_robin_data_pipeline") / "cache" / "sec" / "8k_catalysts"
        _fallback_path = _multi_form_cache_path(_fallback_dir, as_of_date)
        if _fallback_path.exists():
            logger.info(f"SEC multi-form: using fallback cache {_fallback_path}")
            _cache_to_load = _fallback_path

    if _cache_to_load.exists():
        try:
            with open(_cache_to_load, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(f"Loaded {len(cached)} SEC multi-form events from cache")
            return cached
        except Exception as e:
            logger.warning(f"Multi-form cache read error: {e}")

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; skipping SEC multi-form collection")
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Build universe ticker set and CIK mappings
    ticker_set: Set[str] = set()
    ticker_to_cik: Dict[str, str] = {}
    cik_to_ticker: Dict[str, str] = {}
    for entry in universe:
        ticker = entry.get("ticker", "")
        cik = entry.get("cik", "")
        if ticker:
            ticker_set.add(ticker.upper())
        if ticker and cik:
            cik_stripped = str(cik).lstrip("0")
            ticker_to_cik[ticker.upper()] = cik_stripped
            cik_to_ticker[cik_stripped] = ticker.upper()

    # Supplement with SEC's official company_tickers.json
    try:
        ct_resp = _sec_get(session, SEC_COMPANY_TICKERS_URL)
        if ct_resp.status_code == 200:
            for _k, entry in ct_resp.json().items():
                t = entry.get("ticker", "").upper()
                c = str(entry.get("cik_str", "")).lstrip("0")
                if t in ticker_set and c:
                    if t not in ticker_to_cik:
                        ticker_to_cik[t] = c
                    if c not in cik_to_ticker:
                        cik_to_ticker[c] = t
    except Exception as e:
        logger.warning(f"Could not load company_tickers.json: {e}")

    start_date = (as_of_date - timedelta(days=lookback_days)).isoformat()
    end_date = as_of_date.isoformat()

    # Comma-separated form types for EDGAR
    forms_param = ",".join(forms)

    # Reuse the same search queries as 8-K (timing language is form-agnostic)
    search_queries = [
        '"topline data" OR "top-line data"',
        '"topline results" OR "top-line results"',
        '"PDUFA date" OR "PDUFA action date"',
        '"data readout" OR "interim data"',
        '"pivotal trial" AND ("first half" OR "second half")',
        '"Phase 3" AND ("expects" OR "anticipates") AND ("results" OR "data")',
        '"FDA action date" OR "FDA target action date"',
        '"Prescription Drug User Fee Act" AND "date"',
        '"FDA approval" AND ("expected" OR "anticipated")',
        '"regulatory decision" AND ("expected" OR "anticipated")',
        '"Advisory Committee" AND ("meeting" OR "scheduled")',
        '"ADCOM" AND ("meeting" OR "scheduled" OR "expected")',
        '"NDA" AND ("resubmission" OR "resubmitted" OR "accepted")',
        '"BLA" AND ("resubmission" OR "resubmitted" OR "accepted")',
        '"PDUFA" AND ("extended" OR "revised" OR "moved")',
        '"new PDUFA date" OR "revised PDUFA date" OR "PDUFA goal date"',
        '"major amendment" AND "PDUFA"',
        '"three-month extension" OR "3-month extension" OR "review period has been extended"',
        '"Class 2 resubmission" OR "six-month review period"',
        '"target action date"',
        '"mid-cycle review" OR "late-cycle meeting"',
        '"Type A meeting" OR "Type B meeting" OR "Type C meeting"',
        '"Phase 2" AND ("data" OR "results") AND ("expected" OR "anticipated")',
    ]

    _MAX_PAGES_PER_QUERY = 5  # More conservative for multi-form

    filings_to_fetch: Dict[str, Dict[str, str]] = {}

    for query in search_queries:
        try:
            offset = 0
            query_total = None

            while True:
                params = {
                    "q": query,
                    "dateRange": "custom",
                    "startdt": start_date,
                    "enddt": end_date,
                    "forms": forms_param,
                    "from": offset,
                }

                resp = _sec_get(session, SEC_SEARCH_URL, params=params)
                if resp.status_code != 200:
                    break

                results = resp.json()
                hits = results.get("hits", {}).get("hits", [])

                if query_total is None:
                    query_total = results.get("hits", {}).get("total", {}).get("value", 0)
                    if query_total > 0:
                        logger.info(f"EDGAR multi-form '{query[:40]}': {query_total} total hits")

                if not hits:
                    break

                for hit in hits:
                    source = hit.get("_source", {})
                    display_names = source.get("display_names", [])
                    file_date = source.get("file_date", "")
                    adsh = source.get("adsh", "")
                    ciks = source.get("ciks", [])
                    form_type = source.get("form", "") or source.get("form_type", "")

                    if not adsh or not file_date or file_date > end_date:
                        continue

                    ticker = _extract_ticker_from_display_names(display_names)
                    ticker_upper = ticker.upper() if ticker else ""

                    if not ticker_upper or ticker_upper not in ticker_set:
                        ticker_upper = ""
                        for cik_val in ciks:
                            cik_stripped = str(cik_val).lstrip("0")
                            if cik_stripped in cik_to_ticker:
                                ticker_upper = cik_to_ticker[cik_stripped]
                                break
                        if not ticker_upper:
                            continue

                    cik = ticker_to_cik.get(ticker_upper) or (ciks[0] if ciks else "")

                    if adsh not in filings_to_fetch:
                        filings_to_fetch[adsh] = {
                            "ticker": ticker_upper,
                            "cik": cik,
                            "file_date": file_date,
                            "form_type": form_type,
                        }

                offset += len(hits)
                page = offset // 100
                if offset >= query_total or page >= _MAX_PAGES_PER_QUERY:
                    break

        except Exception as e:
            logger.warning(f"SEC multi-form search error for '{query[:40]}': {e}")

    logger.info(f"Multi-form phase 1: {len(filings_to_fetch)} unique filings to fetch")

    # ── Per-ticker cap: keep only _MAX_FILINGS_PER_TICKER most recent per ticker ──
    ticker_filings: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for adsh, meta in filings_to_fetch.items():
        t = meta["ticker"]
        if t not in ticker_filings:
            ticker_filings[t] = []
        ticker_filings[t].append((adsh, meta))

    # Sort each ticker's filings by date (most recent first), keep top N
    capped: Dict[str, Dict[str, str]] = {}
    for t, entries in ticker_filings.items():
        entries.sort(key=lambda x: x[1]["file_date"], reverse=True)
        for adsh, meta in entries[:_MAX_FILINGS_PER_TICKER]:
            capped[adsh] = meta

    pre_cap = len(filings_to_fetch)
    filings_to_fetch = capped
    if pre_cap != len(filings_to_fetch):
        logger.info(
            f"Per-ticker cap ({_MAX_FILINGS_PER_TICKER}/ticker): " f"{pre_cap} → {len(filings_to_fetch)} filings"
        )

    # Global backstop cap (after per-ticker cap)
    if len(filings_to_fetch) > _MAX_FILINGS_PER_MULTI_FORM_RUN:
        logger.warning(
            f"Capping multi-form fetches at {_MAX_FILINGS_PER_MULTI_FORM_RUN} " f"(found {len(filings_to_fetch)})"
        )
        sorted_items = sorted(
            filings_to_fetch.items(),
            key=lambda x: x[1]["file_date"],
            reverse=True,
        )
        filings_to_fetch = dict(sorted_items[:_MAX_FILINGS_PER_MULTI_FORM_RUN])

    # Phase 2: Fetch filing text and extract events (with adsh-level cache)
    all_events = []
    fetch_errors = 0
    cache_hits = 0
    network_fetches = 0
    total_filings = len(filings_to_fetch)

    for filing_idx, (adsh, meta) in enumerate(filings_to_fetch.items(), 1):
        ticker = meta["ticker"]
        cik = meta["cik"]
        file_date = meta["file_date"]
        form_type = meta.get("form_type", "")

        # Determine source label from form type
        source_label = FORM_TO_SOURCE.get(form_type, f"SEC_{form_type.replace('-', '')}_FILING")

        # Check adsh-level cache first
        cached_events = _load_adsh_cache(cache_dir, adsh)
        if cached_events is not None:
            cache_hits += 1
            all_events.extend(cached_events)
            continue

        if filing_idx % 10 == 0 or filing_idx == 1:
            logger.info(
                f"Multi-form [{filing_idx}/{total_filings}] fetching {ticker} " f"{form_type} {adsh} ({file_date})"
            )

        try:
            text = _fetch_filing_text(
                cik,
                adsh,
                session,
                prefer_exhibits_only=form_type in ("10-Q", "10-K"),
            )
            if not text:
                fetch_errors += 1
                _save_adsh_cache(cache_dir, adsh, [])  # Cache empty result
                continue

            network_fetches += 1
            extracted = _extract_timing_events(
                text,
                ticker,
                file_date,
                as_of_date,
                require_biopharma_context=True,
                block_boilerplate=True,
            )
            downside = _extract_downside_events(text, ticker, file_date)

            # Override source to match the filing form type
            filing_events = []
            for ev in extracted:
                ev["source"] = source_label
                ev["filing_form"] = form_type
                filing_events.append(ev)
            for ev in downside:
                ev["source"] = source_label
                ev["filing_form"] = form_type
                filing_events.append(ev)

            # Cache parsed events for this filing
            _save_adsh_cache(cache_dir, adsh, filing_events)

            if filing_events:
                logger.info(
                    f"  {ticker} ({form_type} {adsh}, {file_date}): "
                    f"{len(extracted)} timing + {len(downside)} downside events"
                )
            all_events.extend(filing_events)

        except Exception as e:
            logger.warning(f"Error processing {ticker} {form_type} {adsh}: {e}")
            fetch_errors += 1

    logger.info(
        f"Multi-form phase 2: {cache_hits} cache hits, " f"{network_fetches} network fetches, {fetch_errors} errors"
    )

    # Deduplicate events by (ticker, event_type, event_date)
    seen = set()
    deduped = []
    for event in all_events:
        key = (event["ticker"], event["event_type"], event["event_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(event)

    logger.info(
        f"Collected {len(deduped)} SEC multi-form timing events "
        f"({len(all_events) - len(deduped)} duplicates removed)"
    )

    # Cache results
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

    return deduped
