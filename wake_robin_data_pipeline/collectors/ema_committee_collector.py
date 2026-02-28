"""
ema_committee_collector.py - Collect EMA committee agendas + meeting highlights.

Two AdCom-equivalent catalyst streams:
  1) Committee events (agenda/annex-derived, forward-looking, MED confidence)
  2) Meeting outcomes (meeting highlights-derived, backward-looking, HIGH confidence)

Endpoints:
  - CHMP landing: /en/committees/committee-medicinal-products-human-use-chmp
  - PRAC landing: /en/committees/pharmacovigilance-risk-assessment-committee-prac
  - Meeting highlights pages: /en/news/meeting-highlights-...

Cache: cache/ema/ema_committee_events_{date}.json
       cache/ema/ema_meeting_outcomes_{date}.json

Rate limit: 1.0s between requests. Timeout: 30s.
"""

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RATE_LIMIT_SECS = 1.0
_REQUEST_TIMEOUT = 30
_COLLECTOR_VERSION = "ema_committee_collector.v1"
_MAX_UNMATCHED_LOG = 100

_BASE_URL = "https://www.ema.europa.eu"

_COMMITTEE_URLS = {
    "CHMP": f"{_BASE_URL}/en/committees/committee-medicinal-products-human-use-chmp",
    "PRAC": f"{_BASE_URL}/en/committees/pharmacovigilance-risk-assessment-committee-prac",
}

# Event page URL slugs — used to filter discovered links to committee event pages.
_COMMITTEE_EVENT_SLUGS = {
    "CHMP": "committee-medicinal-products-human-use-chmp",
    "PRAC": "pharmacovigilance-risk-assessment-committee-prac",
}

_MAX_XLSX_ROWS = 2000

# Meeting date range regex: "8 - 11 December 2025" or "8-11 December 2025"
# Also handles "10 November 2025" (single day) and "27 - 30 January 2026"
_DATE_RANGE_RE = re.compile(
    r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
_SINGLE_DATE_RE = re.compile(
    r"(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)

# Outcome section headers → outcome_code mapping
_OUTCOME_SECTION_MAP = {
    "recommended for approval": "positive_opinion",
    "positive opinion": "positive_opinion",
    "recommended for marketing authorisation": "positive_opinion",
    "new indication": "positive_opinion",
    "extension of indication": "positive_opinion",
    "negative opinion": "negative_opinion",
    "refused": "refusal",
    "refusal": "refusal",
    "withdrawn": "withdrawn",
    "withdrawal": "withdrawn",
    "referral": "referral_started",
    "referrals started": "referral_started",
    "referrals concluded": "referral_concluded",
    "safety signal": "safety_signal",
    "signal": "safety_signal",
}

# Procedure type keywords for agenda items
_PROCEDURE_TYPE_MAP = {
    "marketing authorisation application": "MAA",
    "maa": "MAA",
    "initial authorisation": "MAA",
    "variation": "VARIATION",
    "type ii variation": "VARIATION",
    "referral": "REFERRAL",
    "periodic safety update report": "PSUR",
    "psur": "PSUR",
    "signal": "SIGNAL",
    "safety signal": "SIGNAL",
}


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    """Fetch URL content with rate limiting. Returns HTML text."""
    time.sleep(_RATE_LIMIT_SECS)
    resp = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "Wake Robin Research (biotech screener)",
        },
    )
    resp.raise_for_status()
    return resp.text


def _fetch_xlsx_bytes(url: str) -> bytes:
    """Download binary XLSX content with rate limiting. Returns raw bytes."""
    time.sleep(_RATE_LIMIT_SECS)
    resp = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "User-Agent": "Wake Robin Research (biotech screener)",
        },
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_meeting_dates(text: str) -> Optional[Tuple[str, str]]:
    """Extract meeting date range from text like '8 - 11 December 2025'.

    Returns (start_date, end_date) as ISO strings, or None.
    """
    m = _DATE_RANGE_RE.search(text)
    if m:
        day_start, day_end, month_str, year = m.groups()
        try:
            end_dt = datetime.strptime(f"{day_end} {month_str} {year}", "%d %B %Y")
            start_dt = datetime.strptime(f"{day_start} {month_str} {year}", "%d %B %Y")
            return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = _SINGLE_DATE_RE.search(text)
    if m:
        day, month_str, year = m.groups()
        try:
            dt = datetime.strptime(f"{day} {month_str} {year}", "%d %B %Y")
            iso = dt.strftime("%Y-%m-%d")
            return iso, iso
        except ValueError:
            pass

    return None


def _normalize_name(s: str) -> str:
    """Normalize a medicine/substance name for matching."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# Salt forms, counter-ions, and common pharmaceutical suffixes.
_SALT_FORMS = frozenset({
    "hydrochloride", "hcl", "sodium", "potassium", "calcium",
    "succinate", "fumarate", "mesylate", "mesilate", "maleate",
    "tartrate", "acetate", "phosphate", "sulfate", "sulphate",
    "citrate", "besylate", "besilate", "tosylate", "tosylat",
    "dihydrochloride", "disodium", "tromethamine", "meglumine",
    "hemifumarate", "esylate", "edisylate", "xinafoate",
    "bromide", "chloride", "iodide", "nitrate", "carbonate",
    "gluconate", "lactate", "malate", "oxalate", "benzoate",
    "pamoate", "stearate", "valerate", "propionate",
    "dihydrate", "monohydrate", "trihydrate", "hemihydrate",
    "anhydrous",
})

# Dosage / formulation tokens to strip.
_FORMULATION_RE = re.compile(
    r"\b(?:"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|iu|mmol|mg/ml|microgram)"
    r"|tablets?|capsules?|injection|solution|suspension|concentrate"
    r"|powder|granules?|lyophilised|lyophilizate"
    r"|liposomal|pegylated|subcutaneous|intravenous"
    r"|film-coated|modified-release|prolonged-release|gastro-resistant"
    r")\b",
    re.IGNORECASE,
)

# Parenthetical qualifiers: "(liposomal)", "(50 mg)", etc.
_PARENS_RE = re.compile(r"\([^)]*\)")


def _strip_pharma_tokens(s: str) -> str:
    """Strip salt forms, dosage tokens, and formulation suffixes.

    Applied as a fallback normalization layer when base matching fails.
    """
    s = _PARENS_RE.sub(" ", s)
    s = _FORMULATION_RE.sub(" ", s)
    tokens = s.split()
    tokens = [t for t in tokens if t.lower() not in _SALT_FORMS]
    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match_medicine_to_ticker(
    medicine_name: str,
    active_substance: Optional[str],
    product_ticker_map: Dict[str, str],
) -> Optional[str]:
    """Match medicine name or active substance to a ticker.

    Matching order:
      1) exact match on normalized medicine_name
      2) exact match on normalized active_substance
      3) substring match on normalized medicine_name
      4) substring match with hyphens removed
      5) repeat 1-4 with salt/dosage tokens stripped (pharma normalization)
    """
    norm_med = _normalize_name(medicine_name) if medicine_name else ""
    norm_sub = _normalize_name(active_substance) if active_substance else ""

    # --- Pass 1: base normalization --------------------------------------

    # Exact matches first
    if norm_med and norm_med in product_ticker_map:
        return product_ticker_map[norm_med]
    if norm_sub and norm_sub in product_ticker_map:
        return product_ticker_map[norm_sub]

    # Substring matches
    for drug_name, ticker in product_ticker_map.items():
        if norm_med and drug_name in norm_med:
            return ticker
        drug_no_hyphen = drug_name.replace("-", "")
        if norm_med and drug_no_hyphen in norm_med.replace("-", ""):
            return ticker
    for drug_name, ticker in product_ticker_map.items():
        if norm_sub and drug_name in norm_sub:
            return ticker

    # --- Pass 2: pharma-stripped normalization ----------------------------
    # Strip both query and map keys — handles salt forms on either side.
    strip_med = _strip_pharma_tokens(norm_med) if norm_med else ""
    strip_sub = _strip_pharma_tokens(norm_sub) if norm_sub else ""

    if strip_med and strip_med in product_ticker_map:
        return product_ticker_map[strip_med]
    if strip_sub and strip_sub in product_ticker_map:
        return product_ticker_map[strip_sub]

    for drug_name, ticker in product_ticker_map.items():
        drug_stripped = _strip_pharma_tokens(drug_name)
        if strip_med and drug_stripped and drug_stripped in strip_med:
            return ticker
        if strip_med and drug_stripped and drug_stripped.replace("-", "") in strip_med.replace("-", ""):
            return ticker
    for drug_name, ticker in product_ticker_map.items():
        drug_stripped = _strip_pharma_tokens(drug_name)
        if strip_sub and drug_stripped and drug_stripped in strip_sub:
            return ticker

    return None


def _make_event_id(prefix: str, *parts: str) -> str:
    """Generate stable 10-char hex hash from key parts."""
    key = "|".join(str(p).lower() for p in parts)
    return prefix + "_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _classify_procedure_type(text: str) -> str:
    """Classify procedure type from agenda text."""
    lower = text.lower()
    for keyword, ptype in _PROCEDURE_TYPE_MAP.items():
        if keyword in lower:
            return ptype
    return "OTHER"


def _classify_outcome_code(section_heading: str) -> str:
    """Map a section heading to an outcome code."""
    lower = section_heading.lower().strip()
    for keyword, code in _OUTCOME_SECTION_MAP.items():
        if keyword in lower:
            return code
    return "other"


# Tag value for procedure type — lowercase, short.
_PROCEDURE_TAG_MAP = {
    "MAA": "maa",
    "VARIATION": "variation",
    "REFERRAL": "referral",
    "PSUR": "psur",
    "SIGNAL": "signal",
    "OTHER": "other",
}


def _make_tags(procedure_type: Optional[str], committee: str) -> List[str]:
    """Build a tags list for an agenda event.

    Tags emitted:
        procedure:maa | procedure:variation | procedure:referral | ...
        committee:chmp | committee:prac
    """
    tags: List[str] = []
    if procedure_type:
        tag_val = _PROCEDURE_TAG_MAP.get(procedure_type, procedure_type.lower())
        tags.append(f"procedure:{tag_val}")
    tags.append(f"committee:{committee.lower()}")
    return tags


# ---------------------------------------------------------------------------
# XLSX parsing (stdlib only — no openpyxl)
# ---------------------------------------------------------------------------


def _col_letter_to_index(col: str) -> int:
    """Convert Excel column letter(s) to 0-based index. 'A'->0, 'B'->1, ..."""
    idx = 0
    for ch in col.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _cell_ref_to_col_index(ref: str) -> int:
    """Extract column index from a cell reference like 'F25'. Returns 0-based."""
    col_letters = re.match(r"([A-Z]+)", ref)
    if not col_letters:
        return -1
    return _col_letter_to_index(col_letters.group(1))


def _parse_annex_xlsx(xlsx_bytes: bytes) -> List[dict]:
    """Parse an EMA Annex XLSX spreadsheet into product-level items.

    Uses stdlib ``zipfile`` + ``xml.etree.ElementTree`` — no openpyxl.

    Returns list of dicts with keys:
        medicine_name, active_substance, process_type, sponsor
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    except (zipfile.BadZipFile, Exception) as exc:
        logger.warning("XLSX parse: invalid zip — %s", exc)
        return []

    # 1. Build shared-string table ----------------------------------------
    shared_strings: List[str] = []
    try:
        ss_xml = zf.read("xl/sharedStrings.xml")
        ss_root = ET.fromstring(ss_xml)
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for si in ss_root.findall("s:si", ns):
            # <si> may contain <t> or multiple <r><t> elements
            parts = []
            for t_el in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"):
                parts.append(t_el.text or "")
            shared_strings.append("".join(parts))
    except (KeyError, ET.ParseError) as exc:
        logger.warning("XLSX parse: sharedStrings.xml read error — %s", exc)
        return []

    # 2. Determine data start row from table1.xml (if present) ------------
    data_start_row = None  # 1-based
    try:
        tbl_xml = zf.read("xl/tables/table1.xml")
        tbl_root = ET.fromstring(tbl_xml)
        ref = tbl_root.get("ref", "")  # e.g. "A24:O168"
        m = re.match(r"[A-Z]+(\d+):[A-Z]+(\d+)", ref)
        if m:
            header_row = int(m.group(1))
            data_start_row = header_row + 1
    except (KeyError, ET.ParseError):
        pass  # Fall back to header scan below

    # 3. Read sheet1.xml --------------------------------------------------
    try:
        ws_xml = zf.read("xl/worksheets/sheet1.xml")
        ws_root = ET.fromstring(ws_xml)
    except (KeyError, ET.ParseError) as exc:
        logger.warning("XLSX parse: sheet1.xml read error — %s", exc)
        return []

    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def _cell_value(cell_el):
        """Resolve cell value: shared string (t='s') or inline."""
        t_attr = cell_el.get("t", "")
        v_el = cell_el.find("s:v", ns)
        if v_el is None or v_el.text is None:
            return ""
        if t_attr == "s":
            idx = int(v_el.text)
            if 0 <= idx < len(shared_strings):
                return shared_strings[idx]
            return ""
        return v_el.text

    rows = ws_root.findall(".//s:sheetData/s:row", ns)

    # 4. Fallback: scan for header row containing "Invented name" ---------
    #    Column mapping: we need to find which column indices hold the fields.
    col_map = {}  # field_name -> 0-based column index
    _TARGET_HEADERS = {
        "invented name": "medicine_name",
        "active substance": "active_substance",
        "process type": "process_type",
        "customer": "sponsor",
        "customer/sponsor": "sponsor",
    }

    if data_start_row is None:
        # Scan rows for header
        for row_el in rows:
            cells = row_el.findall("s:c", ns)
            cell_texts = {}
            for c in cells:
                ref = c.get("r", "")
                col_idx = _cell_ref_to_col_index(ref)
                cell_texts[col_idx] = _cell_value(c).strip().lower()
            found = {ht for ht in _TARGET_HEADERS if ht in cell_texts.values()}
            if "invented name" in found:
                # This is the header row
                for col_idx, txt in cell_texts.items():
                    for header_key, field_name in _TARGET_HEADERS.items():
                        if header_key == txt:
                            col_map[field_name] = col_idx
                row_num = int(row_el.get("r", "0"))
                data_start_row = row_num + 1
                break

    if data_start_row is None:
        logger.warning("XLSX parse: could not determine data start row")
        return []

    # 5. If no col_map from header scan, also scan header row from table ref
    if not col_map:
        header_row_num = data_start_row - 1
        for row_el in rows:
            if int(row_el.get("r", "0")) == header_row_num:
                cells = row_el.findall("s:c", ns)
                for c in cells:
                    ref = c.get("r", "")
                    col_idx = _cell_ref_to_col_index(ref)
                    txt = _cell_value(c).strip().lower()
                    for header_key, field_name in _TARGET_HEADERS.items():
                        if header_key == txt:
                            col_map[field_name] = col_idx
                break

    # Default column positions matching known EMA Annex layout:
    # E=process_type(4), F=invented_name(5), G=active_substance(6), H=customer(7)
    if "medicine_name" not in col_map:
        col_map.setdefault("medicine_name", 5)      # F
        col_map.setdefault("active_substance", 6)    # G
        col_map.setdefault("process_type", 4)        # E
        col_map.setdefault("sponsor", 7)             # H

    # 6. Extract data rows ------------------------------------------------
    items: List[dict] = []
    for row_el in rows:
        row_num = int(row_el.get("r", "0"))
        if row_num < data_start_row:
            continue
        if len(items) >= _MAX_XLSX_ROWS:
            logger.warning("XLSX parse: truncated at %d rows", _MAX_XLSX_ROWS)
            break

        cells = row_el.findall("s:c", ns)
        cell_vals: Dict[int, str] = {}
        for c in cells:
            ref = c.get("r", "")
            col_idx = _cell_ref_to_col_index(ref)
            cell_vals[col_idx] = _cell_value(c)

        medicine_raw = cell_vals.get(col_map.get("medicine_name", -1), "").strip()
        if not medicine_raw:
            continue

        # Normalize multi-line: take first non-empty line
        medicine_name = medicine_raw
        if "\n" in medicine_raw:
            for line in medicine_raw.split("\n"):
                line = line.strip()
                if line:
                    medicine_name = line
                    break

        active_raw = cell_vals.get(col_map.get("active_substance", -1), "").strip()
        active_substance = active_raw
        if "\n" in active_raw:
            for line in active_raw.split("\n"):
                line = line.strip()
                if line:
                    active_substance = line
                    break

        items.append({
            "medicine_name": medicine_name,
            "active_substance": active_substance or None,
            "process_type": cell_vals.get(col_map.get("process_type", -1), "").strip() or None,
            "sponsor": cell_vals.get(col_map.get("sponsor", -1), "").strip() or None,
        })

    zf.close()
    return items


# ---------------------------------------------------------------------------
# Event page discovery
# ---------------------------------------------------------------------------


def _discover_meeting_event_pages(
    html: str, base_url: str, committee: str,
) -> List[dict]:
    """Extract event page links from a committee landing page.

    Looks for ``<a href="/en/events/...">`` links whose path contains the
    committee slug (e.g. "chmp" or "prac").

    Returns list of dicts: {url, meeting_start, meeting_end, committee}
    """
    slug = _COMMITTEE_EVENT_SLUGS.get(committee, committee.lower())
    soup = BeautifulSoup(html, "html.parser")
    results: List[dict] = []
    seen: set = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/en/events/" not in href:
            continue
        if slug not in href.lower():
            continue
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        if href in seen:
            continue
        seen.add(href)

        text = a_tag.get_text(strip=True)
        dates = _parse_meeting_dates(text) or _parse_meeting_dates(href)
        if not dates:
            continue

        results.append({
            "url": href,
            "meeting_start": dates[0],
            "meeting_end": dates[1],
            "committee": committee,
        })

    return results


def _discover_event_page_documents(
    html: str, base_url: str,
) -> Dict[str, List[dict]]:
    """Extract agenda PDF + annex XLSX links from an event page.

    Returns dict with keys ``agenda`` and ``annex``, each a list of
    ``{url, title, first_published}``.
    """
    soup = BeautifulSoup(html, "html.parser")
    agenda: List[dict] = []
    annex: List[dict] = []
    seen: set = set()

    def _extract_from_block(block, a_tag, href):
        block_text = block.get_text(" ", strip=True) if block else a_tag.get_text(strip=True)
        title = re.sub(r"\s*First published.*", "", block_text, flags=re.IGNORECASE).strip()
        if not title:
            title = block_text[:200]
        first_published = None
        fp = _FIRST_PUBLISHED_RE.search(block_text)
        if fp:
            d, mo, y = fp.groups()
            try:
                first_published = datetime.strptime(f"{d}/{mo}/{y}", "%d/%m/%Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
        return {"url": href, "title": title, "first_published": first_published}

    # Strategy 1: div.bcl-file blocks
    for block in soup.select("div.bcl-file"):
        for a_tag in block.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("/"):
                href = base_url.rstrip("/") + href
            if href in seen:
                continue
            href_lower = href.lower()
            if "/documents/agenda/" in href_lower or "/documents/annex/" in href_lower or "/agenda/" in href_lower or "/annex/" in href_lower:
                seen.add(href)
                doc = _extract_from_block(block, a_tag, href)
                if href_lower.endswith(".xlsx") or "annex" in href_lower:
                    annex.append(doc)
                else:
                    agenda.append(doc)

    # Strategy 2: plain anchor scan fallback
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        if href in seen:
            continue
        href_lower = href.lower()
        if "/documents/agenda/" in href_lower or "/documents/annex/" in href_lower or "/agenda/" in href_lower or "/annex/" in href_lower:
            seen.add(href)
            doc = _extract_from_block(None, a_tag, href)
            if href_lower.endswith(".xlsx") or "annex" in href_lower:
                annex.append(doc)
            else:
                agenda.append(doc)

    return {"agenda": agenda, "annex": annex}


# ---------------------------------------------------------------------------
# Discovery: scrape committee landing page for links (legacy fallback)
# ---------------------------------------------------------------------------


_FIRST_PUBLISHED_RE = re.compile(
    r"[Ff]irst\s+published[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})"
)


def _discover_links(
    html: str, base_url: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Discover agenda docs and meeting highlights links from a committee page.

    EMA committee pages wrap each document in a ``div.bcl-file`` container.
    The anchor text is typically just "View"; the descriptive title and
    "First published" date live in the surrounding ``div.bcl-file`` block.

    Fallback: if no ``div.bcl-file`` wrapper is found (e.g. simpler page
    layouts) the function falls back to matching "agenda" in anchor text.

    Returns dict with keys:
      - agenda_docs: list of {url, title, disclosed_at}
      - highlights: list of {url, title}
    """
    soup = BeautifulSoup(html, "html.parser")

    agenda_docs: List[Dict[str, str]] = []
    highlights: List[Dict[str, str]] = []
    seen_urls: set = set()

    # --- Strategy 1: div.bcl-file containers (current EMA layout) ----------
    for block in soup.select("div.bcl-file"):
        block_text = block.get_text(" ", strip=True)
        for a_tag in block.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("/"):
                href = base_url.rstrip("/") + href

            href_lower = href.lower()

            # Agenda / annex documents — detected via href path segment
            if "/agenda/" in href_lower or "/annex/" in href_lower:
                if href in seen_urls:
                    continue
                seen_urls.add(href)
                # Title from block text, stripping trailing "First published…View"
                title = re.sub(r"\s*First published.*", "", block_text, flags=re.IGNORECASE).strip()
                if not title:
                    title = block_text[:200]
                # Extract disclosed_at from "First published:DD/MM/YYYY"
                disclosed_at = None
                fp = _FIRST_PUBLISHED_RE.search(block_text)
                if fp:
                    d, mo, y = fp.groups()
                    try:
                        disclosed_at = datetime.strptime(f"{d}/{mo}/{y}", "%d/%m/%Y").strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                agenda_docs.append({"url": href, "title": title, "disclosed_at": disclosed_at})

            # Meeting highlights — detected via href path
            if "/en/news/meeting-highlights" in href_lower:
                if href in seen_urls:
                    continue
                seen_urls.add(href)
                title = re.sub(r"\s*First published.*", "", block_text, flags=re.IGNORECASE).strip()
                if not title:
                    title = block_text[:200]
                highlights.append({"url": href, "title": title})

    # --- Strategy 2: plain anchor-text fallback (legacy / simple HTML) -----
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        text = a_tag.get_text(strip=True)
        if not text:
            continue
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        if href in seen_urls:
            continue

        text_lower = text.lower()
        href_lower = href.lower()

        # Agenda by text content
        if "agenda" in text_lower and ("meeting" in text_lower or "annex" in text_lower):
            seen_urls.add(href)
            agenda_docs.append({"url": href, "title": text, "disclosed_at": None})

        # Meeting highlights by text or href
        if "/en/news/meeting-highlights" in href_lower or "meeting highlights" in text_lower:
            if href not in seen_urls:
                seen_urls.add(href)
                highlights.append({"url": href, "title": text})

    return {"agenda_docs": agenda_docs, "highlights": highlights}


# ---------------------------------------------------------------------------
# Parsing: agenda documents (HTML link-based, minimal)
# ---------------------------------------------------------------------------


def _parse_agenda_page(
    html: str,
    url: str,
    committee: str,
    meeting_start: str,
    meeting_end: str,
    product_ticker_map: Dict[str, str],
) -> Tuple[List[dict], List[str]]:
    """Parse an agenda/annex page for medicine items.

    Returns (events, unmatched) where unmatched is list of medicine names
    that couldn't be mapped to tickers.
    """
    soup = BeautifulSoup(html, "html.parser")
    events = []
    unmatched = []

    # Extract text content and look for medicine-like items in lists
    # Agenda pages typically have structured lists or tables
    items = []

    # Try list items first
    for li in soup.find_all("li"):
        text = li.get_text(strip=True)
        if len(text) > 10 and len(text) < 500:
            items.append(text)

    # Also try table rows
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            items.append(" | ".join(cells))

    # Also try paragraphs with capitalized drug-like names
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 10 and len(text) < 500:
            items.append(text)

    seen_ids = set()
    for item_text in items:
        # Try to extract a medicine name — look for capitalized words
        # that match the product map
        procedure_type = _classify_procedure_type(item_text)

        # Try matching the whole text against the product map
        ticker = _match_medicine_to_ticker(item_text, None, product_ticker_map)
        medicine_name = item_text[:100]  # truncate for display

        if ticker is None:
            # Try extracting just capitalized words as potential drug names
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", item_text)
            for word in words:
                ticker = _match_medicine_to_ticker(word, None, product_ticker_map)
                if ticker:
                    medicine_name = word
                    break

        if ticker is None:
            if len(unmatched) < _MAX_UNMATCHED_LOG:
                unmatched.append(item_text[:200])
            continue

        event_id = _make_event_id(
            f"EMA_{committee}_AGENDA_{meeting_end}",
            medicine_name, procedure_type, committee, meeting_start, meeting_end,
        )
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)

        events.append({
            "schema": "ema_committee_event.v1",
            "id": event_id,
            "ticker": ticker,
            "jurisdiction": "EU",
            "committee": committee,
            "meeting_start": meeting_start,
            "meeting_end": meeting_end,
            "event_date": meeting_end,
            "disclosed_at": None,
            "item_type": "agenda_item",
            "procedure_type": procedure_type,
            "medicine_name": medicine_name,
            "active_substance": None,
            "sponsor": None,
            "title": f"{committee} agenda: {procedure_type} — {medicine_name}",
            "source": f"EMA_{committee}_AGENDA",
            "url": url,
            "confidence": "MED",
            "tags": _make_tags(procedure_type, committee),
            "raw": {"snippet": item_text[:300]},
        })

    return events, unmatched


# ---------------------------------------------------------------------------
# Parsing: meeting highlights (outcomes)
# ---------------------------------------------------------------------------


def _parse_meeting_highlights(
    html: str,
    url: str,
    committee: str,
    product_ticker_map: Dict[str, str],
) -> Tuple[List[dict], List[str]]:
    """Parse a meeting highlights page for outcome records.

    Returns (outcomes, unmatched).
    """
    soup = BeautifulSoup(html, "html.parser")
    outcomes = []
    unmatched = []

    # Extract meeting dates from the page title or first heading
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    title_text = ""
    if h1_tag:
        title_text = h1_tag.get_text(strip=True)
    elif title_tag:
        title_text = title_tag.get_text(strip=True)

    dates = _parse_meeting_dates(title_text)
    if dates is None:
        # Try from URL
        dates = _parse_meeting_dates(url)
    meeting_start = dates[0] if dates else ""
    meeting_end = dates[1] if dates else ""

    # Extract "First published" date as disclosed_at
    disclosed_at = None
    for meta in soup.find_all("meta"):
        if meta.get("name", "").lower() in ("dcterms.issued", "dc.date", "article:published_time"):
            raw = meta.get("content", "")
            if raw:
                disclosed_at = raw[:10]
    if not disclosed_at:
        # Look for text like "First published: DD/MM/YYYY"
        for text_node in soup.find_all(string=re.compile(r"first published", re.IGNORECASE)):
            parent_text = text_node.parent.get_text(strip=True) if text_node.parent else str(text_node)
            date_match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", parent_text)
            if date_match:
                d, mo, y = date_match.groups()
                try:
                    disclosed_at = datetime.strptime(f"{d}/{mo}/{y}", "%d/%m/%Y").strftime("%Y-%m-%d")
                except ValueError:
                    pass
    if not disclosed_at:
        disclosed_at = meeting_end or None

    # Walk headings to find outcome sections.
    # Only emit outcomes under decision-grade sections — drop "other",
    # PSUR, administrative, etc. at the collector to keep signal clean.
    _SIGNAL_CODES = frozenset({
        "positive_opinion", "negative_opinion", "refusal",
        "withdrawn", "referral_started", "referral_concluded", "safety_signal",
    })
    current_outcome_code = None
    seen_ids = set()

    for element in soup.find_all(["h2", "h3", "h4", "li", "p"]):
        tag_name = element.name
        text = element.get_text(strip=True)

        if tag_name in ("h2", "h3", "h4"):
            current_outcome_code = _classify_outcome_code(text)
            continue

        if current_outcome_code is None or current_outcome_code not in _SIGNAL_CODES:
            continue
        if not text or len(text) < 3:
            continue

        # Try to match text as a medicine name
        ticker = _match_medicine_to_ticker(text, None, product_ticker_map)
        medicine_name = text

        if ticker is None:
            # Try extracting capitalized words
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            for word in words:
                ticker = _match_medicine_to_ticker(word, None, product_ticker_map)
                if ticker:
                    medicine_name = word
                    break

        if ticker is None:
            if len(unmatched) < _MAX_UNMATCHED_LOG:
                unmatched.append(text[:200])
            continue

        event_id = _make_event_id(
            f"EMA_{committee}_OUTCOME_{meeting_end}_{current_outcome_code}",
            medicine_name, committee, meeting_start, meeting_end,
        )
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)

        outcomes.append({
            "schema": "ema_meeting_outcome.v1",
            "id": event_id,
            "ticker": ticker,
            "jurisdiction": "EU",
            "committee": committee,
            "meeting_start": meeting_start,
            "meeting_end": meeting_end,
            "event_date": meeting_end,
            "disclosed_at": disclosed_at,
            "outcome_code": current_outcome_code,
            "medicine_name": medicine_name,
            "active_substance": None,
            "indication": None,
            "title": f"{committee} meeting highlights: {current_outcome_code} — {medicine_name}",
            "source": "EMA_MEETING_HIGHLIGHTS",
            "url": url,
            "confidence": "HIGH",
            "raw": {"section": current_outcome_code, "snippet": text[:300]},
        })

    return outcomes, unmatched


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def collect_ema_committee_events(
    *,
    as_of_date: date,
    cache_dir: Path,
    product_ticker_map: Dict[str, str],
    committees: Tuple[str, ...] = ("CHMP",),
    lookback_days: int = 365,
) -> List[dict]:
    """Collect EMA committee agenda events.

    Primary path: discover meeting event pages → download Annex XLSX →
    parse product-level items → match to tickers.

    Fallback: if no event pages found, falls back to the landing-page
    document discovery (creates meeting-level placeholder events).

    Args:
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for EMA cache files.
        product_ticker_map: Dict mapping lowercase drug names to ticker symbols.
        committees: Tuple of committee codes to scrape.
        lookback_days: Only include meetings within this window.

    Returns:
        List of event dicts (ema_committee_event.v1 schema).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ema_committee_events_{as_of_date.isoformat()}.json"

    # Short-circuit if cache exists
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            events = payload.get("events", payload) if isinstance(payload, dict) else payload
            logger.info("EMA committee events cache hit: %s (%d events)", cache_path.name, len(events))
            return events
        except (json.JSONDecodeError, OSError):
            pass

    cutoff = (as_of_date - timedelta(days=lookback_days)).isoformat()
    as_of_iso = as_of_date.isoformat()
    all_events: List[dict] = []
    all_unmatched: List[str] = []
    stats = {
        "fetched_pages": 0, "parsed_docs": 0, "matched_tickers": 0,
        "unmatched_items": 0, "meeting_pages_found": 0, "docs_found": 0,
        "items_parsed": 0, "matched": 0, "unmatched": 0,
    }

    for committee in committees:
        landing_url = _COMMITTEE_URLS.get(committee)
        if not landing_url:
            logger.warning("Unknown EMA committee: %s", committee)
            continue

        try:
            landing_html = _fetch(landing_url)
            stats["fetched_pages"] += 1
        except Exception as e:
            logger.warning("EMA %s landing page fetch failed: %s", committee, e)
            continue

        # --- Primary path: event page discovery --------------------------
        meeting_pages = _discover_meeting_event_pages(landing_html, _BASE_URL, committee)
        stats["meeting_pages_found"] += len(meeting_pages)

        used_event_pages = False
        for mp in meeting_pages:
            meeting_start = mp["meeting_start"]
            meeting_end = mp["meeting_end"]

            # Lookback + PIT filter
            if meeting_end < cutoff or meeting_end > as_of_iso:
                continue

            try:
                event_html = _fetch(mp["url"])
                stats["fetched_pages"] += 1
            except Exception as e:
                logger.warning("EMA event page fetch failed for %s: %s", mp["url"], e)
                continue

            docs = _discover_event_page_documents(event_html, _BASE_URL)
            stats["docs_found"] += len(docs.get("agenda", [])) + len(docs.get("annex", []))

            # Try annex XLSX first — product-level data
            annex_parsed = False
            for annex_doc in docs.get("annex", []):
                annex_url = annex_doc["url"]
                if not annex_url.lower().endswith(".xlsx"):
                    continue
                try:
                    xlsx_bytes = _fetch_xlsx_bytes(annex_url)
                    items = _parse_annex_xlsx(xlsx_bytes)
                except Exception as e:
                    logger.warning("EMA annex XLSX fetch/parse failed for %s: %s", annex_url, e)
                    continue

                if not items:
                    continue

                annex_parsed = True
                used_event_pages = True
                stats["items_parsed"] += len(items)
                disclosed_at = annex_doc.get("first_published") or as_of_iso

                seen_ids: set = set()
                for item in items:
                    medicine_name = item["medicine_name"]
                    active_substance = item.get("active_substance")
                    process_type = item.get("process_type") or "OTHER"
                    sponsor = item.get("sponsor")

                    ticker = _match_medicine_to_ticker(
                        medicine_name, active_substance, product_ticker_map,
                    )
                    if ticker is None:
                        stats["unmatched"] += 1
                        if len(all_unmatched) < _MAX_UNMATCHED_LOG:
                            all_unmatched.append(f"{medicine_name} ({active_substance})")
                        continue

                    stats["matched"] += 1

                    event_id = _make_event_id(
                        f"EMA_{committee}_AGENDA_{meeting_end}",
                        medicine_name, process_type, committee, meeting_start, meeting_end,
                    )
                    if event_id in seen_ids:
                        continue
                    seen_ids.add(event_id)

                    classified_type = _classify_procedure_type(process_type)
                    all_events.append({
                        "schema": "ema_committee_event.v1",
                        "id": event_id,
                        "ticker": ticker,
                        "jurisdiction": "EU",
                        "committee": committee,
                        "meeting_start": meeting_start,
                        "meeting_end": meeting_end,
                        "event_date": meeting_end,
                        "disclosed_at": disclosed_at,
                        "item_type": "agenda_item",
                        "procedure_type": classified_type,
                        "medicine_name": medicine_name,
                        "active_substance": active_substance,
                        "sponsor": sponsor,
                        "title": f"{committee} agenda: {process_type} — {medicine_name}",
                        "source": f"EMA_{committee}_AGENDA",
                        "url": annex_url,
                        "confidence": "MED",
                        "tags": _make_tags(classified_type, committee),
                        "raw": {"process_type_raw": process_type},
                    })

                stats["parsed_docs"] += 1
                break  # one annex per meeting is enough

            # If no annex parsed, create meeting-level placeholder from event page
            if not annex_parsed:
                used_event_pages = True
                event_id = _make_event_id(
                    f"EMA_{committee}_AGENDA_{meeting_end}",
                    committee, meeting_start, meeting_end,
                )
                disclosed_at = as_of_iso
                for d in docs.get("agenda", []) + docs.get("annex", []):
                    if d.get("first_published"):
                        disclosed_at = d["first_published"]
                        break
                all_events.append({
                    "schema": "ema_committee_event.v1",
                    "id": event_id,
                    "ticker": None,
                    "jurisdiction": "EU",
                    "committee": committee,
                    "meeting_start": meeting_start,
                    "meeting_end": meeting_end,
                    "event_date": meeting_end,
                    "disclosed_at": disclosed_at,
                    "item_type": "meeting",
                    "procedure_type": None,
                    "medicine_name": None,
                    "active_substance": None,
                    "sponsor": None,
                    "title": f"{committee} meeting {meeting_start} to {meeting_end}",
                    "source": f"EMA_{committee}_AGENDA",
                    "url": mp["url"],
                    "confidence": "MED",
                    "tags": _make_tags(None, committee),
                    "raw": {},
                })

        # --- Fallback: landing page document discovery -------------------
        if not used_event_pages:
            logger.info("EMA %s: no event pages found, falling back to landing page docs", committee)
            links = _discover_links(landing_html, _BASE_URL)
            seen_meetings: set = set()

            for doc in links["agenda_docs"]:
                dates = _parse_meeting_dates(doc["title"])
                if not dates:
                    continue
                meeting_start, meeting_end = dates

                if meeting_end < cutoff or meeting_end > as_of_iso:
                    continue

                doc_url = doc["url"]
                doc_disclosed = doc.get("disclosed_at") or as_of_iso
                is_binary = doc_url.lower().endswith((".pdf", ".xlsx", ".xls"))

                if is_binary:
                    meeting_key = (committee, meeting_end)
                    if meeting_key in seen_meetings:
                        continue
                    seen_meetings.add(meeting_key)

                    event_id = _make_event_id(
                        f"EMA_{committee}_AGENDA_{meeting_end}",
                        committee, meeting_start, meeting_end,
                    )
                    title = doc["title"]
                    title = re.sub(r"\s*\([\d.,]+\s*[KMG]B\s*-\s*\w+\)\s*$", "", title).strip()
                    title = re.sub(r"\s*Draft\s+Reference\s+Number:.*$", "", title).strip()
                    title = re.sub(r"\s*Reference\s+Number:.*$", "", title).strip()

                    all_events.append({
                        "schema": "ema_committee_event.v1",
                        "id": event_id,
                        "ticker": None,
                        "jurisdiction": "EU",
                        "committee": committee,
                        "meeting_start": meeting_start,
                        "meeting_end": meeting_end,
                        "event_date": meeting_end,
                        "disclosed_at": doc_disclosed,
                        "item_type": "meeting",
                        "procedure_type": None,
                        "medicine_name": None,
                        "active_substance": None,
                        "sponsor": None,
                        "title": title,
                        "source": f"EMA_{committee}_AGENDA",
                        "url": doc_url,
                        "confidence": "MED",
                        "tags": _make_tags(None, committee),
                        "raw": {},
                    })
                    stats["parsed_docs"] += 1
                else:
                    try:
                        doc_html = _fetch(doc_url)
                        stats["fetched_pages"] += 1
                        stats["parsed_docs"] += 1
                    except Exception as e:
                        logger.warning("EMA agenda fetch failed for %s: %s", doc_url, e)
                        continue

                    events, unmatched = _parse_agenda_page(
                        doc_html, doc_url, committee,
                        meeting_start, meeting_end, product_ticker_map,
                    )
                    for ev in events:
                        if ev["disclosed_at"] is None:
                            ev["disclosed_at"] = doc_disclosed

                    all_events.extend(events)
                    all_unmatched.extend(unmatched)

    stats["matched_tickers"] = len(set(ev["ticker"] for ev in all_events if ev.get("ticker")))
    stats["meeting_events"] = len([ev for ev in all_events if ev.get("item_type") == "meeting"])
    stats["unmatched_items"] = len(all_unmatched)

    # Top-25 unmatched names by frequency — for surgical map patching.
    unmatched_counts: Counter = Counter()
    for raw in all_unmatched:
        # Normalize to short key: take text before " (" or first 60 chars
        key = raw.split(" (")[0].strip()[:60]
        if key:
            unmatched_counts[key] += 1
    stats["unmatched_top"] = [
        {"name": name, "count": count}
        for name, count in unmatched_counts.most_common(25)
    ]

    # Deterministic sort
    all_events.sort(key=lambda e: (e["event_date"], e["committee"], e.get("ticker") or "", e["id"]))

    # Write cache
    payload = {
        "schema": "ema_committee_events.v1",
        "as_of_date": as_of_date.isoformat(),
        "collector_version": _COLLECTOR_VERSION,
        "events": all_events,
        "stats": stats,
    }
    _write_cache(cache_path, payload)

    logger.info("EMA committee events: %d events (%d product-level) cached -> %s",
                len(all_events), stats["matched"], cache_path)
    return all_events


def collect_ema_meeting_outcomes(
    *,
    as_of_date: date,
    cache_dir: Path,
    product_ticker_map: Dict[str, str],
    committees: Tuple[str, ...] = ("CHMP", "PRAC"),
    lookback_days: int = 365,
) -> List[dict]:
    """Collect EMA meeting outcome events from meeting highlights pages.

    Scrapes committee landing pages for meeting highlights links, parses them,
    and maps medicines to tickers.

    Args:
        as_of_date: PIT date for cache naming.
        cache_dir: Root directory for EMA cache files.
        product_ticker_map: Dict mapping lowercase drug names to ticker symbols.
        committees: Tuple of committee codes to scrape.
        lookback_days: Only include meetings within this window.

    Returns:
        List of outcome dicts (ema_meeting_outcome.v1 schema).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ema_meeting_outcomes_{as_of_date.isoformat()}.json"

    # Short-circuit if cache exists
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            events = payload.get("events", payload) if isinstance(payload, dict) else payload
            logger.info("EMA meeting outcomes cache hit: %s (%d events)", cache_path.name, len(events))
            return events
        except (json.JSONDecodeError, OSError):
            pass

    cutoff = (as_of_date - timedelta(days=lookback_days)).isoformat()
    all_outcomes: List[dict] = []
    all_unmatched: List[str] = []
    stats = {"fetched_pages": 0, "parsed_docs": 0, "matched_tickers": 0, "unmatched_items": 0}

    for committee in committees:
        landing_url = _COMMITTEE_URLS.get(committee)
        if not landing_url:
            logger.warning("Unknown EMA committee: %s", committee)
            continue

        try:
            landing_html = _fetch(landing_url)
            stats["fetched_pages"] += 1
        except Exception as e:
            logger.warning("EMA %s landing page fetch failed: %s", committee, e)
            continue

        links = _discover_links(landing_html, _BASE_URL)

        for hl in links["highlights"]:
            dates = _parse_meeting_dates(hl["title"])
            if not dates:
                continue
            meeting_start, meeting_end = dates

            # Lookback + PIT filter
            if meeting_end < cutoff:
                continue
            # Outcomes are backward-looking: meeting_end <= as_of_date
            if meeting_end > as_of_date.isoformat():
                continue

            try:
                hl_html = _fetch(hl["url"])
                stats["fetched_pages"] += 1
                stats["parsed_docs"] += 1
            except Exception as e:
                logger.warning("EMA highlights fetch failed for %s: %s", hl["url"], e)
                continue

            outcomes, unmatched = _parse_meeting_highlights(
                hl_html, hl["url"], committee, product_ticker_map,
            )
            all_outcomes.extend(outcomes)
            all_unmatched.extend(unmatched)

    stats["matched_tickers"] = len(set(ev["ticker"] for ev in all_outcomes))
    stats["unmatched_items"] = len(all_unmatched)

    unmatched_counts: Counter = Counter()
    for raw in all_unmatched:
        key = raw.split(" (")[0].strip()[:60]
        if key:
            unmatched_counts[key] += 1
    stats["unmatched_top"] = [
        {"name": name, "count": count}
        for name, count in unmatched_counts.most_common(25)
    ]

    # Deterministic sort
    all_outcomes.sort(key=lambda e: (e["event_date"], e["committee"], e["ticker"], e["id"]))

    # Write cache
    payload = {
        "schema": "ema_meeting_outcomes.v1",
        "as_of_date": as_of_date.isoformat(),
        "collector_version": _COLLECTOR_VERSION,
        "events": all_outcomes,
        "stats": stats,
    }
    _write_cache(cache_path, payload)

    logger.info("EMA meeting outcomes: %d events cached -> %s", len(all_outcomes), cache_path)
    return all_outcomes


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _write_cache(cache_path: Path, payload: dict) -> None:
    """Atomic JSON write with sort_keys for determinism."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, str(cache_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
