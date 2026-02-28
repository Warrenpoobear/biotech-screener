"""
conference_program_collector.py - Medical Conference Program + Abstract Collector

Collects sessions, abstracts, and derived catalyst events from major medical
conferences (ASCO, ESMO, AACR, etc.).  Adapter pattern: one common harness
with per-conference parsing adapters.

Pipeline:
  1. Discover program + abstract URLs for a conference edition
  2. Fetch + parse into normalized session / abstract records
  3. Map entities (company / drug / NCT) to tickers via supplied maps
  4. Derive catalyst events (presentation, late-breaker, accepted-abstract)

Cache layout (deterministic, auditable):
  cache/conferences/<conf_slug>/
    sessions_<as_of_date>.json
    abstracts_<as_of_date>.json
    derived_events_<as_of_date>.json
    meta_<as_of_date>.json
    raw/   (optional; captured HTML only in live runs)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_COLLECTOR_VERSION = "conference_program_collector.v1"
_RATE_LIMIT_SECS = 1.0
_REQUEST_TIMEOUT = 30
_MAX_UNMATCHED = 100

# NCT ID regex
_NCT_RE = re.compile(r"NCT\d{8}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_name(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stable_hash10(s: str) -> str:
    """Deterministic 10-char hex hash of a normalized key string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _write_cache(cache_path: Path, payload: dict) -> None:
    """Atomic JSON write with sort_keys for determinism."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, str(cache_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_cache_if_exists(cache_path: Path) -> Optional[List[dict]]:
    """Read cached records from a wrapped JSON payload, or None."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload.get("records", [])
        return payload
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache read error (%s): %s", cache_path.name, e)
        return None


def _fetch(url: str) -> str:
    """Fetch URL content with rate limiting. Returns HTML text."""
    import requests
    time.sleep(_RATE_LIMIT_SECS)
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={
        "User-Agent": "WakeRobinResearch/1.0 (conference-collector)",
    })
    resp.raise_for_status()
    return resp.text


def _extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from free text (NCT IDs, minimal drug/company tokens)."""
    nct_ids = sorted(set(_NCT_RE.findall(text)))
    return {"companies": [], "drugs": [], "nct_ids": nct_ids}


def _map_to_ticker(
    entities: Dict[str, List[str]],
    product_ticker_map: Dict[str, str],
    company_ticker_map: Dict[str, str],
    nct_ticker_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Map entities to ticker using priority: NCT > drug > company.

    Returns (ticker, mapping_method, matched_value) or (None, None, None).
    """
    # 1. NCT exact match (highest precision)
    if nct_ticker_map:
        for nct in entities.get("nct_ids", []):
            if nct in nct_ticker_map:
                return nct_ticker_map[nct], "nct", nct

    # 2. Drug/INN exact match
    for drug in entities.get("drugs", []):
        norm = normalize_name(drug)
        if norm in product_ticker_map:
            return product_ticker_map[norm], "drug", drug

    # 3. Company exact match
    for company in entities.get("companies", []):
        norm = normalize_name(company)
        if norm in company_ticker_map:
            return company_ticker_map[norm], "company", company

    return None, None, None


# ---------------------------------------------------------------------------
# Conference adapter interface
# ---------------------------------------------------------------------------

class ConferenceAdapter(ABC):
    """Base class for per-conference parsing adapters."""

    conf_name: str = ""
    default_tz: str = "UTC"

    @abstractmethod
    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        """Return dict of URLs: {"program": url, "abstracts": url}."""

    @abstractmethod
    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        """Parse program HTML/JSON into conference_session.v1 records."""

    @abstractmethod
    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        """Parse abstract HTML/JSON into conference_abstract.v1 records."""


ALL_CONFERENCE_SLUGS = ("asco", "esmo", "aacr", "ash", "sitc", "sabcs", "aan", "acr")


class ConfexMixin:
    """Shared parser for conferences hosted on confex.com (ASH, SITC).

    Confex sites use a consistent HTML structure:
      Sessions:  <div class="item" data-type="..."> with child spans .title, .date, .time, .location
      Abstracts: <div class="abstract" data-number="..."> with child spans .title, .author, .affiliation, .sessiontype
    """

    def _parse_confex_sessions(
        self, html: str, edition_year: int, conf_name: str, tz: str,
    ) -> List[dict]:
        sessions: List[dict] = []
        try:
            from html.parser import HTMLParser

            class ConfexSessionParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._in_item = False
                    self._current: Dict[str, Any] = {}
                    self._capture_tag: Optional[str] = None
                    self._text_buf: List[str] = []
                    self.items: List[dict] = []

                def handle_starttag(self, tag, attrs):
                    attrs_d = dict(attrs)
                    if tag == "div" and "item" in (attrs_d.get("class") or "").split():
                        self._in_item = True
                        self._current = {
                            "session_type": attrs_d.get("data-type", "other"),
                        }
                    if self._in_item:
                        cls = attrs_d.get("class") or ""
                        if "title" in cls.split():
                            self._capture_tag = "title"
                            self._text_buf = []
                        elif "date" in cls.split():
                            self._capture_tag = "date"
                            self._text_buf = []
                        elif "time" in cls.split():
                            self._capture_tag = "time"
                            self._text_buf = []
                            self._current["start_raw"] = attrs_d.get("data-start", "")
                            self._current["end_raw"] = attrs_d.get("data-end", "")
                        elif "location" in cls.split():
                            self._capture_tag = "location"
                            self._text_buf = []

                def handle_data(self, data):
                    if self._capture_tag:
                        self._text_buf.append(data)

                def handle_endtag(self, tag):
                    if self._capture_tag and tag == "span":
                        text = " ".join(self._text_buf).strip()
                        self._current[self._capture_tag] = text
                        self._capture_tag = None
                    if tag == "div" and self._in_item and self._current.get("title"):
                        self.items.append(dict(self._current))
                        self._in_item = False
                        self._current = {}

            parser = ConfexSessionParser()
            parser.feed(html)

            for s in parser.items:
                title = s.get("title", "")
                stype = _classify_session_type(s.get("session_type", ""), title)
                start_raw = s.get("start_raw", "")
                end_raw = s.get("end_raw", "")

                key = f"{conf_name}|{edition_year}|{stype}|{title}|{start_raw}"
                sid = f"CONF_{conf_name}_{edition_year}_SESSION_{_stable_hash10(key)}"

                entities = _extract_entities(title)

                sessions.append({
                    "schema": "conference_session.v1",
                    "id": sid,
                    "conference": conf_name,
                    "edition_year": edition_year,
                    "session_type": stype,
                    "track": None,
                    "title": title,
                    "start_dt_local": start_raw,
                    "end_dt_local": end_raw or None,
                    "timezone": tz,
                    "start_dt_utc": _local_to_utc(start_raw, tz),
                    "end_dt_utc": _local_to_utc(end_raw, tz) if end_raw else None,
                    "location": s.get("location"),
                    "url": "",
                    "entities": entities,
                })
        except Exception as e:
            logger.warning("%s confex session parse error: %s", conf_name, e)

        return sessions

    def _parse_confex_abstracts(
        self, html: str, edition_year: int, conf_name: str, tz: str,
    ) -> List[dict]:
        abstracts: List[dict] = []
        try:
            from html.parser import HTMLParser

            class ConfexAbstractParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._in_abstract = False
                    self._current: Dict[str, Any] = {}
                    self._capture_tag: Optional[str] = None
                    self._text_buf: List[str] = []
                    self.abstracts: List[dict] = []

                def handle_starttag(self, tag, attrs):
                    attrs_d = dict(attrs)
                    if tag == "div" and "abstract" in (attrs_d.get("class") or "").split():
                        self._in_abstract = True
                        self._current = {
                            "abstract_code": attrs_d.get("data-number", ""),
                            "presentation_type": attrs_d.get("data-type", "other"),
                        }
                    if self._in_abstract:
                        cls = attrs_d.get("class") or ""
                        if "title" in cls.split():
                            self._capture_tag = "title"
                            self._text_buf = []
                        elif "author" in cls.split():
                            self._capture_tag = "presenter"
                            self._text_buf = []
                        elif "affiliation" in cls.split():
                            self._capture_tag = "affiliation"
                            self._text_buf = []
                        elif "sessiontype" in cls.split():
                            self._capture_tag = "session_type"
                            self._text_buf = []

                def handle_data(self, data):
                    if self._capture_tag:
                        self._text_buf.append(data)

                def handle_endtag(self, tag):
                    if self._capture_tag and tag == "span":
                        text = " ".join(self._text_buf).strip()
                        self._current[self._capture_tag] = text
                        self._capture_tag = None
                    if tag == "div" and self._in_abstract and self._current.get("title"):
                        self.abstracts.append(dict(self._current))
                        self._in_abstract = False
                        self._current = {}

            parser = ConfexAbstractParser()
            parser.feed(html)

            for a in parser.abstracts:
                code = a.get("abstract_code", "")
                title = a.get("title", "")
                ptype = _classify_presentation_type(a.get("presentation_type", ""), code)

                key_str = code if code else _stable_hash10(title)
                aid = f"CONF_{conf_name}_{edition_year}_ABS_{key_str}"

                entities = _extract_entities(title)

                abstracts.append({
                    "schema": "conference_abstract.v1",
                    "id": aid,
                    "conference": conf_name,
                    "edition_year": edition_year,
                    "abstract_code": code or None,
                    "title": title,
                    "session_id": None,
                    "presentation_type": ptype,
                    "presenter": a.get("presenter"),
                    "affiliation": a.get("affiliation"),
                    "sponsor_company": None,
                    "entities": entities,
                    "url": "",
                    "disclosed_at": None,
                })
        except Exception as e:
            logger.warning("%s confex abstract parse error: %s", conf_name, e)

        return abstracts


class AscoAdapter(ConferenceAdapter):
    """MVP adapter for ASCO Annual Meeting.

    In live mode, endpoints point to ASCO program pages.
    Parsing is best-effort; tests use inline fixtures.
    """

    conf_name = "ASCO"
    default_tz = "America/Chicago"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://meetings.asco.org/abstracts-presentations/{edition_year}am/program",
            "abstracts": f"https://meetings.asco.org/abstracts-presentations/{edition_year}am/abstracts",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        """Parse session records from program HTML.

        Expected HTML structure (simplified for extraction):
          <div class="session" data-type="..." data-track="...">
            <span class="session-title">...</span>
            <span class="session-time" data-start="..." data-end="...">...</span>
            <span class="session-location">...</span>
          </div>
        """
        sessions: List[dict] = []
        try:
            from html.parser import HTMLParser

            class SessionParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._in_session = False
                    self._current: Dict[str, Any] = {}
                    self._capture_tag: Optional[str] = None
                    self._text_buf: List[str] = []
                    self.sessions: List[dict] = []

                def handle_starttag(self, tag, attrs):
                    attrs_d = dict(attrs)
                    if tag == "div" and "session" in (attrs_d.get("class") or ""):
                        self._in_session = True
                        self._current = {
                            "session_type": attrs_d.get("data-type", "other"),
                            "track": attrs_d.get("data-track"),
                        }
                    if self._in_session:
                        cls = attrs_d.get("class") or ""
                        if "session-title" in cls:
                            self._capture_tag = "title"
                            self._text_buf = []
                        elif "session-time" in cls:
                            self._current["start_raw"] = attrs_d.get("data-start", "")
                            self._current["end_raw"] = attrs_d.get("data-end", "")
                        elif "session-location" in cls:
                            self._capture_tag = "location"
                            self._text_buf = []

                def handle_data(self, data):
                    if self._capture_tag:
                        self._text_buf.append(data)

                def handle_endtag(self, tag):
                    if self._capture_tag and tag == "span":
                        text = " ".join(self._text_buf).strip()
                        self._current[self._capture_tag] = text
                        self._capture_tag = None
                    if tag == "div" and self._in_session and self._current.get("title"):
                        self.sessions.append(dict(self._current))
                        self._in_session = False
                        self._current = {}

            parser = SessionParser()
            parser.feed(program_payload)

            for s in parser.sessions:
                title = s.get("title", "")
                stype = _classify_session_type(s.get("session_type", ""), title)
                start_raw = s.get("start_raw", "")
                end_raw = s.get("end_raw", "")

                key = f"{self.conf_name}|{edition_year}|{stype}|{title}|{start_raw}"
                sid = f"CONF_{self.conf_name}_{edition_year}_SESSION_{_stable_hash10(key)}"

                entities = _extract_entities(title)

                sessions.append({
                    "schema": "conference_session.v1",
                    "id": sid,
                    "conference": self.conf_name,
                    "edition_year": edition_year,
                    "session_type": stype,
                    "track": s.get("track"),
                    "title": title,
                    "start_dt_local": start_raw,
                    "end_dt_local": end_raw or None,
                    "timezone": self.default_tz,
                    "start_dt_utc": _local_to_utc(start_raw, self.default_tz),
                    "end_dt_utc": _local_to_utc(end_raw, self.default_tz) if end_raw else None,
                    "location": s.get("location"),
                    "url": "",
                    "entities": entities,
                })
        except Exception as e:
            logger.warning("ASCO session parse error: %s", e)

        return sessions

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        """Parse abstract records from abstract HTML.

        Expected HTML structure (simplified):
          <div class="abstract" data-code="LBA9001" data-type="...">
            <span class="abstract-title">...</span>
            <span class="abstract-presenter">...</span>
            <span class="abstract-affiliation">...</span>
            <span class="abstract-sponsor">...</span>
          </div>
        """
        abstracts: List[dict] = []
        try:
            from html.parser import HTMLParser

            class AbstractParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._in_abstract = False
                    self._current: Dict[str, Any] = {}
                    self._capture_tag: Optional[str] = None
                    self._text_buf: List[str] = []
                    self.abstracts: List[dict] = []

                def handle_starttag(self, tag, attrs):
                    attrs_d = dict(attrs)
                    if tag == "div" and "abstract" in (attrs_d.get("class") or ""):
                        self._in_abstract = True
                        self._current = {
                            "abstract_code": attrs_d.get("data-code", ""),
                            "presentation_type": attrs_d.get("data-type", "other"),
                        }
                    if self._in_abstract:
                        cls = attrs_d.get("class") or ""
                        for field in ("title", "presenter", "affiliation", "sponsor"):
                            if f"abstract-{field}" in cls:
                                self._capture_tag = field
                                self._text_buf = []
                                break

                def handle_data(self, data):
                    if self._capture_tag:
                        self._text_buf.append(data)

                def handle_endtag(self, tag):
                    if self._capture_tag and tag == "span":
                        text = " ".join(self._text_buf).strip()
                        self._current[self._capture_tag] = text
                        self._capture_tag = None
                    if tag == "div" and self._in_abstract and self._current.get("title"):
                        self.abstracts.append(dict(self._current))
                        self._in_abstract = False
                        self._current = {}

            parser = AbstractParser()
            parser.feed(abstract_payload)

            for a in parser.abstracts:
                code = a.get("abstract_code", "")
                title = a.get("title", "")
                ptype = _classify_presentation_type(a.get("presentation_type", ""), code)
                sponsor = a.get("sponsor", "")

                key_str = code if code else _stable_hash10(title)
                aid = f"CONF_{self.conf_name}_{edition_year}_ABS_{key_str}"

                entities = _extract_entities(title)
                if sponsor:
                    entities["companies"].append(sponsor)

                abstracts.append({
                    "schema": "conference_abstract.v1",
                    "id": aid,
                    "conference": self.conf_name,
                    "edition_year": edition_year,
                    "abstract_code": code or None,
                    "title": title,
                    "session_id": None,
                    "presentation_type": ptype,
                    "presenter": a.get("presenter"),
                    "affiliation": a.get("affiliation"),
                    "sponsor_company": sponsor or None,
                    "entities": entities,
                    "url": "",
                    "disclosed_at": None,
                })
        except Exception as e:
            logger.warning("ASCO abstract parse error: %s", e)

        return abstracts


# ---------------------------------------------------------------------------
# Additional conference adapters
# ---------------------------------------------------------------------------

class _CustomSessionParserBase:
    """Helper to build per-conference HTMLParser session parsers.

    Subclasses set container_tag, container_class, and child class prefixes.
    """

    @staticmethod
    def _build_sessions(
        html: str,
        container_class: str,
        child_prefix: str,
        conf_name: str,
        edition_year: int,
        tz: str,
    ) -> List[dict]:
        """Generic session parser: container div + child spans with prefixed classes."""
        from html.parser import HTMLParser

        class _Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._in = False
                self._cur: Dict[str, Any] = {}
                self._cap: Optional[str] = None
                self._buf: List[str] = []
                self.items: List[dict] = []

            def handle_starttag(self, tag, attrs):
                ad = dict(attrs)
                cls = ad.get("class") or ""
                if tag == "div" and container_class in cls:
                    self._in = True
                    self._cur = {"session_type": ad.get("data-type", "other"),
                                 "track": ad.get("data-track")}
                if self._in:
                    for field in ("title", "time", "location"):
                        if f"{child_prefix}{field}" in cls:
                            self._cap = field
                            self._buf = []
                            if field == "time":
                                self._cur["start_raw"] = ad.get("data-start", "")
                                self._cur["end_raw"] = ad.get("data-end", "")
                            break

            def handle_data(self, data):
                if self._cap:
                    self._buf.append(data)

            def handle_endtag(self, tag):
                if self._cap and tag == "span":
                    self._cur[self._cap] = " ".join(self._buf).strip()
                    self._cap = None
                if tag == "div" and self._in and self._cur.get("title"):
                    self.items.append(dict(self._cur))
                    self._in = False
                    self._cur = {}

        p = _Parser()
        p.feed(html)

        sessions: List[dict] = []
        for s in p.items:
            title = s.get("title", "")
            stype = _classify_session_type(s.get("session_type", ""), title)
            start_raw = s.get("start_raw", "")
            end_raw = s.get("end_raw", "")

            key = f"{conf_name}|{edition_year}|{stype}|{title}|{start_raw}"
            sid = f"CONF_{conf_name}_{edition_year}_SESSION_{_stable_hash10(key)}"

            sessions.append({
                "schema": "conference_session.v1",
                "id": sid,
                "conference": conf_name,
                "edition_year": edition_year,
                "session_type": stype,
                "track": s.get("track"),
                "title": title,
                "start_dt_local": start_raw,
                "end_dt_local": end_raw or None,
                "timezone": tz,
                "start_dt_utc": _local_to_utc(start_raw, tz),
                "end_dt_utc": _local_to_utc(end_raw, tz) if end_raw else None,
                "location": s.get("location"),
                "url": "",
                "entities": _extract_entities(title),
            })
        return sessions

    @staticmethod
    def _build_abstracts(
        html: str,
        container_class: str,
        child_prefix: str,
        conf_name: str,
        edition_year: int,
    ) -> List[dict]:
        """Generic abstract parser: container div + child spans with prefixed classes."""
        from html.parser import HTMLParser

        class _Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._in = False
                self._cur: Dict[str, Any] = {}
                self._cap: Optional[str] = None
                self._buf: List[str] = []
                self.items: List[dict] = []

            def handle_starttag(self, tag, attrs):
                ad = dict(attrs)
                cls = ad.get("class") or ""
                if tag in ("div", "article") and container_class in cls:
                    self._in = True
                    self._cur = {
                        "abstract_code": ad.get("data-code", ""),
                        "presentation_type": ad.get("data-type", "other"),
                    }
                if self._in:
                    for field in ("title", "presenter", "affiliation", "sponsor"):
                        if f"{child_prefix}{field}" in cls:
                            self._cap = field
                            self._buf = []
                            break

            def handle_data(self, data):
                if self._cap:
                    self._buf.append(data)

            def handle_endtag(self, tag):
                if self._cap and tag == "span":
                    self._cur[self._cap] = " ".join(self._buf).strip()
                    self._cap = None
                if tag in ("div", "article") and self._in and self._cur.get("title"):
                    self.items.append(dict(self._cur))
                    self._in = False
                    self._cur = {}

        p = _Parser()
        p.feed(html)

        abstracts: List[dict] = []
        for a in p.items:
            code = a.get("abstract_code", "")
            title = a.get("title", "")
            ptype = _classify_presentation_type(a.get("presentation_type", ""), code)
            sponsor = a.get("sponsor", "")

            key_str = code if code else _stable_hash10(title)
            aid = f"CONF_{conf_name}_{edition_year}_ABS_{key_str}"

            entities = _extract_entities(title)
            if sponsor:
                entities["companies"].append(sponsor)

            abstracts.append({
                "schema": "conference_abstract.v1",
                "id": aid,
                "conference": conf_name,
                "edition_year": edition_year,
                "abstract_code": code or None,
                "title": title,
                "session_id": None,
                "presentation_type": ptype,
                "presenter": a.get("presenter"),
                "affiliation": a.get("affiliation"),
                "sponsor_company": sponsor or None,
                "entities": entities,
                "url": "",
                "disclosed_at": None,
            })
        return abstracts


class EsmoAdapter(ConferenceAdapter):
    """ESMO Congress — esmo.org."""

    conf_name = "ESMO"
    default_tz = "Europe/Berlin"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://www.esmo.org/meeting-calendar/esmo-congress-{edition_year}/programme",
            "abstracts": f"https://www.esmo.org/meeting-calendar/esmo-congress-{edition_year}/abstracts",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_sessions(
                program_payload, "session-item", "session-item-",
                self.conf_name, edition_year, self.default_tz,
            )
        except Exception as e:
            logger.warning("ESMO session parse error: %s", e)
            return []

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_abstracts(
                abstract_payload, "abstract-item", "abstract-item-",
                self.conf_name, edition_year,
            )
        except Exception as e:
            logger.warning("ESMO abstract parse error: %s", e)
            return []


class AacrAdapter(ConferenceAdapter):
    """AACR Annual Meeting — aacr.org."""

    conf_name = "AACR"
    default_tz = "America/New_York"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://www.aacr.org/meeting/aacr-annual-meeting-{edition_year}/program/",
            "abstracts": f"https://www.aacr.org/meeting/aacr-annual-meeting-{edition_year}/abstracts/",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_sessions(
                program_payload, "program-session", "program-session-",
                self.conf_name, edition_year, self.default_tz,
            )
        except Exception as e:
            logger.warning("AACR session parse error: %s", e)
            return []

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_abstracts(
                abstract_payload, "abstract-entry", "abstract-entry-",
                self.conf_name, edition_year,
            )
        except Exception as e:
            logger.warning("AACR abstract parse error: %s", e)
            return []


class AshAdapter(ConfexMixin, ConferenceAdapter):
    """ASH Annual Meeting — ash.confex.com."""

    conf_name = "ASH"
    default_tz = "America/New_York"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://ash.confex.com/ash/{edition_year}/webprogram/start.html",
            "abstracts": f"https://ash.confex.com/ash/{edition_year}/webprogram/ALLABSTRACTS.html",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        return self._parse_confex_sessions(
            program_payload, edition_year, self.conf_name, self.default_tz,
        )

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        return self._parse_confex_abstracts(
            abstract_payload, edition_year, self.conf_name, self.default_tz,
        )


class SitcAdapter(ConfexMixin, ConferenceAdapter):
    """SITC Annual Meeting — sitcancer.confex.com."""

    conf_name = "SITC"
    default_tz = "America/New_York"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://sitcancer.confex.com/sitcancer/{edition_year}/meetingapp.cgi/Home",
            "abstracts": f"https://sitcancer.confex.com/sitcancer/{edition_year}/meetingapp.cgi/Paper",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        return self._parse_confex_sessions(
            program_payload, edition_year, self.conf_name, self.default_tz,
        )

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        return self._parse_confex_abstracts(
            abstract_payload, edition_year, self.conf_name, self.default_tz,
        )


class SabcsAdapter(ConferenceAdapter):
    """SABCS — San Antonio Breast Cancer Symposium."""

    conf_name = "SABCS"
    default_tz = "America/Chicago"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        yr_short = str(edition_year)
        return {
            "program": f"https://www.sabcs.org/Portals/SABCS{yr_short}/program.aspx",
            "abstracts": f"https://www.sabcs.org/Portals/SABCS{yr_short}/abstracts.aspx",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_sessions(
                program_payload, "program-item", "program-item-",
                self.conf_name, edition_year, self.default_tz,
            )
        except Exception as e:
            logger.warning("SABCS session parse error: %s", e)
            return []

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_abstracts(
                abstract_payload, "abstract-card", "abstract-card-",
                self.conf_name, edition_year,
            )
        except Exception as e:
            logger.warning("SABCS abstract parse error: %s", e)
            return []


class AanAdapter(ConferenceAdapter):
    """AAN — American Academy of Neurology."""

    conf_name = "AAN"
    default_tz = "America/New_York"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://submissions.mirasmart.com/AAN{edition_year}/Itinerary/SearchResults.aspx",
            "abstracts": f"https://submissions.mirasmart.com/AAN{edition_year}/Itinerary/SubmissionList.aspx",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_sessions(
                program_payload, "itinerary-item", "itinerary-item-",
                self.conf_name, edition_year, self.default_tz,
            )
        except Exception as e:
            logger.warning("AAN session parse error: %s", e)
            return []

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_abstracts(
                abstract_payload, "submission-item", "submission-item-",
                self.conf_name, edition_year,
            )
        except Exception as e:
            logger.warning("AAN abstract parse error: %s", e)
            return []


class AcrAdapter(ConferenceAdapter):
    """ACR Convergence — acrabstracts.org."""

    conf_name = "ACR"
    default_tz = "America/New_York"

    def discover_endpoints(self, edition_year: int) -> Dict[str, str]:
        return {
            "program": f"https://acrabstracts.org/meeting/acr-convergence-{edition_year}/program/",
            "abstracts": f"https://acrabstracts.org/meeting/acr-convergence-{edition_year}/abstracts/",
        }

    def parse_sessions(self, program_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_sessions(
                program_payload, "session-entry", "session-entry-",
                self.conf_name, edition_year, self.default_tz,
            )
        except Exception as e:
            logger.warning("ACR session parse error: %s", e)
            return []

    def parse_abstracts(self, abstract_payload: str, edition_year: int) -> List[dict]:
        try:
            return _CustomSessionParserBase._build_abstracts(
                abstract_payload, "abstract-entry", "abstract-entry-",
                self.conf_name, edition_year,
            )
        except Exception as e:
            logger.warning("ACR abstract parse error: %s", e)
            return []


def _classify_session_type(raw_type: str, title: str) -> str:
    """Normalize session type to canonical values."""
    low = (raw_type + " " + title).lower()
    if "late-breaking" in low or "lbct" in low or "lba" in low:
        return "lbct"
    if "plenary" in low:
        return "plenary"
    if "oral" in low:
        return "oral"
    if "poster" in low:
        return "poster"
    if "symposium" in low:
        return "symposium"
    return raw_type.lower() if raw_type else "other"


def _classify_presentation_type(raw_type: str, abstract_code: str) -> str:
    """Normalize presentation type, boost by abstract code prefix."""
    low = raw_type.lower()
    code_up = (abstract_code or "").upper()
    if code_up.startswith(("LBA", "LB")):
        return "lbct"
    if "plenary" in low:
        return "plenary"
    if "oral" in low:
        return "oral"
    if "poster" in low:
        return "poster"
    return low if low else "other"


def _local_to_utc(dt_str: str, tz: str) -> Optional[str]:
    """Best-effort local datetime string to UTC ISO string.

    Falls back gracefully if zoneinfo is unavailable.
    """
    if not dt_str:
        return None
    try:
        from zoneinfo import ZoneInfo
        naive = datetime.fromisoformat(dt_str)
        local = naive.replace(tzinfo=ZoneInfo(tz))
        utc = local.astimezone(ZoneInfo("UTC"))
        return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # Fallback: return as-is with Z suffix
        return dt_str + "Z" if dt_str and not dt_str.endswith("Z") else dt_str


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

CONFERENCE_ADAPTERS: Dict[str, ConferenceAdapter] = {
    "asco": AscoAdapter(),
    "esmo": EsmoAdapter(),
    "aacr": AacrAdapter(),
    "ash": AshAdapter(),
    "sitc": SitcAdapter(),
    "sabcs": SabcsAdapter(),
    "aan": AanAdapter(),
    "acr": AcrAdapter(),
}


# ---------------------------------------------------------------------------
# Derived event generation
# ---------------------------------------------------------------------------

def derive_events_from_records(
    sessions: List[dict],
    abstracts: List[dict],
    product_ticker_map: Dict[str, str],
    company_ticker_map: Dict[str, str],
    nct_ticker_map: Optional[Dict[str, str]],
    conference: str,
    edition_year: int,
    as_of_date: date,
) -> Tuple[List[dict], dict]:
    """Derive catalyst events from session + abstract records.

    Returns (derived_events, stats_dict).
    """
    events: List[dict] = []
    mapping_breakdown: Dict[str, int] = {"nct": 0, "drug": 0, "company": 0}
    matched_tickers: set = set()
    unmatched: List[dict] = []
    parse_errors = 0

    # Build session lookup by id for linking
    session_by_id = {s["id"]: s for s in sessions}

    # Process abstracts (primary mapping source)
    for abstract in abstracts:
        entities = dict(abstract.get("entities", {}))
        sponsor = abstract.get("sponsor_company")
        if sponsor and sponsor not in entities.get("companies", []):
            entities.setdefault("companies", []).append(sponsor)

        ticker, method, matched_val = _map_to_ticker(
            entities, product_ticker_map, company_ticker_map, nct_ticker_map,
        )

        if not ticker:
            if len(unmatched) < _MAX_UNMATCHED:
                context = abstract.get("title", "")[:80]
                for kind, vals in entities.items():
                    for v in vals:
                        unmatched.append({"kind": kind, "value": v, "context": context})
                if sponsor:
                    unmatched.append({"kind": "company", "value": sponsor, "context": context})
            continue

        matched_tickers.add(ticker)
        if method:
            mapping_breakdown[method] = mapping_breakdown.get(method, 0) + 1

        ptype = abstract.get("presentation_type", "other")
        is_late_breaker = ptype in ("lbct", "plenary")
        event_type = "CONF_LATE_BREAKER" if is_late_breaker else "CONF_PRESENTATION"
        confidence = "HIGH" if is_late_breaker else "MED"

        # Determine event date from linked session or conference dates
        event_date_str = None
        session_id = abstract.get("session_id")
        if session_id and session_id in session_by_id:
            sess = session_by_id[session_id]
            start = sess.get("start_dt_local", "")
            if start:
                try:
                    event_date_str = start[:10]
                except (IndexError, TypeError):
                    pass

        if not event_date_str:
            # Fallback: use as_of_date (conference hasn't happened yet)
            event_date_str = as_of_date.isoformat()

        code = abstract.get("abstract_code") or _stable_hash10(abstract.get("title", ""))
        key = f"{conference}|{edition_year}|{event_type}|{event_date_str}|{ticker}|{code}"
        eid = f"CONF_{conference}_{edition_year}_{event_type}_{event_date_str}_{_stable_hash10(key)}"

        events.append({
            "schema": "conference_catalyst_event.v1",
            "id": eid,
            "ticker": ticker,
            "conference": conference,
            "edition_year": edition_year,
            "event_type": event_type,
            "event_date": event_date_str,
            "event_dt_utc": None,
            "disclosed_at": abstract.get("disclosed_at") or as_of_date.isoformat(),
            "title": abstract.get("title", ""),
            "confidence": confidence,
            "source": "CONF_ABSTRACTS",
            "url": abstract.get("url", ""),
            "session_id": session_id,
            "abstract_id": abstract.get("id"),
            "mapping_method": method,
        })

    # Process sessions without abstract linkage (orphan late-breaking sessions)
    abstract_session_ids = {a.get("session_id") for a in abstracts if a.get("session_id")}
    for session in sessions:
        if session["id"] in abstract_session_ids:
            continue  # Already covered via abstracts
        if session.get("session_type") not in ("lbct", "plenary"):
            continue  # Only derive events for high-impact orphan sessions

        entities = session.get("entities", {})
        ticker, method, matched_val = _map_to_ticker(
            entities, product_ticker_map, company_ticker_map, nct_ticker_map,
        )
        if not ticker:
            continue

        matched_tickers.add(ticker)
        if method:
            mapping_breakdown[method] = mapping_breakdown.get(method, 0) + 1

        start = session.get("start_dt_local", "")
        event_date_str = start[:10] if start else as_of_date.isoformat()

        key = f"{conference}|{edition_year}|CONF_LATE_BREAKER|{event_date_str}|{ticker}|{session['id']}"
        eid = f"CONF_{conference}_{edition_year}_CONF_LATE_BREAKER_{event_date_str}_{_stable_hash10(key)}"

        events.append({
            "schema": "conference_catalyst_event.v1",
            "id": eid,
            "ticker": ticker,
            "conference": conference,
            "edition_year": edition_year,
            "event_type": "CONF_LATE_BREAKER",
            "event_date": event_date_str,
            "event_dt_utc": session.get("start_dt_utc"),
            "disclosed_at": as_of_date.isoformat(),
            "title": session.get("title", ""),
            "confidence": "HIGH",
            "source": "CONF_PROGRAM",
            "url": session.get("url", ""),
            "session_id": session["id"],
            "abstract_id": None,
            "mapping_method": method,
        })

    # Sort deterministically
    events.sort(key=lambda e: (e["event_date"], e["ticker"], e["event_type"], e["id"]))

    # Bound unmatched sample
    unmatched = unmatched[:_MAX_UNMATCHED]

    stats = {
        "sessions": len(sessions),
        "abstracts": len(abstracts),
        "derived_events": len(events),
        "matched_tickers": len(matched_tickers),
        "mapping_breakdown": mapping_breakdown,
        "unmatched_items": len(unmatched),
        "parse_errors": parse_errors,
    }

    return events, stats


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

def collect_conference_sessions(
    *,
    conference_slug: str,
    edition_year: int,
    as_of_date: date,
    cache_dir: Path,
    fetch_live: bool = True,
) -> List[dict]:
    """Collect and cache conference session records."""
    adapter = CONFERENCE_ADAPTERS.get(conference_slug)
    if not adapter:
        logger.warning("No adapter for conference '%s'", conference_slug)
        return []

    conf_cache = cache_dir / conference_slug
    cache_path = conf_cache / f"sessions_{as_of_date.isoformat()}.json"

    cached = _read_cache_if_exists(cache_path)
    if cached is not None:
        logger.info("Conference sessions cache hit: %s (%d records)", cache_path.name, len(cached))
        return cached

    if not fetch_live:
        logger.debug("Conference sessions: no cache, fetch_live=False")
        return []

    endpoints = adapter.discover_endpoints(edition_year)
    program_url = endpoints.get("program")
    if not program_url:
        return []

    try:
        html = _fetch(program_url)
    except Exception as e:
        logger.warning("Conference program fetch error (%s): %s", program_url, e)
        return []

    sessions = adapter.parse_sessions(html, edition_year)
    sessions.sort(key=lambda s: (s.get("start_dt_utc") or "", s["id"]))

    payload = {
        "schema": "conference_sessions.v1",
        "as_of_date": as_of_date.isoformat(),
        "conference": adapter.conf_name,
        "edition_year": edition_year,
        "collector_version": _COLLECTOR_VERSION,
        "records": sessions,
        "stats": {"fetched_pages": 1, "sessions": len(sessions)},
    }
    _write_cache(cache_path, payload)
    logger.info("Conference sessions: %d records cached → %s", len(sessions), cache_path.name)
    return sessions


def collect_conference_abstracts(
    *,
    conference_slug: str,
    edition_year: int,
    as_of_date: date,
    cache_dir: Path,
    fetch_live: bool = True,
) -> List[dict]:
    """Collect and cache conference abstract records."""
    adapter = CONFERENCE_ADAPTERS.get(conference_slug)
    if not adapter:
        logger.warning("No adapter for conference '%s'", conference_slug)
        return []

    conf_cache = cache_dir / conference_slug
    cache_path = conf_cache / f"abstracts_{as_of_date.isoformat()}.json"

    cached = _read_cache_if_exists(cache_path)
    if cached is not None:
        logger.info("Conference abstracts cache hit: %s (%d records)", cache_path.name, len(cached))
        return cached

    if not fetch_live:
        logger.debug("Conference abstracts: no cache, fetch_live=False")
        return []

    endpoints = adapter.discover_endpoints(edition_year)
    abstracts_url = endpoints.get("abstracts")
    if not abstracts_url:
        return []

    try:
        html = _fetch(abstracts_url)
    except Exception as e:
        logger.warning("Conference abstracts fetch error (%s): %s", abstracts_url, e)
        return []

    abstracts = adapter.parse_abstracts(html, edition_year)
    abstracts.sort(key=lambda a: (a.get("abstract_code") or "", a["id"]))

    payload = {
        "schema": "conference_abstracts.v1",
        "as_of_date": as_of_date.isoformat(),
        "conference": adapter.conf_name,
        "edition_year": edition_year,
        "collector_version": _COLLECTOR_VERSION,
        "records": abstracts,
        "stats": {"fetched_pages": 1, "abstracts": len(abstracts)},
    }
    _write_cache(cache_path, payload)
    logger.info("Conference abstracts: %d records cached → %s", len(abstracts), cache_path.name)
    return abstracts


def collect_conference_derived_events(
    *,
    conference_slug: str,
    edition_year: int,
    as_of_date: date,
    cache_dir: Path,
    product_ticker_map: Dict[str, str],
    company_ticker_map: Dict[str, str],
    nct_ticker_map: Optional[Dict[str, str]] = None,
    fetch_live: bool = True,
) -> List[dict]:
    """Collect sessions + abstracts, derive catalyst events, cache all three.

    Returns list of derived catalyst event dicts.
    """
    adapter = CONFERENCE_ADAPTERS.get(conference_slug)
    if not adapter:
        logger.warning("No adapter for conference '%s'", conference_slug)
        return []

    conf_cache = cache_dir / conference_slug
    derived_path = conf_cache / f"derived_events_{as_of_date.isoformat()}.json"

    # Short-circuit: if derived events cache exists, return it
    cached = _read_cache_if_exists(derived_path)
    if cached is not None:
        logger.info("Conference derived events cache hit: %s (%d events)", derived_path.name, len(cached))
        return cached

    # Collect upstream data
    sessions = collect_conference_sessions(
        conference_slug=conference_slug,
        edition_year=edition_year,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        fetch_live=fetch_live,
    )
    abstracts = collect_conference_abstracts(
        conference_slug=conference_slug,
        edition_year=edition_year,
        as_of_date=as_of_date,
        cache_dir=cache_dir,
        fetch_live=fetch_live,
    )

    # Derive events
    events, stats = derive_events_from_records(
        sessions=sessions,
        abstracts=abstracts,
        product_ticker_map=product_ticker_map,
        company_ticker_map=company_ticker_map,
        nct_ticker_map=nct_ticker_map,
        conference=adapter.conf_name,
        edition_year=edition_year,
        as_of_date=as_of_date,
    )

    # Collect unmatched sample for the meta sidecar
    unmatched_sample: List[dict] = []
    # Re-derive unmatched for logging (bounded in derive_events_from_records)
    # — stats already contain the count

    payload = {
        "schema": "conference_derived_events.v1",
        "as_of_date": as_of_date.isoformat(),
        "conference": adapter.conf_name,
        "edition_year": edition_year,
        "collector_version": _COLLECTOR_VERSION,
        "records": events,
        "stats": stats,
    }
    _write_cache(derived_path, payload)

    # Write meta sidecar
    meta_path = conf_cache / f"meta_{as_of_date.isoformat()}.json"
    meta = {
        "schema": "conference_meta.v1",
        "as_of_date": as_of_date.isoformat(),
        "conference": adapter.conf_name,
        "edition_year": edition_year,
        "collector_version": _COLLECTOR_VERSION,
        "stats": stats,
    }
    _write_cache(meta_path, meta)

    logger.info(
        "Conference derived events: %d events (%d tickers) cached → %s",
        len(events), stats["matched_tickers"], derived_path.name,
    )
    return events
