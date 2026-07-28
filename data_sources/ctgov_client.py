"""
ClinicalTrials.gov API Client

Fetches trial data for biotech tickers from ClinicalTrials.gov v2 API.
Registry-anchored source for catalyst timing and clinical development scoring.

API Docs: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# API configuration
CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2"
CTGOV_RATE_LIMIT = 0.2  # 5 requests/second max

# Cache directory — resolved relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _REPO_ROOT / "data" / "cache" / "ctgov"

# --- Shadow-only date-precision provenance -----------------------------------
# _parse_date() below snaps "2026-08" to "2026-08-01" and returns a bare ISO
# string, discarding both the month-only granularity and the ACTUAL/ESTIMATED
# type. Nothing downstream can recover either, so month placeholders end up
# asserted as exact-day catalysts.
#
# These helpers record what is being discarded. They are NON-ROUTING: no
# scoring, sizing, catalyst mode, decay or tilt path reads them. Production
# behaviour is unchanged, including _parse_date()'s return contract. The
# corrected-parser proposal is specs/changes/spec_114_catalyst_date_precision_provenance_2026_07_28.md,
# which is freeze-gated; this capture builds the forward evidence for it.

# Shape-matched, not length-matched: "not-a-date" is also ten characters.
_DAY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_YEAR_PATTERN = re.compile(r"^\d{4}$")

# The source these date structs come from, for the normalized candidate.
_CTGOV_DEFAULT_SOURCE = "CTGOV_CALENDAR"


def derive_date_precision(date_str: Optional[str]) -> str:
    """Granularity actually present in a CT.gov date string.

    ``"2026-08-14"`` -> ``DAY``, ``"2026-08"`` -> ``MONTH``, ``"2026"`` -> ``YEAR``.
    A full timestamp is ``DAY``. Anything unrecognised is ``UNKNOWN`` rather
    than an assumed ``DAY``.
    """
    if not isinstance(date_str, str):
        return "UNKNOWN"
    trimmed = date_str.strip()
    if _DAY_PATTERN.match(trimmed):
        return "DAY"
    if _MONTH_PATTERN.match(trimmed):
        return "MONTH"
    if _YEAR_PATTERN.match(trimmed):
        return "YEAR"
    return "UNKNOWN"


def date_provenance_shadow(
    date_struct: Optional[Dict[str, Any]],
    source: str = _CTGOV_DEFAULT_SOURCE,
) -> Dict[str, Optional[str]]:
    """Non-routing provenance for one CT.gov date struct.

    ``ctgov_precision_normalized_candidate`` is the precision a corrected
    parser *would* assign: the weaker of the observed granularity and the
    source's declared precision floor. Recording it in shadow is how Spec 114
    can be validated before anything is switched over.
    """
    # Imported lazily: this module is otherwise stdlib-only, and data_sources
    # sits below common in the layering. Keeping the import here preserves
    # standalone importability rather than inverting that dependency at import
    # time. _SOURCE_PRECISION stays the single authority either way.
    from common.event_quality_features import source_precision_floor, weakest_precision

    struct = date_struct or {}
    raw = struct.get("date")
    precision_raw = derive_date_precision(raw)
    return {
        "ctgov_raw_date": raw,
        "ctgov_date_precision_raw": precision_raw,
        "ctgov_date_type": struct.get("type") or "UNKNOWN",
        "ctgov_precision_normalized_candidate": weakest_precision(precision_raw, source_precision_floor(source)),
    }


_ctgov_breaker = None


def _get_ctgov_breaker():
    """Lazy-init circuit breaker for CTgov API."""
    global _ctgov_breaker
    if _ctgov_breaker is None:
        try:
            from common.robustness_extended import StatefulCircuitBreaker, StatefulCircuitBreakerConfig

            _ctgov_breaker = StatefulCircuitBreaker(
                "ctgov_api",
                config=StatefulCircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=2,
                    timeout_seconds=120.0,
                    window_size_seconds=300.0,
                ),
            )
        except ImportError:
            _ctgov_breaker = None
    return _ctgov_breaker


class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov v2 API.
    Handles rate limiting, caching, and circuit breaking.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self._breaker = _get_ctgov_breaker()

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < CTGOV_RATE_LIMIT:
            time.sleep(CTGOV_RATE_LIMIT - elapsed)
        self._last_request = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with caching and circuit breaker protection."""
        # Circuit breaker: fail fast if API is down
        if self._breaker and not self._breaker.allow_request():
            logging.getLogger(__name__).warning(
                "CTgov circuit breaker OPEN — skipping request for %s",
                endpoint,
            )
            return {}

        self._rate_limit()

        url = f"{CTGOV_API_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                # Sanity check: if we expected studies but got none, log it
                if not data:
                    logging.getLogger(__name__).warning("CTgov API returned empty response for %s", endpoint)
                if self._breaker:
                    self._breaker.record_success()
                return data
        except urllib.error.URLError as e:
            logging.getLogger(__name__).warning("CTgov network error for %s: %s", endpoint, e)
            if self._breaker:
                self._breaker.record_failure(e)
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logging.getLogger(__name__).error("CTgov parse error for %s: %s", endpoint, e)
            return {}
        except Exception as e:
            logging.getLogger(__name__).error("CTgov unexpected error for %s: %s", endpoint, e)
            if self._breaker:
                self._breaker.record_failure(e)
            return {}

    def search_by_sponsor(
        self,
        sponsor_name: str,
        status_filter: Optional[List[str]] = None,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Search trials by sponsor/company name with pagination.

        Args:
            sponsor_name: Company name to search
            status_filter: Optional list of statuses (e.g., ["RECRUITING", "ACTIVE_NOT_RECRUITING"])
            max_results: Maximum results to return (default 1000)

        Returns:
            List of trial records
        """
        # Check cache
        cache_key = f"sponsor_{sponsor_name.replace(' ', '_')}.json"
        cache_file = self.cache_dir / cache_key

        if cache_file.exists():
            # Check if cache is fresh (< 24 hours)
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - mtime).days < 1:
                with open(cache_file) as f:
                    return json.load(f)

        # Build base query params
        base_params = {
            "query.spons": sponsor_name,
            "pageSize": "100",  # API max is 100 per page
            "format": "json",
        }

        if status_filter:
            base_params["filter.overallStatus"] = ",".join(status_filter)

        # Paginate through all results
        all_studies = []
        next_page_token = None

        while len(all_studies) < max_results:
            params = base_params.copy()
            if next_page_token:
                params["pageToken"] = next_page_token

            result = self._make_request("studies", params)
            studies = result.get("studies", [])

            if not studies:
                break

            all_studies.extend(studies)

            # Check for next page
            next_page_token = result.get("nextPageToken")
            if not next_page_token:
                break

        # Limit to max_results
        studies = all_studies[:max_results]

        # Parse into simplified records
        trials = []
        for study in studies:
            protocol = study.get("protocolSection", {})

            # Extract key fields
            identification = protocol.get("identificationModule", {})
            status_mod = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            outcomes = protocol.get("outcomesModule", {})

            trial = {
                "nct_id": identification.get("nctId"),
                "title": identification.get("briefTitle"),
                "sponsor": sponsor_name,
                "phase": self._normalize_phase(design.get("phases", [])),
                "status": status_mod.get("overallStatus"),
                "first_posted": self._parse_date(status_mod.get("studyFirstPostDateStruct", {})),
                "start_date": self._parse_date(status_mod.get("startDateStruct", {})),
                "study_start_date": self._parse_date(status_mod.get("startDateStruct", {})),
                "primary_completion_date": self._parse_date(status_mod.get("primaryCompletionDateStruct", {})),
                "last_update_posted": self._parse_date(status_mod.get("lastUpdatePostDateStruct", {})),
                "results_first_posted": self._parse_date(status_mod.get("resultsFirstPostDateStruct", {})),
                "enrollment": design.get("enrollmentInfo", {}).get("count"),
                "randomized": "RANDOMIZED" in design.get("designInfo", {}).get("allocation", ""),
                "blinded": self._parse_blinding(design.get("designInfo", {})),
                "primary_endpoint": self._get_primary_endpoint(outcomes),
                "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
            }

            # Shadow-only provenance for the primary completion date — the field
            # that drives catalyst timing. Non-routing; recorded so the precision
            # discarded by _parse_date() stops being unrecoverable. See Spec 114.
            trial.update(date_provenance_shadow(status_mod.get("primaryCompletionDateStruct", {})))

            trials.append(trial)

        # Cache results
        with open(cache_file, "w") as f:
            json.dump(trials, f, indent=2)

        return trials

    def get_trial_by_nct(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Get single trial by NCT ID."""
        cache_file = self.cache_dir / f"{nct_id}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        result = self._make_request(f"studies/{nct_id}", {"format": "json"})

        if result:
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)

        return result

    def _normalize_phase(self, phases: List[str]) -> str:
        """Normalize phase to standard format."""
        if not phases:
            return "unknown"

        phase_str = " ".join(phases).lower()

        if "phase3" in phase_str or "phase 3" in phase_str:
            if "phase2" in phase_str or "phase 2" in phase_str:
                return "phase 2/3"
            return "phase 3"
        elif "phase2" in phase_str or "phase 2" in phase_str:
            if "phase1" in phase_str or "phase 1" in phase_str:
                return "phase 1/2"
            return "phase 2"
        elif "phase1" in phase_str or "phase 1" in phase_str:
            return "phase 1"
        elif "phase4" in phase_str or "phase 4" in phase_str:
            return "approved"

        return "unknown"

    def _parse_date(self, date_struct: Dict[str, Any]) -> Optional[str]:
        """Parse date struct to ISO format."""
        if not date_struct:
            return None

        date_str = date_struct.get("date")
        if not date_str:
            return None

        # Handle various formats
        try:
            if len(date_str) == 7:  # "2024-06"
                return f"{date_str}-01"
            elif len(date_str) == 4:  # "2024"
                return f"{date_str}-01-01"
            else:
                return date_str[:10]
        except (TypeError, AttributeError):
            # date_str might not be a string or have expected attributes
            return None

    def _parse_blinding(self, design_info: Dict[str, Any]) -> str:
        """Parse blinding/masking info."""
        masking = design_info.get("maskingInfo", {})
        masking_str = masking.get("masking", "").upper()

        if "DOUBLE" in masking_str:
            return "double"
        elif "SINGLE" in masking_str:
            return "single"
        elif "TRIPLE" in masking_str or "QUADRUPLE" in masking_str:
            return "double"  # Treat as double
        else:
            return "open"

    def _get_primary_endpoint(self, outcomes: Dict[str, Any]) -> str:
        """Extract primary endpoint description."""
        primary = outcomes.get("primaryOutcomes", [])
        if primary:
            return primary[0].get("measure", "")
        return ""


def fetch_trials_for_tickers(
    ticker_to_company: Dict[str, str],
    as_of_date: str,
) -> List[Dict[str, Any]]:
    """
    Fetch trial data for multiple tickers.

    Args:
        ticker_to_company: Mapping of ticker → company name
        as_of_date: Analysis date (for PIT filtering)

    Returns:
        List of trial records with ticker field added
    """
    _log = logging.getLogger(__name__)
    client = ClinicalTrialsClient()
    all_trials = []
    failed_tickers = []

    for ticker, company in ticker_to_company.items():
        _log.info("Fetching trials for %s (%s)...", ticker, company)

        try:
            trials = client.search_by_sponsor(company)

            # Add ticker field
            for trial in trials:
                trial["ticker"] = ticker

            all_trials.extend(trials)
            _log.info("  Found %d trials for %s", len(trials), ticker)

        except Exception as e:
            _log.warning("Failed to fetch trials for %s (%s): %s", ticker, company, e)
            failed_tickers.append(ticker)

    if failed_tickers:
        _log.warning(
            "CTgov fetch: %d/%d tickers failed: %s",
            len(failed_tickers),
            len(ticker_to_company),
            failed_tickers,
        )

    return all_trials


# Ticker to company name mapping for biotech universe
BIOTECH_TICKER_MAP = {
    "AMGN": "Amgen",
    "GILD": "Gilead Sciences",
    "VRTX": "Vertex Pharmaceuticals",
    "REGN": "Regeneron Pharmaceuticals",
    "BIIB": "Biogen",
    "ALNY": "Alnylam Pharmaceuticals",
    "BMRN": "BioMarin Pharmaceutical",
    "SGEN": "Seagen",
    "INCY": "Incyte Corporation",
    "EXEL": "Exelixis",
    "MRNA": "Moderna",
    "BNTX": "BioNTech",
    "IONS": "Ionis Pharmaceuticals",
    "SRPT": "Sarepta Therapeutics",
    "RARE": "Ultragenyx Pharmaceutical",
    "BLUE": "bluebird bio",
    "FOLD": "Amicus Therapeutics",
    "ACAD": "ACADIA Pharmaceuticals",
    "HALO": "Halozyme Therapeutics",
    "KRTX": "Karuna Therapeutics",
    "IMVT": "Immunovant",
    "ARWR": "Arrowhead Pharmaceuticals",
    "PCVX": "Vaxcyte",
    "BEAM": "Beam Therapeutics",
    "EDIT": "Editas Medicine",
}


if __name__ == "__main__":
    # Test the client
    print("Testing ClinicalTrials.gov API client...")

    client = ClinicalTrialsClient()

    # Test single company
    trials = client.search_by_sponsor("Vertex Pharmaceuticals", max_results=10)

    print(f"\nFound {len(trials)} trials for Vertex:")
    for t in trials[:3]:
        print(f"  {t['nct_id']}: {t['phase']} - {t['title'][:50]}...")
        print(f"    Status: {t['status']}, PCD: {t['primary_completion_date']}")
