"""
ClinicalTrials.gov API Client

Fetches trial data for biotech tickers from ClinicalTrials.gov v2 API.
Registry-anchored source for catalyst timing and clinical development scoring.

API Docs: https://clinicaltrials.gov/data-api/api
"""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# API configuration
CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2"
CTGOV_RATE_LIMIT = 0.2  # 5 requests/second max

# Cache directory
CACHE_DIR = Path("/home/claude/biotech_screener/data/cache/ctgov")


class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov v2 API.
    Handles rate limiting and caching.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < CTGOV_RATE_LIMIT:
            time.sleep(CTGOV_RATE_LIMIT - elapsed)
        self._last_request = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with caching."""
        self._rate_limit()
        
        url = f"{CTGOV_API_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
        
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"API request failed: {e}")
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
                "primary_completion_date": self._parse_date(
                    status_mod.get("primaryCompletionDateStruct", {})
                ),
                "study_start_date": self._parse_date(
                    status_mod.get("startDateStruct", {})
                ),
                "enrollment": design.get("enrollmentInfo", {}).get("count"),
                "randomized": "RANDOMIZED" in design.get("designInfo", {}).get("allocation", ""),
                "blinded": self._parse_blinding(design.get("designInfo", {})),
                "primary_endpoint": self._get_primary_endpoint(outcomes),
                "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
            }
            
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
    client = ClinicalTrialsClient()
    all_trials = []
    
    for ticker, company in ticker_to_company.items():
        print(f"  Fetching trials for {ticker} ({company})...")
        
        try:
            trials = client.search_by_sponsor(company)
            
            # Add ticker field
            for trial in trials:
                trial["ticker"] = ticker
            
            all_trials.extend(trials)
            print(f"    Found {len(trials)} trials")
            
        except Exception as e:
            print(f"    Error: {e}")
    
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
