#!/usr/bin/env python3
"""
ClinicalTrials.gov API integration.

Free, public data, no API key required.

Fetches clinical trial data including:
- Trial status
- Phase
- Enrollment
- Completion dates
- Results posting

Uses ClinicalTrials.gov API v2.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import urllib.request
import urllib.parse
import json
import ssl
from datetime import datetime
from typing import Dict, List, Optional, Any


# Module metadata
__version__ = "1.0.0"

# API base URL
API_BASE = "https://clinicaltrials.gov/api/v2"


def _create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for HTTPS requests."""
    context = ssl.create_default_context()
    return context


def fetch_trial_status(nct_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch trial status from ClinicalTrials.gov API.

    Args:
        nct_id: NCT number (e.g., "NCT04567890")

    Returns:
        Dict containing:
        - nct_id: str
        - status: str (e.g., "Recruiting", "Completed")
        - phase: str (e.g., "Phase 3")
        - enrollment: int
        - primary_completion_date: str
        - last_update: str
        - title: str
        - sponsor: str
        - conditions: List[str]
        - source: str

        Returns None if fetch fails.
    """
    try:
        # Normalize NCT ID
        nct_id = nct_id.upper().strip()
        if not nct_id.startswith("NCT"):
            nct_id = f"NCT{nct_id}"

        url = f"{API_BASE}/studies/{nct_id}"

        headers = {
            "User-Agent": "Wake Robin Capital (compliance@wakerobincapital.com)",
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        # Extract protocol section
        protocol = data.get("protocolSection", {})

        # Identification module
        id_module = protocol.get("identificationModule", {})
        nct_id_result = id_module.get("nctId", nct_id)
        title = id_module.get("briefTitle", "")

        # Status module
        status_module = protocol.get("statusModule", {})
        overall_status = status_module.get("overallStatus", "Unknown")
        last_update = status_module.get("lastUpdatePostDateStruct", {}).get("date")
        primary_completion = status_module.get("primaryCompletionDateStruct", {}).get("date")

        # Design module
        design_module = protocol.get("designModule", {})
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else "N/A"

        # Enrollment
        enrollment_info = design_module.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count", 0)

        # Sponsor
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        sponsor_name = lead_sponsor.get("name", "Unknown")

        # Conditions
        conditions_module = protocol.get("conditionsModule", {})
        conditions = conditions_module.get("conditions", [])

        return {
            "nct_id": nct_id_result,
            "status": overall_status,
            "phase": phase,
            "enrollment": enrollment,
            "primary_completion_date": primary_completion,
            "last_update": last_update,
            "title": title,
            "sponsor": sponsor_name,
            "conditions": conditions,
            "collected_at": datetime.now().isoformat(),
            "source": "clinicaltrials_gov"
        }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  ClinicalTrials.gov: Trial not found: {nct_id}")
        else:
            print(f"  ClinicalTrials.gov HTTP error for {nct_id}: {e.code}")
        return None
    except urllib.error.URLError as e:
        print(f"  ClinicalTrials.gov URL error for {nct_id}: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ClinicalTrials.gov JSON error for {nct_id}: {e}")
        return None
    except Exception as e:
        print(f"  ClinicalTrials.gov fetch failed for {nct_id}: {e}")
        return None


def search_trials_by_sponsor(sponsor: str, max_results: int = 100) -> List[str]:
    """
    Search for all trials by sponsor name.

    Args:
        sponsor: Sponsor name to search for
        max_results: Maximum number of NCT IDs to return

    Returns:
        List of NCT IDs
    """
    try:
        # Build search query
        query = f'AREA[LeadSponsorName]"{sponsor}"'
        encoded_query = urllib.parse.quote(query)

        url = f"{API_BASE}/studies?query.term={encoded_query}&countTotal=true&pageSize={max_results}"

        headers = {
            "User-Agent": "Wake Robin Capital (compliance@wakerobincapital.com)",
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        studies = data.get("studies", [])

        nct_ids = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId")
            if nct_id:
                nct_ids.append(nct_id)

        return nct_ids

    except Exception as e:
        print(f"  ClinicalTrials.gov search failed for sponsor {sponsor}: {e}")
        return []


def search_trials_by_condition(condition: str, status: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
    """
    Search for trials by condition/disease.

    Args:
        condition: Condition name to search for
        status: Filter by status (e.g., "Recruiting", "Completed")
        max_results: Maximum number of results to return

    Returns:
        List of trial summary dicts
    """
    try:
        # Build search query
        parts = [f'AREA[Condition]"{condition}"']
        if status:
            parts.append(f'AREA[OverallStatus]{status}')

        query = " AND ".join(parts)
        encoded_query = urllib.parse.quote(query)

        url = f"{API_BASE}/studies?query.term={encoded_query}&countTotal=true&pageSize={max_results}"

        headers = {
            "User-Agent": "Wake Robin Capital (compliance@wakerobincapital.com)",
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        studies = data.get("studies", [])

        results = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})

            phases = design_module.get("phases", [])

            results.append({
                "nct_id": id_module.get("nctId"),
                "title": id_module.get("briefTitle"),
                "status": status_module.get("overallStatus"),
                "phase": phases[0] if phases else "N/A",
            })

        return results

    except Exception as e:
        print(f"  ClinicalTrials.gov search failed for condition {condition}: {e}")
        return []


def fetch_multiple_trials(nct_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch status for multiple trials.

    Args:
        nct_ids: List of NCT IDs

    Returns:
        List of trial status dicts
    """
    results = []

    for nct_id in nct_ids:
        trial = fetch_trial_status(nct_id)
        if trial:
            results.append(trial)

    return results


def get_trial_results_status(nct_id: str) -> Optional[Dict[str, Any]]:
    """
    Check if trial results have been posted.

    Args:
        nct_id: NCT number

    Returns:
        Dict with results posting info, or None
    """
    try:
        nct_id = nct_id.upper().strip()
        if not nct_id.startswith("NCT"):
            nct_id = f"NCT{nct_id}"

        url = f"{API_BASE}/studies/{nct_id}"

        headers = {
            "User-Agent": "Wake Robin Capital (compliance@wakerobincapital.com)",
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        # Check for results section
        has_results = "resultsSection" in data

        # Check results posting info
        results_module = data.get("protocolSection", {}).get("statusModule", {})
        results_first_post = results_module.get("resultsFirstPostDateStruct", {})

        return {
            "nct_id": nct_id,
            "has_results": has_results,
            "results_posted_date": results_first_post.get("date"),
            "collected_at": datetime.now().isoformat(),
            "source": "clinicaltrials_gov"
        }

    except Exception as e:
        print(f"  ClinicalTrials.gov results check failed for {nct_id}: {e}")
        return None


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CLINICALTRIALS.GOV DATA FETCHER TEST")
    print("=" * 60)

    # Test with a known trial
    test_nct = "NCT04368728"  # Example trial

    print(f"\nFetching trial status for {test_nct}...")
    trial = fetch_trial_status(test_nct)
    if trial:
        print(f"  Title: {trial['title'][:60]}...")
        print(f"  Status: {trial['status']}")
        print(f"  Phase: {trial['phase']}")
        print(f"  Enrollment: {trial['enrollment']}")
        print(f"  Sponsor: {trial['sponsor']}")
        print(f"  Last Update: {trial['last_update']}")
    else:
        print("  Failed to fetch trial status")

    # Test sponsor search
    test_sponsor = "Pfizer"
    print(f"\nSearching trials by sponsor: {test_sponsor}...")
    nct_ids = search_trials_by_sponsor(test_sponsor, max_results=5)
    if nct_ids:
        print(f"  Found {len(nct_ids)} trials:")
        for nct_id in nct_ids[:5]:
            print(f"    - {nct_id}")
    else:
        print("  No trials found or search failed")

    # Test condition search
    test_condition = "breast cancer"
    print(f"\nSearching Phase 3 trials for: {test_condition}...")
    trials = search_trials_by_condition(test_condition, max_results=5)
    if trials:
        print(f"  Found {len(trials)} trials:")
        for t in trials[:3]:
            print(f"    - {t['nct_id']}: {t['status']} ({t['phase']})")
    else:
        print("  No trials found or search failed")
