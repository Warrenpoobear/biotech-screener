#!/usr/bin/env python3
"""
indication_mapper.py

Maps clinical trial conditions to standardized indication categories
for use with PoS scoring and other enhancement modules.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() - uses as_of_date for timestamps
- STDLIB-ONLY: No external dependencies
- FAIL LOUDLY: Clear error states
- AUDITABLE: Track all mapping decisions

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import re
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from datetime import date
from pathlib import Path


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class IndicationMapper:
    """
    Maps clinical trial conditions to standardized indication categories.

    Usage:
        mapper = IndicationMapper()
        indication = mapper.map_conditions(
            conditions=["breast cancer", "solid tumor"],
            ticker="VRTX"
        )
        # Returns: "oncology"
    """

    VERSION = "1.0.0"
    MAPPING_FILE = "data/indication_mapping.json"

    def __init__(self, mapping_path: Optional[str] = None):
        """
        Initialize the indication mapper.

        Args:
            mapping_path: Optional path to indication mapping JSON file.
        """
        self.mapping_path = mapping_path or self.MAPPING_FILE
        self.condition_patterns: Dict[str, List[str]] = {}
        self.ticker_overrides: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []

        self._load_mappings()

    def _load_mappings(self) -> None:
        """Load indication mappings from external file."""
        try:
            paths_to_try = [
                Path(__file__).parent / self.mapping_path,
                Path(self.mapping_path)
            ]

            for path in paths_to_try:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    self.metadata = data.get("provenance", {})
                    self.condition_patterns = data.get("condition_patterns", {})
                    self.ticker_overrides = data.get("ticker_overrides", {})
                    return

            # Fallback to empty mappings
            self._use_fallback_mappings()

        except Exception:
            self._use_fallback_mappings()

    def _use_fallback_mappings(self) -> None:
        """Use hardcoded fallback mappings when file unavailable."""
        self.metadata = {
            "source": "FALLBACK_HARDCODED",
            "warning": "External mapping file not loaded"
        }
        # Minimal fallback patterns
        self.condition_patterns = {
            "oncology": ["cancer", "tumor", "carcinoma", "leukemia", "lymphoma"],
            "rare_disease": ["rare", "orphan", "duchenne", "cystic fibrosis"],
            "infectious_disease": ["infection", "viral", "bacterial", "hiv", "hepatitis"],
            "autoimmune": ["autoimmune", "rheumatoid", "lupus", "psoriasis"],
            "cns": ["alzheimer", "parkinson", "epilepsy", "depression"]
        }
        self.ticker_overrides = {}

    def map_ticker(
        self,
        ticker: str,
        conditions: Optional[List[str]] = None,
        as_of_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Map a ticker to its primary indication category.

        Args:
            ticker: Company ticker symbol
            conditions: List of trial conditions/diseases
            as_of_date: Point-in-time date for audit

        Returns:
            Dict with indication, confidence, and audit info
        """
        if as_of_date is None:
            as_of_date = date.today()

        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z"

        # Check ticker override first
        ticker_upper = ticker.upper() if ticker else ""
        if ticker_upper in self.ticker_overrides:
            indication = self.ticker_overrides[ticker_upper]
            audit_entry = {
                "timestamp": deterministic_timestamp,
                "ticker": ticker_upper,
                "indication": indication,
                "source": "ticker_override",
                "confidence": "HIGH",
                "module_version": self.VERSION
            }
            self.audit_trail.append(audit_entry)
            return {
                "indication": indication,
                "confidence": "HIGH",
                "source": "ticker_override",
                "conditions_analyzed": 0
            }

        # Map from conditions
        if conditions:
            indication, match_info = self._map_conditions(conditions)
            confidence = "HIGH" if match_info["match_count"] >= 2 else "MEDIUM"

            audit_entry = {
                "timestamp": deterministic_timestamp,
                "ticker": ticker_upper,
                "indication": indication,
                "source": "condition_patterns",
                "confidence": confidence,
                "match_info": match_info,
                "module_version": self.VERSION
            }
            self.audit_trail.append(audit_entry)

            return {
                "indication": indication,
                "confidence": confidence,
                "source": "condition_patterns",
                "conditions_analyzed": len(conditions),
                "match_info": match_info
            }

        # No data available
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "ticker": ticker_upper,
            "indication": None,
            "source": "no_data",
            "confidence": "NONE",
            "module_version": self.VERSION
        }
        self.audit_trail.append(audit_entry)

        return {
            "indication": None,
            "confidence": "NONE",
            "source": "no_data",
            "conditions_analyzed": 0
        }

    def _map_conditions(
        self,
        conditions: List[str]
    ) -> tuple:
        """
        Map a list of conditions to an indication category.

        Returns:
            Tuple of (indication, match_info)
        """
        # Count matches per category
        category_scores: Dict[str, int] = {}
        matched_patterns: Dict[str, List[str]] = {}

        for condition in conditions:
            if not condition:
                continue
            condition_lower = condition.lower()

            for category, patterns in self.condition_patterns.items():
                for pattern in patterns:
                    # Use word boundary matching for patterns
                    if re.search(rf"\b{re.escape(pattern)}\b", condition_lower):
                        category_scores[category] = category_scores.get(category, 0) + 1
                        if category not in matched_patterns:
                            matched_patterns[category] = []
                        if pattern not in matched_patterns[category]:
                            matched_patterns[category].append(pattern)

        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            match_info = {
                "match_count": category_scores[best_category],
                "category_scores": category_scores,
                "matched_patterns": matched_patterns.get(best_category, [])
            }
            return best_category, match_info

        # No matches
        return None, {"match_count": 0, "category_scores": {}, "matched_patterns": []}

    def map_universe(
        self,
        tickers: List[str],
        trial_records: List[Dict[str, Any]],
        as_of_date: date
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map indications for an entire universe of tickers.

        Args:
            tickers: List of ticker symbols
            trial_records: Trial records with sponsor_ticker and conditions
            as_of_date: Point-in-time date

        Returns:
            Dict mapping ticker -> indication result
        """
        # Build ticker -> conditions map
        ticker_conditions: Dict[str, Set[str]] = {}
        for trial in trial_records:
            ticker = trial.get("sponsor_ticker") or trial.get("ticker")
            if ticker and trial.get("conditions"):
                ticker_upper = ticker.upper()
                if ticker_upper not in ticker_conditions:
                    ticker_conditions[ticker_upper] = set()
                for cond in trial["conditions"]:
                    if cond:
                        ticker_conditions[ticker_upper].add(cond.lower())

        # Map each ticker
        results = {}
        for ticker in tickers:
            ticker_upper = ticker.upper()
            conditions = list(ticker_conditions.get(ticker_upper, []))
            results[ticker_upper] = self.map_ticker(
                ticker=ticker_upper,
                conditions=conditions,
                as_of_date=as_of_date
            )

        return results

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []

    def get_mapping_info(self) -> Dict[str, Any]:
        """Return mapping metadata."""
        return {
            "metadata": self.metadata,
            "categories_available": list(self.condition_patterns.keys()),
            "ticker_overrides_count": len(self.ticker_overrides)
        }


def demonstration():
    """Demonstrate the indication mapper."""
    print("=" * 70)
    print("INDICATION MAPPER v1.0.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    mapper = IndicationMapper()

    # Show mapping info
    info = mapper.get_mapping_info()
    print(f"Mapping Source: {info['metadata'].get('source', 'UNKNOWN')}")
    print(f"Categories: {', '.join(info['categories_available'])}")
    print(f"Ticker Overrides: {info['ticker_overrides_count']}")
    print()

    # Test cases
    test_cases = [
        ("MRNA", ["covid-19", "vaccine"]),
        ("REGN", ["macular degeneration", "eye disease"]),
        ("CRSP", ["sickle cell disease", "beta thalassemia"]),
        ("UNKNOWN", ["breast cancer", "solid tumor"]),
    ]

    as_of = date(2026, 1, 11)

    print("Test Mappings:")
    print("-" * 70)
    for ticker, conditions in test_cases:
        result = mapper.map_ticker(ticker, conditions, as_of)
        print(f"{ticker}: {result['indication']} (confidence: {result['confidence']}, source: {result['source']})")
    print()


if __name__ == "__main__":
    demonstration()
