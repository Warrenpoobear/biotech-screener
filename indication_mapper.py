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
- PIT SAFE: Ticker overrides are time-bounded with evidence

Precedence Rules (highest to lowest):
1. nct_overrides - NCT-level overrides (program-specific)
2. ticker_overrides_v3 - PIT-safe ticker overrides with effective_from
3. ticker_overrides - Legacy ticker overrides (backwards compatibility)
4. condition_patterns - Pattern matching against trial conditions
5. category_fallback - Default to 'other'

Author: Wake Robin Capital Management
Version: 2.0.0
"""

import json
import re
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import date
from pathlib import Path


__version__ = "2.0.0"
__author__ = "Wake Robin Capital Management"


class MappingValidationError(Exception):
    """Raised when mapping configuration is invalid."""
    pass


class IndicationMapper:
    """
    Maps clinical trial conditions to standardized indication categories.

    Implements precedence-based mapping with PIT safety for ticker overrides.

    Usage:
        mapper = IndicationMapper()
        result = mapper.map_ticker(
            ticker="VRTX",
            conditions=["cystic fibrosis"],
            as_of_date=date(2026, 1, 15)
        )
        # Returns: {"indication": "rare_disease", "confidence": "HIGH", ...}
    """

    VERSION = "2.0.0"
    MAPPING_FILE = "data/indication_mapping.json"

    # Confidence tiers for different mapping sources
    CONFIDENCE_TIERS = {
        "ticker_override_v3": 0.95,
        "ticker_override": 0.85,
        "pattern_match_multi": 0.80,
        "pattern_match_single": 0.65,
        "ta_fallback": 0.50,
        "phase_only": 0.30
    }

    def __init__(self, mapping_path: Optional[str] = None):
        """
        Initialize the indication mapper.

        Args:
            mapping_path: Optional path to indication mapping JSON file.
        """
        self.mapping_path = mapping_path or self.MAPPING_FILE
        self.condition_patterns: Dict[str, List[str]] = {}
        self.ticker_overrides: Dict[str, str] = {}
        self.ticker_overrides_v3: Dict[str, Dict[str, Any]] = {}
        self.category_aliases: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        self.validation_errors: List[str] = []

        self._load_mappings()
        self._validate_mappings()

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

                    # Load PIT-safe v3 overrides (excluding _schema key)
                    raw_v3 = data.get("ticker_overrides_v3", {})
                    self.ticker_overrides_v3 = {
                        k: v for k, v in raw_v3.items()
                        if not k.startswith("_")
                    }

                    # Load category aliases for PoS engine compatibility
                    self.category_aliases = data.get("category_aliases", {})
                    # Remove description key if present
                    self.category_aliases = {
                        k: v for k, v in self.category_aliases.items()
                        if not k.startswith("_")
                    }
                    return

            # Fallback to empty mappings
            self._use_fallback_mappings()

        except (json.JSONDecodeError, IOError, OSError) as e:
            # Track the error instead of silently ignoring it
            self.validation_errors.append(
                f"Failed to load mapping file: {type(e).__name__}: {e}"
            )
            self._use_fallback_mappings()

    def _validate_mappings(self) -> None:
        """Validate mapping configuration for consistency and PIT safety."""
        self.validation_errors = []

        # Validate v3 overrides have required fields
        required_v3_fields = {"indication", "effective_from", "evidence"}
        for ticker, override in self.ticker_overrides_v3.items():
            missing = required_v3_fields - set(override.keys())
            if missing:
                self.validation_errors.append(
                    f"ticker_overrides_v3[{ticker}] missing required fields: {missing}"
                )

            # Validate effective_from is valid date format
            effective_from = override.get("effective_from", "")
            try:
                if effective_from:
                    date.fromisoformat(effective_from)
            except ValueError:
                self.validation_errors.append(
                    f"ticker_overrides_v3[{ticker}].effective_from invalid date: {effective_from}"
                )

        # Check for pattern overlaps (patterns that could match the same string)
        self._check_pattern_overlaps()

    def _check_pattern_overlaps(self) -> None:
        """Check for patterns that could cause ambiguous matches."""
        # Build pattern -> category mapping
        pattern_to_category: Dict[str, str] = {}
        for category, patterns in sorted(self.condition_patterns.items()):
            for pattern in sorted(patterns):
                if pattern in pattern_to_category:
                    self.validation_errors.append(
                        f"Pattern '{pattern}' appears in both "
                        f"'{pattern_to_category[pattern]}' and '{category}'"
                    )
                pattern_to_category[pattern] = category

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
        self.ticker_overrides_v3 = {}
        self.category_aliases = {
            "cns": "neurology",
            "autoimmune": "immunology",
            "gi_hepatology": "gastroenterology"
        }

    def _is_v3_override_pit_admissible(
        self,
        override: Dict[str, Any],
        as_of_date: date
    ) -> bool:
        """Check if a v3 override is PIT-admissible for the given as_of_date."""
        effective_from = override.get("effective_from")
        if not effective_from:
            return False

        try:
            effective_date = date.fromisoformat(effective_from)
        except ValueError:
            return False

        # Override must be effective before or on the as_of_date
        if effective_date > as_of_date:
            return False

        # Check effective_until if present
        effective_until = override.get("effective_until")
        if effective_until:
            try:
                until_date = date.fromisoformat(effective_until)
                if as_of_date > until_date:
                    return False
            except ValueError:
                # Invalid date format - treat as if no end date (override still active)
                # This is logged as validation error during init
                self.validation_errors.append(
                    f"Invalid effective_until date format: {effective_until}"
                )

        return True

    def _resolve_category_alias(self, indication: Optional[str]) -> Optional[str]:
        """Resolve category to PoS engine benchmark category via alias."""
        if not indication:
            return indication
        return self.category_aliases.get(indication, indication)

    def map_ticker(
        self,
        ticker: str,
        conditions: Optional[List[str]] = None,
        as_of_date: Optional[date] = None,
        resolve_alias: bool = True
    ) -> Dict[str, Any]:
        """
        Map a ticker to its primary indication category.

        Implements precedence rules:
        1. ticker_overrides_v3 (PIT-safe, with effective_from check)
        2. ticker_overrides (legacy, for backwards compatibility)
        3. condition_patterns (pattern matching against trial conditions)
        4. No data fallback

        Args:
            ticker: Company ticker symbol
            conditions: List of trial conditions/diseases
            as_of_date: Point-in-time date for audit
            resolve_alias: If True, resolve category aliases for PoS compatibility

        Returns:
            Dict with indication, confidence, and audit info
        """
        if as_of_date is None:
            raise ValueError(
                "as_of_date is REQUIRED for determinism. "
                "Do not use date.today() - pass explicit date."
            )

        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z"
        ticker_upper = ticker.upper() if ticker else ""

        # PRECEDENCE 1: Check PIT-safe v3 ticker override first
        if ticker_upper in self.ticker_overrides_v3:
            override = self.ticker_overrides_v3[ticker_upper]
            if self._is_v3_override_pit_admissible(override, as_of_date):
                indication_raw = override.get("indication")
                indication = self._resolve_category_alias(indication_raw) if resolve_alias else indication_raw

                audit_entry = {
                    "timestamp": deterministic_timestamp,
                    "ticker": ticker_upper,
                    "indication": indication,
                    "indication_raw": indication_raw,
                    "source": "ticker_override_v3",
                    "confidence": "HIGHEST",
                    "confidence_score": self.CONFIDENCE_TIERS["ticker_override_v3"],
                    "effective_from": override.get("effective_from"),
                    "evidence": override.get("evidence"),
                    "module_version": self.VERSION
                }
                self.audit_trail.append(audit_entry)
                return {
                    "indication": indication,
                    "indication_raw": indication_raw,
                    "confidence": "HIGHEST",
                    "confidence_score": self.CONFIDENCE_TIERS["ticker_override_v3"],
                    "source": "ticker_override_v3",
                    "effective_from": override.get("effective_from"),
                    "evidence": override.get("evidence"),
                    "conditions_analyzed": 0
                }

        # PRECEDENCE 2: Check legacy ticker override
        if ticker_upper in self.ticker_overrides:
            indication_raw = self.ticker_overrides[ticker_upper]
            indication = self._resolve_category_alias(indication_raw) if resolve_alias else indication_raw

            audit_entry = {
                "timestamp": deterministic_timestamp,
                "ticker": ticker_upper,
                "indication": indication,
                "indication_raw": indication_raw,
                "source": "ticker_override",
                "confidence": "HIGH",
                "confidence_score": self.CONFIDENCE_TIERS["ticker_override"],
                "module_version": self.VERSION
            }
            self.audit_trail.append(audit_entry)
            return {
                "indication": indication,
                "indication_raw": indication_raw,
                "confidence": "HIGH",
                "confidence_score": self.CONFIDENCE_TIERS["ticker_override"],
                "source": "ticker_override",
                "conditions_analyzed": 0
            }

        # PRECEDENCE 3: Map from conditions via pattern matching
        if conditions:
            indication_raw, match_info = self._map_conditions(conditions)
            indication = self._resolve_category_alias(indication_raw) if resolve_alias else indication_raw

            multi_match = match_info["match_count"] >= 2
            confidence = "HIGH" if multi_match else "MEDIUM"
            confidence_key = "pattern_match_multi" if multi_match else "pattern_match_single"

            audit_entry = {
                "timestamp": deterministic_timestamp,
                "ticker": ticker_upper,
                "indication": indication,
                "indication_raw": indication_raw,
                "source": "condition_patterns",
                "confidence": confidence,
                "confidence_score": self.CONFIDENCE_TIERS[confidence_key],
                "match_info": match_info,
                "module_version": self.VERSION
            }
            self.audit_trail.append(audit_entry)

            return {
                "indication": indication,
                "indication_raw": indication_raw,
                "confidence": confidence,
                "confidence_score": self.CONFIDENCE_TIERS[confidence_key],
                "source": "condition_patterns",
                "conditions_analyzed": len(conditions),
                "match_info": match_info
            }

        # PRECEDENCE 4: No data available
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "ticker": ticker_upper,
            "indication": None,
            "indication_raw": None,
            "source": "no_data",
            "confidence": "NONE",
            "confidence_score": 0.0,
            "module_version": self.VERSION
        }
        self.audit_trail.append(audit_entry)

        return {
            "indication": None,
            "indication_raw": None,
            "confidence": "NONE",
            "confidence_score": 0.0,
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
            "ticker_overrides_count": len(self.ticker_overrides),
            "ticker_overrides_v3_count": len(self.ticker_overrides_v3),
            "category_aliases": self.category_aliases,
            "validation_errors": self.validation_errors,
            "mapper_version": self.VERSION
        }

    def get_validation_errors(self) -> List[str]:
        """Return any validation errors found during initialization."""
        return self.validation_errors.copy()

    def is_valid(self) -> bool:
        """Return True if no validation errors were found."""
        return len(self.validation_errors) == 0


def demonstration():
    """Demonstrate the indication mapper."""
    print("=" * 70)
    print(f"INDICATION MAPPER v{__version__} - DEMONSTRATION")
    print("=" * 70)
    print()

    mapper = IndicationMapper()

    # Show mapping info
    info = mapper.get_mapping_info()
    print(f"Mapping Source: {info['metadata'].get('source', 'UNKNOWN')}")
    print(f"Schema Version: {info['metadata'].get('schema_version', 'UNKNOWN')}")
    print(f"Categories: {', '.join(info['categories_available'])}")
    print(f"Ticker Overrides (legacy): {info['ticker_overrides_count']}")
    print(f"Ticker Overrides (v3/PIT-safe): {info['ticker_overrides_v3_count']}")
    print(f"Category Aliases: {info['category_aliases']}")
    print()

    # Show validation status
    if mapper.is_valid():
        print("Validation: PASSED (no errors)")
    else:
        print("Validation: FAILED")
        for error in mapper.get_validation_errors():
            print(f"  - {error}")
    print()

    # Test cases demonstrating precedence
    test_cases = [
        ("MRNA", ["covid-19", "vaccine"]),      # v3 override
        ("BIIB", ["alzheimer", "neurology"]),   # v3 override with alias
        ("REGN", ["macular degeneration"]),     # legacy override
        ("UNKNOWN", ["breast cancer", "solid tumor"]),  # pattern match
    ]

    as_of = date(2026, 1, 15)

    print("Test Mappings (demonstrating precedence):")
    print("-" * 70)
    for ticker, conditions in test_cases:
        result = mapper.map_ticker(ticker, conditions, as_of)
        print(f"{ticker}:")
        print(f"  indication: {result['indication']} (raw: {result.get('indication_raw', 'N/A')})")
        print(f"  confidence: {result['confidence']} ({result['confidence_score']})")
        print(f"  source: {result['source']}")
        if result.get('evidence'):
            print(f"  evidence: {result['evidence']}")
    print()

    # Demonstrate PIT safety - v3 override not effective before effective_from
    print("PIT Safety Demonstration:")
    print("-" * 70)
    # MRNA's v3 override is effective from 2020-01-01
    before_effective = date(2019, 6, 1)
    after_effective = date(2021, 1, 1)

    result_before = mapper.map_ticker("MRNA", ["vaccine"], before_effective)
    result_after = mapper.map_ticker("MRNA", ["vaccine"], after_effective)

    print(f"MRNA as of 2019-06-01 (before v3 effective_from):")
    print(f"  source: {result_before['source']}")
    print(f"MRNA as of 2021-01-01 (after v3 effective_from):")
    print(f"  source: {result_after['source']}")
    print()


if __name__ == "__main__":
    demonstration()
