#!/usr/bin/env python3
"""
fda_designation_engine.py

FDA Designation Scoring Engine for Biotech Screener

Tracks and scores FDA regulatory designations that significantly impact
approval probability and timeline. Designations provide objective,
regulatory-validated signals of drug potential.

Designation Types and Impact:
- Breakthrough Therapy (BTD): ~20% higher approval rate, intensive FDA guidance
- Fast Track (FT): Rolling review, more frequent FDA meetings
- Orphan Drug (ODD): Market exclusivity, tax credits, fee waivers
- Priority Review (PR): 6-month vs 10-month review timeline
- Accelerated Approval (AA): Surrogate endpoint pathway
- RMAT (Regenerative Medicine): Expedited for cell/gene therapies

Data Sources:
- FDA designation announcements
- ClinicalTrials.gov (limited designation data)
- SEC filings (8-K, 10-K risk factors)
- Company press releases

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Set
from datetime import date
from enum import Enum
from dataclasses import dataclass


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class DesignationType(Enum):
    """FDA designation types with regulatory significance."""
    BREAKTHROUGH_THERAPY = "BTD"      # Breakthrough Therapy Designation
    FAST_TRACK = "FT"                 # Fast Track Designation
    ORPHAN_DRUG = "ODD"               # Orphan Drug Designation
    PRIORITY_REVIEW = "PR"            # Priority Review
    ACCELERATED_APPROVAL = "AA"       # Accelerated Approval pathway
    RMAT = "RMAT"                     # Regenerative Medicine Advanced Therapy


@dataclass
class DesignationRecord:
    """Record of an FDA designation for a specific program."""
    ticker: str
    designation_type: DesignationType
    indication: str
    drug_name: Optional[str]
    nct_id: Optional[str]
    grant_date: Optional[date]
    source: str  # "fda", "sec_filing", "press_release", "clinicaltrials"
    confidence: str  # "confirmed", "announced", "inferred"


class FDADesignationEngine:
    """
    FDA Designation scoring engine for biotech valuation.

    Designations provide regulatory validation of drug potential and
    significantly impact approval probability. This engine:
    1. Tracks designations by company and program
    2. Calculates PoS adjustments based on designation portfolio
    3. Provides timeline acceleration estimates

    Usage:
        engine = FDADesignationEngine()
        engine.load_designations(designations_data)
        result = engine.score_ticker("ACME", as_of_date=date(2026, 1, 26))
    """

    VERSION = "1.0.0"

    # PoS adjustment factors by designation type
    # Based on FDA designation approval rate differentials (BIO/FDA data)
    POS_ADJUSTMENT_FACTORS: Dict[DesignationType, Decimal] = {
        DesignationType.BREAKTHROUGH_THERAPY: Decimal("1.25"),   # +25% approval lift
        DesignationType.FAST_TRACK: Decimal("1.12"),             # +12% approval lift
        DesignationType.ORPHAN_DRUG: Decimal("1.18"),            # +18% approval lift
        DesignationType.PRIORITY_REVIEW: Decimal("1.08"),        # +8% (timeline, not rate)
        DesignationType.ACCELERATED_APPROVAL: Decimal("1.15"),   # +15% (surrogate endpoints)
        DesignationType.RMAT: Decimal("1.20"),                   # +20% for CGT
    }

    # Confidence multipliers for designation source reliability
    CONFIDENCE_MULTIPLIERS: Dict[str, Decimal] = {
        "confirmed": Decimal("1.00"),    # FDA announcement or database
        "announced": Decimal("0.95"),    # Company announcement, not yet in FDA database
        "inferred": Decimal("0.70"),     # Inferred from trial characteristics
    }

    # Timeline acceleration (months saved) by designation
    TIMELINE_ACCELERATION: Dict[DesignationType, int] = {
        DesignationType.BREAKTHROUGH_THERAPY: 6,   # Intensive FDA engagement
        DesignationType.FAST_TRACK: 3,             # Rolling review
        DesignationType.ORPHAN_DRUG: 0,            # No timeline impact
        DesignationType.PRIORITY_REVIEW: 4,        # 6 vs 10 month review
        DesignationType.ACCELERATED_APPROVAL: 12,  # Surrogate endpoints
        DesignationType.RMAT: 6,                   # Expedited development
    }

    # Score contribution caps (prevent over-weighting)
    MAX_DESIGNATION_BONUS = Decimal("35")  # Max 35 points from designations
    DIMINISHING_RETURNS_THRESHOLD = 3      # After 3 designations, diminishing returns

    def __init__(self):
        """Initialize the FDA designation engine."""
        self.designations: Dict[str, List[DesignationRecord]] = {}  # ticker -> records
        self.audit_trail: List[Dict[str, Any]] = []

    def load_designations(self, data: List[Dict[str, Any]]) -> int:
        """
        Load designation data from external source.

        Args:
            data: List of designation records with keys:
                - ticker: str
                - designation_type: str (BTD, FT, ODD, PR, AA, RMAT)
                - indication: str
                - drug_name: Optional[str]
                - nct_id: Optional[str]
                - grant_date: Optional[str] (YYYY-MM-DD)
                - source: str
                - confidence: str

        Returns:
            Number of designations loaded
        """
        self.designations.clear()
        loaded = 0

        for record in data:
            ticker = record.get("ticker", "").upper()
            if not ticker:
                continue

            try:
                des_type = DesignationType(record.get("designation_type", ""))
            except ValueError:
                continue  # Skip invalid designation types

            grant_date = None
            if record.get("grant_date"):
                try:
                    grant_date = date.fromisoformat(record["grant_date"])
                except ValueError:
                    pass

            designation = DesignationRecord(
                ticker=ticker,
                designation_type=des_type,
                indication=record.get("indication", "unknown"),
                drug_name=record.get("drug_name"),
                nct_id=record.get("nct_id"),
                grant_date=grant_date,
                source=record.get("source", "unknown"),
                confidence=record.get("confidence", "announced"),
            )

            if ticker not in self.designations:
                self.designations[ticker] = []
            self.designations[ticker].append(designation)
            loaded += 1

        return loaded

    def score_ticker(
        self,
        ticker: str,
        as_of_date: date,
        base_pos: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Calculate designation score and PoS adjustment for a ticker.

        Args:
            ticker: Stock ticker
            as_of_date: Point-in-time date
            base_pos: Optional base PoS score to adjust

        Returns:
            Dict containing:
            - designation_score: 0-100 score based on designation portfolio
            - pos_multiplier: Multiplier to apply to base PoS
            - adjusted_pos: base_pos * pos_multiplier (if base_pos provided)
            - designations: List of active designations
            - timeline_acceleration_months: Estimated months saved
            - audit_entry: Full calculation trace
        """
        ticker = ticker.upper()
        records = self.designations.get(ticker, [])

        # Filter to active designations (granted before as_of_date)
        active = [
            r for r in records
            if r.grant_date is None or r.grant_date <= as_of_date
        ]

        if not active:
            return self._no_designations_result(ticker, base_pos, as_of_date)

        # Calculate composite PoS multiplier
        # Use max multiplier approach with diminishing returns for additional designations
        designation_types = set(r.designation_type for r in active)

        # Get base multiplier from strongest designation
        multipliers = [
            self.POS_ADJUSTMENT_FACTORS[d] * self.CONFIDENCE_MULTIPLIERS.get(
                next((r.confidence for r in active if r.designation_type == d), "announced"),
                Decimal("0.90")
            )
            for d in designation_types
        ]

        # Primary multiplier is the max, additional provide diminishing returns
        multipliers_sorted = sorted(multipliers, reverse=True)
        primary_mult = multipliers_sorted[0] if multipliers_sorted else Decimal("1.0")

        # Additional designations add 50% of their excess over 1.0, then 25%, etc.
        composite_mult = primary_mult
        for i, mult in enumerate(multipliers_sorted[1:], start=1):
            if i >= self.DIMINISHING_RETURNS_THRESHOLD:
                break
            excess = mult - Decimal("1.0")
            diminishing_factor = Decimal("0.5") ** i
            composite_mult += excess * diminishing_factor

        # Cap the multiplier
        composite_mult = min(composite_mult, Decimal("1.50"))

        # Calculate designation score (0-100)
        # Base: 50 (neutral) + bonus per designation type
        designation_score = Decimal("50")

        type_bonuses = {
            DesignationType.BREAKTHROUGH_THERAPY: Decimal("15"),
            DesignationType.FAST_TRACK: Decimal("8"),
            DesignationType.ORPHAN_DRUG: Decimal("10"),
            DesignationType.PRIORITY_REVIEW: Decimal("5"),
            DesignationType.ACCELERATED_APPROVAL: Decimal("10"),
            DesignationType.RMAT: Decimal("12"),
        }

        total_bonus = Decimal("0")
        for des_type in designation_types:
            bonus = type_bonuses.get(des_type, Decimal("5"))
            # Apply diminishing returns after first
            if total_bonus > 0:
                bonus = bonus * Decimal("0.6")
            total_bonus += bonus

        total_bonus = min(total_bonus, self.MAX_DESIGNATION_BONUS)
        designation_score += total_bonus
        designation_score = min(designation_score, Decimal("100"))

        # Calculate timeline acceleration
        timeline_months = sum(
            self.TIMELINE_ACCELERATION.get(d, 0) for d in designation_types
        )
        # Cap at 18 months (don't double-count overlapping benefits)
        timeline_months = min(timeline_months, 18)

        # Adjusted PoS
        adjusted_pos = None
        if base_pos is not None:
            adjusted_pos = (base_pos * composite_mult).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            adjusted_pos = min(adjusted_pos, Decimal("100"))

        # Build audit entry
        audit_entry = {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "active_designations": len(active),
            "designation_types": [d.value for d in designation_types],
            "primary_multiplier": str(primary_mult),
            "composite_multiplier": str(composite_mult),
            "designation_score": str(designation_score),
            "timeline_acceleration_months": timeline_months,
            "base_pos": str(base_pos) if base_pos else None,
            "adjusted_pos": str(adjusted_pos) if adjusted_pos else None,
        }

        self.audit_trail.append(audit_entry)

        return {
            "ticker": ticker,
            "designation_score": designation_score,
            "pos_multiplier": composite_mult,
            "adjusted_pos": adjusted_pos,
            "designation_count": len(active),
            "designation_types": [d.value for d in designation_types],
            "designations": [
                {
                    "type": r.designation_type.value,
                    "indication": r.indication,
                    "drug_name": r.drug_name,
                    "grant_date": r.grant_date.isoformat() if r.grant_date else None,
                    "confidence": r.confidence,
                }
                for r in active
            ],
            "timeline_acceleration_months": timeline_months,
            "audit_entry": audit_entry,
        }

    def _no_designations_result(
        self,
        ticker: str,
        base_pos: Optional[Decimal],
        as_of_date: date
    ) -> Dict[str, Any]:
        """Return result for ticker with no designations."""
        return {
            "ticker": ticker,
            "designation_score": Decimal("50"),  # Neutral
            "pos_multiplier": Decimal("1.0"),
            "adjusted_pos": base_pos,
            "designation_count": 0,
            "designation_types": [],
            "designations": [],
            "timeline_acceleration_months": 0,
            "audit_entry": {
                "ticker": ticker,
                "as_of_date": as_of_date.isoformat(),
                "active_designations": 0,
                "note": "no_designations_found",
            },
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        as_of_date: date
    ) -> Dict[str, Any]:
        """
        Score an entire universe of tickers.

        Args:
            universe: List of dicts with at least 'ticker' key,
                     optionally 'base_pos' for PoS adjustment
            as_of_date: Point-in-time date

        Returns:
            Dict with scores, diagnostics, and provenance
        """
        scores = []
        designation_coverage = 0
        type_distribution: Dict[str, int] = {}

        for company in universe:
            ticker = company.get("ticker", "").upper()
            base_pos = company.get("base_pos")
            if base_pos is not None:
                base_pos = Decimal(str(base_pos))

            result = self.score_ticker(ticker, as_of_date, base_pos)

            scores.append({
                "ticker": ticker,
                "designation_score": result["designation_score"],
                "pos_multiplier": result["pos_multiplier"],
                "adjusted_pos": result["adjusted_pos"],
                "designation_count": result["designation_count"],
                "designation_types": result["designation_types"],
                "timeline_acceleration_months": result["timeline_acceleration_months"],
            })

            if result["designation_count"] > 0:
                designation_coverage += 1
                for des_type in result["designation_types"]:
                    type_distribution[des_type] = type_distribution.get(des_type, 0) + 1

        # Content hash for determinism
        scores_json = json.dumps(
            [{"t": s["ticker"], "s": str(s["designation_score"])} for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "with_designations": designation_coverage,
                "coverage_pct": f"{designation_coverage / max(1, len(scores)) * 100:.1f}%",
                "type_distribution": type_distribution,
            },
            "provenance": {
                "module": "fda_designation_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
            },
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


# =============================================================================
# SAMPLE DATA GENERATOR (for testing/bootstrapping)
# =============================================================================

def generate_sample_designations() -> List[Dict[str, Any]]:
    """
    Generate sample FDA designation data for common biotech tickers.

    In production, this would be replaced with actual data from:
    - FDA designation database
    - SEC filings parser
    - Press release NLP
    """
    return [
        # Breakthrough Therapy examples
        {"ticker": "MRNA", "designation_type": "BTD", "indication": "oncology",
         "drug_name": "mRNA-4157", "grant_date": "2023-02-01", "source": "fda", "confidence": "confirmed"},
        {"ticker": "BEAM", "designation_type": "BTD", "indication": "rare_disease",
         "drug_name": "BEAM-101", "grant_date": "2023-06-15", "source": "fda", "confidence": "confirmed"},
        {"ticker": "EDIT", "designation_type": "BTD", "indication": "rare_disease",
         "drug_name": "EDIT-101", "grant_date": "2022-11-01", "source": "fda", "confidence": "confirmed"},

        # Orphan Drug examples
        {"ticker": "RARE", "designation_type": "ODD", "indication": "rare_disease",
         "drug_name": "RARx-01", "grant_date": "2024-01-15", "source": "fda", "confidence": "confirmed"},
        {"ticker": "FOLD", "designation_type": "ODD", "indication": "rare_disease",
         "drug_name": "Folotyn", "grant_date": "2020-03-01", "source": "fda", "confidence": "confirmed"},
        {"ticker": "IONS", "designation_type": "ODD", "indication": "neurology",
         "drug_name": "IONIS-HTTRx", "grant_date": "2021-08-01", "source": "fda", "confidence": "confirmed"},

        # Fast Track examples
        {"ticker": "NVAX", "designation_type": "FT", "indication": "infectious_disease",
         "drug_name": "NVX-CoV2373", "grant_date": "2020-11-01", "source": "fda", "confidence": "confirmed"},
        {"ticker": "SIGA", "designation_type": "FT", "indication": "infectious_disease",
         "drug_name": "TPOXX", "grant_date": "2018-05-01", "source": "fda", "confidence": "confirmed"},

        # RMAT examples (regenerative medicine)
        {"ticker": "BLUE", "designation_type": "RMAT", "indication": "rare_disease",
         "drug_name": "LentiGlobin", "grant_date": "2019-04-01", "source": "fda", "confidence": "confirmed"},
        {"ticker": "CRSP", "designation_type": "RMAT", "indication": "rare_disease",
         "drug_name": "CTX001", "grant_date": "2020-02-01", "source": "fda", "confidence": "confirmed"},

        # Multiple designations for same company
        {"ticker": "MRNA", "designation_type": "FT", "indication": "infectious_disease",
         "drug_name": "mRNA-1273", "grant_date": "2020-05-01", "source": "fda", "confidence": "confirmed"},
        {"ticker": "BEAM", "designation_type": "ODD", "indication": "rare_disease",
         "drug_name": "BEAM-101", "grant_date": "2023-04-01", "source": "fda", "confidence": "confirmed"},
    ]


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("FDA DESIGNATION ENGINE v1.0.0 - DEMONSTRATION")
    print("=" * 70)

    engine = FDADesignationEngine()

    # Load sample data
    sample_data = generate_sample_designations()
    loaded = engine.load_designations(sample_data)
    print(f"\nLoaded {loaded} designations")

    # Score some tickers
    test_tickers = ["MRNA", "BEAM", "FOLD", "NVAX", "UNKNOWN"]
    as_of = date(2026, 1, 26)

    print(f"\nScoring tickers as of {as_of}:")
    print("-" * 70)

    for ticker in test_tickers:
        result = engine.score_ticker(ticker, as_of, base_pos=Decimal("50"))
        print(f"\n{ticker}:")
        print(f"  Designation Score: {result['designation_score']}")
        print(f"  PoS Multiplier: {result['pos_multiplier']}")
        print(f"  Adjusted PoS: {result['adjusted_pos']} (from base 50)")
        print(f"  Designations: {result['designation_types']}")
        print(f"  Timeline Acceleration: {result['timeline_acceleration_months']} months")
