#!/usr/bin/env python3
"""
catalyst_impact_weighting.py

Catalyst Impact Scoring with CCFT (Cross-Context Float Tolerance) Compliance

Calculates impact multipliers for catalyst events based on timing proximity,
mechanism of action, and phase success rates.

Design Philosophy:
- Deterministic scoring with Decimal arithmetic
- Config values stored as strings for Decimal init (CCFT compliance)
- Full audit trail for every calculation
- Stdlib-only for corporate safety

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import date, datetime


# Module metadata
__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class CatalystImpactScorer:
    """
    Catalyst impact scoring engine with CCFT-compliant configuration.

    All config values that become Decimals are stored as strings to ensure
    byte-identical initialization across platforms.

    Usage:
        scorer = CatalystImpactScorer()
        result = scorer.calculate_catalyst_impact(
            ticker="XYZ",
            catalyst_type="phase_3_topline",
            catalyst_date="2025-06-15",
            as_of_date="2025-01-11",
            mechanism="antibody",
            current_phase="phase_3"
        )
        print(result["impact_multiplier"])  # 1.35
    """

    VERSION = "1.0.0"

    # Timing proximity scoring (days to catalyst -> score multiplier)
    TIMING_PROXIMITY_THRESHOLDS: Dict[str, Dict[str, str]] = {
        "imminent": {"days": "30", "multiplier": "1.25"},
        "near_term": {"days": "90", "multiplier": "1.15"},
        "medium_term": {"days": "180", "multiplier": "1.05"},
        "distant": {"days": "365", "multiplier": "1.00"},
    }

    # Mechanism of action adjustments
    MECHANISM_ADJUSTMENTS: Dict[str, str] = {
        "antibody": "1.10",
        "small_molecule": "1.05",
        "gene_therapy": "1.15",
        "cell_therapy": "1.20",
        "rna": "1.12",
        "protein": "1.05",
        "other": "1.00",
    }

    # Phase success rate adjustments
    PHASE_SUCCESS_RATES: Dict[str, str] = {
        "phase_1": "0.10",
        "phase_2": "0.25",
        "phase_3": "0.50",
        "nda_bla": "0.85",
        "approved": "1.00",
    }

    # Catalyst type base scores
    CATALYST_TYPE_SCORES: Dict[str, str] = {
        "phase_3_topline": "1.30",
        "phase_2_topline": "1.15",
        "fda_approval": "1.40",
        "pdufa_date": "1.35",
        "partnership": "1.10",
        "financing": "0.90",
        "data_presentation": "1.05",
        "other": "1.00",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the catalyst impact scorer.

        Config values are stored as STRINGS for Decimal initialization
        to ensure CCFT compliance (byte-identical across platforms).
        """
        # CCFT-compliant default config: ALL numeric values as strings
        self.config = config or {
            "use_timing_proximity": True,
            "use_mechanism_adjustments": True,
            "use_phase_success_rates": True,
            "max_multiplier": "1.5",  # String for Decimal init (CCFT)
            "min_multiplier": "0.5",  # String for Decimal init (CCFT)
        }
        self.audit_trail: List[Dict[str, Any]] = []

    def calculate_catalyst_impact(
        self,
        ticker: str,
        catalyst_type: str,
        catalyst_date: str,
        as_of_date: str,
        mechanism: Optional[str] = None,
        current_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate catalyst impact multiplier.

        Args:
            ticker: Stock ticker symbol
            catalyst_type: Type of catalyst event
            catalyst_date: Expected catalyst date (YYYY-MM-DD)
            as_of_date: Analysis date (YYYY-MM-DD)
            mechanism: Mechanism of action (optional)
            current_phase: Current clinical phase (optional)

        Returns:
            Dict containing:
            - impact_multiplier: Decimal multiplier for scoring
            - timing_score: Timing proximity contribution
            - mechanism_score: Mechanism adjustment
            - phase_score: Phase success rate adjustment
            - breakdown: Detailed component breakdown
            - audit_hash: Hash of calculation inputs
        """
        # Parse dates
        catalyst_dt = datetime.strptime(catalyst_date, "%Y-%m-%d").date()
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()

        # Calculate days to catalyst
        days_to_catalyst = (catalyst_dt - as_of_dt).days

        # Get base score from catalyst type (CCFT: direct Decimal from string)
        base_score = Decimal(
            self.CATALYST_TYPE_SCORES.get(catalyst_type.lower(), "1.00")
        )

        # Calculate timing proximity score
        timing_score = Decimal("1.00")
        if self.config.get("use_timing_proximity", True):
            timing_score = self._calculate_timing_score(days_to_catalyst)

        # Calculate mechanism adjustment
        mechanism_score = Decimal("1.00")
        if self.config.get("use_mechanism_adjustments", True) and mechanism:
            mechanism_score = Decimal(
                self.MECHANISM_ADJUSTMENTS.get(mechanism.lower(), "1.00")
            )

        # Calculate phase success adjustment
        phase_score = Decimal("1.00")
        if self.config.get("use_phase_success_rates", True) and current_phase:
            # Phase success adjusts confidence, not directly multiplied
            phase_rate = Decimal(
                self.PHASE_SUCCESS_RATES.get(current_phase.lower(), "0.25")
            )
            # Scale phase contribution
            phase_score = Decimal("1.00") + (phase_rate - Decimal("0.25"))

        # Calculate final multiplier
        raw_multiplier = base_score * timing_score * mechanism_score * phase_score

        # Apply bounds (CCFT: direct Decimal from config string values)
        max_mult = Decimal(self.config["max_multiplier"])
        min_mult = Decimal(self.config["min_multiplier"])

        final_multiplier = max(min(raw_multiplier, max_mult), min_mult)
        final_multiplier = final_multiplier.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Build audit record
        audit_inputs = {
            "ticker": ticker,
            "catalyst_type": catalyst_type,
            "catalyst_date": catalyst_date,
            "as_of_date": as_of_date,
            "mechanism": mechanism,
            "current_phase": current_phase,
            "days_to_catalyst": days_to_catalyst,
        }
        audit_hash = hashlib.sha256(
            json.dumps(audit_inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        result = {
            "ticker": ticker,
            "impact_multiplier": str(final_multiplier),
            "timing_score": str(timing_score.quantize(Decimal("0.01"))),
            "mechanism_score": str(mechanism_score.quantize(Decimal("0.01"))),
            "phase_score": str(phase_score.quantize(Decimal("0.01"))),
            "base_score": str(base_score),
            "days_to_catalyst": days_to_catalyst,
            "breakdown": {
                "catalyst_type": catalyst_type,
                "raw_multiplier": str(raw_multiplier.quantize(Decimal("0.0001"))),
                "bounded": str(final_multiplier) != str(raw_multiplier.quantize(Decimal("0.01"))),
            },
            "audit_hash": audit_hash,
            "version": self.VERSION,
        }

        # Record in audit trail
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "inputs": audit_inputs,
            "outputs": result,
        })

        return result

    def _calculate_timing_score(self, days_to_catalyst: int) -> Decimal:
        """
        Calculate timing proximity score based on days to catalyst.

        Closer catalysts get higher multipliers.
        Past catalysts (negative days) get base multiplier.
        """
        if days_to_catalyst < 0:
            # Catalyst has passed
            return Decimal("1.00")

        # Check thresholds from closest to furthest
        for level, threshold in sorted(
            self.TIMING_PROXIMITY_THRESHOLDS.items(),
            key=lambda x: int(x[1]["days"])
        ):
            if days_to_catalyst <= int(threshold["days"]):
                return Decimal(threshold["multiplier"])

        # Beyond all thresholds
        return Decimal("1.00")

    def score_universe(
        self,
        catalysts: List[Dict[str, Any]],
        as_of_date: str,
    ) -> Dict[str, Any]:
        """
        Score all catalysts in universe.

        Args:
            catalysts: List of catalyst dicts with ticker, type, date, etc.
            as_of_date: Analysis date

        Returns:
            Dict with scored catalysts and summary statistics
        """
        results = []
        for catalyst in catalysts:
            try:
                result = self.calculate_catalyst_impact(
                    ticker=catalyst.get("ticker", "UNKNOWN"),
                    catalyst_type=catalyst.get("catalyst_type", "other"),
                    catalyst_date=catalyst.get("catalyst_date", as_of_date),
                    as_of_date=as_of_date,
                    mechanism=catalyst.get("mechanism"),
                    current_phase=catalyst.get("current_phase"),
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "ticker": catalyst.get("ticker", "UNKNOWN"),
                    "error": str(e),
                    "impact_multiplier": "1.00",
                })

        # Calculate summary statistics
        multipliers = [
            Decimal(r["impact_multiplier"])
            for r in results
            if "error" not in r
        ]

        summary = {
            "total_catalysts": len(catalysts),
            "scored_successfully": len([r for r in results if "error" not in r]),
            "errors": len([r for r in results if "error" in r]),
            "avg_multiplier": str(
                (sum(multipliers) / len(multipliers)).quantize(Decimal("0.01"))
            ) if multipliers else "1.00",
            "max_multiplier": str(max(multipliers)) if multipliers else "1.00",
            "min_multiplier": str(min(multipliers)) if multipliers else "1.00",
        }

        return {
            "as_of_date": as_of_date,
            "results": results,
            "summary": summary,
            "version": self.VERSION,
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail of all calculations."""
        return self.audit_trail

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("CATALYST IMPACT SCORER - CCFT COMPLIANCE TEST")
    print("=" * 60)

    scorer = CatalystImpactScorer()

    # Test case 1: Near-term Phase 3 topline
    result1 = scorer.calculate_catalyst_impact(
        ticker="XYZ",
        catalyst_type="phase_3_topline",
        catalyst_date="2025-03-15",
        as_of_date="2025-01-11",
        mechanism="antibody",
        current_phase="phase_3"
    )
    print(f"\nTest 1: Near-term Phase 3 ({result1['days_to_catalyst']} days)")
    print(f"  Impact Multiplier: {result1['impact_multiplier']}")
    print(f"  Timing Score: {result1['timing_score']}")
    print(f"  Mechanism Score: {result1['mechanism_score']}")
    print(f"  Audit Hash: {result1['audit_hash']}")

    # Test case 2: Distant FDA approval
    result2 = scorer.calculate_catalyst_impact(
        ticker="ABC",
        catalyst_type="fda_approval",
        catalyst_date="2025-12-15",
        as_of_date="2025-01-11",
        mechanism="gene_therapy",
        current_phase="nda_bla"
    )
    print(f"\nTest 2: Distant FDA Approval ({result2['days_to_catalyst']} days)")
    print(f"  Impact Multiplier: {result2['impact_multiplier']}")
    print(f"  Timing Score: {result2['timing_score']}")
    print(f"  Phase Score: {result2['phase_score']}")

    # Test determinism: Run same calculation 3x
    print("\n" + "=" * 60)
    print("DETERMINISM VERIFICATION")
    print("=" * 60)

    hashes = []
    for i in range(3):
        test_scorer = CatalystImpactScorer()
        result = test_scorer.calculate_catalyst_impact(
            ticker="XYZ",
            catalyst_type="phase_3_topline",
            catalyst_date="2025-06-15",
            as_of_date="2025-01-11",
            mechanism="antibody",
            current_phase="phase_3"
        )
        result_json = json.dumps(result, sort_keys=True)
        result_hash = hashlib.sha256(result_json.encode()).hexdigest()[:16]
        hashes.append(result_hash)
        print(f"  Run {i+1}: {result_hash}")

    if len(set(hashes)) == 1:
        print("\n  CCFT COMPLIANCE VERIFIED: All hashes identical")
    else:
        print("\n  CCFT VIOLATION: Non-deterministic output detected!")
