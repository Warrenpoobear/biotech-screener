#!/usr/bin/env python3
"""
competitive_pressure_engine.py

Competitive Pressure Engine for Biotech Screener

Wraps competitive landscape analysis from accuracy_improvements.py
into a standardized engine interface for pipeline integration.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import date
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from common.accuracy_improvements import (
    compute_competition_penalty,
    CompetitiveLandscapeResult,
)

__version__ = "1.0.0"


@dataclass
class TickerCompetitiveScore:
    """Competitive pressure score for a ticker."""
    ticker: str
    competitive_pressure_score: Decimal  # 0-100, higher=less competition
    competitor_count: int
    competition_level: str
    penalty_applied: Decimal
    market_share_estimate: Decimal
    is_first_in_class: bool
    confidence: str
    flags: List[str] = field(default_factory=list)


class CompetitivePressureEngine:
    """
    Engine for scoring competitive pressure/crowding risk.

    Usage:
        engine = CompetitivePressureEngine()
        result = engine.calculate_competitive_score(
            ticker="ACME",
            indication="oncology",
            phase="phase 3",
            competitor_programs=[...],
            as_of_date=date(2026, 1, 15)
        )
    """

    VERSION = "1.0.0"

    def __init__(self):
        """Initialize the engine."""
        self.audit_trail: List[Dict[str, Any]] = []

    def calculate_competitive_score(
        self,
        ticker: str,
        indication: str,
        phase: str,
        competitor_programs: List[Dict[str, Any]],
        is_first_in_class: bool = False,
        as_of_date: Union[str, date] = None,
    ) -> TickerCompetitiveScore:
        """
        Calculate competitive pressure score for a ticker.

        Args:
            ticker: Stock ticker
            indication: Target indication
            phase: Current development phase
            competitor_programs: List of competitor program records
            is_first_in_class: Whether this is a novel mechanism
            as_of_date: Analysis date

        Returns:
            TickerCompetitiveScore with competitive assessment
        """
        if as_of_date is None:
            raise ValueError("as_of_date is required for PIT discipline")

        if isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        # Use existing competition penalty calculation
        result = compute_competition_penalty(
            indication=indication,
            phase=phase,
            competitor_programs=competitor_programs,
            is_first_in_class=is_first_in_class,
        )

        # Convert penalty to score (100 = no competition, 0 = max competition)
        # Penalty ranges from 0 to 20, so score = 100 - (penalty * 5) to spread the range
        score = Decimal("100") - (result.penalty_points * Decimal("5"))
        score = max(Decimal("0"), min(Decimal("100"), score))

        # Determine confidence based on data quality
        if competitor_programs and len(competitor_programs) > 0:
            confidence = "high"
        elif indication and phase:
            confidence = "medium"
        else:
            confidence = "low"

        flags = []
        if result.competition_level == "hyper_competitive":
            flags.append("HYPER_COMPETITIVE_MARKET")
        if result.is_first_in_class:
            flags.append("FIRST_IN_CLASS")
        if result.competitor_count == 0:
            flags.append("NO_KNOWN_COMPETITORS")

        ticker_score = TickerCompetitiveScore(
            ticker=ticker,
            competitive_pressure_score=score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            competitor_count=result.competitor_count,
            competition_level=result.competition_level,
            penalty_applied=result.penalty_points,
            market_share_estimate=result.market_share_estimate,
            is_first_in_class=result.is_first_in_class,
            confidence=confidence,
            flags=flags,
        )

        # Add to audit trail
        self.audit_trail.append({
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "competitive_pressure_score": str(ticker_score.competitive_pressure_score),
            "competitor_count": result.competitor_count,
            "competition_level": result.competition_level,
            "confidence": confidence,
        })

        return ticker_score

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        competitor_data_by_indication: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        as_of_date: Union[str, date] = None,
    ) -> Dict[str, Any]:
        """
        Score competitive pressure for entire universe.

        Args:
            universe: List of company dicts with ticker, indication, phase
            competitor_data_by_indication: {indication: [competitor_programs]}
            as_of_date: Analysis date

        Returns:
            Dict with scores and diagnostics
        """
        if as_of_date is None:
            raise ValueError("as_of_date is required")

        if isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        competitor_data_by_indication = competitor_data_by_indication or {}

        scores = []
        for company in universe:
            ticker = company.get("ticker")
            if not ticker:
                continue

            indication = company.get("indication", "unknown")
            phase = company.get("phase", "phase 2")
            is_fic = company.get("is_first_in_class", False)

            # Get competitors for this indication
            competitors = competitor_data_by_indication.get(indication, [])

            result = self.calculate_competitive_score(
                ticker=ticker,
                indication=indication,
                phase=phase,
                competitor_programs=competitors,
                is_first_in_class=is_fic,
                as_of_date=as_of_date,
            )

            scores.append({
                "ticker": ticker,
                "competitive_pressure_score": str(result.competitive_pressure_score),
                "competitor_count": result.competitor_count,
                "competition_level": result.competition_level,
                "confidence": result.confidence,
                "flags": result.flags,
            })

        # Diagnostics
        total = len(scores)
        hyper_competitive = sum(1 for s in scores if s["competition_level"] == "hyper_competitive")
        first_in_class = sum(1 for s in scores if "FIRST_IN_CLASS" in s["flags"])

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": total,
                "hyper_competitive": hyper_competitive,
                "first_in_class": first_in_class,
            },
            "provenance": {
                "engine": "CompetitivePressureEngine",
                "version": self.VERSION,
            },
        }

    def get_diagnostic_counts(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        if not self.audit_trail:
            return {"total": 0}

        total = len(self.audit_trail)
        high_competition = sum(
            1 for a in self.audit_trail
            if a.get("competition_level") in ("high", "hyper_competitive")
        )

        return {
            "total": total,
            "high_competition_pct": round(high_competition / total * 100, 1) if total else 0,
        }


# =============================================================================
# SELF-CHECKS
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify engine correctness."""
    errors = []

    engine = CompetitivePressureEngine()

    # CHECK 1: No competitors = high score
    result1 = engine.calculate_competitive_score(
        ticker="TEST1",
        indication="rare_disease",
        phase="phase 3",
        competitor_programs=[],
        as_of_date=date(2026, 1, 15),
    )

    if result1.competitive_pressure_score < Decimal("90"):
        errors.append(f"CHECK1 FAIL: No competitors should have high score, got {result1.competitive_pressure_score}")

    # CHECK 2: Many competitors = low score
    competitors = [{"phase": "phase 3"} for _ in range(15)]
    result2 = engine.calculate_competitive_score(
        ticker="TEST2",
        indication="oncology",
        phase="phase 3",
        competitor_programs=competitors,
        as_of_date=date(2026, 1, 15),
    )

    if result2.competition_level != "hyper_competitive":
        errors.append(f"CHECK2 FAIL: 15 competitors should be hyper_competitive, got {result2.competition_level}")

    # CHECK 3: Score bounds
    if result2.competitive_pressure_score < Decimal("0") or result2.competitive_pressure_score > Decimal("100"):
        errors.append(f"CHECK3 FAIL: Score out of bounds: {result2.competitive_pressure_score}")

    return errors


if __name__ == "__main__":
    errors = _run_self_checks()
    if errors:
        print("SELF-CHECK FAILURES:")
        for e in errors:
            print(f"  {e}")
        exit(1)
    else:
        print("All self-checks passed!")
        exit(0)
