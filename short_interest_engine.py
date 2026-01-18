#!/usr/bin/env python3
"""
short_interest_engine.py

Short Interest Signal Engine for Biotech Screener

Detects squeeze potential and crowding via short interest data. Provides
supplementary signal layer for institutional-grade screening.

Signal Components:
- Short interest % of float
- Days to cover (short interest / avg daily volume)
- Rate of change in short interest
- Institutional longs vs shorts (from 13F + short data)

Data Sources (Integration Points):
- Fintel (subscription, real-time)
- FINRA (free, 2-week lag)
- S3 Partners (premium, daily)
- Exchange short interest reports (bi-monthly)

Design Philosophy:
- Deterministic scoring with explicit thresholds
- Stdlib-only for corporate safety
- Decimal arithmetic for precision
- Full audit trail for every calculation

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import date


# Module metadata
__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class ShortInterestSignalEngine:
    """
    Short interest signal scoring for squeeze potential and crowding detection.

    Provides bullish/bearish signals based on short interest dynamics:
    - High short interest + high days-to-cover = squeeze potential (bullish)
    - Rising short interest = bearish sentiment indicator
    - Shorts covering rapidly = bullish momentum

    Usage:
        engine = ShortInterestSignalEngine()
        result = engine.calculate_short_signal(
            ticker="ACME",
            short_interest_pct=Decimal("25.5"),
            days_to_cover=Decimal("8.2"),
            short_interest_change_pct=Decimal("-15.0")
        )
        print(result["short_signal_score"])  # 65.0 (bullish squeeze potential)
    """

    VERSION = "1.0.0"

    # Squeeze potential thresholds
    SQUEEZE_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
        "extreme": {"si_pct": Decimal("40"), "dtc": Decimal("10")},
        "high": {"si_pct": Decimal("20"), "dtc": Decimal("7")},
        "moderate": {"si_pct": Decimal("10"), "dtc": Decimal("5")},
        "low": {"si_pct": Decimal("0"), "dtc": Decimal("0")}
    }

    # Crowding risk thresholds (high SI = crowded short)
    CROWDING_THRESHOLDS: Dict[str, Decimal] = {
        "high": Decimal("30"),
        "medium": Decimal("15"),
        "low": Decimal("0")
    }

    # Signal contribution weights
    SIGNAL_WEIGHTS: Dict[str, Decimal] = {
        "squeeze_potential": Decimal("0.40"),
        "trend": Decimal("0.30"),
        "institutional_support": Decimal("0.20"),
        "days_to_cover": Decimal("0.10")
    }

    # Score contributions by component
    SQUEEZE_CONTRIBUTIONS: Dict[str, Decimal] = {
        "extreme": Decimal("25"),
        "high": Decimal("15"),
        "moderate": Decimal("8"),
        "low": Decimal("0")
    }

    def __init__(self):
        """Initialize the short interest engine with empty audit trail."""
        self.audit_trail: List[Dict[str, Any]] = []

    def calculate_short_signal(
        self,
        ticker: str,
        short_interest_pct: Optional[Decimal],  # % of float shorted
        days_to_cover: Optional[Decimal],  # Short interest / avg daily volume
        short_interest_change_pct: Optional[Decimal] = None,  # Period-over-period change
        institutional_long_pct: Optional[Decimal] = None,  # From 13F analysis
        avg_daily_volume: Optional[Decimal] = None,  # For context
        as_of_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Calculate short interest signal score.

        Args:
            ticker: Stock ticker symbol
            short_interest_pct: Short interest as % of float (0-100)
            days_to_cover: Days to cover ratio (SI / ADV)
            short_interest_change_pct: Change in SI vs prior period (can be negative)
            institutional_long_pct: Institutional ownership % (from 13F)
            avg_daily_volume: Average daily volume in shares
            as_of_date: Point-in-time date for audit

        Returns:
            Dict containing:
            - status: "SUCCESS" or "INSUFFICIENT_DATA"
            - short_signal_score: Decimal 0-100
            - squeeze_potential: str ("EXTREME", "HIGH", "MODERATE", "LOW", "UNKNOWN")
            - crowding_risk: str ("HIGH", "MEDIUM", "LOW", "UNKNOWN")
            - signal_direction: str ("BULLISH", "NEUTRAL", "BEARISH")
            - component_contributions: Dict of score breakdowns
            - audit_entry: Dict of full calculation trace
        """

        # Handle insufficient data
        if short_interest_pct is None and days_to_cover is None:
            return self._insufficient_data_result(ticker, as_of_date)

        # Use defaults for missing values
        si_pct = short_interest_pct if short_interest_pct is not None else Decimal("0")
        dtc = days_to_cover if days_to_cover is not None else Decimal("0")

        # Component 1: Squeeze potential assessment
        squeeze_potential = self._assess_squeeze_potential(si_pct, dtc)
        squeeze_contrib = self.SQUEEZE_CONTRIBUTIONS.get(squeeze_potential.lower(), Decimal("0"))

        # Component 2: Crowding risk assessment
        crowding_risk = self._assess_crowding_risk(si_pct)

        # Component 3: Short interest trend contribution
        trend_contrib = self._calculate_trend_contribution(short_interest_change_pct)
        trend_direction = self._get_trend_direction(short_interest_change_pct)

        # Component 4: Institutional support contribution
        inst_contrib = self._calculate_institutional_contribution(institutional_long_pct)

        # Component 5: Days-to-cover contribution (higher = more squeeze pressure)
        dtc_contrib = self._calculate_dtc_contribution(dtc)

        # Calculate composite score (base 50 = neutral)
        base_score = Decimal("50")

        composite_score = (
            base_score +
            squeeze_contrib +
            trend_contrib +
            inst_contrib +
            dtc_contrib
        )

        # Clamp to 0-100 range
        composite_score = max(Decimal("0"), min(Decimal("100"), composite_score))
        composite_score = composite_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Determine overall signal direction
        if composite_score >= Decimal("60"):
            signal_direction = "BULLISH"
        elif composite_score <= Decimal("40"):
            signal_direction = "BEARISH"
        else:
            signal_direction = "NEUTRAL"

        # Generate flags
        flags = self._generate_flags(
            si_pct, dtc, squeeze_potential, crowding_risk, short_interest_change_pct
        )

        # Deterministic timestamp from as_of_date
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z" if as_of_date else None

        # Audit trail (deterministic!)
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "ticker": ticker,
            "input": {
                "short_interest_pct": str(si_pct),
                "days_to_cover": str(dtc),
                "short_interest_change_pct": str(short_interest_change_pct) if short_interest_change_pct else None,
                "institutional_long_pct": str(institutional_long_pct) if institutional_long_pct else None,
                "avg_daily_volume": str(avg_daily_volume) if avg_daily_volume else None
            },
            "calculation": {
                "squeeze_potential": squeeze_potential,
                "squeeze_contrib": str(squeeze_contrib),
                "crowding_risk": crowding_risk,
                "trend_contrib": str(trend_contrib),
                "trend_direction": trend_direction,
                "inst_contrib": str(inst_contrib),
                "dtc_contrib": str(dtc_contrib),
                "base_score": str(base_score),
                "composite_score": str(composite_score),
                "signal_direction": signal_direction
            },
            "flags": flags,
            "module_version": self.VERSION
        }

        self.audit_trail.append(audit_entry)

        return {
            "status": "SUCCESS",
            "ticker": ticker,
            "short_signal_score": composite_score,
            "squeeze_potential": squeeze_potential,
            "crowding_risk": crowding_risk,
            "signal_direction": signal_direction,
            "trend_direction": trend_direction,
            "component_contributions": {
                "squeeze": squeeze_contrib,
                "trend": trend_contrib,
                "institutional": inst_contrib,
                "days_to_cover": dtc_contrib
            },
            "flags": flags,
            "audit_entry": audit_entry
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        as_of_date: date
    ) -> Dict[str, Any]:
        """
        Score an entire universe of companies.

        Args:
            universe: List of dicts with keys:
                - ticker: str
                - short_interest_pct: Optional[Decimal]
                - days_to_cover: Optional[Decimal]
                - short_interest_change_pct: Optional[Decimal]
                - institutional_long_pct: Optional[Decimal]
            as_of_date: Point-in-time date

        Returns:
            Dict with:
            - as_of_date: str
            - scores: List of scored companies
            - diagnostic_counts: Dict of metrics
            - provenance: Dict with version and hash
        """

        scores = []
        squeeze_distribution: Dict[str, int] = {
            "EXTREME": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "UNKNOWN": 0
        }
        signal_distribution: Dict[str, int] = {
            "BULLISH": 0, "NEUTRAL": 0, "BEARISH": 0
        }
        data_coverage = 0

        for company in universe:
            ticker = company.get("ticker", "UNKNOWN")

            # Convert to Decimal if needed
            si_pct = self._to_decimal(company.get("short_interest_pct"))
            dtc = self._to_decimal(company.get("days_to_cover"))
            si_change = self._to_decimal(company.get("short_interest_change_pct"))
            inst_long = self._to_decimal(company.get("institutional_long_pct"))
            adv = self._to_decimal(company.get("avg_daily_volume"))

            result = self.calculate_short_signal(
                ticker=ticker,
                short_interest_pct=si_pct,
                days_to_cover=dtc,
                short_interest_change_pct=si_change,
                institutional_long_pct=inst_long,
                avg_daily_volume=adv,
                as_of_date=as_of_date
            )

            scores.append({
                "ticker": ticker,
                "short_signal_score": result["short_signal_score"],
                "squeeze_potential": result["squeeze_potential"],
                "crowding_risk": result["crowding_risk"],
                "signal_direction": result["signal_direction"],
                "flags": result.get("flags", [])
            })

            # Track metrics
            squeeze_distribution[result["squeeze_potential"]] += 1
            if result["status"] == "SUCCESS":
                signal_distribution[result["signal_direction"]] += 1
                data_coverage += 1

        # Calculate content hash
        scores_json = json.dumps(
            [{"t": s["ticker"], "s": str(s["short_signal_score"])} for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "data_coverage": data_coverage,
                "data_coverage_pct": f"{data_coverage / max(1, len(scores)) * 100:.1f}%",
                "squeeze_distribution": squeeze_distribution,
                "signal_distribution": signal_distribution
            },
            "provenance": {
                "module": "short_interest_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat()
            }
        }

    def _insufficient_data_result(
        self,
        ticker: str,
        as_of_date: Optional[date]
    ) -> Dict[str, Any]:
        """Return standardized result when data is insufficient."""
        # Deterministic timestamp
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z" if as_of_date else None
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "ticker": ticker,
            "status": "INSUFFICIENT_DATA",
            "module_version": self.VERSION
        }
        self.audit_trail.append(audit_entry)

        return {
            "status": "INSUFFICIENT_DATA",
            "ticker": ticker,
            "short_signal_score": Decimal("50"),  # Neutral default
            "squeeze_potential": "UNKNOWN",
            "crowding_risk": "UNKNOWN",
            "signal_direction": "NEUTRAL",
            "trend_direction": "UNKNOWN",
            "component_contributions": {},
            "flags": ["SI_DATA_MISSING"],
            "audit_entry": audit_entry
        }

    def _assess_squeeze_potential(
        self,
        si_pct: Decimal,
        dtc: Decimal
    ) -> str:
        """Assess squeeze potential category based on SI% and days-to-cover."""
        # Check thresholds from highest to lowest
        if (si_pct >= self.SQUEEZE_THRESHOLDS["extreme"]["si_pct"] and
            dtc >= self.SQUEEZE_THRESHOLDS["extreme"]["dtc"]):
            return "EXTREME"
        elif (si_pct >= self.SQUEEZE_THRESHOLDS["high"]["si_pct"] and
              dtc >= self.SQUEEZE_THRESHOLDS["high"]["dtc"]):
            return "HIGH"
        elif (si_pct >= self.SQUEEZE_THRESHOLDS["moderate"]["si_pct"] and
              dtc >= self.SQUEEZE_THRESHOLDS["moderate"]["dtc"]):
            return "MODERATE"
        else:
            return "LOW"

    def _assess_crowding_risk(self, si_pct: Decimal) -> str:
        """Assess crowding risk based on short interest percentage."""
        if si_pct >= self.CROWDING_THRESHOLDS["high"]:
            return "HIGH"
        elif si_pct >= self.CROWDING_THRESHOLDS["medium"]:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_trend_contribution(
        self,
        si_change_pct: Optional[Decimal]
    ) -> Decimal:
        """
        Calculate trend contribution to score.

        Shorts covering (negative change) = bullish = positive contribution
        Shorts increasing (positive change) = bearish = negative contribution
        """
        if si_change_pct is None:
            return Decimal("0")

        # Shorts covering rapidly = bullish (positive contribution)
        if si_change_pct <= Decimal("-20"):
            return Decimal("15")  # Strong covering
        elif si_change_pct <= Decimal("-10"):
            return Decimal("8")   # Moderate covering
        elif si_change_pct <= Decimal("-5"):
            return Decimal("4")   # Light covering
        # Shorts increasing = bearish (negative contribution)
        elif si_change_pct >= Decimal("20"):
            return Decimal("-12")  # Strong buildup
        elif si_change_pct >= Decimal("10"):
            return Decimal("-6")   # Moderate buildup
        elif si_change_pct >= Decimal("5"):
            return Decimal("-3")   # Light buildup
        else:
            return Decimal("0")    # Stable

    def _get_trend_direction(
        self,
        si_change_pct: Optional[Decimal]
    ) -> str:
        """Determine trend direction from SI change."""
        if si_change_pct is None:
            return "UNKNOWN"
        elif si_change_pct <= Decimal("-5"):
            return "COVERING"
        elif si_change_pct >= Decimal("5"):
            return "BUILDING"
        else:
            return "STABLE"

    def _calculate_institutional_contribution(
        self,
        inst_long_pct: Optional[Decimal]
    ) -> Decimal:
        """
        Calculate institutional support contribution.

        Higher institutional ownership = more support = positive contribution
        """
        if inst_long_pct is None:
            return Decimal("0")

        if inst_long_pct >= Decimal("70"):
            return Decimal("10")  # Very strong institutional support
        elif inst_long_pct >= Decimal("50"):
            return Decimal("6")   # Strong support
        elif inst_long_pct >= Decimal("30"):
            return Decimal("3")   # Moderate support
        else:
            return Decimal("0")   # Low support

    def _calculate_dtc_contribution(self, dtc: Decimal) -> Decimal:
        """
        Calculate days-to-cover contribution.

        Higher DTC = more squeeze pressure = positive contribution (bullish)
        """
        if dtc >= Decimal("15"):
            return Decimal("8")   # Extreme squeeze pressure
        elif dtc >= Decimal("10"):
            return Decimal("5")   # High squeeze pressure
        elif dtc >= Decimal("7"):
            return Decimal("3")   # Moderate squeeze pressure
        elif dtc >= Decimal("5"):
            return Decimal("1")   # Light squeeze pressure
        else:
            return Decimal("0")   # No significant pressure

    def _generate_flags(
        self,
        si_pct: Decimal,
        dtc: Decimal,
        squeeze_potential: str,
        crowding_risk: str,
        si_change_pct: Optional[Decimal]
    ) -> List[str]:
        """Generate warning/info flags based on analysis."""
        flags = []

        # Squeeze-related flags
        if squeeze_potential == "EXTREME":
            flags.append("EXTREME_SQUEEZE_POTENTIAL")
        elif squeeze_potential == "HIGH":
            flags.append("HIGH_SQUEEZE_POTENTIAL")

        # Crowding flags
        if crowding_risk == "HIGH":
            flags.append("HIGH_SHORT_CROWDING")

        # Specific condition flags
        if si_pct >= Decimal("50"):
            flags.append("MAJORITY_FLOAT_SHORTED")
        if dtc >= Decimal("15"):
            flags.append("EXTENDED_DAYS_TO_COVER")

        # Trend flags
        if si_change_pct is not None:
            if si_change_pct <= Decimal("-25"):
                flags.append("RAPID_SHORT_COVERING")
            elif si_change_pct >= Decimal("25"):
                flags.append("RAPID_SHORT_BUILDUP")

        return flags

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        """Safely convert value to Decimal."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return None

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the full audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


def demonstration() -> None:
    """Demonstrate the short interest engine capabilities."""
    print("=" * 70)
    print("SHORT INTEREST SIGNAL ENGINE - DEMONSTRATION")
    print("=" * 70)
    print()

    engine = ShortInterestSignalEngine()

    # Example 1: High squeeze potential
    print("Example 1: High Squeeze Potential Stock")
    print("-" * 70)

    result1 = engine.calculate_short_signal(
        ticker="SQUEEZE",
        short_interest_pct=Decimal("35.5"),
        days_to_cover=Decimal("12.3"),
        short_interest_change_pct=Decimal("-18.0"),
        institutional_long_pct=Decimal("55.0")
    )

    print(f"Ticker: {result1['ticker']}")
    print(f"Signal Score: {result1['short_signal_score']}")
    print(f"Squeeze Potential: {result1['squeeze_potential']}")
    print(f"Crowding Risk: {result1['crowding_risk']}")
    print(f"Signal Direction: {result1['signal_direction']}")
    print(f"Trend: {result1['trend_direction']}")
    print(f"Flags: {result1['flags']}")
    print()

    # Example 2: Bearish short buildup
    print("Example 2: Bearish Short Buildup")
    print("-" * 70)

    result2 = engine.calculate_short_signal(
        ticker="BEARISH",
        short_interest_pct=Decimal("15.0"),
        days_to_cover=Decimal("4.5"),
        short_interest_change_pct=Decimal("22.0"),
        institutional_long_pct=Decimal("25.0")
    )

    print(f"Ticker: {result2['ticker']}")
    print(f"Signal Score: {result2['short_signal_score']}")
    print(f"Squeeze Potential: {result2['squeeze_potential']}")
    print(f"Signal Direction: {result2['signal_direction']}")
    print(f"Trend: {result2['trend_direction']}")
    print()

    # Example 3: Low short interest (neutral)
    print("Example 3: Low Short Interest (Neutral)")
    print("-" * 70)

    result3 = engine.calculate_short_signal(
        ticker="NEUTRAL",
        short_interest_pct=Decimal("3.5"),
        days_to_cover=Decimal("2.1"),
        short_interest_change_pct=Decimal("1.0")
    )

    print(f"Ticker: {result3['ticker']}")
    print(f"Signal Score: {result3['short_signal_score']}")
    print(f"Signal Direction: {result3['signal_direction']}")
    print()


if __name__ == "__main__":
    demonstration()
