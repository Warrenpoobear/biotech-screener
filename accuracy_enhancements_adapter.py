#!/usr/bin/env python3
"""
accuracy_enhancements_adapter.py

Adapter to integrate accuracy improvements into the screening pipeline.

Connects the 8 accuracy improvement functions from common/accuracy_improvements.py
into Module 5 composite scoring for enhanced IC performance.

Improvements Applied:
1. Indication-specific endpoint weighting
2. Phase-dependent staleness thresholds
3. Regulatory pathway scoring
4. Regime-adaptive catalyst decay
5. Competitive landscape penalty
6. Dynamic dilution (VIX-adjusted)
7. Burn seasonality adjustment
8. Binary event proximity boost

Design Philosophy:
- Deterministic: No datetime.now(), timestamps from as_of_date
- Fail-closed: Missing data = no adjustment (not penalty)
- Auditable: Every adjustment tracked with confidence
- Stdlib-only: No external dependencies

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import date
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from common.accuracy_improvements import (
    apply_all_accuracy_improvements,
    MarketRegimeType,
    classify_therapeutic_area,
    compute_regulatory_pathway_score,
)

__version__ = "1.0.0"


@dataclass
class AccuracyAdjustment:
    """Result of accuracy enhancement calculations."""
    ticker: str
    clinical_adjustment: Decimal  # Multiplier for clinical score
    financial_adjustment: Decimal  # Multiplier for financial score
    catalyst_adjustment: Decimal  # Multiplier for catalyst score
    regulatory_bonus: Decimal  # Additive bonus for regulatory designations
    confidence: str  # "high", "medium", "low"
    adjustments_applied: List[str] = field(default_factory=list)
    audit_details: Dict[str, Any] = field(default_factory=dict)


class AccuracyEnhancementsAdapter:
    """
    Adapter for integrating accuracy improvements into the scoring pipeline.

    Usage:
        adapter = AccuracyEnhancementsAdapter()
        adjustment = adapter.compute_adjustments(
            ticker="ACME",
            trial_data=trial_record,
            financial_data=financial_record,
            market_data=market_data,
            as_of_date=date(2026, 1, 15),
            vix_current=Decimal("18.5"),
            market_regime="BULL"
        )

        # Apply to scores
        adjusted_clinical = clinical_score * adjustment.clinical_adjustment
        adjusted_financial = financial_score * adjustment.financial_adjustment
        adjusted_catalyst = catalyst_score * adjustment.catalyst_adjustment
    """

    VERSION = "1.0.0"

    # Adjustment bounds (prevent extreme moves)
    MIN_MULTIPLIER = Decimal("0.70")
    MAX_MULTIPLIER = Decimal("1.30")
    MAX_REGULATORY_BONUS = Decimal("15")

    def __init__(self, enable_staleness: bool = True, enable_regulatory: bool = True,
                 enable_vix_adjustment: bool = True, enable_seasonality: bool = True,
                 enable_proximity_boost: bool = True):
        """
        Initialize adapter with feature flags.

        Args:
            enable_staleness: Apply phase-dependent staleness penalties
            enable_regulatory: Apply regulatory pathway bonuses
            enable_vix_adjustment: Apply VIX-based dilution adjustment
            enable_seasonality: Apply burn seasonality adjustment
            enable_proximity_boost: Apply catalyst proximity boost
        """
        self.enable_staleness = enable_staleness
        self.enable_regulatory = enable_regulatory
        self.enable_vix_adjustment = enable_vix_adjustment
        self.enable_seasonality = enable_seasonality
        self.enable_proximity_boost = enable_proximity_boost
        self.audit_trail: List[Dict[str, Any]] = []

    def _normalize_regime(self, regime: Optional[str]) -> Optional[MarketRegimeType]:
        """Convert regime string to enum."""
        if regime is None:
            return None
        regime_map = {
            "BULL": MarketRegimeType.BULL,
            "BEAR": MarketRegimeType.BEAR,
            "NEUTRAL": MarketRegimeType.UNKNOWN,  # Map NEUTRAL to UNKNOWN
            "UNKNOWN": MarketRegimeType.UNKNOWN,
            "SECTOR_ROTATION": MarketRegimeType.SECTOR_ROTATION,
            "VOLATILITY_SPIKE": MarketRegimeType.VOLATILITY_SPIKE,
        }
        return regime_map.get(regime.upper(), MarketRegimeType.UNKNOWN)

    def compute_adjustments(
        self,
        ticker: str,
        trial_data: Dict[str, Any],
        financial_data: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        as_of_date: Union[str, date] = None,
        vix_current: Optional[Decimal] = None,
        market_regime: Optional[str] = None,
        competitor_programs: Optional[List[Dict[str, Any]]] = None,
    ) -> AccuracyAdjustment:
        """
        Compute accuracy adjustments for a single ticker.

        Args:
            ticker: Stock ticker
            trial_data: Dict with clinical trial fields
            financial_data: Dict with financial fields
            market_data: Optional market data dict
            as_of_date: Analysis date
            vix_current: Current VIX level
            market_regime: Market regime string ("BULL", "BEAR", etc.)
            competitor_programs: List of competitor program dicts

        Returns:
            AccuracyAdjustment with multipliers and audit info
        """
        if as_of_date is None:
            raise ValueError("as_of_date is required for PIT discipline")

        if isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        market_data = market_data or {}
        regime_enum = self._normalize_regime(market_regime)

        # Get raw accuracy improvement results
        raw_results = apply_all_accuracy_improvements(
            ticker=ticker,
            trial_data=trial_data,
            financial_data=financial_data,
            market_data=market_data,
            as_of_date=as_of_date,
            vix_current=vix_current,
            market_regime=regime_enum,
            competitor_programs=competitor_programs,
        )

        # Initialize adjustments
        clinical_mult = Decimal("1.00")
        financial_mult = Decimal("1.00")
        catalyst_mult = Decimal("1.00")
        regulatory_bonus = Decimal("0")
        adjustments_applied = []
        confidence = "medium"

        # 1. Apply staleness penalty to clinical score
        if self.enable_staleness and "staleness_analysis" in raw_results:
            staleness = raw_results["staleness_analysis"]
            if staleness.get("is_stale"):
                penalty = Decimal(staleness.get("staleness_penalty", "1.00"))
                clinical_mult = clinical_mult * penalty
                adjustments_applied.append(f"staleness_penalty:{penalty}")

        # 2. Apply regulatory bonus
        if self.enable_regulatory and "regulatory_analysis" in raw_results:
            reg = raw_results["regulatory_analysis"]
            modifier = Decimal(reg.get("total_modifier", "0"))
            if modifier != Decimal("0"):
                # Cap regulatory bonus
                regulatory_bonus = min(modifier, self.MAX_REGULATORY_BONUS)
                regulatory_bonus = max(regulatory_bonus, -Decimal("10"))
                adjustments_applied.append(f"regulatory_bonus:{regulatory_bonus}")

                # Expedited pathway = higher confidence
                if reg.get("is_expedited"):
                    confidence = "high"

        # 3. Apply VIX adjustment to financial score
        if self.enable_vix_adjustment and "vix_adjustment" in raw_results:
            vix = raw_results["vix_adjustment"]
            adj_factor = Decimal(vix.get("adjustment_factor", "1.00"))
            financial_mult = financial_mult * adj_factor
            adjustments_applied.append(f"vix_adjustment:{adj_factor}")

        # 4. Apply seasonality to financial score
        if self.enable_seasonality and "seasonality_analysis" in raw_results:
            seas = raw_results["seasonality_analysis"]
            adj_factor = Decimal(seas.get("adjustment_factor", "1.00"))
            # Seasonality affects burn rate interpretation
            financial_mult = financial_mult * adj_factor
            adjustments_applied.append(f"seasonality:{adj_factor}")

        # 5. Apply proximity boost to catalyst score
        if self.enable_proximity_boost and "proximity_analysis" in raw_results:
            prox = raw_results["proximity_analysis"]
            boost = Decimal(prox.get("boost_percentage", "0"))
            if boost > Decimal("0"):
                catalyst_mult = catalyst_mult * (Decimal("1") + boost)
                adjustments_applied.append(f"proximity_boost:{boost}")

        # 6. Apply decay to catalyst score
        if "decay_analysis" in raw_results:
            decay = raw_results["decay_analysis"]
            decay_weight = Decimal(decay.get("decay_weight", "1.00"))
            catalyst_mult = catalyst_mult * decay_weight
            adjustments_applied.append(f"decay_weight:{decay_weight}")

        # 7. Apply competition penalty to clinical score
        if "competition_analysis" in raw_results:
            comp = raw_results["competition_analysis"]
            penalty_pts = Decimal(comp.get("penalty_points", "0"))
            if penalty_pts > Decimal("0"):
                # Convert penalty points to multiplier (20 pts = 0.80x)
                penalty_mult = Decimal("1") - (penalty_pts / Decimal("100"))
                clinical_mult = clinical_mult * penalty_mult
                adjustments_applied.append(f"competition_penalty:{penalty_pts}pts")

        # 8. Apply endpoint weight to clinical score
        if "endpoint_analysis" in raw_results:
            ep = raw_results["endpoint_analysis"]
            ep_weight = Decimal(ep.get("endpoint_weight", "0.70"))
            if ep.get("is_strong"):
                # Strong endpoint = modest boost
                clinical_mult = clinical_mult * Decimal("1.05")
                adjustments_applied.append("strong_endpoint_boost")

        # Clamp multipliers
        clinical_mult = self._clamp(clinical_mult)
        financial_mult = self._clamp(financial_mult)
        catalyst_mult = self._clamp(catalyst_mult)

        # Build audit entry
        audit_entry = {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "clinical_adjustment": str(clinical_mult),
            "financial_adjustment": str(financial_mult),
            "catalyst_adjustment": str(catalyst_mult),
            "regulatory_bonus": str(regulatory_bonus),
            "adjustments_applied": adjustments_applied,
            "raw_results": raw_results,
        }
        self.audit_trail.append(audit_entry)

        return AccuracyAdjustment(
            ticker=ticker,
            clinical_adjustment=clinical_mult,
            financial_adjustment=financial_mult,
            catalyst_adjustment=catalyst_mult,
            regulatory_bonus=regulatory_bonus,
            confidence=confidence,
            adjustments_applied=adjustments_applied,
            audit_details=audit_entry,
        )

    def _clamp(self, value: Decimal) -> Decimal:
        """Clamp multiplier to valid range."""
        clamped = max(self.MIN_MULTIPLIER, min(self.MAX_MULTIPLIER, value))
        return clamped.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def compute_universe_adjustments(
        self,
        universe: List[Dict[str, Any]],
        trial_data_map: Dict[str, Dict[str, Any]],
        financial_data_map: Dict[str, Dict[str, Any]],
        as_of_date: Union[str, date],
        vix_current: Optional[Decimal] = None,
        market_regime: Optional[str] = None,
    ) -> Dict[str, AccuracyAdjustment]:
        """
        Compute adjustments for entire universe.

        Args:
            universe: List of company dicts with 'ticker' field
            trial_data_map: {ticker: trial_data} mapping
            financial_data_map: {ticker: financial_data} mapping
            as_of_date: Analysis date
            vix_current: Current VIX
            market_regime: Market regime

        Returns:
            {ticker: AccuracyAdjustment} mapping
        """
        results = {}

        for company in universe:
            ticker = company.get("ticker")
            if not ticker:
                continue

            ticker_upper = ticker.upper()
            trial_data = trial_data_map.get(ticker_upper, trial_data_map.get(ticker, {}))
            financial_data = financial_data_map.get(ticker_upper, financial_data_map.get(ticker, {}))

            try:
                adjustment = self.compute_adjustments(
                    ticker=ticker,
                    trial_data=trial_data,
                    financial_data=financial_data,
                    as_of_date=as_of_date,
                    vix_current=vix_current,
                    market_regime=market_regime,
                )
                results[ticker] = adjustment
            except Exception as e:
                # Fail-closed: no adjustment on error, but log for debugging
                import logging
                logging.getLogger(__name__).debug(f"Accuracy adjustment error for {ticker}: {e}")
                results[ticker] = AccuracyAdjustment(
                    ticker=ticker,
                    clinical_adjustment=Decimal("1.00"),
                    financial_adjustment=Decimal("1.00"),
                    catalyst_adjustment=Decimal("1.00"),
                    regulatory_bonus=Decimal("0"),
                    confidence="low",
                    adjustments_applied=[f"error:{str(e)}"],
                )
                # Also add to audit trail for diagnostics
                self.audit_trail.append({
                    "ticker": ticker,
                    "error": str(e),
                })

        return results

    def get_diagnostic_counts(self) -> Dict[str, Any]:
        """Get diagnostic summary of adjustments applied."""
        if not self.audit_trail:
            return {
                "total": 0,
                "with_staleness_penalty": 0,
                "with_regulatory_bonus": 0,
                "with_vix_adjustment": 0,
                "with_proximity_boost": 0,
                "staleness_coverage_pct": 0.0,
                "regulatory_coverage_pct": 0.0,
            }

        total = len(self.audit_trail)
        with_staleness = sum(1 for a in self.audit_trail
                           if any("staleness" in adj for adj in a.get("adjustments_applied", [])))
        with_regulatory = sum(1 for a in self.audit_trail
                            if any("regulatory" in adj for adj in a.get("adjustments_applied", [])))
        with_vix = sum(1 for a in self.audit_trail
                      if any("vix" in adj for adj in a.get("adjustments_applied", [])))
        with_proximity = sum(1 for a in self.audit_trail
                           if any("proximity" in adj for adj in a.get("adjustments_applied", [])))

        return {
            "total": total,
            "with_staleness_penalty": with_staleness,
            "with_regulatory_bonus": with_regulatory,
            "with_vix_adjustment": with_vix,
            "with_proximity_boost": with_proximity,
            "staleness_coverage_pct": round(with_staleness / total * 100, 1) if total else 0,
            "regulatory_coverage_pct": round(with_regulatory / total * 100, 1) if total else 0,
        }


def apply_accuracy_to_scores(
    base_clinical: Decimal,
    base_financial: Decimal,
    base_catalyst: Decimal,
    adjustment: AccuracyAdjustment,
) -> Dict[str, Decimal]:
    """
    Apply accuracy adjustments to base scores.

    Args:
        base_clinical: Clinical score (0-100)
        base_financial: Financial score (0-100)
        base_catalyst: Catalyst score (0-100)
        adjustment: AccuracyAdjustment from adapter

    Returns:
        Dict with adjusted scores
    """
    adjusted_clinical = (base_clinical * adjustment.clinical_adjustment +
                        adjustment.regulatory_bonus)
    adjusted_financial = base_financial * adjustment.financial_adjustment
    adjusted_catalyst = base_catalyst * adjustment.catalyst_adjustment

    # Clamp to 0-100
    adjusted_clinical = max(Decimal("0"), min(Decimal("100"), adjusted_clinical))
    adjusted_financial = max(Decimal("0"), min(Decimal("100"), adjusted_financial))
    adjusted_catalyst = max(Decimal("0"), min(Decimal("100"), adjusted_catalyst))

    return {
        "clinical": adjusted_clinical.quantize(Decimal("0.01")),
        "financial": adjusted_financial.quantize(Decimal("0.01")),
        "catalyst": adjusted_catalyst.quantize(Decimal("0.01")),
    }


# =============================================================================
# SELF-CHECKS
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify adapter correctness."""
    errors = []

    # CHECK 1: Adapter initialization
    adapter = AccuracyEnhancementsAdapter()
    if adapter.VERSION != "1.0.0":
        errors.append(f"CHECK1 FAIL: Wrong version {adapter.VERSION}")

    # CHECK 2: Basic adjustment computation
    trial_data = {
        "phase": "phase 3",
        "conditions": ["oncology"],
        "last_update_posted": "2025-06-01",  # 7 months old
    }
    financial_data = {"dilution_score": 50}

    adjustment = adapter.compute_adjustments(
        ticker="TEST",
        trial_data=trial_data,
        financial_data=financial_data,
        as_of_date=date(2026, 1, 15),
    )

    if adjustment.ticker != "TEST":
        errors.append(f"CHECK2 FAIL: Wrong ticker {adjustment.ticker}")

    # CHECK 3: Staleness should apply penalty for 7-month-old Phase 3
    # (Phase 3 has 180-day max staleness)
    if "staleness_penalty" not in str(adjustment.adjustments_applied):
        errors.append(f"CHECK3 FAIL: Staleness penalty not applied for stale Phase 3")

    # CHECK 4: Clamp bounds
    if adjustment.clinical_adjustment < Decimal("0.70"):
        errors.append(f"CHECK4 FAIL: Clinical adjustment below min: {adjustment.clinical_adjustment}")
    if adjustment.clinical_adjustment > Decimal("1.30"):
        errors.append(f"CHECK4 FAIL: Clinical adjustment above max: {adjustment.clinical_adjustment}")

    # CHECK 5: Score application
    scores = apply_accuracy_to_scores(
        base_clinical=Decimal("75"),
        base_financial=Decimal("60"),
        base_catalyst=Decimal("50"),
        adjustment=adjustment,
    )
    if scores["clinical"] < Decimal("0") or scores["clinical"] > Decimal("100"):
        errors.append(f"CHECK5 FAIL: Clinical score out of bounds: {scores['clinical']}")

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
