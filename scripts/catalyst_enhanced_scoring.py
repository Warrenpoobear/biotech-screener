#!/usr/bin/env python3
"""
catalyst_enhanced_scoring.py - Enhanced Catalyst Scoring Integration

Integrates all catalyst enhancement engines:
- Probability Engine (FDA base rates, TDQS, competitive, sponsor)
- Timing Engine (enrollment-based, sponsor delays, PDUFA, clustering)
- Governance Engine (fail-closed, black swan, audit trails)

Provides unified scoring with complete audit trails.

Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import logging

from catalyst_probability_engine import (
    ProbabilityEngine,
    ProbabilityEstimate,
    Phase,
    TherapeuticArea,
    TrialDesignProfile,
    CompetitiveLandscape,
    SponsorTrackRecord,
    parse_phase,
    parse_therapeutic_area,
)
from catalyst_timing_engine import (
    TimingEngine,
    TimingEstimate,
    CatalystCluster,
    PDUFADate,
    SponsorDelayProfile,
)
from catalyst_governance_engine import (
    GovernanceEngine,
    ValidationResult,
    BlackSwanEvent,
    fail_closed_validate,
    FailClosedError,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED CATALYST SCORE
# =============================================================================

@dataclass
class EnhancedCatalystScore:
    """
    Complete enhanced catalyst score with all factors.
    """
    ticker: str
    as_of_date: str

    # === BASE SCORES (from existing module_3) ===
    base_catalyst_score: Decimal = Decimal("50")
    base_proximity_score: Decimal = Decimal("0")
    base_delta_score: Decimal = Decimal("0")

    # === PROBABILITY ENHANCEMENT ===
    probability_of_success: Decimal = Decimal("0.50")
    probability_confidence_low: Decimal = Decimal("0.30")
    probability_confidence_high: Decimal = Decimal("0.70")
    probability_adjustment_factor: Decimal = Decimal("1.0")

    # === TIMING ENHANCEMENT ===
    estimated_readout_days: Optional[int] = None
    timing_confidence: str = "SPECULATIVE"
    has_pdufa: bool = False
    pdufa_days: Optional[int] = None
    n_catalyst_clusters: int = 0
    cluster_convexity_bonus: Decimal = Decimal("0")

    # === GOVERNANCE OVERLAY ===
    governance_passed: bool = True
    n_validation_errors: int = 0
    n_black_swans: int = 0
    governance_penalty: Decimal = Decimal("0")

    # === FINAL COMPOSITE ===
    enhanced_score: Decimal = Decimal("50")
    score_components: Dict[str, str] = field(default_factory=dict)

    # === AUDIT ===
    calculation_hash: str = ""
    calculation_log: List[str] = field(default_factory=list)

    @property
    def score_id(self) -> str:
        """Stable score ID."""
        canonical = f"{self.ticker}|{self.as_of_date}|{self.enhanced_score}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "base_scores": {
                "catalyst_score": str(self.base_catalyst_score),
                "proximity_score": str(self.base_proximity_score),
                "delta_score": str(self.base_delta_score),
            },
            "probability": {
                "pos": str(self.probability_of_success),
                "ci_low": str(self.probability_confidence_low),
                "ci_high": str(self.probability_confidence_high),
                "adjustment_factor": str(self.probability_adjustment_factor),
            },
            "timing": {
                "estimated_readout_days": self.estimated_readout_days,
                "confidence": self.timing_confidence,
                "has_pdufa": self.has_pdufa,
                "pdufa_days": self.pdufa_days,
                "n_clusters": self.n_catalyst_clusters,
                "cluster_convexity_bonus": str(self.cluster_convexity_bonus),
            },
            "governance": {
                "passed": self.governance_passed,
                "n_errors": self.n_validation_errors,
                "n_black_swans": self.n_black_swans,
                "penalty": str(self.governance_penalty),
            },
            "enhanced_score": str(self.enhanced_score),
            "score_components": self.score_components,
            "calculation_hash": self.calculation_hash,
            "calculation_log": self.calculation_log,
        }


# =============================================================================
# ENHANCED SCORING ENGINE
# =============================================================================

class EnhancedCatalystScoringEngine:
    """
    Unified enhanced catalyst scoring engine.

    Combines:
    - Base catalyst scores (from module_3)
    - Probability of success adjustments
    - Timing-based adjustments
    - Governance overlay

    Weights (configurable):
    - Base catalyst: 40%
    - Probability adjustment: 25%
    - Timing adjustment: 20%
    - Governance overlay: 15%
    """

    # Default weights
    WEIGHT_BASE = Decimal("0.40")
    WEIGHT_PROBABILITY = Decimal("0.25")
    WEIGHT_TIMING = Decimal("0.20")
    WEIGHT_GOVERNANCE = Decimal("0.15")

    def __init__(
        self,
        as_of_date: date,
        weights: Optional[Dict[str, Decimal]] = None,
    ):
        self.as_of_date = as_of_date
        self.probability_engine = ProbabilityEngine()
        self.timing_engine = TimingEngine()
        self.governance_engine = GovernanceEngine(as_of_date)

        # Configure weights
        if weights:
            self.weight_base = weights.get("base", self.WEIGHT_BASE)
            self.weight_probability = weights.get("probability", self.WEIGHT_PROBABILITY)
            self.weight_timing = weights.get("timing", self.WEIGHT_TIMING)
            self.weight_governance = weights.get("governance", self.WEIGHT_GOVERNANCE)
        else:
            self.weight_base = self.WEIGHT_BASE
            self.weight_probability = self.WEIGHT_PROBABILITY
            self.weight_timing = self.WEIGHT_TIMING
            self.weight_governance = self.WEIGHT_GOVERNANCE

    def compute_probability_adjustment(
        self,
        probability_estimate: ProbabilityEstimate,
    ) -> Tuple[Decimal, List[str]]:
        """
        Compute score adjustment from probability of success.

        High PoS (>70%) → positive adjustment
        Low PoS (<30%) → negative adjustment
        """
        log = []
        pos = probability_estimate.adjusted_probability

        # Base adjustment: (PoS - 0.5) * 40
        # Range: -20 to +20
        adjustment = (pos - Decimal("0.5")) * Decimal("40")
        adjustment = max(Decimal("-20"), min(Decimal("20"), adjustment))

        log.append(f"PoS: {pos:.2%} → adjustment: {adjustment:+.1f}")

        return (adjustment.quantize(Decimal("0.01")), log)

    def compute_timing_adjustment(
        self,
        timing_estimate: TimingEstimate,
        clusters: List[CatalystCluster],
        pdufas: List[PDUFADate],
    ) -> Tuple[Decimal, List[str]]:
        """
        Compute score adjustment from timing factors.

        Near-term catalysts → positive
        Distant/uncertain catalysts → negative
        Clusters → convexity bonus
        PDUFA → additional boost
        """
        log = []
        adjustment = Decimal("0")

        # Timing proximity adjustment
        if timing_estimate.estimated_readout_days is not None:
            days = timing_estimate.estimated_readout_days

            if days <= 30:
                time_adj = Decimal("15")
            elif days <= 90:
                time_adj = Decimal("10")
            elif days <= 180:
                time_adj = Decimal("5")
            elif days <= 365:
                time_adj = Decimal("0")
            else:
                time_adj = Decimal("-5")

            # Confidence adjustment
            if timing_estimate.confidence == "HIGH":
                time_adj *= Decimal("1.2")
            elif timing_estimate.confidence == "SPECULATIVE":
                time_adj *= Decimal("0.6")

            adjustment += time_adj
            log.append(f"Timing ({days} days, {timing_estimate.confidence}): {time_adj:+.1f}")

        # Cluster convexity bonus
        if clusters:
            convex_clusters = [c for c in clusters if c.is_convex]
            if convex_clusters:
                cluster_bonus = min(Decimal("10"), Decimal(len(convex_clusters)) * Decimal("5"))
                adjustment += cluster_bonus
                log.append(f"Cluster convexity ({len(convex_clusters)} clusters): +{cluster_bonus:.1f}")

        # PDUFA bonus
        if pdufas:
            nearest_pdufa = min(p.days_until(self.as_of_date) or 999 for p in pdufas)
            if nearest_pdufa is not None and nearest_pdufa <= 180:
                pdufa_bonus = Decimal("8") if nearest_pdufa <= 90 else Decimal("4")
                adjustment += pdufa_bonus
                log.append(f"PDUFA in {nearest_pdufa} days: +{pdufa_bonus:.1f}")

        adjustment = max(Decimal("-20"), min(Decimal("25"), adjustment))
        return (adjustment.quantize(Decimal("0.01")), log)

    def compute_governance_adjustment(
        self,
        validation_result: ValidationResult,
        black_swans: List[BlackSwanEvent],
    ) -> Tuple[Decimal, List[str]]:
        """
        Compute score adjustment from governance factors.

        Clean validation → small bonus
        Errors → penalties
        Black swans → large penalties
        """
        log = []
        adjustment = Decimal("0")

        # Clean validation bonus
        if validation_result.passed and validation_result.n_error == 0:
            adjustment += Decimal("5")
            log.append("Clean validation: +5")

        # Error penalties
        if validation_result.n_error > 0:
            error_penalty = min(Decimal("15"), Decimal(validation_result.n_error) * Decimal("3"))
            adjustment -= error_penalty
            log.append(f"Validation errors ({validation_result.n_error}): -{error_penalty:.1f}")

        # Black swan penalties
        if black_swans:
            bs_penalty = min(Decimal("30"), Decimal(len(black_swans)) * Decimal("10"))
            adjustment -= bs_penalty
            log.append(f"Black swans ({len(black_swans)}): -{bs_penalty:.1f}")

        # Fatal validation = severe penalty
        if validation_result.n_fatal > 0:
            adjustment = Decimal("-40")
            log.append(f"FATAL violations ({validation_result.n_fatal}): -40 (capped)")

        adjustment = max(Decimal("-40"), min(Decimal("10"), adjustment))
        return (adjustment.quantize(Decimal("0.01")), log)

    def compute_enhanced_score(
        self,
        ticker: str,
        base_catalyst_score: Decimal,
        base_proximity_score: Decimal,
        base_delta_score: Decimal,
        trial_data: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> EnhancedCatalystScore:
        """
        Compute complete enhanced catalyst score.

        Args:
            ticker: Ticker symbol
            base_catalyst_score: Base score from module_3
            base_proximity_score: Proximity score from module_3
            base_delta_score: Delta score from module_3
            trial_data: Trial metadata for probability/timing
            events: Catalyst events for governance validation

        Returns:
            EnhancedCatalystScore with complete breakdown
        """
        calculation_log = []
        trial_data = trial_data or {}
        events = events or []

        result = EnhancedCatalystScore(
            ticker=ticker,
            as_of_date=self.as_of_date.isoformat(),
            base_catalyst_score=base_catalyst_score,
            base_proximity_score=base_proximity_score,
            base_delta_score=base_delta_score,
        )

        # === 1. PROBABILITY ADJUSTMENT ===
        phase_str = trial_data.get("phase", "UNKNOWN")
        indication = trial_data.get("indication", "")
        nct_id = trial_data.get("nct_id", "")

        phase = parse_phase(phase_str)
        therapeutic_area = parse_therapeutic_area(indication)

        prob_estimate = self.probability_engine.estimate_probability(
            ticker=ticker,
            nct_id=nct_id,
            phase=phase,
            therapeutic_area=therapeutic_area,
            as_of_date=self.as_of_date,
        )

        result.probability_of_success = prob_estimate.adjusted_probability
        result.probability_confidence_low = prob_estimate.confidence_interval_low
        result.probability_confidence_high = prob_estimate.confidence_interval_high

        prob_adj, prob_log = self.compute_probability_adjustment(prob_estimate)
        result.probability_adjustment_factor = prob_adj
        calculation_log.extend(prob_log)

        # === 2. TIMING ADJUSTMENT ===
        timing_estimate = self.timing_engine.estimate_readout_date(
            ticker=ticker,
            nct_id=nct_id,
            phase=phase_str,
            as_of_date=self.as_of_date,
            target_enrollment=trial_data.get("target_enrollment", 0),
            current_enrollment=trial_data.get("current_enrollment", 0),
            enrollment_rate_per_month=Decimal(str(trial_data.get("enrollment_rate", 0))),
            expected_primary_completion=trial_data.get("primary_completion_date"),
            sponsor_name=trial_data.get("sponsor", ""),
        )

        result.estimated_readout_days = timing_estimate.estimated_readout_days
        result.timing_confidence = timing_estimate.confidence.value

        # Get clusters and PDUFAs
        catalyst_dates = [
            (e.get("event_date", ""), e.get("event_type", ""), Decimal(str(e.get("impact", 1))))
            for e in events
            if e.get("event_date")
        ]
        clusters = self.timing_engine.detect_clusters(catalyst_dates, ticker)
        pdufas = self.timing_engine.get_upcoming_pdufas(ticker, self.as_of_date)

        result.n_catalyst_clusters = len(clusters)
        result.has_pdufa = len(pdufas) > 0
        if pdufas:
            result.pdufa_days = pdufas[0].days_until(self.as_of_date)

        timing_adj, timing_log = self.compute_timing_adjustment(timing_estimate, clusters, pdufas)
        calculation_log.extend(timing_log)

        # Cluster convexity bonus
        if clusters:
            result.cluster_convexity_bonus = sum(c.total_impact_score for c in clusters if c.is_convex)

        # === 3. GOVERNANCE ADJUSTMENT ===
        validation_result = self.governance_engine.run_all_validations(events)
        black_swans = self.governance_engine.black_swans

        result.governance_passed = validation_result.passed
        result.n_validation_errors = validation_result.n_error
        result.n_black_swans = len([bs for bs in black_swans if bs.ticker == ticker])

        gov_adj, gov_log = self.compute_governance_adjustment(
            validation_result,
            [bs for bs in black_swans if bs.ticker == ticker],
        )
        result.governance_penalty = gov_adj
        calculation_log.extend(gov_log)

        # === 4. COMPOSITE SCORE ===
        # Weighted combination
        base_component = base_catalyst_score * self.weight_base
        prob_component = (Decimal("50") + prob_adj) * self.weight_probability
        timing_component = (Decimal("50") + timing_adj) * self.weight_timing
        gov_component = (Decimal("50") + gov_adj) * self.weight_governance

        enhanced = base_component + prob_component + timing_component + gov_component

        # Clamp to [0, 100]
        enhanced = max(Decimal("0"), min(Decimal("100"), enhanced))
        result.enhanced_score = enhanced.quantize(Decimal("0.01"))

        # Store components
        result.score_components = {
            "base": str(base_component.quantize(Decimal("0.01"))),
            "probability": str(prob_component.quantize(Decimal("0.01"))),
            "timing": str(timing_component.quantize(Decimal("0.01"))),
            "governance": str(gov_component.quantize(Decimal("0.01"))),
        }

        calculation_log.append(
            f"Final: {base_component:.1f} + {prob_component:.1f} + {timing_component:.1f} + {gov_component:.1f} = {enhanced:.1f}"
        )

        result.calculation_log = calculation_log
        result.calculation_hash = hashlib.sha256(
            json.dumps(result.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]

        return result

    def batch_compute_enhanced_scores(
        self,
        tickers_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, EnhancedCatalystScore]:
        """
        Batch compute enhanced scores for multiple tickers.

        Args:
            tickers_data: {ticker: {base_scores, trial_data, events}}

        Returns:
            {ticker: EnhancedCatalystScore}
        """
        results = {}

        for ticker in sorted(tickers_data.keys()):
            data = tickers_data[ticker]

            try:
                score = self.compute_enhanced_score(
                    ticker=ticker,
                    base_catalyst_score=Decimal(str(data.get("base_catalyst_score", 50))),
                    base_proximity_score=Decimal(str(data.get("base_proximity_score", 0))),
                    base_delta_score=Decimal(str(data.get("base_delta_score", 0))),
                    trial_data=data.get("trial_data"),
                    events=data.get("events", []),
                )
                results[ticker] = score
            except Exception as e:
                logger.error(f"Error computing enhanced score for {ticker}: {e}")
                # Return default score on error
                results[ticker] = EnhancedCatalystScore(
                    ticker=ticker,
                    as_of_date=self.as_of_date.isoformat(),
                    calculation_log=[f"ERROR: {str(e)}"],
                )

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def canonical_json_dumps(obj: Any) -> str:
    """Serialize to canonical JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
