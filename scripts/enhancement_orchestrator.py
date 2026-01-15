#!/usr/bin/env python3
"""
enhancement_orchestrator.py

Master Orchestrator for Enhanced Wake Robin Screening System

Coordinates all enhancement modules (PoS, Short Interest, Regime Detection)
and produces final ranked output with complete audit trails.

Integration Architecture:
    run_screen.py (Main Pipeline)
        ↓
    Module 1-4 (Universe, Financial, Catalyst, Clinical)
        ↓
    EnhancementOrchestrator
        ├── RegimeDetectionEngine (market conditions)
        ├── ProbabilityOfSuccessEngine (indication-adjusted PoS)
        ├── ShortInterestSignalEngine (squeeze/crowding)
        └── Enhanced Composite Scoring
        ↓
    Module 5 (Final Composite + Defensive)

Design Philosophy:
- Governed over smart (determinism is non-negotiable)
- Fail loudly, audit everything
- Stdlib-only for corporate safety
- Point-in-time discipline for backtesting

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import date
from dataclasses import dataclass, field, asdict

# Import enhancement modules
from pos_engine import ProbabilityOfSuccessEngine
from short_interest_engine import ShortInterestSignalEngine
from regime_engine import RegimeDetectionEngine


# Module metadata
__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


@dataclass
class EnhancedCompanyScore:
    """Enhanced company score with all signal layers."""
    ticker: str
    composite_score: Decimal
    pos_score: Decimal
    pos_multiplier: Decimal
    short_interest_score: Decimal
    squeeze_potential: str
    signal_direction: str
    severity: str
    flags: List[str] = field(default_factory=list)
    layer_scores: Dict[str, Decimal] = field(default_factory=dict)
    audit_hash: str = ""


@dataclass
class EnhancedScreeningResult:
    """Complete enhanced screening result."""
    as_of_date: str
    regime: str
    regime_confidence: Decimal
    signal_adjustments: Dict[str, Decimal]
    scores: List[EnhancedCompanyScore]
    diagnostic_counts: Dict[str, Any]
    provenance: Dict[str, str]


class EnhancementOrchestrator:
    """
    Master orchestrator for enhanced Wake Robin screening system.

    Coordinates all enhancement modules and produces final ranked output
    with complete audit trails.

    Usage:
        orchestrator = EnhancementOrchestrator()

        # Step 1: Detect market regime
        regime_result = orchestrator.detect_regime(
            vix_current=Decimal("18.5"),
            xbi_vs_spy_30d=Decimal("3.2")
        )

        # Step 2: Score universe with all enhancements
        results = orchestrator.score_universe(
            universe=universe_data,
            regime_result=regime_result,
            as_of_date=date(2026, 1, 11)
        )

        # Step 3: Get ranked output
        for score in results.scores[:10]:
            print(f"{score.ticker}: {score.composite_score}")
    """

    VERSION = "1.0.0"

    # Default composite weights (regime-adjustable)
    DEFAULT_WEIGHTS = {
        "clinical": Decimal("0.35"),
        "financial": Decimal("0.25"),
        "catalyst": Decimal("0.15"),
        "pos": Decimal("0.15"),
        "short_interest": Decimal("0.10")
    }

    # Severity multipliers (aligned with existing system)
    SEVERITY_MULTIPLIERS = {
        "none": Decimal("1.00"),
        "sev1": Decimal("0.90"),
        "sev2": Decimal("0.50"),
        "sev3": Decimal("0.00")
    }

    def __init__(self, enable_short_interest: bool = True):
        """
        Initialize the enhancement orchestrator.

        Args:
            enable_short_interest: Whether to include short interest signals
        """
        # Initialize enhancement engines
        self.pos_engine = ProbabilityOfSuccessEngine()
        self.regime_engine = RegimeDetectionEngine()
        self.short_interest_engine = ShortInterestSignalEngine() if enable_short_interest else None

        self.enable_short_interest = enable_short_interest
        self.audit_trail: List[Dict[str, Any]] = []

    def detect_regime(
        self,
        vix_current: Decimal,
        xbi_vs_spy_30d: Decimal,
        fed_rate_change_3m: Optional[Decimal] = None,
        as_of_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Detect current market regime.

        Should be called first before scoring to establish signal weights.

        Returns:
            Dict with regime classification and signal adjustments
        """
        return self.regime_engine.detect_regime(
            vix_current=vix_current,
            xbi_vs_spy_30d=xbi_vs_spy_30d,
            fed_rate_change_3m=fed_rate_change_3m,
            as_of_date=as_of_date
        )

    def score_company(
        self,
        ticker: str,
        company_data: Dict[str, Any],
        base_scores: Dict[str, Decimal],
        regime_result: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[date] = None
    ) -> EnhancedCompanyScore:
        """
        Score a single company through all enhancement layers.

        Args:
            ticker: Stock ticker
            company_data: Dict containing:
                - base_stage: str (development phase)
                - indication: Optional[str]
                - short_interest_pct: Optional[Decimal]
                - days_to_cover: Optional[Decimal]
                - short_interest_change_pct: Optional[Decimal]
                - institutional_long_pct: Optional[Decimal]
                - trial_design_quality: Optional[Decimal]
                - competitive_intensity: Optional[Decimal]
            base_scores: Dict of existing module scores:
                - clinical: Decimal (from module_4)
                - financial: Decimal (from module_2)
                - catalyst: Decimal (from module_3)
            regime_result: Output from detect_regime() or None for defaults
            as_of_date: Point-in-time date

        Returns:
            EnhancedCompanyScore with composite score and all layer details
        """

        layer_scores = {}
        flags = []

        # Layer 1: PoS-Adjusted Score
        pos_result = self.pos_engine.calculate_adjusted_stage_score(
            base_stage=company_data.get("base_stage", "phase_2"),
            indication=company_data.get("indication"),
            trial_design_quality=self._to_decimal(company_data.get("trial_design_quality")),
            competitive_intensity=self._to_decimal(company_data.get("competitive_intensity")),
            as_of_date=as_of_date
        )
        pos_score = pos_result["stage_score_adjusted"]
        pos_multiplier = pos_result["pos_multiplier"]
        layer_scores["pos"] = pos_score

        # Layer 2: Short Interest Signal (if enabled)
        if self.enable_short_interest and self.short_interest_engine:
            short_result = self.short_interest_engine.calculate_short_signal(
                ticker=ticker,
                short_interest_pct=self._to_decimal(company_data.get("short_interest_pct")),
                days_to_cover=self._to_decimal(company_data.get("days_to_cover")),
                short_interest_change_pct=self._to_decimal(company_data.get("short_interest_change_pct")),
                institutional_long_pct=self._to_decimal(company_data.get("institutional_long_pct")),
                as_of_date=as_of_date
            )
            short_score = short_result["short_signal_score"]
            squeeze_potential = short_result["squeeze_potential"]
            signal_direction = short_result["signal_direction"]
            flags.extend(short_result.get("flags", []))
        else:
            short_score = Decimal("50")  # Neutral
            squeeze_potential = "DISABLED"
            signal_direction = "NEUTRAL"

        layer_scores["short_interest"] = short_score

        # Layer 3: Include base scores
        layer_scores["clinical"] = base_scores.get("clinical", Decimal("50"))
        layer_scores["financial"] = base_scores.get("financial", Decimal("50"))
        layer_scores["catalyst"] = base_scores.get("catalyst", Decimal("50"))

        # Get weights (regime-adjusted if available)
        if regime_result:
            weights = self.regime_engine.get_composite_weight_adjustments(
                regime=regime_result.get("regime", "UNKNOWN"),
                base_weights=self.DEFAULT_WEIGHTS
            )
        else:
            weights = self.DEFAULT_WEIGHTS.copy()

        # Calculate weighted composite
        composite_score = Decimal("0")
        for component, weight in weights.items():
            if component in layer_scores:
                composite_score += layer_scores[component] * weight

        # Apply severity multiplier
        severity = company_data.get("severity", "none")
        severity_mult = self.SEVERITY_MULTIPLIERS.get(severity, Decimal("1.0"))
        composite_score = composite_score * severity_mult

        # Apply regime-based adjustment (bonus/penalty)
        if regime_result:
            regime = regime_result.get("regime", "UNKNOWN")
            if regime == "VOLATILITY_SPIKE":
                # Penalty for uncertainty
                composite_score = composite_score * Decimal("0.95")
                flags.append("REGIME_PENALTY_APPLIED")
            elif regime == "BULL" and signal_direction == "BULLISH":
                # Bonus for momentum alignment
                composite_score = composite_score * Decimal("1.02")

        # Round and clamp
        composite_score = max(Decimal("0"), min(Decimal("100"), composite_score))
        composite_score = composite_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Generate audit hash
        audit_data = {
            "ticker": ticker,
            "composite": str(composite_score),
            "layers": {k: str(v) for k, v in layer_scores.items()}
        }
        audit_hash = hashlib.sha256(
            json.dumps(audit_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return EnhancedCompanyScore(
            ticker=ticker,
            composite_score=composite_score,
            pos_score=pos_score,
            pos_multiplier=pos_multiplier,
            short_interest_score=short_score,
            squeeze_potential=squeeze_potential,
            signal_direction=signal_direction,
            severity=severity,
            flags=flags,
            layer_scores={k: v for k, v in layer_scores.items()},
            audit_hash=audit_hash
        )

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        base_scores_map: Dict[str, Dict[str, Decimal]],
        regime_result: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[date] = None
    ) -> EnhancedScreeningResult:
        """
        Score an entire universe of companies.

        Args:
            universe: List of company data dicts
            base_scores_map: Dict of ticker → base_scores from modules 2-4
            regime_result: Output from detect_regime()
            as_of_date: Point-in-time date

        Returns:
            EnhancedScreeningResult with ranked scores and diagnostics
        """

        if as_of_date is None:
            as_of_date = date.today()

        # Get regime info
        regime = regime_result.get("regime", "UNKNOWN") if regime_result else "UNKNOWN"
        regime_confidence = regime_result.get("confidence", Decimal("0")) if regime_result else Decimal("0")
        signal_adjustments = regime_result.get("signal_adjustments", {}) if regime_result else {}

        scores: List[EnhancedCompanyScore] = []
        squeeze_distribution: Dict[str, int] = {}
        signal_distribution: Dict[str, int] = {}
        severity_distribution: Dict[str, int] = {}

        for company in universe:
            ticker = company.get("ticker", "UNKNOWN")

            # Get base scores for this ticker
            base_scores = base_scores_map.get(ticker, {
                "clinical": Decimal("50"),
                "financial": Decimal("50"),
                "catalyst": Decimal("50")
            })

            # Score company
            score = self.score_company(
                ticker=ticker,
                company_data=company,
                base_scores=base_scores,
                regime_result=regime_result,
                as_of_date=as_of_date
            )
            scores.append(score)

            # Track distributions
            squeeze_distribution[score.squeeze_potential] = \
                squeeze_distribution.get(score.squeeze_potential, 0) + 1
            signal_distribution[score.signal_direction] = \
                signal_distribution.get(score.signal_direction, 0) + 1
            severity_distribution[score.severity] = \
                severity_distribution.get(score.severity, 0) + 1

        # Sort by composite score (ASCENDING: lower score = better = rank 1)
        # Validation showed inverted ranking: high scores predicted underperformance
        scores.sort(key=lambda x: x.composite_score, reverse=False)

        # Assign ranks
        for i, score in enumerate(scores):
            score.flags.append(f"RANK_{i + 1}")

        # Calculate content hash for determinism verification
        scores_json = json.dumps(
            [{"t": s.ticker, "s": str(s.composite_score)} for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        # Deterministic timestamp
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z"

        # Audit entry (deterministic!)
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat(),
            "regime": regime,
            "regime_confidence": str(regime_confidence),
            "universe_size": len(universe),
            "content_hash": content_hash,
            "module_version": self.VERSION
        }
        self.audit_trail.append(audit_entry)

        return EnhancedScreeningResult(
            as_of_date=as_of_date.isoformat(),
            regime=regime,
            regime_confidence=regime_confidence,
            signal_adjustments=signal_adjustments,
            scores=scores,
            diagnostic_counts={
                "total_scored": len(scores),
                "squeeze_distribution": squeeze_distribution,
                "signal_distribution": signal_distribution,
                "severity_distribution": severity_distribution,
                "regime": regime,
                "regime_confidence": str(regime_confidence)
            },
            provenance={
                "module": "enhancement_orchestrator",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
                "pos_engine_version": self.pos_engine.VERSION,
                "regime_engine_version": self.regime_engine.VERSION,
                "short_interest_enabled": str(self.enable_short_interest)
            }
        )

    def integrate_with_module5(
        self,
        enhanced_scores: EnhancedScreeningResult,
        module5_scores: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Integrate enhanced scores with existing Module 5 composite scores.

        This allows the enhancement layer to augment rather than replace
        the existing scoring pipeline.

        Args:
            enhanced_scores: Output from score_universe()
            module5_scores: Existing Module 5 output scores

        Returns:
            Merged scores with enhancement data added
        """

        # Create lookup for enhanced scores
        enhanced_lookup = {
            s.ticker: s for s in enhanced_scores.scores
        }

        merged_scores = []
        for m5_score in module5_scores:
            ticker = m5_score.get("ticker", "")
            enhanced = enhanced_lookup.get(ticker)

            if enhanced:
                # Merge enhanced data into module 5 score
                merged = m5_score.copy()
                merged["pos_score"] = float(enhanced.pos_score)
                merged["pos_multiplier"] = float(enhanced.pos_multiplier)
                merged["short_interest_score"] = float(enhanced.short_interest_score)
                merged["squeeze_potential"] = enhanced.squeeze_potential
                merged["signal_direction"] = enhanced.signal_direction
                merged["enhanced_composite"] = float(enhanced.composite_score)
                merged["enhancement_flags"] = enhanced.flags
                merged["enhancement_audit_hash"] = enhanced.audit_hash
                merged_scores.append(merged)
            else:
                # No enhancement data, pass through
                merged_scores.append(m5_score)

        return merged_scores

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
        """Return combined audit trail from all engines."""
        return {
            "orchestrator": self.audit_trail.copy(),
            "pos_engine": self.pos_engine.get_audit_trail(),
            "regime_engine": self.regime_engine.get_audit_trail(),
            "short_interest_engine": (
                self.short_interest_engine.get_audit_trail()
                if self.short_interest_engine else []
            )
        }

    def clear_audit_trail(self) -> None:
        """Clear all audit trails."""
        self.audit_trail = []
        self.pos_engine.clear_audit_trail()
        self.regime_engine.clear_state()
        if self.short_interest_engine:
            self.short_interest_engine.clear_audit_trail()


def demonstration():
    """Demonstrate the enhancement orchestrator capabilities."""
    print("=" * 70)
    print("ENHANCEMENT ORCHESTRATOR - DEMONSTRATION")
    print("=" * 70)
    print()

    orchestrator = EnhancementOrchestrator()

    # Step 1: Detect regime
    print("Step 1: Detect Market Regime")
    print("-" * 70)

    regime_result = orchestrator.detect_regime(
        vix_current=Decimal("18.5"),
        xbi_vs_spy_30d=Decimal("3.2"),
        fed_rate_change_3m=Decimal("-0.25"),
        as_of_date=date(2026, 1, 11)
    )

    print(f"Regime: {regime_result['regime']}")
    print(f"Confidence: {regime_result['confidence']}")
    print(f"Description: {regime_result['regime_description']}")
    print()

    # Step 2: Score sample universe
    print("Step 2: Score Sample Universe")
    print("-" * 70)

    universe = [
        {
            "ticker": "ACME",
            "base_stage": "phase_3",
            "indication": "oncology",
            "short_interest_pct": Decimal("25.5"),
            "days_to_cover": Decimal("8.2"),
            "short_interest_change_pct": Decimal("-12.0"),
            "institutional_long_pct": Decimal("55.0"),
            "severity": "none"
        },
        {
            "ticker": "BIOTECH",
            "base_stage": "phase_3",
            "indication": "rare disease",
            "short_interest_pct": Decimal("8.5"),
            "days_to_cover": Decimal("3.1"),
            "short_interest_change_pct": Decimal("5.0"),
            "institutional_long_pct": Decimal("70.0"),
            "severity": "none"
        },
        {
            "ticker": "PHARMA",
            "base_stage": "phase_2",
            "indication": "neurology",
            "short_interest_pct": Decimal("35.0"),
            "days_to_cover": Decimal("15.5"),
            "short_interest_change_pct": Decimal("-25.0"),
            "severity": "sev1"
        }
    ]

    base_scores_map = {
        "ACME": {"clinical": Decimal("75"), "financial": Decimal("65"), "catalyst": Decimal("80")},
        "BIOTECH": {"clinical": Decimal("85"), "financial": Decimal("70"), "catalyst": Decimal("60")},
        "PHARMA": {"clinical": Decimal("55"), "financial": Decimal("40"), "catalyst": Decimal("70")}
    }

    results = orchestrator.score_universe(
        universe=universe,
        base_scores_map=base_scores_map,
        regime_result=regime_result,
        as_of_date=date(2026, 1, 11)
    )

    print(f"Regime: {results.regime}")
    print(f"Total Scored: {results.diagnostic_counts['total_scored']}")
    print()
    print("Ranked Results:")
    print("-" * 70)
    for i, score in enumerate(results.scores, 1):
        print(f"{i}. {score.ticker}")
        print(f"   Composite: {score.composite_score}")
        print(f"   PoS Score: {score.pos_score} (mult: {score.pos_multiplier})")
        print(f"   Short Interest: {score.short_interest_score}")
        print(f"   Squeeze Potential: {score.squeeze_potential}")
        print(f"   Signal Direction: {score.signal_direction}")
        print(f"   Flags: {score.flags}")
        print()

    print("Diagnostic Counts:")
    print(f"  Squeeze Distribution: {results.diagnostic_counts['squeeze_distribution']}")
    print(f"  Signal Distribution: {results.diagnostic_counts['signal_distribution']}")
    print()

    print("Provenance:")
    for key, value in results.provenance.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    demonstration()
