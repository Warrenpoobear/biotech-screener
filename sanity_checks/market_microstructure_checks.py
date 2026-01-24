"""
Query 7.5: Insider/Market Microstructure Checks

Validates that market signals align with fundamental scoring:

1. Options Flow Validation
2. Insider Activity Coherence
3. Conference Presentation Calendar
4. Sentiment Divergence Checks

These checks ensure the model output is consistent with
observable market signals and sentiment.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class OptionsFlow:
    """Options trading flow data."""
    ticker: str
    call_volume_ratio: Decimal  # vs. historical avg (1.0 = avg)
    put_volume_ratio: Decimal
    call_put_ratio: Decimal
    unusual_activity: bool
    activity_type: Optional[str]  # "bullish", "bearish", "neutral"


@dataclass
class InsiderTransaction:
    """Insider trading transaction."""
    ticker: str
    insider_name: str
    insider_title: str
    transaction_type: str  # "buy", "sell"
    shares: int
    value_usd: Decimal
    is_c_suite: bool
    is_director: bool


@dataclass
class ConferencePresentation:
    """Conference presentation event."""
    ticker: str
    conference_name: str
    presentation_date: str
    conference_tier: str  # "major", "mid", "small"
    indication_focus: Optional[str]


@dataclass
class AnalystRating:
    """Analyst rating and target."""
    ticker: str
    rating: str  # "buy", "hold", "sell"
    target_price: Optional[Decimal]
    prior_rating: Optional[str]
    rating_change_date: Optional[str]


class MarketMicrostructureChecker:
    """
    Market microstructure sanity checker.

    Validates that market signals are coherent with model rankings.
    """

    # Major biotech conferences
    MAJOR_CONFERENCES = {
        "ash": "hematology",
        "asco": "oncology",
        "aacr": "oncology",
        "easl": "hepatology",
        "ada": "diabetes",
        "aan": "neurology",
        "aha": "cardiology",
        "jpm": "general",
        "jefferies": "general",
        "goldman": "general",
    }

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
        options_data: Optional[List[OptionsFlow]] = None,
        insider_transactions: Optional[List[InsiderTransaction]] = None,
        upcoming_conferences: Optional[List[ConferencePresentation]] = None,
        analyst_ratings: Optional[List[AnalystRating]] = None,
    ) -> SanityCheckResult:
        """
        Run all market microstructure checks.

        Args:
            securities: List of security contexts
            options_data: Optional options flow data
            insider_transactions: Optional insider transactions
            upcoming_conferences: Optional conference presentations
            analyst_ratings: Optional analyst ratings

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []

        # 1. Options Flow Validation
        if options_data:
            flags.extend(self._check_options_flow(securities, options_data))

        # 2. Insider Activity Coherence
        if insider_transactions:
            flags.extend(self._check_insider_activity(securities, insider_transactions))

        # 3. Conference Calendar Check
        if upcoming_conferences:
            flags.extend(self._check_conference_calendar(securities, upcoming_conferences))

        # 4. Sentiment Divergence
        if analyst_ratings:
            flags.extend(self._check_sentiment_divergence(securities, analyst_ratings))

        # Calculate metrics
        metrics = self._calculate_metrics(flags, securities)

        passed = not any(f.severity == FlagSeverity.CRITICAL for f in flags)

        return SanityCheckResult(
            check_name="market_microstructure",
            category=CheckCategory.MARKET_MICROSTRUCTURE,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_options_flow(
        self,
        securities: List[SecurityContext],
        options_data: List[OptionsFlow],
    ) -> List[SanityFlag]:
        """
        Check options flow coherence.

        Flags:
        - Unusual call buying in top 10 - validate
        - Heavy put activity in top 5 - negative signal
        """
        flags: List[SanityFlag] = []

        sec_lookup = {s.ticker: s for s in securities}

        for flow in options_data:
            sec = sec_lookup.get(flow.ticker)
            if not sec or sec.rank is None:
                continue

            # Check unusual call activity in highly ranked names
            if flow.unusual_activity and flow.activity_type == "bullish":
                if sec.rank <= 10:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.LOW,
                        category=CheckCategory.MARKET_MICROSTRUCTURE,
                        ticker=flow.ticker,
                        check_name="unusual_call_activity_validation",
                        message=f"Unusual call buying ({flow.call_volume_ratio:.1f}x avg) in top 10 candidate",
                        details={
                            "rank": sec.rank,
                            "call_volume_ratio": float(flow.call_volume_ratio),
                            "call_put_ratio": float(flow.call_put_ratio),
                        },
                        recommendation="Validate: Market anticipating catalyst or rumor?",
                    ))

            # Check heavy put activity in top-ranked names
            if flow.put_volume_ratio > Decimal("2.0") and flow.call_put_ratio < Decimal("0.5"):
                if sec.rank <= 5:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.MARKET_MICROSTRUCTURE,
                        ticker=flow.ticker,
                        check_name="put_activity_contradiction",
                        message=f"Heavy put activity ({flow.put_volume_ratio:.1f}x avg) in top 5 candidate #{sec.rank}",
                        details={
                            "rank": sec.rank,
                            "put_volume_ratio": float(flow.put_volume_ratio),
                            "call_put_ratio": float(flow.call_put_ratio),
                        },
                        recommendation="Negative sentiment contradiction - investigate",
                    ))

        return flags

    def _check_insider_activity(
        self,
        securities: List[SecurityContext],
        transactions: List[InsiderTransaction],
    ) -> List[SanityFlag]:
        """
        Check insider activity coherence.

        Flags:
        - C-suite selling aggressively in top-ranked name
        - Directors buying in top 10 - positive validation
        """
        flags: List[SanityFlag] = []

        # Aggregate by ticker
        insider_by_ticker: Dict[str, Dict[str, List[InsiderTransaction]]] = {}
        for txn in transactions:
            if txn.ticker not in insider_by_ticker:
                insider_by_ticker[txn.ticker] = {"buy": [], "sell": []}
            insider_by_ticker[txn.ticker][txn.transaction_type].append(txn)

        sec_lookup = {s.ticker: s for s in securities}

        for ticker, txns in insider_by_ticker.items():
            sec = sec_lookup.get(ticker)
            if not sec or sec.rank is None:
                continue

            # Count C-suite transactions
            csuite_sells = [t for t in txns["sell"] if t.is_c_suite]
            csuite_buys = [t for t in txns["buy"] if t.is_c_suite]

            # Check aggressive C-suite selling
            if len(csuite_sells) >= 2:
                total_sell_value = sum(t.value_usd for t in csuite_sells)
                if total_sell_value > Decimal("1000000"):  # > $1M in sales
                    if sec.rank <= 10:
                        flags.append(SanityFlag(
                            severity=FlagSeverity.HIGH,
                            category=CheckCategory.MARKET_MICROSTRUCTURE,
                            ticker=ticker,
                            check_name="csuite_selling_mismatch",
                            message=f"C-suite selling ${total_sell_value/1000000:.1f}M BUT ranked #{sec.rank}",
                            details={
                                "rank": sec.rank,
                                "csuite_sell_count": len(csuite_sells),
                                "total_sell_value": float(total_sell_value),
                                "sellers": [t.insider_name for t in csuite_sells],
                            },
                            recommendation="Insider confidence mismatch - investigate",
                        ))

            # Check director buying (positive validation)
            director_buys = [t for t in txns["buy"] if t.is_director]
            if len(director_buys) >= 2:
                total_buy_value = sum(t.value_usd for t in director_buys)
                if total_buy_value > Decimal("100000"):  # > $100K in buys
                    if sec.rank <= 10:
                        flags.append(SanityFlag(
                            severity=FlagSeverity.LOW,
                            category=CheckCategory.MARKET_MICROSTRUCTURE,
                            ticker=ticker,
                            check_name="director_buying_validation",
                            message=f"Directors buying ${total_buy_value/1000:.0f}K + ranked #{sec.rank}",
                            details={
                                "rank": sec.rank,
                                "director_buy_count": len(director_buys),
                                "total_buy_value": float(total_buy_value),
                            },
                            recommendation="Positive signal confirmation",
                        ))

        return flags

    def _check_conference_calendar(
        self,
        securities: List[SecurityContext],
        conferences: List[ConferencePresentation],
    ) -> List[SanityFlag]:
        """
        Check conference calendar coherence.

        Flags:
        - Major conference upcoming but relevant sector not ranked highly
        - Top candidate presenting at major conference - validation
        """
        flags: List[SanityFlag] = []

        # Check for major conferences in next 2 weeks
        upcoming_major = [
            c for c in conferences
            if c.conference_tier == "major"
        ]

        if not upcoming_major:
            return flags

        sec_lookup = {s.ticker: s for s in securities}

        # Group securities by indication
        top20_indications: Dict[str, List[str]] = {}
        for sec in securities:
            if sec.rank is not None and sec.rank <= 20 and sec.indication:
                if sec.indication not in top20_indications:
                    top20_indications[sec.indication] = []
                top20_indications[sec.indication].append(sec.ticker)

        for conf in upcoming_major:
            # Check if conference focus area is represented in top 20
            conf_key = conf.conference_name.lower()
            expected_indication = None

            for conf_name, indication in self.MAJOR_CONFERENCES.items():
                if conf_name in conf_key:
                    expected_indication = indication
                    break

            if expected_indication and expected_indication not in ("general",):
                # Check if any top 20 names are in this indication
                has_representation = any(
                    expected_indication.lower() in ind.lower()
                    for ind in top20_indications.keys()
                )

                if not has_representation:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.MARKET_MICROSTRUCTURE,
                        ticker=None,
                        check_name="conference_sector_gap",
                        message=f"{conf.conference_name} in 2 weeks but no {expected_indication} names in top 20",
                        details={
                            "conference": conf.conference_name,
                            "presentation_date": conf.presentation_date,
                            "expected_focus": expected_indication,
                            "top20_indications": list(top20_indications.keys()),
                        },
                        recommendation="Missing sector catalyst concentration",
                    ))

            # Check if presenting ticker is highly ranked
            if conf.ticker:
                sec = sec_lookup.get(conf.ticker)
                if sec and sec.rank is not None and sec.rank <= 10:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.LOW,
                        category=CheckCategory.MARKET_MICROSTRUCTURE,
                        ticker=conf.ticker,
                        check_name="conference_validation",
                        message=f"Top {sec.rank} candidate presenting at {conf.conference_name}",
                        details={
                            "rank": sec.rank,
                            "conference": conf.conference_name,
                            "presentation_date": conf.presentation_date,
                        },
                        recommendation="Near-term attention catalyst captured",
                    ))

        return flags

    def _check_sentiment_divergence(
        self,
        securities: List[SecurityContext],
        ratings: List[AnalystRating],
    ) -> List[SanityFlag]:
        """
        Check sentiment divergence.

        Flags:
        - Analyst consensus = Sell BUT model ranks highly
        - Multiple upgrades but ranking dropped
        """
        flags: List[SanityFlag] = []

        sec_lookup = {s.ticker: s for s in securities}

        # Aggregate ratings by ticker
        ratings_by_ticker: Dict[str, List[AnalystRating]] = {}
        for rating in ratings:
            if rating.ticker not in ratings_by_ticker:
                ratings_by_ticker[rating.ticker] = []
            ratings_by_ticker[rating.ticker].append(rating)

        for ticker, ticker_ratings in ratings_by_ticker.items():
            sec = sec_lookup.get(ticker)
            if not sec or sec.rank is None:
                continue

            # Calculate consensus
            buy_count = sum(1 for r in ticker_ratings if r.rating.lower() in ("buy", "strong buy", "overweight"))
            sell_count = sum(1 for r in ticker_ratings if r.rating.lower() in ("sell", "underperform", "reduce"))
            total = len(ticker_ratings)

            # Check contrarian position
            if total >= 3 and sell_count >= buy_count * 2:
                if sec.rank <= 10:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.MARKET_MICROSTRUCTURE,
                        ticker=ticker,
                        check_name="contrarian_position",
                        message=f"Analyst consensus bearish ({sell_count} sells vs {buy_count} buys) BUT ranked #{sec.rank}",
                        details={
                            "rank": sec.rank,
                            "buy_count": buy_count,
                            "sell_count": sell_count,
                            "total_ratings": total,
                        },
                        recommendation="Contrarian position - document rationale",
                    ))

            # Check recent upgrades
            recent_upgrades = [
                r for r in ticker_ratings
                if r.prior_rating and r.prior_rating.lower() in ("hold", "neutral", "sell")
                and r.rating.lower() in ("buy", "overweight")
            ]

            if len(recent_upgrades) >= 3:
                # Multiple upgrades - should be positive for rank
                if sec.rank is not None and sec.rank > 30:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.MARKET_MICROSTRUCTURE,
                        ticker=ticker,
                        check_name="upgrades_vs_rank",
                        message=f"{len(recent_upgrades)} recent upgrades BUT ranked #{sec.rank}",
                        details={
                            "rank": sec.rank,
                            "upgrade_count": len(recent_upgrades),
                        },
                        recommendation="Lagging indicator vs. fundamental deterioration - investigate",
                    ))

        return flags

    def _calculate_metrics(
        self,
        flags: List[SanityFlag],
        securities: List[SecurityContext],
    ) -> Dict[str, Any]:
        """Calculate summary metrics."""
        return {
            "total_flags": len(flags),
            "by_check": {
                "options_flow": sum(1 for f in flags if "call" in f.check_name or "put" in f.check_name),
                "insider_activity": sum(1 for f in flags if "csuite" in f.check_name or "director" in f.check_name),
                "conference": sum(1 for f in flags if "conference" in f.check_name),
                "sentiment": sum(1 for f in flags if "contrarian" in f.check_name or "upgrade" in f.check_name),
            },
            "by_severity": {
                "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
                "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
                "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
                "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
            },
        }
