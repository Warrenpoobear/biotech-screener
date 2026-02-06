#!/usr/bin/env python3
"""
morningstar_signal_engine.py

Morningstar Quantitative Signal Engine for Biotech Screener

Integrates Morningstar MCP datapoints into the screening model to improve
valuation accuracy and cross-validate financial health. Blends into existing
weight allocations (valuation 5%, financial 35%) rather than adding new weight.

Signal Components (internal composite 0-100):
- Fair Value Discount (40%): QV009 (Quantitative FV) vs current price
- Capital Efficiency (25%): STA4Z (ROIC), HS08F (ROE)
- Leverage Health (20%): ST389 (D/E), HS06U (D/Capital)
- Growth Quality (15%): HS035 (Sales Growth), HS08D (Net Margin)

Design Philosophy:
- Confidence-gated: missing sub-signals redistribute weight to available ones
- Regime-aware: dev-stage tickers get neutral scores on commercial-only metrics
- Deterministic Decimal arithmetic throughout
- Full audit trail for every calculation

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _clamp(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal to [lo, hi]."""
    return max(lo, min(hi, value))


def _quantize(value: Decimal) -> Decimal:
    """Quantize to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class MorningstarSignalEngine:
    """
    Morningstar quantitative signal scoring engine.

    Loads Morningstar MCP data and produces a composite signal (0-100) for each
    ticker based on fair value discount, capital efficiency, leverage health,
    and growth quality. Confidence-gated and regime-aware.

    Usage:
        engine = MorningstarSignalEngine()
        loaded = engine.load_data(Path("production_data"))
        if loaded > 0:
            result = engine.score_universe(universe, market_data, as_of_date)
    """

    VERSION = "1.0.0"

    # Sub-signal weights (must sum to 1.0)
    SUB_SIGNAL_WEIGHTS: Dict[str, Decimal] = {
        "fair_value_discount": Decimal("0.40"),
        "capital_efficiency": Decimal("0.25"),
        "leverage_health": Decimal("0.20"),
        "growth_quality": Decimal("0.15"),
    }

    # Datapoint IDs used per sub-signal
    DATAPOINTS = {
        "fair_value_discount": ["QV009"],          # Quantitative Fair Value
        "capital_efficiency": ["STA4Z", "HS08F"],  # ROIC, ROE
        "leverage_health": ["ST389", "HS06U"],     # D/E, D/Capital
        "growth_quality": ["HS035", "HS08D"],      # Sales Growth, Net Margin
    }

    # Thresholds for fair value discount scoring
    # discount_pct = (QV - price) / QV * 100
    # Higher discount = more undervalued = higher score
    FV_DISCOUNT_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        # (discount_pct_threshold, score)
        (Decimal("50"), Decimal("95")),   # 50%+ discount → 95
        (Decimal("30"), Decimal("80")),   # 30%+ → 80
        (Decimal("15"), Decimal("65")),   # 15%+ → 65
        (Decimal("5"), Decimal("55")),    # 5%+ → 55
        (Decimal("0"), Decimal("50")),    # At fair value → 50
        (Decimal("-15"), Decimal("35")),  # 15% premium → 35
        (Decimal("-30"), Decimal("20")),  # 30% premium → 20
        (Decimal("-50"), Decimal("10")),  # 50%+ premium → 10
    ]

    # ROIC scoring breakpoints (higher = better)
    ROIC_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        (Decimal("20"), Decimal("90")),
        (Decimal("10"), Decimal("75")),
        (Decimal("5"), Decimal("60")),
        (Decimal("0"), Decimal("45")),
        (Decimal("-5"), Decimal("35")),
        (Decimal("-20"), Decimal("20")),
    ]

    # ROE scoring breakpoints (higher = better, but negative is common in biotech)
    ROE_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        (Decimal("25"), Decimal("90")),
        (Decimal("15"), Decimal("75")),
        (Decimal("5"), Decimal("60")),
        (Decimal("0"), Decimal("45")),
        (Decimal("-10"), Decimal("35")),
        (Decimal("-30"), Decimal("20")),
    ]

    # D/E scoring breakpoints (lower = better for health)
    DE_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        (Decimal("0"), Decimal("85")),     # No debt
        (Decimal("0.3"), Decimal("70")),   # Low leverage
        (Decimal("0.7"), Decimal("55")),   # Moderate
        (Decimal("1.0"), Decimal("45")),   # At equity
        (Decimal("2.0"), Decimal("30")),   # High leverage
        (Decimal("5.0"), Decimal("15")),   # Very high
    ]

    # D/Capital scoring breakpoints (lower = better, expressed as %)
    DCAP_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        (Decimal("0"), Decimal("85")),
        (Decimal("15"), Decimal("70")),
        (Decimal("30"), Decimal("55")),
        (Decimal("50"), Decimal("40")),
        (Decimal("70"), Decimal("25")),
    ]

    # Sales growth scoring breakpoints (higher = better, %)
    SALES_GROWTH_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        (Decimal("50"), Decimal("90")),
        (Decimal("20"), Decimal("75")),
        (Decimal("10"), Decimal("60")),
        (Decimal("0"), Decimal("45")),
        (Decimal("-10"), Decimal("35")),
        (Decimal("-30"), Decimal("20")),
    ]

    # Net margin scoring breakpoints (higher = better, %)
    NET_MARGIN_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
        (Decimal("20"), Decimal("90")),
        (Decimal("10"), Decimal("75")),
        (Decimal("0"), Decimal("55")),
        (Decimal("-20"), Decimal("40")),
        (Decimal("-50"), Decimal("30")),
        (Decimal("-100"), Decimal("20")),
    ]

    # Dev-stage regime: metrics that require commercial operations
    COMMERCIAL_ONLY_SIGNALS = {"capital_efficiency", "growth_quality"}

    def __init__(self):
        """Initialize the Morningstar signal engine."""
        self._data: Dict[str, Dict[str, str]] = {}
        self._price_history: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}
        self._snapshot_date: Optional[str] = None
        self.audit_trail: List[Dict[str, Any]] = []

    def load_data(self, data_dir: Path) -> int:
        """
        Load morningstar_mcp_data.json from the given data directory.

        Args:
            data_dir: Path to the data directory containing the JSON file.

        Returns:
            Number of ticker records loaded (0 if file missing/invalid).
        """
        data_file = data_dir / "morningstar_mcp_data.json"
        if not data_file.exists():
            logger.info("Morningstar MCP data file not found: %s", data_file)
            return 0

        try:
            with open(data_file) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load Morningstar MCP data: %s", e)
            return 0

        self._metadata = raw.get("metadata", {})
        self._data = raw.get("records", {})
        self._snapshot_date = self._metadata.get("pull_date")

        count = len(self._data)
        logger.info("Loaded Morningstar MCP data: %d tickers (snapshot: %s)",
                     count, self._snapshot_date)

        # Load price history for FV discount fallback
        price_file = data_dir / "morningstar_price_history.json"
        if price_file.exists():
            try:
                with open(price_file) as f:
                    price_raw = json.load(f)
                self._price_history = price_raw.get("records", {})
                logger.info("Loaded Morningstar price history: %d tickers",
                            len(self._price_history))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load Morningstar price history: %s", e)

        return count

    def score_ticker(
        self,
        ticker: str,
        current_price: Optional[Decimal] = None,
        market_cap_mm: Optional[Decimal] = None,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Score a single ticker using Morningstar datapoints.

        Args:
            ticker: Equity ticker symbol.
            current_price: Current share price (for FV discount calc).
            market_cap_mm: Market cap in millions (for regime detection).
            as_of_date: Point-in-time date for audit trail.

        Returns:
            Dict with morningstar_score (0-100), sub-scores, confidence,
            regime, flags, and data_coverage.
        """
        ticker_upper = ticker.upper()
        record = self._data.get(ticker_upper)

        if not record:
            return self._no_data_result(ticker_upper, as_of_date)

        flags: List[str] = []

        # Determine regime: dev-stage vs commercial
        regime = self._classify_regime(record, market_cap_mm)
        if regime == "development":
            flags.append("ms_regime_development")
        else:
            flags.append("ms_regime_commercial")

        # Resolve current_price: caller-provided > MS price history > None
        effective_price = current_price
        price_source = "caller"
        if effective_price is None:
            ms_price = self._get_latest_price(ticker_upper, as_of_date)
            if ms_price is not None:
                effective_price = ms_price
                price_source = "ms_price_history"
                flags.append("ms_price_from_history")

        # Score each sub-signal
        sub_scores: Dict[str, Optional[Decimal]] = {}
        sub_details: Dict[str, Dict[str, Any]] = {}

        # 1. Fair Value Discount
        fv_result = self._score_fair_value_discount(record, effective_price)
        sub_scores["fair_value_discount"] = fv_result["score"]
        sub_details["fair_value_discount"] = fv_result
        flags.extend(fv_result.get("flags", []))

        # 2. Capital Efficiency
        ce_result = self._score_capital_efficiency(record, regime)
        sub_scores["capital_efficiency"] = ce_result["score"]
        sub_details["capital_efficiency"] = ce_result
        flags.extend(ce_result.get("flags", []))

        # 3. Leverage Health
        lh_result = self._score_leverage_health(record)
        sub_scores["leverage_health"] = lh_result["score"]
        sub_details["leverage_health"] = lh_result
        flags.extend(lh_result.get("flags", []))

        # 4. Growth Quality
        gq_result = self._score_growth_quality(record, regime)
        sub_scores["growth_quality"] = gq_result["score"]
        sub_details["growth_quality"] = gq_result
        flags.extend(gq_result.get("flags", []))

        # Confidence-gated composite: redistribute weight from missing sub-signals
        composite, confidence, available_count = self._compute_composite(sub_scores)

        # Data coverage
        total_datapoints = sum(
            len(dps) for dps in self.DATAPOINTS.values()
        )
        available_datapoints = sum(
            1 for signal, dps in self.DATAPOINTS.items()
            for dp in dps
            if record.get(dp) is not None
        )
        data_coverage = Decimal(str(available_datapoints)) / Decimal(str(total_datapoints))
        data_coverage = _quantize(data_coverage * Decimal("100"))

        # Audit entry
        deterministic_ts = f"{as_of_date.isoformat()}T00:00:00Z" if as_of_date else None
        audit_entry = {
            "timestamp": deterministic_ts,
            "ticker": ticker_upper,
            "regime": regime,
            "composite_score": str(composite),
            "confidence": str(confidence),
            "sub_scores": {k: str(v) if v is not None else None for k, v in sub_scores.items()},
            "data_coverage_pct": str(data_coverage),
            "module_version": self.VERSION,
        }
        self.audit_trail.append(audit_entry)

        return {
            "status": "SUCCESS",
            "ticker": ticker_upper,
            "morningstar_score": composite,
            "fair_value_discount_score": sub_scores.get("fair_value_discount"),
            "capital_efficiency_score": sub_scores.get("capital_efficiency"),
            "leverage_health_score": sub_scores.get("leverage_health"),
            "growth_quality_score": sub_scores.get("growth_quality"),
            "sub_details": sub_details,
            "confidence": confidence,
            "regime": regime,
            "flags": flags,
            "data_coverage_pct": data_coverage,
            "snapshot_date": self._snapshot_date,
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        market_data_by_ticker: Dict[str, Dict[str, Any]],
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Score an entire universe of tickers.

        Args:
            universe: List of dicts with at least a "ticker" key.
            market_data_by_ticker: Dict mapping ticker -> market data dict
                with optional "current_price" and "market_cap_mm" keys.
            as_of_date: Point-in-time date.

        Returns:
            Dict with scores_by_ticker, diagnostic_counts, and provenance.
        """
        scores_by_ticker: Dict[str, Dict[str, Any]] = {}
        regime_distribution: Dict[str, int] = {
            "commercial": 0, "development": 0, "unknown": 0
        }
        total_scored = 0
        total_with_fv = 0

        for item in universe:
            ticker = item.get("ticker", "UNKNOWN").upper()
            mkt = market_data_by_ticker.get(ticker, {})

            current_price = _to_decimal(mkt.get("current_price"))
            market_cap_mm = _to_decimal(mkt.get("market_cap_mm"))

            result = self.score_ticker(
                ticker=ticker,
                current_price=current_price,
                market_cap_mm=market_cap_mm,
                as_of_date=as_of_date,
            )

            scores_by_ticker[ticker] = result

            if result["status"] == "SUCCESS":
                total_scored += 1
                regime_distribution[result["regime"]] += 1
                if result.get("fair_value_discount_score") is not None:
                    total_with_fv += 1

        # Content hash for provenance
        hash_input = json.dumps(
            [{"t": t, "s": str(s.get("morningstar_score", ""))}
             for t, s in sorted(scores_by_ticker.items())],
            sort_keys=True,
        )
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "scores_by_ticker": scores_by_ticker,
            "diagnostic_counts": {
                "total_in_universe": len(universe),
                "total_scored": total_scored,
                "total_with_fair_value": total_with_fv,
                "fair_value_coverage_pct": f"{total_with_fv / max(1, len(universe)) * 100:.1f}%",
                "regime_distribution": regime_distribution,
            },
            "provenance": {
                "module": "morningstar_signal_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "snapshot_date": self._snapshot_date,
                "pit_cutoff": as_of_date.isoformat() if as_of_date else None,
            },
        }

    # =========================================================================
    # SUB-SIGNAL SCORING
    # =========================================================================

    def _score_fair_value_discount(
        self,
        record: Dict[str, str],
        current_price: Optional[Decimal],
    ) -> Dict[str, Any]:
        """Score fair value discount: (QV - price) / QV."""
        qv_raw = _to_decimal(record.get("QV009"))
        flags: List[str] = []

        if qv_raw is None or qv_raw <= Decimal("0"):
            return {"score": None, "flags": ["ms_fv_missing_qv"]}

        if current_price is None or current_price <= Decimal("0"):
            # Use Morningstar's market cap to infer a relative score from P/FV
            pf_raw = _to_decimal(record.get("OS603"))
            if pf_raw is not None and pf_raw > Decimal("0"):
                # OS603 = Price/Fair Value. discount = (1 - P/FV) * 100
                discount_pct = (Decimal("1") - pf_raw) * Decimal("100")
                flags.append("ms_fv_from_pfv_ratio")
            else:
                return {"score": None, "flags": ["ms_fv_missing_price"]}
        else:
            discount_pct = (qv_raw - current_price) / qv_raw * Decimal("100")

        score = self._interpolate_breakpoints(discount_pct, self.FV_DISCOUNT_BREAKPOINTS)
        score = _clamp(score, Decimal("5"), Decimal("95"))

        if discount_pct >= Decimal("30"):
            flags.append("ms_deeply_undervalued")
        elif discount_pct <= Decimal("-30"):
            flags.append("ms_deeply_overvalued")

        return {
            "score": _quantize(score),
            "discount_pct": _quantize(discount_pct),
            "quantitative_fv": str(qv_raw),
            "current_price": str(current_price) if current_price else None,
            "flags": flags,
        }

    def _score_capital_efficiency(
        self,
        record: Dict[str, str],
        regime: str,
    ) -> Dict[str, Any]:
        """Score capital efficiency from ROIC and ROE."""
        flags: List[str] = []

        # Dev-stage: return neutral
        if regime == "development":
            return {
                "score": Decimal("50"),
                "flags": ["ms_ce_neutral_dev_stage"],
                "roic": None,
                "roe": None,
            }

        roic = _to_decimal(record.get("STA4Z"))
        roe = _to_decimal(record.get("HS08F"))

        scores: List[Decimal] = []

        if roic is not None:
            roic_score = self._interpolate_breakpoints(roic, self.ROIC_BREAKPOINTS)
            scores.append(_clamp(roic_score, Decimal("5"), Decimal("95")))

        if roe is not None:
            roe_score = self._interpolate_breakpoints(roe, self.ROE_BREAKPOINTS)
            scores.append(_clamp(roe_score, Decimal("5"), Decimal("95")))

        if not scores:
            return {"score": None, "flags": ["ms_ce_no_data"], "roic": None, "roe": None}

        avg = sum(scores) / Decimal(str(len(scores)))

        return {
            "score": _quantize(avg),
            "roic": str(roic) if roic is not None else None,
            "roe": str(roe) if roe is not None else None,
            "flags": flags,
        }

    def _score_leverage_health(
        self,
        record: Dict[str, str],
    ) -> Dict[str, Any]:
        """Score leverage health from D/E and D/Capital."""
        de = _to_decimal(record.get("ST389"))
        dcap = _to_decimal(record.get("HS06U"))
        flags: List[str] = []

        scores: List[Decimal] = []

        if de is not None:
            # D/E can be negative (negative equity) — treat as very high leverage
            if de < Decimal("0"):
                de_score = Decimal("10")
                flags.append("ms_negative_equity")
            else:
                de_score = self._interpolate_breakpoints(de, self.DE_BREAKPOINTS, ascending=False)
            scores.append(_clamp(de_score, Decimal("5"), Decimal("95")))

        if dcap is not None:
            if dcap < Decimal("0"):
                dcap_score = Decimal("10")
            else:
                dcap_score = self._interpolate_breakpoints(dcap, self.DCAP_BREAKPOINTS, ascending=False)
            scores.append(_clamp(dcap_score, Decimal("5"), Decimal("95")))

        if not scores:
            return {"score": None, "flags": ["ms_lh_no_data"], "de": None, "dcap": None}

        avg = sum(scores) / Decimal(str(len(scores)))

        if avg < Decimal("30"):
            flags.append("ms_high_leverage")
        elif avg > Decimal("70"):
            flags.append("ms_low_leverage")

        return {
            "score": _quantize(avg),
            "de": str(de) if de is not None else None,
            "dcap": str(dcap) if dcap is not None else None,
            "flags": flags,
        }

    def _score_growth_quality(
        self,
        record: Dict[str, str],
        regime: str,
    ) -> Dict[str, Any]:
        """Score growth quality from sales growth and net margin."""
        flags: List[str] = []

        # Dev-stage: return neutral
        if regime == "development":
            return {
                "score": Decimal("50"),
                "flags": ["ms_gq_neutral_dev_stage"],
                "sales_growth": None,
                "net_margin": None,
            }

        sales_growth = _to_decimal(record.get("HS035"))
        net_margin = _to_decimal(record.get("HS08D"))

        scores: List[Decimal] = []

        if sales_growth is not None:
            sg_score = self._interpolate_breakpoints(sales_growth, self.SALES_GROWTH_BREAKPOINTS)
            scores.append(_clamp(sg_score, Decimal("5"), Decimal("95")))

        if net_margin is not None:
            nm_score = self._interpolate_breakpoints(net_margin, self.NET_MARGIN_BREAKPOINTS)
            scores.append(_clamp(nm_score, Decimal("5"), Decimal("95")))

        if not scores:
            return {"score": None, "flags": ["ms_gq_no_data"], "sales_growth": None, "net_margin": None}

        avg = sum(scores) / Decimal(str(len(scores)))

        return {
            "score": _quantize(avg),
            "sales_growth": str(sales_growth) if sales_growth is not None else None,
            "net_margin": str(net_margin) if net_margin is not None else None,
            "flags": flags,
        }

    # =========================================================================
    # COMPOSITE COMPUTATION
    # =========================================================================

    def _compute_composite(
        self,
        sub_scores: Dict[str, Optional[Decimal]],
    ) -> Tuple[Decimal, Decimal, int]:
        """
        Compute confidence-gated weighted composite from sub-scores.

        Missing sub-signals have their weight redistributed proportionally
        to the available sub-signals.

        Returns:
            (composite_score, confidence, available_count)
        """
        available: Dict[str, Decimal] = {}
        for name, score in sub_scores.items():
            if score is not None:
                available[name] = score

        if not available:
            return Decimal("50"), Decimal("0.1"), 0

        # Redistribute weights to available signals only
        total_available_weight = sum(
            self.SUB_SIGNAL_WEIGHTS[name] for name in available
        )

        if total_available_weight <= Decimal("0"):
            return Decimal("50"), Decimal("0.1"), 0

        composite = Decimal("0")
        for name, score in available.items():
            w = self.SUB_SIGNAL_WEIGHTS[name] / total_available_weight
            composite += w * score

        composite = _clamp(_quantize(composite), Decimal("5"), Decimal("95"))

        # Confidence based on coverage (more sub-signals = higher confidence)
        total_signals = len(self.SUB_SIGNAL_WEIGHTS)
        coverage_ratio = Decimal(str(len(available))) / Decimal(str(total_signals))
        confidence = _clamp(
            _quantize(Decimal("0.3") + coverage_ratio * Decimal("0.7")),
            Decimal("0.1"),
            Decimal("1.0"),
        )

        return composite, confidence, len(available)

    # =========================================================================
    # PRICE HISTORY LOOKUP
    # =========================================================================

    def _get_latest_price(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Decimal]:
        """
        Get the latest closing price from Morningstar price history (HS377).

        Uses the most recent entry on or before as_of_date. If as_of_date is
        None, uses the last available entry. Rejects prices older than 7 days
        from as_of_date to avoid stale data.

        Returns:
            Decimal price or None if unavailable/stale.
        """
        price_rec = self._price_history.get(ticker)
        if not price_rec:
            return None

        hs377 = price_rec.get("HS377")
        if not hs377:
            return None

        # Handle both formats: dict with time_series or direct string
        if isinstance(hs377, dict):
            ts = hs377.get("time_series", [])
        else:
            return _to_decimal(hs377)

        if not ts:
            return None

        if as_of_date is None:
            # Use the last entry
            entry = ts[-1]
        else:
            # Find the latest entry on or before as_of_date
            cutoff = as_of_date.isoformat()
            entry = None
            for e in reversed(ts):
                if e.get("date", "") <= cutoff:
                    entry = e
                    break

        if entry is None:
            return None

        # Staleness check: reject prices older than 7 calendar days
        if as_of_date is not None:
            entry_date_str = entry.get("date", "")
            if entry_date_str:
                try:
                    entry_date = date.fromisoformat(entry_date_str)
                    if (as_of_date - entry_date).days > 7:
                        return None
                except ValueError:
                    pass

        return _to_decimal(entry.get("value"))

    # =========================================================================
    # REGIME CLASSIFICATION
    # =========================================================================

    def _classify_regime(
        self,
        record: Dict[str, str],
        market_cap_mm: Optional[Decimal],
    ) -> str:
        """
        Classify ticker as 'commercial' or 'development' based on available data.

        Commercial indicators: positive revenue metrics (sales growth, net margin,
        positive ROIC/ROE). Development: deeply negative margins, no revenue data.
        """
        net_margin = _to_decimal(record.get("HS08D"))
        sales_growth = _to_decimal(record.get("HS035"))
        roic = _to_decimal(record.get("STA4Z"))
        ps_ratio = _to_decimal(record.get("HS05U"))
        pe_ratio = _to_decimal(record.get("HS05X"))

        # Strong commercial signals
        commercial_signals = 0

        if net_margin is not None and net_margin > Decimal("-50"):
            commercial_signals += 1
        if sales_growth is not None:
            commercial_signals += 1  # Has measurable revenue
        if roic is not None and roic > Decimal("-5"):
            commercial_signals += 1
        if ps_ratio is not None and ps_ratio > Decimal("0") and ps_ratio < Decimal("50"):
            commercial_signals += 1
        if pe_ratio is not None and pe_ratio > Decimal("0"):
            commercial_signals += 1  # Profitable enough for P/E

        # Deeply negative margins are a strong dev-stage signal
        if net_margin is not None and net_margin < Decimal("-200"):
            return "development"

        if commercial_signals >= 3:
            return "commercial"

        return "development"

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _interpolate_breakpoints(
        self,
        value: Decimal,
        breakpoints: List[Tuple[Decimal, Decimal]],
        ascending: bool = True,
    ) -> Decimal:
        """
        Piecewise-linear interpolation between breakpoints.

        Args:
            value: The input metric value.
            breakpoints: List of (threshold, score) tuples, sorted descending
                         by threshold for ascending metrics (higher value = higher score)
                         or ascending for descending metrics.
            ascending: If True, higher values get higher scores. If False, lower
                       values get higher scores (e.g., D/E ratio).
        """
        if not breakpoints:
            return Decimal("50")

        # For descending metrics (lower = better), we reverse the value comparison
        # but the breakpoints are already ordered high-threshold-first
        if not ascending:
            # Breakpoints are (threshold, score) where lower threshold = higher score
            # We iterate from lowest threshold to highest
            for i, (thresh, score) in enumerate(breakpoints):
                if value <= thresh:
                    if i == 0:
                        return score
                    # Interpolate between this and previous breakpoint
                    prev_thresh, prev_score = breakpoints[i - 1]
                    if thresh == prev_thresh:
                        return score
                    frac = (value - prev_thresh) / (thresh - prev_thresh)
                    return prev_score + frac * (score - prev_score)
            # Beyond last breakpoint
            return breakpoints[-1][1]

        # Ascending: higher value = higher score
        # Breakpoints sorted descending by threshold
        for i, (thresh, score) in enumerate(breakpoints):
            if value >= thresh:
                if i == 0:
                    return score
                # Interpolate between this and previous (higher) breakpoint
                prev_thresh, prev_score = breakpoints[i - 1]
                if prev_thresh == thresh:
                    return score
                frac = (value - thresh) / (prev_thresh - thresh)
                return score + frac * (prev_score - score)

        # Below all breakpoints
        return breakpoints[-1][1]

    def _no_data_result(
        self,
        ticker: str,
        as_of_date: Optional[date],
    ) -> Dict[str, Any]:
        """Return standardized result when no Morningstar data exists."""
        deterministic_ts = f"{as_of_date.isoformat()}T00:00:00Z" if as_of_date else None
        audit_entry = {
            "timestamp": deterministic_ts,
            "ticker": ticker,
            "status": "NO_DATA",
            "module_version": self.VERSION,
        }
        self.audit_trail.append(audit_entry)

        return {
            "status": "NO_DATA",
            "ticker": ticker,
            "morningstar_score": None,
            "fair_value_discount_score": None,
            "capital_efficiency_score": None,
            "leverage_health_score": None,
            "growth_quality_score": None,
            "sub_details": {},
            "confidence": Decimal("0"),
            "regime": "unknown",
            "flags": ["ms_no_data"],
            "data_coverage_pct": Decimal("0"),
            "snapshot_date": self._snapshot_date,
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the full audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


# =============================================================================
# STANDALONE DEMONSTRATION
# =============================================================================

def demonstration() -> None:
    """Demonstrate the Morningstar signal engine against production data."""
    import sys

    data_dir = Path("production_data")
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    print("=" * 70)
    print("MORNINGSTAR SIGNAL ENGINE - DEMONSTRATION")
    print("=" * 70)
    print()

    engine = MorningstarSignalEngine()
    loaded = engine.load_data(data_dir)
    print(f"Loaded: {loaded} tickers")

    if loaded == 0:
        print("No data found. Pass data directory as argument.")
        return

    # Score a few example tickers
    test_tickers = ["VRTX", "BMRN", "ALKS", "ACRS", "ABVX"]
    as_of = date(2026, 2, 6)

    for ticker in test_tickers:
        result = engine.score_ticker(ticker, as_of_date=as_of)
        if result["status"] == "SUCCESS":
            print(f"{ticker}: score={result['morningstar_score']}  "
                  f"conf={result['confidence']}  regime={result['regime']}  "
                  f"FV={result['fair_value_discount_score']}  "
                  f"CE={result['capital_efficiency_score']}  "
                  f"LH={result['leverage_health_score']}  "
                  f"GQ={result['growth_quality_score']}")
        else:
            print(f"{ticker}: {result['status']}")

    print()

    # Score full universe
    universe = [{"ticker": t} for t in engine._data.keys()]
    result = engine.score_universe(universe, {}, as_of)
    diag = result["diagnostic_counts"]
    print(f"Universe scored: {diag['total_scored']}/{diag['total_in_universe']}")
    print(f"Fair value coverage: {diag['fair_value_coverage_pct']}")
    print(f"Regime distribution: {diag['regime_distribution']}")
    print(f"Content hash: {result['provenance']['content_hash']}")


if __name__ == "__main__":
    demonstration()
