"""
macro_data_collector.py - Collect macro data for regime detection

Collects:
1. Yield curve slope (10Y-2Y spread) from FRED
2. High yield credit spread (OAS) from FRED
3. Biotech fund flows (calculated from ETF data)

Data Sources:
- FRED API (free, requires API key): https://fred.stlouisfed.org/docs/api/
- Yahoo Finance (for fund flow estimation)

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MacroDataPoint:
    """A single macro data observation."""
    series_id: str
    date: str  # ISO format
    value: str  # String to preserve precision
    source: str
    retrieved_at: str


@dataclass
class MacroSnapshot:
    """Complete macro data snapshot for regime detection."""
    as_of_date: str
    yield_curve_slope_bps: Optional[str]  # 10Y-2Y in basis points
    hy_credit_spread_bps: Optional[str]   # HY OAS in basis points
    biotech_fund_flows_mm: Optional[str]  # Weekly flows in $MM
    data_quality: Dict[str, Any]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# FRED API Collector
# =============================================================================

class FREDCollector:
    """
    Collector for Federal Reserve Economic Data (FRED).

    Provides yield curve and credit spread data.
    Free API with key registration at: https://fred.stlouisfed.org/docs/api/api_key.html

    Rate limit: 120 requests per minute
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # FRED series IDs
    SERIES = {
        "yield_curve_10y2y": "T10Y2Y",      # 10Y-2Y Treasury spread (already in %)
        "yield_10y": "DGS10",                # 10-Year Treasury yield
        "yield_2y": "DGS2",                  # 2-Year Treasury yield
        "hy_spread": "BAMLH0A0HYM2",        # ICE BofA US High Yield OAS (in %)
        "hy_spread_bb": "BAMLH0A1HYBB",     # BB-rated OAS
        "hy_spread_b": "BAMLH0A2HYB",       # B-rated OAS
        "hy_spread_ccc": "BAMLH0A3HYC",     # CCC & lower OAS
        "ig_spread": "BAMLC0A0CM",          # Investment grade OAS
    }

    # Default FRED API key (registered for Wake Robin Capital Management)
    DEFAULT_API_KEY = "329d666b93ff448cecfab1004f81f670"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize FRED collector.

        Args:
            api_key: FRED API key. If None, uses default key or FRED_API_KEY env var.
            cache_dir: Directory for caching. Defaults to module cache dir.
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY") or self.DEFAULT_API_KEY
        if not self.api_key:
            logger.warning("No FRED API key provided. Set FRED_API_KEY environment variable.")

        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache" / "fred"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._request_count = 0
        self._last_request_time = 0.0

    def _get_cache_path(self, series_id: str, as_of_date: date) -> Path:
        """Get cache file path for a series and date."""
        return self.cache_dir / f"{series_id}_{as_of_date.isoformat()}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age < timedelta(hours=max_age_hours)

    def _rate_limit(self) -> None:
        """Enforce rate limiting (120 req/min = 0.5 sec between requests)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    def _fetch_series(
        self,
        series_id: str,
        observation_start: date,
        observation_end: date,
    ) -> List[Dict[str, str]]:
        """
        Fetch series data from FRED API.

        Args:
            series_id: FRED series ID
            observation_start: Start date for observations
            observation_end: End date for observations

        Returns:
            List of observation dicts with 'date' and 'value' keys
        """
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY environment variable.")

        try:
            import requests
        except ImportError:
            raise ImportError("requests library required: pip install requests")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start.isoformat(),
            "observation_end": observation_end.isoformat(),
            "sort_order": "desc",  # Most recent first
        }

        self._rate_limit()

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])
            # Filter out missing values (FRED uses "." for missing)
            return [
                {"date": obs["date"], "value": obs["value"]}
                for obs in observations
                if obs.get("value") and obs["value"] != "."
            ]

        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            raise

    def get_yield_curve_slope(
        self,
        as_of_date: date,
        use_cache: bool = True,
    ) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """
        Get 10Y-2Y Treasury spread (yield curve slope) in basis points.

        Args:
            as_of_date: Point-in-time date
            use_cache: Whether to use cached data

        Returns:
            Tuple of (value in bps, metadata dict)
        """
        series_id = self.SERIES["yield_curve_10y2y"]
        cache_path = self._get_cache_path(series_id, as_of_date)

        metadata = {
            "series_id": series_id,
            "source": "FRED",
            "as_of_date": as_of_date.isoformat(),
        }

        # Check cache
        if use_cache and self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                    metadata["from_cache"] = True
                    metadata["cache_timestamp"] = cached.get("retrieved_at")
                    value_pct = cached.get("value")
                    if value_pct:
                        # Convert from percentage to basis points
                        value_bps = Decimal(value_pct) * Decimal("100")
                        return value_bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), metadata
                    return None, metadata
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Fetch from API
        try:
            # Look back up to 7 days to find data (markets closed on weekends/holidays)
            start_date = as_of_date - timedelta(days=7)
            observations = self._fetch_series(series_id, start_date, as_of_date)

            if not observations:
                metadata["error"] = "No observations found"
                return None, metadata

            # Get most recent observation on or before as_of_date
            latest = observations[0]
            value_pct = Decimal(latest["value"])
            value_bps = value_pct * Decimal("100")  # Convert to basis points

            # Cache the result
            cache_data = {
                "series_id": series_id,
                "date": latest["date"],
                "value": latest["value"],
                "retrieved_at": datetime.now().isoformat(),
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            metadata["observation_date"] = latest["date"]
            metadata["from_cache"] = False

            return value_bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), metadata

        except Exception as e:
            logger.error(f"Failed to fetch yield curve data: {e}")
            metadata["error"] = str(e)
            return None, metadata

    def get_hy_credit_spread(
        self,
        as_of_date: date,
        use_cache: bool = True,
    ) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """
        Get High Yield credit spread (OAS) in basis points.

        Args:
            as_of_date: Point-in-time date
            use_cache: Whether to use cached data

        Returns:
            Tuple of (value in bps, metadata dict)
        """
        series_id = self.SERIES["hy_spread"]
        cache_path = self._get_cache_path(series_id, as_of_date)

        metadata = {
            "series_id": series_id,
            "source": "FRED (ICE BofA)",
            "as_of_date": as_of_date.isoformat(),
        }

        # Check cache
        if use_cache and self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                    metadata["from_cache"] = True
                    metadata["cache_timestamp"] = cached.get("retrieved_at")
                    value_pct = cached.get("value")
                    if value_pct:
                        # FRED reports OAS in percentage, convert to bps
                        value_bps = Decimal(value_pct) * Decimal("100")
                        return value_bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), metadata
                    return None, metadata
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Fetch from API
        try:
            start_date = as_of_date - timedelta(days=7)
            observations = self._fetch_series(series_id, start_date, as_of_date)

            if not observations:
                metadata["error"] = "No observations found"
                return None, metadata

            latest = observations[0]
            value_pct = Decimal(latest["value"])
            value_bps = value_pct * Decimal("100")

            # Cache
            cache_data = {
                "series_id": series_id,
                "date": latest["date"],
                "value": latest["value"],
                "retrieved_at": datetime.now().isoformat(),
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            metadata["observation_date"] = latest["date"]
            metadata["from_cache"] = False

            return value_bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), metadata

        except Exception as e:
            logger.error(f"Failed to fetch HY spread data: {e}")
            metadata["error"] = str(e)
            return None, metadata


# =============================================================================
# ETF Fund Flow Collector
# =============================================================================

class FundFlowCollector:
    """
    Collector for biotech ETF fund flows.

    IMPORTANT: This provides a ROUGH ESTIMATE only.
    True fund flow data requires paid subscriptions (Bloomberg, Refinitiv, ETF.com).

    Method:
    - Aggregates flows from multiple biotech ETFs (XBI + IBB by default)
    - Trading volume ≠ fund flows (most trading is secondary market)
    - Only Authorized Participant (AP) creation/redemption creates flows
    - We estimate AP activity as ~2% of dollar volume (industry typical: 1-3%)
    - Direction inferred from price movement

    Accuracy: ±50% at best. Use for directional signal only, not precise amounts.
    """

    # Default biotech ETFs to aggregate for sector-wide flow signal
    DEFAULT_BIOTECH_ETFS = ["XBI", "IBB"]

    # AP activity rate estimate (1-3% typical, we use 2%)
    AP_ACTIVITY_RATE = Decimal("0.02")

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize fund flow collector.

        Args:
            cache_dir: Directory for caching. Defaults to module cache dir.
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache" / "fund_flows"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, ticker: str, as_of_date: date) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{ticker}_{as_of_date.isoformat()}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache is valid."""
        if not cache_path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age < timedelta(hours=max_age_hours)

    def get_weekly_fund_flow(
        self,
        ticker: str,
        as_of_date: date,
        use_cache: bool = True,
    ) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """
        Calculate approximate weekly fund flow for an ETF.

        Args:
            ticker: ETF ticker (e.g., "XBI")
            as_of_date: Point-in-time date
            use_cache: Whether to use cached data

        Returns:
            Tuple of (flow in $MM, metadata dict)
        """
        cache_path = self._get_cache_path(ticker, as_of_date)

        metadata = {
            "ticker": ticker,
            "source": "Calculated from Yahoo Finance",
            "as_of_date": as_of_date.isoformat(),
            "method": "AUM delta minus price return",
        }

        # Check cache
        if use_cache and self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                    metadata["from_cache"] = True
                    metadata["cache_timestamp"] = cached.get("retrieved_at")
                    flow = cached.get("flow_mm")
                    if flow is not None:
                        return Decimal(str(flow)), metadata
                    return None, metadata
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Calculate flow from Yahoo Finance data
        try:
            import yfinance as yf
        except ImportError:
            metadata["error"] = "yfinance not installed"
            return None, metadata

        try:
            etf = yf.Ticker(ticker)

            # Get historical data for the week
            end_date = as_of_date + timedelta(days=1)  # yfinance end is exclusive
            start_date = as_of_date - timedelta(days=10)  # Extra days for holidays

            hist = etf.history(start=start_date.isoformat(), end=end_date.isoformat())

            if hist.empty or len(hist) < 2:
                metadata["error"] = "Insufficient historical data"
                return None, metadata

            # Get current info for shares outstanding
            info = etf.info
            shares_outstanding = info.get("sharesOutstanding")

            # For ETFs, sharesOutstanding may not be available
            # Calculate from totalAssets / navPrice if needed
            if not shares_outstanding:
                total_assets = info.get("totalAssets") or info.get("netAssets")
                nav_price = info.get("navPrice")
                if total_assets and nav_price and nav_price > 0:
                    shares_outstanding = total_assets / nav_price
                    metadata["shares_calculated"] = True
                else:
                    metadata["error"] = "Missing shares outstanding and cannot calculate from AUM/NAV"
                    return None, metadata

            # Calculate weekly return and volume-based flow estimate
            # Use last 5 trading days
            recent = hist.tail(6)  # 6 rows to get 5-day return

            if len(recent) < 2:
                metadata["error"] = "Insufficient data for weekly calculation"
                return None, metadata

            price_start = float(recent.iloc[0]["Close"])
            price_end = float(recent.iloc[-1]["Close"])

            # Weekly price return
            price_return = (price_end - price_start) / price_start

            # Fund flow estimation
            # IMPORTANT: Trading volume ≠ fund flows
            # - Most ETF trading occurs on secondary market (no flow impact)
            # - Only Authorized Participant (AP) creation/redemption affects flows
            # - AP activity is typically 1-3% of daily volume
            #
            # Method: Estimate AP activity from volume + price pressure
            recent_volume = recent["Volume"].sum() if "Volume" in recent.columns else 0
            avg_price = (price_start + price_end) / 2
            dollar_volume = recent_volume * avg_price

            # Get current AUM for context
            current_aum = info.get("totalAssets") or info.get("netAssets") or 0

            if current_aum > 0 and dollar_volume > 0:
                # Direction based on price trend
                flow_direction = 1 if price_return > 0 else -1

                # Estimated flow = AP activity portion of volume
                estimated_flow = flow_direction * dollar_volume * float(self.AP_ACTIVITY_RATE)
            else:
                estimated_flow = 0

            # Convert to millions
            flow_mm = Decimal(str(estimated_flow / 1_000_000))
            flow_mm = flow_mm.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Cache result
            cache_data = {
                "ticker": ticker,
                "flow_mm": str(flow_mm),
                "price_start": price_start,
                "price_end": price_end,
                "price_return": price_return,
                "dollar_volume": dollar_volume,
                "current_aum": current_aum,
                "retrieved_at": datetime.now().isoformat(),
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            metadata["from_cache"] = False
            metadata["price_return"] = f"{price_return:.4f}"
            metadata["dollar_volume"] = f"{dollar_volume:,.0f}"
            metadata["ap_rate"] = "2%"
            metadata["calculation_note"] = "ESTIMATE: ~2% of volume as AP activity. For accurate data use Bloomberg/ETF.com"

            return flow_mm, metadata

        except Exception as e:
            logger.error(f"Failed to calculate fund flow for {ticker}: {e}")
            metadata["error"] = str(e)
            return None, metadata

    def get_aggregated_biotech_flows(
        self,
        as_of_date: date,
        etfs: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """
        Calculate aggregated fund flows across multiple biotech ETFs.

        Aggregates flows from XBI and IBB (by default) for a sector-wide signal.
        This provides a more robust estimate than a single ETF.

        Args:
            as_of_date: Point-in-time date
            etfs: List of ETF tickers to aggregate (default: ["XBI", "IBB"])
            use_cache: Whether to use cached data

        Returns:
            Tuple of (total flow in $MM, metadata dict)
        """
        etfs = etfs or self.DEFAULT_BIOTECH_ETFS

        metadata = {
            "etfs": etfs,
            "source": "Aggregated from Yahoo Finance",
            "as_of_date": as_of_date.isoformat(),
            "method": "Sum of individual ETF flow estimates",
        }

        total_flow = Decimal("0")
        etf_details = {}
        errors = []

        for ticker in etfs:
            flow, etf_meta = self.get_weekly_fund_flow(ticker, as_of_date, use_cache)
            etf_details[ticker] = etf_meta

            if flow is not None:
                total_flow += flow
            else:
                errors.append(f"{ticker}: {etf_meta.get('error', 'Unknown error')}")

        metadata["etf_details"] = etf_details
        metadata["successful_etfs"] = len(etfs) - len(errors)
        metadata["total_etfs"] = len(etfs)

        if errors:
            metadata["errors"] = errors

        # If no ETFs returned data, return None
        if metadata["successful_etfs"] == 0:
            metadata["error"] = "No ETF data available"
            return None, metadata

        metadata["calculation_note"] = (
            f"ESTIMATE: Aggregated flows from {metadata['successful_etfs']}/{len(etfs)} ETFs. "
            "~2% of volume as AP activity. For accurate data use Bloomberg/ETF.com"
        )

        return total_flow, metadata


# =============================================================================
# Main Collector Interface
# =============================================================================

class MacroDataCollector:
    """
    Unified collector for all macro data needed by the regime engine.

    Usage:
        collector = MacroDataCollector(fred_api_key="your_key")
        snapshot = collector.collect_snapshot(date(2026, 1, 15))

        # Use in regime engine
        engine.detect_regime(
            vix_current=...,
            xbi_vs_spy_30d=...,
            yield_curve_slope=snapshot.yield_curve_slope_bps,
            hy_credit_spread=snapshot.hy_credit_spread_bps,
            biotech_fund_flows=snapshot.biotech_fund_flows_mm,
        )
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize macro data collector.

        Args:
            fred_api_key: FRED API key. Reads from FRED_API_KEY env var if not provided.
            cache_dir: Base cache directory. Defaults to module cache dir.
        """
        base_cache = cache_dir or Path(__file__).parent.parent / "cache"

        self.fred = FREDCollector(
            api_key=fred_api_key,
            cache_dir=base_cache / "fred",
        )
        self.fund_flow = FundFlowCollector(cache_dir=base_cache / "fund_flows")

    def collect_snapshot(
        self,
        as_of_date: date,
        biotech_etfs: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> MacroSnapshot:
        """
        Collect all macro data for a given date.

        Args:
            as_of_date: Point-in-time date
            biotech_etfs: List of biotech ETF tickers for fund flow aggregation
                         (default: ["XBI", "IBB"])
            use_cache: Whether to use cached data

        Returns:
            MacroSnapshot with all available data
        """
        data_quality = {
            "has_yield_curve": False,
            "has_hy_spread": False,
            "has_fund_flows": False,
            "errors": [],
        }
        provenance = {
            "collected_at": datetime.now().isoformat(),
            "sources": {},
        }

        # Yield curve slope
        yield_curve_bps = None
        try:
            value, meta = self.fred.get_yield_curve_slope(as_of_date, use_cache)
            if value is not None:
                yield_curve_bps = str(value)
                data_quality["has_yield_curve"] = True
            provenance["sources"]["yield_curve"] = meta
        except Exception as e:
            data_quality["errors"].append(f"yield_curve: {e}")

        # HY credit spread
        hy_spread_bps = None
        try:
            value, meta = self.fred.get_hy_credit_spread(as_of_date, use_cache)
            if value is not None:
                hy_spread_bps = str(value)
                data_quality["has_hy_spread"] = True
            provenance["sources"]["hy_spread"] = meta
        except Exception as e:
            data_quality["errors"].append(f"hy_spread: {e}")

        # Fund flows (aggregated from multiple biotech ETFs)
        fund_flows_mm = None
        try:
            value, meta = self.fund_flow.get_aggregated_biotech_flows(
                as_of_date, etfs=biotech_etfs, use_cache=use_cache
            )
            if value is not None:
                fund_flows_mm = str(value)
                data_quality["has_fund_flows"] = True
            provenance["sources"]["fund_flows"] = meta
        except Exception as e:
            data_quality["errors"].append(f"fund_flows: {e}")

        # Calculate completeness score
        complete_fields = sum([
            data_quality["has_yield_curve"],
            data_quality["has_hy_spread"],
            data_quality["has_fund_flows"],
        ])
        data_quality["completeness"] = f"{complete_fields}/3"

        return MacroSnapshot(
            as_of_date=as_of_date.isoformat(),
            yield_curve_slope_bps=yield_curve_bps,
            hy_credit_spread_bps=hy_spread_bps,
            biotech_fund_flows_mm=fund_flows_mm,
            data_quality=data_quality,
            provenance=provenance,
        )

    def to_regime_engine_params(
        self,
        snapshot: MacroSnapshot,
    ) -> Dict[str, Optional[Decimal]]:
        """
        Convert snapshot to parameters for detect_regime().

        Args:
            snapshot: MacroSnapshot from collect_snapshot()

        Returns:
            Dict ready to pass to detect_regime(**kwargs)
        """
        return {
            "yield_curve_slope": (
                Decimal(snapshot.yield_curve_slope_bps)
                if snapshot.yield_curve_slope_bps else None
            ),
            "hy_credit_spread": (
                Decimal(snapshot.hy_credit_spread_bps)
                if snapshot.hy_credit_spread_bps else None
            ),
            "biotech_fund_flows": (
                Decimal(snapshot.biotech_fund_flows_mm)
                if snapshot.biotech_fund_flows_mm else None
            ),
        }


# =============================================================================
# CLI and Testing
# =============================================================================

def demonstration() -> None:
    """Demonstrate the macro data collector."""
    print("=" * 70)
    print("MACRO DATA COLLECTOR - DEMONSTRATION")
    print("=" * 70)
    print()

    # Check for API key
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("NOTE: FRED_API_KEY not set. FRED data will not be available.")
        print("      Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print()

    collector = MacroDataCollector()

    # Use a recent date
    test_date = date.today() - timedelta(days=1)
    print(f"Collecting macro data for: {test_date}")
    print("-" * 70)

    snapshot = collector.collect_snapshot(test_date)

    print(f"\nResults:")
    print(f"  Yield Curve Slope: {snapshot.yield_curve_slope_bps or 'N/A'} bps")
    print(f"  HY Credit Spread:  {snapshot.hy_credit_spread_bps or 'N/A'} bps")
    print(f"  Fund Flows (XBI):  {snapshot.biotech_fund_flows_mm or 'N/A'} MM")

    print(f"\nData Quality: {snapshot.data_quality.get('completeness', 'N/A')}")
    if snapshot.data_quality.get("errors"):
        print(f"  Errors:")
        for err in snapshot.data_quality["errors"]:
            print(f"    - {err}")

    # Show provenance details
    print(f"\nData Sources:")
    for source_name, meta in snapshot.provenance.get("sources", {}).items():
        status = "OK" if not meta.get("error") else f"Error: {meta.get('error', '')[:50]}"
        obs_date = meta.get("observation_date", meta.get("as_of_date", "N/A"))
        print(f"  {source_name}: {status} (date: {obs_date})")

    print(f"\nRegime Engine Parameters:")
    params = collector.to_regime_engine_params(snapshot)
    for key, value in params.items():
        print(f"  {key}: {value}")

    print()
    print("Data Availability Notes:")
    print("  - FRED (yield curve, HY spread): Free with API key, updated daily")
    print("  - Fund flows: Requires yfinance; approximation only")
    print("  - VIX (from separate source) serves as sentiment indicator")
    print()
    print("=" * 70)


if __name__ == "__main__":
    demonstration()
