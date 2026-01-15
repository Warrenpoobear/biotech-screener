#!/usr/bin/env python3
"""
Wake Robin - Morningstar Returns Signal Generator (CORRECTED)

Implements 4 high-ROI momentum signals with proper governance:
1. Multi-horizon risk-adjusted momentum (dimensionally consistent Sharpe)
2. Relative strength vs XBI (benchmark-aligned windows)
3. Beta + idiosyncratic momentum (with beta clamping)
4. Drawdown risk gate (benchmark-aligned)

CORRECTIONS FROM V1:
- Proper Decimal sqrt with fixed context
- Weights sum to 1.0 (not 0.85)
- Benchmark-aligned windows throughout
- Dimensionally consistent Sharpe ratios
- Confidence tiers instead of fail-loud
- Deterministic quantization
- Input hashing for provenance
- Beta clamping

GOVERNANCE: stdlib-only, Decimal arithmetic, complete audit trails
"""

from decimal import Decimal, ROUND_HALF_UP, getcontext, InvalidOperation
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Union
from typing_extensions import TypedDict
import json
import hashlib


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Type aliases
ReturnsData = Dict[str, Decimal]


class MultiHorizonSharpe(TypedDict, total=False):
    """Multi-horizon Sharpe ratio results."""
    short_20d: Decimal
    medium_60d: Decimal
    long_120d: Decimal
    composite: Decimal


class RelativeStrengthResult(TypedDict, total=False):
    """Relative strength calculation results."""
    rs_20d: Decimal
    rs_60d: Decimal
    rs_120d: Decimal
    composite: Decimal


class IdiosyncraticMomentum(TypedDict):
    """Idiosyncratic momentum calculation results."""
    beta_60d: Decimal
    idio_return_60d: Decimal
    idio_sharpe: Decimal
    n_aligned_obs: int


class DrawdownGate(TypedDict, total=False):
    """Drawdown gate calculation results."""
    max_drawdown_120d: Decimal
    max_drawdown_180d: Decimal
    risk_penalty: Decimal


class AlignedWindow(TypedDict):
    """Benchmark-aligned window data."""
    ticker: Dict[str, Decimal]
    xbi: Dict[str, Decimal]
    n_missing: int
    coverage_pct: Decimal


class Provenance(TypedDict):
    """Calculation provenance information."""
    ticker: str
    calc_date: str
    n_observations: int
    ticker_window_hash: str
    xbi_window_hash: str
    calculation_timestamp: str


class AllSignalsResult(TypedDict):
    """Complete momentum signals result."""
    multi_horizon_sharpe: MultiHorizonSharpe
    relative_strength_vs_xbi: RelativeStrengthResult
    idiosyncratic_momentum: IdiosyncraticMomentum
    drawdown_gate: DrawdownGate
    composite_momentum_score: Decimal
    confidence_tier: str
    confidence_multiplier: Decimal
    provenance: Provenance


class AuditEntry(TypedDict):
    """Audit trail entry."""
    ticker: str
    calc_date: str
    composite_score: str
    confidence_tier: str
    timestamp: str

# Set fixed Decimal context for deterministic calculations
getcontext().prec = 28  # High precision for sqrt operations


class MorningstarMomentumSignals:
    """Generate momentum signals from Morningstar returns data."""

    # Time horizons (trading days)
    HORIZONS = {
        "short": 20,    # ~1 month
        "medium": 60,   # ~3 months
        "long": 120     # ~6 months
    }

    # Annualization factor (proper Decimal sqrt)
    ANNUAL_FACTOR = Decimal("252")
    SQRT_ANNUAL = Decimal("252").sqrt()  # Computed with fixed precision

    # Confidence tiers based on data availability
    CONFIDENCE_TIERS = {
        "HIGH": {"min_obs": 120, "weight_multiplier": Decimal("1.0")},
        "MEDIUM": {"min_obs": 60, "weight_multiplier": Decimal("0.75")},
        "LOW": {"min_obs": 20, "weight_multiplier": Decimal("0.5")},
        "UNKNOWN": {"min_obs": 0, "weight_multiplier": Decimal("0.25")}
    }

    # Beta clamping bounds (prevent insane idiosyncratic math)
    BETA_MIN = Decimal("-1.0")
    BETA_MAX = Decimal("3.0")

    # Quantization precisions (deterministic rounding)
    PRECISION = {
        "daily_return": Decimal("0.0000001"),
        "volatility": Decimal("0.0001"),
        "sharpe": Decimal("0.0001"),
        "score": Decimal("0.01")
    }

    def __init__(self) -> None:
        self.audit_trail: List[AuditEntry] = []

    def calculate_all_signals(
        self,
        ticker: str,
        returns_data: ReturnsData,
        xbi_returns_data: ReturnsData,
        calc_date: str
    ) -> AllSignalsResult:
        """
        Calculate all 4 momentum signals for a ticker.

        Args:
            ticker: Stock symbol (e.g., "IONS")
            returns_data: Dict {date_str: Decimal(return)} for ticker
            xbi_returns_data: Dict {date_str: Decimal(return)} for XBI benchmark
            calc_date: Calculation date (YYYY-MM-DD string)

        Returns:
            dict: {
                "multi_horizon_sharpe": {...},
                "relative_strength_vs_xbi": {...},
                "idiosyncratic_momentum": {...},
                "drawdown_gate": {...},
                "composite_momentum_score": Decimal (0-100),
                "confidence_tier": str,
                "confidence_multiplier": Decimal,
                "provenance": {...}
            }
        """
        # Determine confidence tier based on data availability
        confidence_tier, confidence_multiplier, n_obs = self._determine_confidence_tier(
            returns_data, xbi_returns_data, calc_date
        )

        # Calculate input hashes for provenance
        ticker_hash = self._hash_window_data(returns_data, calc_date, 120)
        xbi_hash = self._hash_window_data(xbi_returns_data, calc_date, 120)

        signals = {}

        # Signal 1: Multi-horizon risk-adjusted momentum (corrected Sharpe)
        signals["multi_horizon_sharpe"] = self._calculate_multi_horizon_sharpe(
            returns_data, xbi_returns_data, calc_date
        )

        # Signal 2: Relative strength vs XBI (benchmark-aligned)
        signals["relative_strength_vs_xbi"] = self._calculate_relative_strength(
            returns_data, xbi_returns_data, calc_date
        )

        # Signal 3: Idiosyncratic momentum (with beta clamping)
        signals["idiosyncratic_momentum"] = self._calculate_idiosyncratic_momentum(
            returns_data, xbi_returns_data, calc_date
        )

        # Signal 4: Drawdown risk gate (benchmark-aligned)
        signals["drawdown_gate"] = self._calculate_drawdown_gate(
            returns_data, xbi_returns_data, calc_date
        )

        # Calculate composite score (weights sum to 1.0)
        signals["composite_momentum_score"] = self._calculate_composite_score(signals)

        # Add confidence info
        signals["confidence_tier"] = confidence_tier
        signals["confidence_multiplier"] = confidence_multiplier

        # Add provenance with input hashes
        signals["provenance"] = {
            "ticker": ticker,
            "calc_date": calc_date,
            "n_observations": n_obs,
            "ticker_window_hash": ticker_hash,
            "xbi_window_hash": xbi_hash,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Log for audit
        self._log_calculation(ticker, calc_date, signals)

        return signals

    def _calculate_multi_horizon_sharpe(
        self,
        returns_data: ReturnsData,
        xbi_returns_data: ReturnsData,
        calc_date: str
    ) -> MultiHorizonSharpe:
        """
        Signal 1: Multi-horizon risk-adjusted momentum (CORRECTED).

        FIXES:
        - Dimensionally consistent Sharpe: (mean_daily / std_daily) * sqrt(252)
        - Uses benchmark-aligned windows
        - Proper Decimal sqrt
        - Quantized outputs
        """
        sharpes = {}

        for horizon_name, days in self.HORIZONS.items():
            # Get BENCHMARK-ALIGNED window
            aligned_window = self._get_benchmark_aligned_window(
                returns_data, xbi_returns_data, calc_date, days
            )

            if len(aligned_window["ticker"]) < 5:  # Need minimum data
                sharpes[f"{horizon_name}_{days}d"] = Decimal("0.0")
                continue

            ticker_returns = list(aligned_window["ticker"].values())

            # Calculate mean daily return
            mean_daily = sum(ticker_returns) / len(ticker_returns)

            # Calculate standard deviation of daily returns
            squared_diffs = [(r - mean_daily) ** 2 for r in ticker_returns]
            variance = sum(squared_diffs) / (len(squared_diffs) - 1)  # Sample variance

            if variance <= 0:
                sharpes[f"{horizon_name}_{days}d"] = Decimal("0.0")
                continue

            # Proper Decimal sqrt
            std_daily = variance.sqrt()

            # Dimensionally consistent Sharpe: (mean / std) * sqrt(252)
            if std_daily > 0:
                sharpe = (mean_daily / std_daily) * self.SQRT_ANNUAL
            else:
                sharpe = Decimal("0.0")

            # Quantize for deterministic output
            sharpe = sharpe.quantize(self.PRECISION["sharpe"], rounding=ROUND_HALF_UP)
            sharpes[f"{horizon_name}_{days}d"] = sharpe

        # Composite: weighted average (20% short, 40% medium, 40% long)
        weights = {
            "short_20d": Decimal("0.20"),
            "medium_60d": Decimal("0.40"),
            "long_120d": Decimal("0.40")
        }

        composite = sum(
            sharpes.get(k, Decimal("0.0")) * weights[k]
            for k in weights.keys()
        )
        composite = composite.quantize(self.PRECISION["sharpe"], rounding=ROUND_HALF_UP)
        sharpes["composite"] = composite

        return sharpes

    def _calculate_relative_strength(
        self,
        ticker_returns: ReturnsData,
        xbi_returns: ReturnsData,
        calc_date: str
    ) -> RelativeStrengthResult:
        """
        Signal 2: Relative strength vs XBI (CORRECTED).

        FIXES:
        - Benchmark-aligned windows throughout
        - Missing dates reduce confidence (handled in alignment)
        - Quantized outputs
        """
        rs_signals = {}

        for horizon_name, days in self.HORIZONS.items():
            # Get BENCHMARK-ALIGNED window
            aligned_window = self._get_benchmark_aligned_window(
                ticker_returns, xbi_returns, calc_date, days
            )

            if len(aligned_window["ticker"]) < 5:
                rs_signals[f"rs_{days}d"] = Decimal("0.0")
                continue

            # Calculate cumulative returns on aligned windows
            ticker_cum = self._calculate_cumulative_return(aligned_window["ticker"])
            xbi_cum = self._calculate_cumulative_return(aligned_window["xbi"])

            # Relative strength = ticker - benchmark
            rs = ticker_cum - xbi_cum
            rs = rs.quantize(self.PRECISION["daily_return"], rounding=ROUND_HALF_UP)
            rs_signals[f"rs_{days}d"] = rs

        # Composite: weighted average
        weights = {
            "rs_20d": Decimal("0.20"),
            "rs_60d": Decimal("0.40"),
            "rs_120d": Decimal("0.40")
        }

        composite = sum(
            rs_signals.get(k, Decimal("0.0")) * weights[k]
            for k in weights.keys()
        )
        composite = composite.quantize(self.PRECISION["daily_return"], rounding=ROUND_HALF_UP)
        rs_signals["composite"] = composite

        return rs_signals

    def _calculate_idiosyncratic_momentum(
        self,
        ticker_returns: ReturnsData,
        xbi_returns: ReturnsData,
        calc_date: str
    ) -> IdiosyncraticMomentum:
        """
        Signal 3: Beta + idiosyncratic momentum (CORRECTED).

        FIXES:
        - Beta clamping to prevent explosions
        - Benchmark-aligned windows
        - Dimensionally consistent idio Sharpe
        """
        horizon_days = 60  # Use 60d for beta estimation

        # Get BENCHMARK-ALIGNED window
        aligned_window = self._get_benchmark_aligned_window(
            ticker_returns, xbi_returns, calc_date, horizon_days
        )

        if len(aligned_window["ticker"]) < 30:  # Need minimum data
            return {
                "beta_60d": Decimal("1.0"),
                "idio_return_60d": Decimal("0.0"),
                "idio_sharpe": Decimal("0.0"),
                "n_aligned_obs": 0
            }

        ticker_series = list(aligned_window["ticker"].values())
        xbi_series = list(aligned_window["xbi"].values())

        # Calculate beta with CLAMPING
        beta = self._calculate_beta(ticker_series, xbi_series)
        beta_clamped = max(self.BETA_MIN, min(self.BETA_MAX, beta))

        # Calculate idiosyncratic returns (using aligned series)
        idio_returns = {}
        dates = sorted(aligned_window["ticker"].keys())
        for date in dates:
            ticker_ret = aligned_window["ticker"][date]
            xbi_ret = aligned_window["xbi"][date]
            idio_ret = ticker_ret - (beta_clamped * xbi_ret)
            idio_returns[date] = idio_ret

        # Calculate idiosyncratic Sharpe (dimensionally consistent)
        idio_list = list(idio_returns.values())
        mean_idio = sum(idio_list) / len(idio_list)

        squared_diffs = [(r - mean_idio) ** 2 for r in idio_list]
        variance_idio = sum(squared_diffs) / (len(squared_diffs) - 1)

        if variance_idio > 0:
            std_idio = variance_idio.sqrt()
            idio_sharpe = (mean_idio / std_idio) * self.SQRT_ANNUAL
        else:
            idio_sharpe = Decimal("0.0")

        # Calculate cumulative idio return for reference
        idio_cumulative = self._calculate_cumulative_return(idio_returns)

        return {
            "beta_60d": beta_clamped.quantize(self.PRECISION["volatility"], rounding=ROUND_HALF_UP),
            "idio_return_60d": idio_cumulative.quantize(self.PRECISION["daily_return"], rounding=ROUND_HALF_UP),
            "idio_sharpe": idio_sharpe.quantize(self.PRECISION["sharpe"], rounding=ROUND_HALF_UP),
            "n_aligned_obs": len(aligned_window["ticker"])
        }

    def _calculate_drawdown_gate(
        self,
        returns_data: ReturnsData,
        xbi_returns_data: ReturnsData,
        calc_date: str
    ) -> DrawdownGate:
        """
        Signal 4: Maximum drawdown risk gate (CORRECTED).

        FIXES:
        - Benchmark-aligned windows
        - Uses XBI calendar as clock
        """
        horizons = {"120d": 120, "180d": 180}
        drawdowns = {}

        for horizon_name, days in horizons.items():
            # Get BENCHMARK-ALIGNED window
            aligned_window = self._get_benchmark_aligned_window(
                returns_data, xbi_returns_data, calc_date, days
            )

            if len(aligned_window["ticker"]) < 10:
                drawdowns[f"max_drawdown_{horizon_name}"] = Decimal("0.0")
                continue

            # Calculate cumulative returns to get price path
            dates_sorted = sorted(aligned_window["ticker"].keys())
            cumulative_path = []
            cum_return = Decimal("1.0")

            for date in dates_sorted:
                cum_return *= (Decimal("1.0") + aligned_window["ticker"][date])
                cumulative_path.append(cum_return)

            # Calculate max drawdown
            max_dd = Decimal("0.0")
            peak = cumulative_path[0]

            for value in cumulative_path:
                if value > peak:
                    peak = value

                drawdown = (peak - value) / peak if peak > 0 else Decimal("0.0")
                max_dd = max(max_dd, drawdown)

            max_dd = max_dd.quantize(self.PRECISION["daily_return"], rounding=ROUND_HALF_UP)
            drawdowns[f"max_drawdown_{horizon_name}"] = max_dd

        # Calculate risk penalty (0.5 to 1.0 scale)
        dd_120 = drawdowns.get("max_drawdown_120d", Decimal("0.0"))

        if dd_120 < Decimal("0.30"):  # <30% drawdown
            risk_penalty = Decimal("1.0")
        elif dd_120 > Decimal("0.60"):  # >60% drawdown
            risk_penalty = Decimal("0.5")
        else:
            # Linear interpolation
            risk_penalty = Decimal("1.0") - ((dd_120 - Decimal("0.30")) / Decimal("0.30")) * Decimal("0.5")

        risk_penalty = risk_penalty.quantize(self.PRECISION["volatility"], rounding=ROUND_HALF_UP)
        drawdowns["risk_penalty"] = risk_penalty

        return drawdowns

    def _calculate_composite_score(
        self,
        signals: Dict[str, Union[MultiHorizonSharpe, RelativeStrengthResult, IdiosyncraticMomentum, DrawdownGate]]
    ) -> Decimal:
        """
        Combine all 4 signals into single 0-100 score.

        FIXES:
        - Weights sum to 1.0 (not 0.85)
        - Drawdown gate is separate multiplier
        - Properly bounded [0, 100]
        """
        # Extract composite values
        sharpe_composite = signals["multi_horizon_sharpe"]["composite"]
        rs_composite = signals["relative_strength_vs_xbi"]["composite"]
        idio_sharpe = signals["idiosyncratic_momentum"]["idio_sharpe"]
        risk_penalty = signals["drawdown_gate"]["risk_penalty"]

        # Normalize each signal to 0-100 scale
        sharpe_score = self._normalize_sharpe_to_score(sharpe_composite)
        rs_score = self._normalize_return_to_score(rs_composite)
        idio_score = self._normalize_sharpe_to_score(idio_sharpe)

        # Weighted average (MUST SUM TO 1.0)
        weights = {
            "sharpe": Decimal("0.35"),
            "rs": Decimal("0.35"),
            "idio": Decimal("0.30")
        }
        # Verify: 0.35 + 0.35 + 0.30 = 1.0 ✓

        base_score = (
            sharpe_score * weights["sharpe"] +
            rs_score * weights["rs"] +
            idio_score * weights["idio"]
        )

        # Apply risk penalty as SEPARATE multiplier
        final_score = base_score * risk_penalty

        # Ensure bounds and quantize
        final_score = max(Decimal("0.0"), min(Decimal("100.0"), final_score))
        final_score = final_score.quantize(self.PRECISION["score"], rounding=ROUND_HALF_UP)

        return final_score

    # ===== HELPER FUNCTIONS =====

    def _get_benchmark_aligned_window(
        self,
        ticker_returns: ReturnsData,
        xbi_returns: ReturnsData,
        calc_date: str,
        lookback_days: int
    ) -> AlignedWindow:
        """
        Get benchmark-aligned window for calculations.

        CRITICAL FIX: Uses XBI dates as the clock, then subsets ticker returns.
        This ensures we're comparing apples-to-apples even when ticker has gaps.

        Returns:
            dict: {
                "ticker": {date: return},
                "xbi": {date: return},
                "n_missing": int,
                "coverage_pct": Decimal
            }
        """
        # Get XBI window (benchmark is the clock)
        xbi_dates_all = sorted([d for d in xbi_returns.keys() if d <= calc_date])

        if len(xbi_dates_all) < lookback_days:
            return {"ticker": {}, "xbi": {}, "n_missing": lookback_days, "coverage_pct": Decimal("0.0")}

        # Last N trading days from benchmark
        xbi_window_dates = xbi_dates_all[-lookback_days:]

        # Subset ticker returns to these dates
        ticker_aligned = {}
        xbi_aligned = {}
        n_missing = 0

        for date in xbi_window_dates:
            xbi_aligned[date] = xbi_returns[date]

            if date in ticker_returns:
                ticker_aligned[date] = ticker_returns[date]
            else:
                n_missing += 1

        coverage_pct = Decimal(len(ticker_aligned)) / Decimal(len(xbi_window_dates)) if xbi_window_dates else Decimal("0.0")

        return {
            "ticker": ticker_aligned,
            "xbi": xbi_aligned,
            "n_missing": n_missing,
            "coverage_pct": coverage_pct
        }

    def _determine_confidence_tier(
        self,
        returns_data: ReturnsData,
        xbi_returns_data: ReturnsData,
        calc_date: str
    ) -> Tuple[str, Decimal, int]:
        """
        Determine confidence tier based on data availability.

        REPLACES fail-loud approach with graceful degradation.
        """
        # Check longest horizon needed
        aligned_120d = self._get_benchmark_aligned_window(
            returns_data, xbi_returns_data, calc_date, 120
        )

        n_obs = len(aligned_120d["ticker"])
        coverage = aligned_120d["coverage_pct"]

        # Determine tier
        if n_obs >= 120 and coverage > Decimal("0.90"):
            tier = "HIGH"
        elif n_obs >= 60 and coverage > Decimal("0.85"):
            tier = "MEDIUM"
        elif n_obs >= 20 and coverage > Decimal("0.75"):
            tier = "LOW"
        else:
            tier = "UNKNOWN"

        multiplier = self.CONFIDENCE_TIERS[tier]["weight_multiplier"]

        return tier, multiplier, n_obs

    def _calculate_cumulative_return(self, returns_dict: Dict[str, Decimal]) -> Decimal:
        """Calculate cumulative return from daily returns."""
        cum_return = Decimal("1.0")

        for date in sorted(returns_dict.keys()):
            daily_return = returns_dict[date]
            cum_return *= (Decimal("1.0") + daily_return)

        return cum_return - Decimal("1.0")

    def _calculate_beta(self, ticker_series: List[Decimal], xbi_series: List[Decimal]) -> Decimal:
        """
        Calculate beta: Cov(ticker, xbi) / Var(xbi).

        Beta measures sensitivity to market movements.
        """
        if len(ticker_series) != len(xbi_series) or len(ticker_series) < 2:
            return Decimal("1.0")  # Default to market beta

        n = len(ticker_series)

        # Calculate means
        ticker_mean = sum(ticker_series) / n
        xbi_mean = sum(xbi_series) / n

        # Calculate covariance
        covariance = sum(
            (ticker_series[i] - ticker_mean) * (xbi_series[i] - xbi_mean)
            for i in range(n)
        ) / (n - 1)

        # Calculate variance of XBI
        xbi_variance = sum(
            (xbi_series[i] - xbi_mean) ** 2
            for i in range(n)
        ) / (n - 1)

        # Beta = Cov / Var
        if xbi_variance > 0:
            beta = covariance / xbi_variance
        else:
            beta = Decimal("1.0")

        return beta

    def _normalize_sharpe_to_score(self, sharpe: Decimal) -> Decimal:
        """
        Normalize Sharpe ratio to 0-100 score.

        Mapping:
        - Sharpe < -2: Score ~0
        - Sharpe = 0: Score = 50
        - Sharpe > +2: Score ~100
        """
        sharpe_clipped = max(Decimal("-3.0"), min(Decimal("3.0"), sharpe))
        score = Decimal("50.0") + (sharpe_clipped * Decimal("16.67"))
        return max(Decimal("0.0"), min(Decimal("100.0"), score))

    def _normalize_return_to_score(self, return_val: Decimal) -> Decimal:
        """Normalize return (for relative strength) to 0-100 score."""
        return_clipped = max(Decimal("-0.50"), min(Decimal("0.50"), return_val))
        score = Decimal("50.0") + (return_clipped * Decimal("100.0"))
        return max(Decimal("0.0"), min(Decimal("100.0"), score))

    def _hash_window_data(self, returns_data: ReturnsData, calc_date: str, lookback_days: int) -> str:
        """
        Hash input window for provenance.

        Enables deterministic replay validation.
        """
        dates_sorted = sorted([d for d in returns_data.keys() if d <= calc_date])

        if len(dates_sorted) < lookback_days:
            window_dates = dates_sorted
        else:
            window_dates = dates_sorted[-lookback_days:]

        # Create deterministic string representation
        window_str = "|".join([
            f"{date}:{returns_data[date]}"
            for date in window_dates
        ])

        return hashlib.sha256(window_str.encode()).hexdigest()[:16]

    def _log_calculation(self, ticker: str, calc_date: str, signals: AllSignalsResult) -> None:
        """Log calculation for audit trail."""
        entry = {
            "ticker": ticker,
            "calc_date": calc_date,
            "composite_score": str(signals["composite_momentum_score"]),
            "confidence_tier": signals["confidence_tier"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.audit_trail.append(entry)

    def write_audit_trail(self, output_file: str) -> None:
        """Write audit trail to file."""
        with open(output_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)


# ===== ACCEPTANCE TESTS =====

def run_acceptance_tests():
    """Run minimal acceptance tests before production deployment."""

    print("=" * 70)
    print("MOMENTUM SIGNAL ACCEPTANCE TESTS")
    print("=" * 70)

    # Test 1: Monotonic return stream should rank positive
    print("\n[Test 1] Monotonic Returns → Positive Composite")
    monotonic_returns = {}
    base_date = datetime(2024, 1, 1)
    for i in range(130):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        monotonic_returns[date_str] = Decimal("0.005")  # Steady +0.5%/day

    xbi_flat = {date: Decimal("0.0") for date in monotonic_returns.keys()}

    calc = MorningstarMomentumSignals()
    signals = calc.calculate_all_signals("TEST1", monotonic_returns, xbi_flat, "2024-05-10")

    if signals["composite_momentum_score"] <= Decimal("50.0"):
        raise ValueError(f"FAIL: Monotonic positive returns should score >50, got {signals['composite_momentum_score']}")
    print(f"  ✓ PASS: Score = {signals['composite_momentum_score']:.2f} (>50)")

    # Test 2: Score range must be [0, 100]
    print("\n[Test 2] Score Range [0, 100] (Not Capped at 85)")
    if not (Decimal("0.0") <= signals["composite_momentum_score"] <= Decimal("100.0")):
        raise ValueError("FAIL: Score out of bounds")

    # Check that high-Sharpe returns with variance produce high scores
    # Using alternating positive returns to create variance while maintaining high mean
    high_sharpe_returns = {}
    for i, date in enumerate(sorted(monotonic_returns.keys())):
        # Alternate between 0.8% and 1.2% daily returns (mean=1%, std>0)
        if i % 2 == 0:
            high_sharpe_returns[date] = Decimal("0.008")
        else:
            high_sharpe_returns[date] = Decimal("0.012")

    signals_high_sharpe = calc.calculate_all_signals("TEST2", high_sharpe_returns, xbi_flat, "2024-05-10")

    # High Sharpe + high relative strength should produce score > 75
    if signals_high_sharpe["composite_momentum_score"] <= Decimal("75.0"):
        raise ValueError(f"FAIL: High Sharpe returns should score >75, got {signals_high_sharpe['composite_momentum_score']}")
    print(f"  ✓ PASS: High Sharpe score = {signals_high_sharpe['composite_momentum_score']:.2f} (>75, demonstrates full range)")

    # Test 3: Alignment - missing dates should affect confidence
    print("\n[Test 3] Alignment → Missing Dates Reduce Confidence")
    sparse_returns = {
        date: monotonic_returns[date]
        for i, date in enumerate(sorted(monotonic_returns.keys()))
        if i % 3 == 0  # Keep only every 3rd date (67% missing)
    }

    signals_sparse = calc.calculate_all_signals("TEST3", sparse_returns, monotonic_returns, "2024-05-10")

    if signals_sparse["confidence_tier"] not in ["LOW", "UNKNOWN"]:
        raise ValueError(f"FAIL: Sparse data should be LOW/UNKNOWN, got {signals_sparse['confidence_tier']}")
    if signals_sparse["confidence_multiplier"] >= Decimal("1.0"):
        raise ValueError("FAIL: Sparse data should have reduced confidence multiplier")
    print(f"  ✓ PASS: Confidence tier = {signals_sparse['confidence_tier']}, "
          f"multiplier = {signals_sparse['confidence_multiplier']:.2f}")

    # Test 4: Reproducibility - same inputs = identical output
    print("\n[Test 4] Reproducibility → Same Inputs = Identical Output")
    signals_a = calc.calculate_all_signals("TEST4", monotonic_returns, xbi_flat, "2024-05-10")
    signals_b = calc.calculate_all_signals("TEST4", monotonic_returns, xbi_flat, "2024-05-10")

    if signals_a["composite_momentum_score"] != signals_b["composite_momentum_score"]:
        raise ValueError("FAIL: Non-deterministic output")
    if signals_a["provenance"]["ticker_window_hash"] != signals_b["provenance"]["ticker_window_hash"]:
        raise ValueError("FAIL: Input hashes differ")
    print(f"  ✓ PASS: Identical scores and input hashes")

    print("\n" + "=" * 70)
    print("ALL ACCEPTANCE TESTS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    run_acceptance_tests()
