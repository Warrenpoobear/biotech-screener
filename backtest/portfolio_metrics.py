#!/usr/bin/env python3
"""
portfolio_metrics.py - Portfolio Performance and Drawdown Analysis

P3 Enhancement: Provides comprehensive portfolio performance metrics
including drawdown analysis for backtest validation.

Key Features:
1. Maximum drawdown calculation
2. Drawdown duration and recovery
3. Regime-conditional drawdowns
4. Sharpe and Sortino ratios
5. Rolling performance windows

Design Philosophy:
- DETERMINISTIC: All calculations use explicit parameters
- AUDITABLE: Full history of performance metrics
- PIT-SAFE: Forward returns only, no lookahead

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


@dataclass
class DrawdownEvent:
    """A single drawdown event."""
    start_date: str
    trough_date: str
    recovery_date: Optional[str]  # None if not recovered
    peak_value: Decimal
    trough_value: Decimal
    drawdown_pct: Decimal  # As negative percentage
    duration_days: int  # Days from peak to trough
    recovery_days: Optional[int]  # Days from trough to recovery
    is_recovered: bool
    regime_at_start: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "trough_date": self.trough_date,
            "recovery_date": self.recovery_date,
            "peak_value": str(self.peak_value),
            "trough_value": str(self.trough_value),
            "drawdown_pct": str(self.drawdown_pct),
            "duration_days": self.duration_days,
            "recovery_days": self.recovery_days,
            "is_recovered": self.is_recovered,
            "regime_at_start": self.regime_at_start,
        }


@dataclass
class DrawdownAnalysis:
    """Complete drawdown analysis."""
    max_drawdown_pct: Decimal
    max_drawdown_event: Optional[DrawdownEvent]
    avg_drawdown_pct: Decimal
    avg_recovery_days: Optional[int]
    drawdown_events: List[DrawdownEvent]
    current_drawdown_pct: Decimal  # Current drawdown from peak
    days_since_peak: int
    regime_drawdowns: Dict[str, List[DrawdownEvent]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "max_drawdown_event": self.max_drawdown_event.to_dict() if self.max_drawdown_event else None,
            "avg_drawdown_pct": str(self.avg_drawdown_pct),
            "avg_recovery_days": self.avg_recovery_days,
            "n_drawdown_events": len(self.drawdown_events),
            "current_drawdown_pct": str(self.current_drawdown_pct),
            "days_since_peak": self.days_since_peak,
            "regime_summary": {
                regime: {
                    "count": len(events),
                    "avg_drawdown": str(Decimal(str(mean([float(e.drawdown_pct) for e in events])))
                                        .quantize(Decimal("0.01"))) if events else None
                }
                for regime, events in self.regime_drawdowns.items()
            },
        }


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""
    sharpe_ratio: Optional[Decimal]
    sortino_ratio: Optional[Decimal]
    calmar_ratio: Optional[Decimal]  # Return / Max Drawdown
    volatility_annualized: Optional[Decimal]
    downside_deviation: Optional[Decimal]
    var_95: Optional[Decimal]  # 5% Value at Risk
    cvar_95: Optional[Decimal]  # Conditional VaR (Expected Shortfall)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio else None,
            "sortino_ratio": str(self.sortino_ratio) if self.sortino_ratio else None,
            "calmar_ratio": str(self.calmar_ratio) if self.calmar_ratio else None,
            "volatility_annualized": str(self.volatility_annualized) if self.volatility_annualized else None,
            "downside_deviation": str(self.downside_deviation) if self.downside_deviation else None,
            "var_95": str(self.var_95) if self.var_95 else None,
            "cvar_95": str(self.cvar_95) if self.cvar_95 else None,
        }


@dataclass
class PerformanceSummary:
    """Complete performance summary."""
    start_date: str
    end_date: str
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    n_periods: int
    positive_periods: int
    hit_rate: Decimal  # % positive periods
    best_period_return: Decimal
    worst_period_return: Decimal
    avg_period_return: Decimal
    median_period_return: Decimal
    drawdown_analysis: DrawdownAnalysis
    risk_metrics: RiskMetrics
    regime_returns: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "n_periods": self.n_periods,
            },
            "returns": {
                "total_return_pct": str(self.total_return_pct),
                "annualized_return_pct": str(self.annualized_return_pct),
                "best_period": str(self.best_period_return),
                "worst_period": str(self.worst_period_return),
                "avg_period": str(self.avg_period_return),
                "median_period": str(self.median_period_return),
            },
            "hit_rate": str(self.hit_rate),
            "positive_periods": self.positive_periods,
            "drawdown": self.drawdown_analysis.to_dict(),
            "risk": self.risk_metrics.to_dict(),
            "regime_returns": self.regime_returns,
        }


def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return default
            return Decimal(str(value))
        return Decimal(str(value))
    except Exception:
        return default


def _quantize(value: Decimal, precision: str = "0.01") -> Decimal:
    """Quantize decimal to specified precision."""
    return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)


class DrawdownCalculator:
    """
    Calculate drawdown metrics from portfolio values or returns.

    Tracks peak values and computes drawdown events with
    duration and recovery analysis.

    Usage:
        calc = DrawdownCalculator()

        # Process portfolio values over time
        for date_str, value in portfolio_history:
            calc.update(date_str, value)

        # Get analysis
        analysis = calc.get_analysis()
    """

    def __init__(
        self,
        drawdown_threshold_pct: Decimal = Decimal("5.0"),
    ):
        """
        Initialize drawdown calculator.

        Args:
            drawdown_threshold_pct: Minimum drawdown % to record as event
        """
        self.threshold = drawdown_threshold_pct
        self._values: List[Tuple[str, Decimal]] = []
        self._regimes: Dict[str, str] = {}  # date -> regime

    def update(
        self,
        date_str: str,
        value: Decimal,
        regime: Optional[str] = None,
    ) -> None:
        """
        Update with new portfolio value.

        Args:
            date_str: Date as ISO string
            value: Portfolio value
            regime: Optional market regime at this date
        """
        self._values.append((date_str, value))
        if regime:
            self._regimes[date_str] = regime

    def compute_drawdown_events(self) -> List[DrawdownEvent]:
        """Compute all drawdown events from value history."""
        if len(self._values) < 2:
            return []

        events = []
        current_peak = self._values[0][1]
        current_peak_date = self._values[0][0]
        in_drawdown = False
        drawdown_start_date = None
        trough_value = current_peak
        trough_date = current_peak_date

        for i, (dt, value) in enumerate(self._values):
            if value > current_peak:
                # New peak
                if in_drawdown:
                    # Recovered from drawdown
                    drawdown_pct = ((trough_value - current_peak) / current_peak * Decimal("100"))
                    if abs(drawdown_pct) >= self.threshold:
                        start_dt = date.fromisoformat(drawdown_start_date)
                        trough_dt = date.fromisoformat(trough_date)
                        recovery_dt = date.fromisoformat(dt)

                        events.append(DrawdownEvent(
                            start_date=drawdown_start_date,
                            trough_date=trough_date,
                            recovery_date=dt,
                            peak_value=current_peak,
                            trough_value=trough_value,
                            drawdown_pct=_quantize(drawdown_pct),
                            duration_days=(trough_dt - start_dt).days,
                            recovery_days=(recovery_dt - trough_dt).days,
                            is_recovered=True,
                            regime_at_start=self._regimes.get(drawdown_start_date),
                        ))

                    in_drawdown = False

                current_peak = value
                current_peak_date = dt
                trough_value = value
                trough_date = dt

            elif value < trough_value:
                # New trough
                if not in_drawdown:
                    # Start of new drawdown
                    in_drawdown = True
                    drawdown_start_date = current_peak_date

                trough_value = value
                trough_date = dt

        # Handle ongoing drawdown
        if in_drawdown:
            drawdown_pct = ((trough_value - current_peak) / current_peak * Decimal("100"))
            if abs(drawdown_pct) >= self.threshold:
                start_dt = date.fromisoformat(drawdown_start_date)
                trough_dt = date.fromisoformat(trough_date)

                events.append(DrawdownEvent(
                    start_date=drawdown_start_date,
                    trough_date=trough_date,
                    recovery_date=None,
                    peak_value=current_peak,
                    trough_value=trough_value,
                    drawdown_pct=_quantize(drawdown_pct),
                    duration_days=(trough_dt - start_dt).days,
                    recovery_days=None,
                    is_recovered=False,
                    regime_at_start=self._regimes.get(drawdown_start_date),
                ))

        return events

    def get_analysis(self) -> DrawdownAnalysis:
        """Get complete drawdown analysis."""
        events = self.compute_drawdown_events()

        # Max drawdown
        if events:
            max_event = min(events, key=lambda e: e.drawdown_pct)
            max_dd = max_event.drawdown_pct
        else:
            max_event = None
            max_dd = Decimal("0")

        # Average drawdown
        if events:
            avg_dd = _quantize(Decimal(str(mean([float(e.drawdown_pct) for e in events]))))
        else:
            avg_dd = Decimal("0")

        # Average recovery
        recovered_events = [e for e in events if e.is_recovered and e.recovery_days is not None]
        avg_recovery = None
        if recovered_events:
            avg_recovery = int(mean([e.recovery_days for e in recovered_events]))

        # Current drawdown
        current_dd = Decimal("0")
        days_since_peak = 0
        if self._values:
            peak = max(v for _, v in self._values)
            current = self._values[-1][1]
            if peak > Decimal("0"):
                current_dd = _quantize((current - peak) / peak * Decimal("100"))

            # Find peak date
            for dt, v in reversed(self._values):
                if v == peak:
                    days_since_peak = (date.fromisoformat(self._values[-1][0]) -
                                      date.fromisoformat(dt)).days
                    break

        # Group by regime
        regime_drawdowns: Dict[str, List[DrawdownEvent]] = {}
        for event in events:
            regime = event.regime_at_start or "UNKNOWN"
            if regime not in regime_drawdowns:
                regime_drawdowns[regime] = []
            regime_drawdowns[regime].append(event)

        return DrawdownAnalysis(
            max_drawdown_pct=max_dd,
            max_drawdown_event=max_event,
            avg_drawdown_pct=avg_dd,
            avg_recovery_days=avg_recovery,
            drawdown_events=events,
            current_drawdown_pct=current_dd,
            days_since_peak=days_since_peak,
            regime_drawdowns=regime_drawdowns,
        )


class PortfolioMetricsEngine:
    """
    Engine for computing comprehensive portfolio metrics.

    Calculates returns, risk metrics, and drawdown analysis
    from portfolio return series.

    Usage:
        engine = PortfolioMetricsEngine()
        summary = engine.compute_summary(
            returns=[
                ("2025-01-01", Decimal("0.02")),
                ("2025-02-01", Decimal("-0.01")),
                ...
            ],
            regimes={"2025-01-01": "BULL", "2025-02-01": "BEAR", ...}
        )
    """

    VERSION = "1.0.0"

    # Annualization factors
    TRADING_DAYS_YEAR = 252
    MONTHS_YEAR = 12

    # Risk-free rate assumption (annualized)
    RISK_FREE_RATE = Decimal("0.05")

    def __init__(
        self,
        period_type: str = "monthly",  # "daily", "weekly", "monthly"
        risk_free_rate: Optional[Decimal] = None,
    ):
        """
        Initialize portfolio metrics engine.

        Args:
            period_type: Type of return periods
            risk_free_rate: Annualized risk-free rate
        """
        self.period_type = period_type
        self.risk_free_rate = risk_free_rate or self.RISK_FREE_RATE

        # Annualization factor based on period type
        self._annual_factor = {
            "daily": self.TRADING_DAYS_YEAR,
            "weekly": 52,
            "monthly": 12,
        }.get(period_type, 12)

    def compute_risk_metrics(
        self,
        returns: List[Decimal],
    ) -> RiskMetrics:
        """
        Compute risk-adjusted performance metrics.

        Args:
            returns: List of period returns (as decimals, e.g., 0.05 = 5%)

        Returns:
            RiskMetrics with all risk measures
        """
        if len(returns) < 3:
            return RiskMetrics(
                sharpe_ratio=None,
                sortino_ratio=None,
                calmar_ratio=None,
                volatility_annualized=None,
                downside_deviation=None,
                var_95=None,
                cvar_95=None,
            )

        returns_float = [float(r) for r in returns]

        # Mean return (annualized)
        mean_return = mean(returns_float) * self._annual_factor
        period_rf = float(self.risk_free_rate) / self._annual_factor

        # Volatility (annualized)
        if len(returns_float) > 1:
            vol = stdev(returns_float) * math.sqrt(self._annual_factor)
        else:
            vol = 0

        # Sharpe Ratio
        sharpe = None
        if vol > 0:
            excess_return = mean_return - float(self.risk_free_rate)
            sharpe = _quantize(Decimal(str(excess_return / vol)), "0.01")

        # Downside deviation (for Sortino)
        downside_returns = [r - period_rf for r in returns_float if r < period_rf]
        downside_dev = None
        sortino = None

        if downside_returns and len(downside_returns) > 1:
            downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
            downside_dev = math.sqrt(downside_var) * math.sqrt(self._annual_factor)
            if downside_dev > 0:
                excess_return = mean_return - float(self.risk_free_rate)
                sortino = _quantize(Decimal(str(excess_return / downside_dev)), "0.01")

        # VaR and CVaR at 95%
        sorted_returns = sorted(returns_float)
        var_index = int(len(sorted_returns) * 0.05)
        var_95 = None
        cvar_95 = None

        if var_index > 0:
            var_95 = _quantize(Decimal(str(sorted_returns[var_index] * 100)), "0.01")
            tail_returns = sorted_returns[:var_index + 1]
            if tail_returns:
                cvar_95 = _quantize(Decimal(str(mean(tail_returns) * 100)), "0.01")

        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=None,  # Computed with drawdown
            volatility_annualized=_quantize(Decimal(str(vol * 100)), "0.01") if vol else None,
            downside_deviation=_quantize(Decimal(str(downside_dev * 100)), "0.01") if downside_dev else None,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def compute_summary(
        self,
        returns: List[Tuple[str, Decimal]],  # (date, return)
        regimes: Optional[Dict[str, str]] = None,
    ) -> PerformanceSummary:
        """
        Compute complete performance summary.

        Args:
            returns: List of (date, return) tuples
            regimes: Optional dict of date -> regime

        Returns:
            PerformanceSummary with all metrics
        """
        if not returns:
            raise ValueError("No returns provided")

        regimes = regimes or {}
        dates = [dt for dt, _ in returns]
        return_values = [r for _, r in returns]
        return_floats = [float(r) for r in return_values]

        # Basic stats
        n_periods = len(returns)
        positive = sum(1 for r in return_values if r > Decimal("0"))
        hit_rate = _quantize(Decimal(str(positive / n_periods)), "0.01")

        total_return = Decimal("1")
        for r in return_values:
            total_return *= (Decimal("1") + r)
        total_return_pct = _quantize((total_return - Decimal("1")) * Decimal("100"))

        # Annualized return
        years = n_periods / self._annual_factor
        if years > 0 and total_return > 0:
            annualized = (float(total_return) ** (1 / years) - 1) * 100
            annualized_return = _quantize(Decimal(str(annualized)))
        else:
            annualized_return = Decimal("0")

        # Drawdown analysis
        dd_calc = DrawdownCalculator()
        portfolio_value = Decimal("100")  # Start at 100
        for dt, r in returns:
            portfolio_value *= (Decimal("1") + r)
            dd_calc.update(dt, portfolio_value, regimes.get(dt))

        drawdown_analysis = dd_calc.get_analysis()

        # Risk metrics
        risk_metrics = self.compute_risk_metrics(return_values)

        # Add Calmar ratio (return / max drawdown)
        if drawdown_analysis.max_drawdown_pct < Decimal("0"):
            calmar = _quantize(annualized_return / abs(drawdown_analysis.max_drawdown_pct), "0.01")
            risk_metrics = RiskMetrics(
                sharpe_ratio=risk_metrics.sharpe_ratio,
                sortino_ratio=risk_metrics.sortino_ratio,
                calmar_ratio=calmar,
                volatility_annualized=risk_metrics.volatility_annualized,
                downside_deviation=risk_metrics.downside_deviation,
                var_95=risk_metrics.var_95,
                cvar_95=risk_metrics.cvar_95,
            )

        # Regime-conditional returns
        regime_returns: Dict[str, Dict[str, Any]] = {}
        for regime in set(regimes.values()):
            regime_rets = [r for dt, r in returns if regimes.get(dt) == regime]
            if regime_rets:
                regime_returns[regime] = {
                    "count": len(regime_rets),
                    "avg_return": str(_quantize(Decimal(str(mean([float(r) for r in regime_rets]) * 100)))),
                    "hit_rate": str(_quantize(Decimal(str(sum(1 for r in regime_rets if r > 0) / len(regime_rets))))),
                }

        return PerformanceSummary(
            start_date=dates[0],
            end_date=dates[-1],
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return,
            n_periods=n_periods,
            positive_periods=positive,
            hit_rate=hit_rate,
            best_period_return=_quantize(max(return_values) * Decimal("100")),
            worst_period_return=_quantize(min(return_values) * Decimal("100")),
            avg_period_return=_quantize(Decimal(str(mean(return_floats) * 100))),
            median_period_return=_quantize(Decimal(str(sorted(return_floats)[len(return_floats) // 2] * 100))),
            drawdown_analysis=drawdown_analysis,
            risk_metrics=risk_metrics,
            regime_returns=regime_returns,
        )


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducibility

    print("=" * 70)
    print("PORTFOLIO METRICS ENGINE - DEMONSTRATION")
    print("=" * 70)

    engine = PortfolioMetricsEngine(period_type="monthly")

    # Generate sample monthly returns
    base_date = date(2024, 1, 1)
    regimes = {}
    returns = []

    for i in range(24):  # 2 years of monthly data
        dt = base_date + timedelta(days=i * 30)
        dt_str = dt.isoformat()

        # Simulate returns with regime dependency
        if i < 8:
            regime = "BULL"
            ret = Decimal(str(0.02 + random.uniform(-0.03, 0.05)))
        elif i < 16:
            regime = "BEAR"
            ret = Decimal(str(-0.01 + random.uniform(-0.05, 0.02)))
        else:
            regime = "NEUTRAL"
            ret = Decimal(str(0.005 + random.uniform(-0.02, 0.03)))

        returns.append((dt_str, ret))
        regimes[dt_str] = regime

    summary = engine.compute_summary(returns, regimes)

    print(f"\nPeriod: {summary.start_date} to {summary.end_date}")
    print(f"Total Return: {summary.total_return_pct}%")
    print(f"Annualized Return: {summary.annualized_return_pct}%")
    print(f"Hit Rate: {summary.hit_rate}")

    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio: {summary.risk_metrics.sharpe_ratio}")
    print(f"  Sortino Ratio: {summary.risk_metrics.sortino_ratio}")
    print(f"  Volatility: {summary.risk_metrics.volatility_annualized}%")
    print(f"  Max Drawdown: {summary.drawdown_analysis.max_drawdown_pct}%")
    print(f"  Calmar Ratio: {summary.risk_metrics.calmar_ratio}")

    print("\nRegime Performance:")
    for regime, metrics in summary.regime_returns.items():
        print(f"  {regime}: avg={metrics['avg_return']}%, hit_rate={metrics['hit_rate']}")
