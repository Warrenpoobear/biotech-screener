"""
PATCH 001: PIT-Safe IC Validation
==================================

CRITICAL FIX: The original validate_production_momentum.py uses forward returns
to validate current momentum signals, which is textbook look-ahead bias.

This patch provides a PIT-safe IC validation approach that:
1. Only uses data available at the calculation date
2. Validates using PAST IC performance (walk-forward style)
3. Never peeks into future returns for current signals

Usage:
    from patches.patch_001_pit_safe_ic_validation import PITSafeICValidator

    validator = PITSafeICValidator(data_dir=Path('production_data'))
    result = validator.validate_momentum_signal(as_of_date='2026-01-15')
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ICValidationResult:
    """Result of PIT-safe IC validation."""
    validation_date: str
    lookback_start: str
    lookback_end: str
    mean_ic: Decimal
    std_ic: Decimal
    t_statistic: Decimal
    n_periods: int
    hit_rate: Decimal
    is_significant: bool
    passes_threshold: bool
    threshold: Decimal
    message: str


class PITSafeICValidator:
    """
    PIT-Safe Information Coefficient Validator.

    CRITICAL DESIGN PRINCIPLE:
    At any as_of_date, we ONLY use data that was available BEFORE that date.

    For IC validation, this means:
    - At as_of_date, we calculate momentum signals using data up to as_of_date - 1
    - To validate these signals, we look at HISTORICAL IC (not future returns)
    - We use a walk-forward approach: past signals correlated with their subsequent returns

    Example:
        as_of_date = 2026-01-15

        CORRECT (PIT-safe):
        - Calculate momentum signals using data up to 2026-01-14
        - Validate by looking at IC from 2025-01-15 to 2025-12-15 (past year)
        - Each past IC calculation: signal on date D, return from D to D+21 trading days

        WRONG (look-ahead bias):
        - Calculate momentum signals on 2026-01-15
        - Validate by correlating with returns from 2026-01-15 to 2026-02-15 (FUTURE!)
    """

    # Minimum lookback for IC validation (252 trading days = ~1 year)
    MIN_LOOKBACK_DAYS = 252

    # Minimum IC periods for statistical significance
    MIN_IC_PERIODS = 12

    # Default IC significance threshold
    DEFAULT_IC_THRESHOLD = Decimal("0.05")

    def __init__(
        self,
        data_dir: Path,
        momentum_lookback_days: int = 60,
        forward_horizon_days: int = 21,
        ic_significance_level: Decimal = Decimal("2.0"),  # t-stat threshold
    ):
        """
        Initialize PIT-safe IC validator.

        Args:
            data_dir: Directory containing price data
            momentum_lookback_days: Days used to calculate momentum signal
            forward_horizon_days: Days forward for return calculation
            ic_significance_level: t-statistic threshold for significance
        """
        self.data_dir = Path(data_dir)
        self.momentum_lookback = momentum_lookback_days
        self.forward_horizon = forward_horizon_days
        self.significance_level = ic_significance_level

    def _compute_pit_cutoff(self, as_of_date: str) -> str:
        """Compute PIT cutoff: as_of_date - 1 day."""
        dt = date.fromisoformat(as_of_date)
        cutoff = dt - timedelta(days=1)
        return cutoff.isoformat()

    def _compute_lookback_window(
        self,
        as_of_date: str
    ) -> Tuple[str, str]:
        """
        Compute the historical lookback window for IC validation.

        Returns:
            (lookback_start, lookback_end)

        The window ends at PIT cutoff minus forward_horizon (to have complete returns).
        The window starts MIN_LOOKBACK_DAYS before that.
        """
        pit_cutoff = date.fromisoformat(self._compute_pit_cutoff(as_of_date))

        # End of lookback window: need complete forward returns
        # So end date is pit_cutoff minus forward horizon
        lookback_end = pit_cutoff - timedelta(days=self.forward_horizon + 1)

        # Start of lookback window
        lookback_start = lookback_end - timedelta(days=self.MIN_LOOKBACK_DAYS)

        return lookback_start.isoformat(), lookback_end.isoformat()

    def validate_momentum_signal(
        self,
        as_of_date: str,
        ic_threshold: Optional[Decimal] = None,
    ) -> ICValidationResult:
        """
        Validate momentum signal using PIT-safe historical IC.

        This does NOT use future returns. Instead, it:
        1. Looks at the past year of momentum signals
        2. For each past date, calculates IC of that day's signal vs subsequent returns
        3. Aggregates to get mean IC and significance

        Args:
            as_of_date: The date for which we're validating (YYYY-MM-DD)
            ic_threshold: Minimum acceptable mean IC

        Returns:
            ICValidationResult with validation outcome
        """
        threshold = ic_threshold or self.DEFAULT_IC_THRESHOLD

        # Compute PIT-safe lookback window
        lookback_start, lookback_end = self._compute_lookback_window(as_of_date)

        logger.info(
            f"PIT-safe IC validation for {as_of_date}: "
            f"using historical IC from {lookback_start} to {lookback_end}"
        )

        # Load historical IC results (pre-computed)
        ic_series = self._load_historical_ic(lookback_start, lookback_end)

        if len(ic_series) < self.MIN_IC_PERIODS:
            return ICValidationResult(
                validation_date=as_of_date,
                lookback_start=lookback_start,
                lookback_end=lookback_end,
                mean_ic=Decimal("0"),
                std_ic=Decimal("0"),
                t_statistic=Decimal("0"),
                n_periods=len(ic_series),
                hit_rate=Decimal("0"),
                is_significant=False,
                passes_threshold=False,
                threshold=threshold,
                message=f"Insufficient IC periods: {len(ic_series)} < {self.MIN_IC_PERIODS}",
            )

        # Calculate IC statistics
        mean_ic = sum(ic_series) / len(ic_series)

        # Standard deviation
        variance = sum((ic - mean_ic) ** 2 for ic in ic_series) / (len(ic_series) - 1)
        std_ic = variance ** Decimal("0.5") if variance > 0 else Decimal("0.0001")

        # t-statistic
        n = Decimal(str(len(ic_series)))
        t_stat = mean_ic / (std_ic / (n ** Decimal("0.5")))

        # Hit rate (% of positive IC periods)
        hit_rate = Decimal(str(sum(1 for ic in ic_series if ic > 0))) / n

        # Significance check
        is_significant = abs(t_stat) > self.significance_level
        passes_threshold = mean_ic >= threshold and is_significant

        # Build message
        if passes_threshold:
            message = f"PASS: Mean IC {mean_ic:.4f} >= {threshold}, t-stat {t_stat:.2f}"
        elif not is_significant:
            message = f"FAIL: Not statistically significant (|t| = {abs(t_stat):.2f} < {self.significance_level})"
        else:
            message = f"FAIL: Mean IC {mean_ic:.4f} < threshold {threshold}"

        return ICValidationResult(
            validation_date=as_of_date,
            lookback_start=lookback_start,
            lookback_end=lookback_end,
            mean_ic=mean_ic.quantize(Decimal("0.0001")),
            std_ic=std_ic.quantize(Decimal("0.0001")),
            t_statistic=t_stat.quantize(Decimal("0.01")),
            n_periods=len(ic_series),
            hit_rate=hit_rate.quantize(Decimal("0.01")),
            is_significant=is_significant,
            passes_threshold=passes_threshold,
            threshold=threshold,
            message=message,
        )

    def _load_historical_ic(
        self,
        start_date: str,
        end_date: str
    ) -> List[Decimal]:
        """
        Load pre-computed historical IC values.

        Historical IC should be computed offline in a separate process that:
        1. For each historical date D
        2. Calculate momentum signals using data up to D-1
        3. Calculate returns from D to D+forward_horizon
        4. Compute cross-sectional IC (Spearman correlation)
        5. Store in ic_history.json

        This avoids any look-ahead bias because:
        - At date D, signal uses data up to D-1 (PIT safe)
        - Return from D to D+forward is known ex-post (this is historical)
        - We're looking at PAST ICs, not computing future returns now
        """
        ic_file = self.data_dir / "ic_history.json"

        if not ic_file.exists():
            logger.warning(
                f"IC history file not found: {ic_file}. "
                "Run backtest/compute_historical_ic.py to generate."
            )
            return []

        with open(ic_file) as f:
            ic_data = json.load(f)

        # Filter to date range
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)

        ic_values = []
        for entry in ic_data.get("ic_series", []):
            entry_date = date.fromisoformat(entry["date"])
            if start_dt <= entry_date <= end_dt:
                ic_values.append(Decimal(str(entry["ic"])))

        return ic_values


def _warn_about_original_implementation():
    """
    WARNING: The original validate_production_momentum.py has look-ahead bias!

    The original code at lines 174-183 does:

        forward_date = self.calendar.add_trading_days(calc_date, forward_horizon_days)
        ret = (prices_df[ticker].loc[forward_date] / prices_df[ticker].loc[calc_date]) - 1

    This calculates FUTURE returns from the validation date, which is:
    1. Look-ahead bias in validation
    2. Gives false confidence in signal quality
    3. Production IC will be lower than validated IC

    USE THIS PATCH INSTEAD for production validation.
    """
    pass


if __name__ == "__main__":
    print("=" * 70)
    print("PATCH 001: PIT-Safe IC Validation")
    print("=" * 70)
    print()
    print("This patch fixes CRITICAL look-ahead bias in IC validation.")
    print()
    print("PROBLEM with original implementation:")
    print("  - Uses FUTURE returns to validate CURRENT signals")
    print("  - IC appears artificially high")
    print("  - Production performance will disappoint")
    print()
    print("SOLUTION (this patch):")
    print("  - Uses HISTORICAL IC only (walk-forward validation)")
    print("  - At any date, only uses data available before that date")
    print("  - Honest IC estimate of out-of-sample performance")
    print()
    print("Usage:")
    print("  from patches.patch_001_pit_safe_ic_validation import PITSafeICValidator")
    print("  validator = PITSafeICValidator(data_dir=Path('production_data'))")
    print("  result = validator.validate_momentum_signal(as_of_date='2026-01-15')")
    print()
