# extensions/bootstrap_scoring.py
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

from common.random_state import DeterministicRNG


def _decimal_mean(values: List[Decimal]) -> Decimal:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / Decimal(len(values))


def _quantile(sorted_vals: List[Decimal], q: Decimal) -> Decimal:
    """
    Deterministic quantile via 'nearest rank' on a sorted list.
    q in [0,1]. For CI bounds, this is plenty and reproducible.
    """
    if not sorted_vals:
        raise ValueError("quantile of empty list")
    if q < 0 or q > 1:
        raise ValueError("q must be in [0,1]")

    n = len(sorted_vals)
    # nearest-rank: k = ceil(q*n) with k in [1..n]
    k = (q * Decimal(n)).to_integral_value(rounding=ROUND_HALF_UP)
    k_int = int(k)
    if k_int <= 0:
        return sorted_vals[0]
    if k_int >= n:
        return sorted_vals[-1]
    return sorted_vals[k_int - 1]


def compute_bootstrap_ci_decimal(
    scores: List[Decimal],
    rng: DeterministicRNG,
    n_bootstrap: int = 1000,
    confidence: Decimal = Decimal("0.95"),
) -> Dict[str, Any]:
    """
    Decimal-only bootstrap CI for the mean.

    Returns:
      {
        "mean": str,
        "ci_lower": str,
        "ci_upper": str,
        "bootstrap_samples": int,
        "rng_audit": {...}
      }
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if not scores:
        return {
            "error": "no_scores",
            "bootstrap_samples": n_bootstrap,
            "rng_audit": rng.audit().as_dict(),
        }

    base_mean = _decimal_mean(scores)
    n = len(scores)

    boot_means: List[Decimal] = []
    for _ in range(n_bootstrap):
        idx = rng.bootstrap_sample_indices(n, n)
        sample = [scores[i] for i in idx]
        boot_means.append(_decimal_mean(sample))

    boot_means.sort()

    alpha = Decimal("1") - confidence
    lower_q = alpha / Decimal("2")
    upper_q = Decimal("1") - (alpha / Decimal("2"))

    ci_lower = _quantile(boot_means, lower_q)
    ci_upper = _quantile(boot_means, upper_q)

    # Serialize as strings to avoid Decimal JSON issues + keep canonical dumps stable.
    return {
        "mean": str(base_mean),
        "ci_lower": str(ci_lower),
        "ci_upper": str(ci_upper),
        "bootstrap_samples": n_bootstrap,
        "rng_audit": rng.audit().as_dict(),
    }