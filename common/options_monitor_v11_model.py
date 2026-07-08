"""Options Monitor v1.1 — Trade verdict + probability model (Spec 040 Sprint 3).

Separates monitoring verdict from trade verdict. Provides:
  1. Deterministic trade verdict rules (live now)
  2. Probability model stub (trained after outcome data accumulates)
  3. State transition tracking (new/ongoing/resolved)

Trade verdicts are advisory-only in v1.1.

Usage:
    from common.options_monitor_v11_model import compute_trade_verdict, track_state
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_D = Decimal
_D0 = _D("0")


# ---------------------------------------------------------------------------
# Trade verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeVerdict:
    """Advisory trade verdict for one ticker."""

    bias: str  # LONG_GAMMA | SHORT_PREMIUM_AVOID | POST_EVENT_SHORT_VOL | NO_ACTION
    confidence: float  # 0-1
    reason: str  # human-readable explanation
    primary_factor: str  # EP | SR | SK | DV


def compute_trade_verdict(
    *,
    s_final: float,
    f_ep: float,
    f_sr: float,
    f_sk: float,
    f_dv: float,
    chain_quality: float,
    event_window_flag: bool = False,
    hard_catalyst_flag: bool = False,
    days_to_catalyst: Optional[int] = None,
    catalyst_class: str = "other",
    post_event: bool = False,
    # Probability model outputs (None until model is trained)
    p_move_gt_implied: Optional[float] = None,
    p_iv_crush: Optional[float] = None,
    p_false_positive: Optional[float] = None,
) -> TradeVerdict:
    """Compute advisory trade verdict from v1.1 features.

    Uses probability model when available, falls back to deterministic
    rules when probabilities are None (pre-training phase).
    """
    # Identify primary factor
    factors = {"EP": f_ep, "SR": f_sr, "SK": f_sk, "DV": f_dv}
    primary = max(factors, key=factors.get)

    # --- Probability-based rules (when model is trained) ---
    if p_move_gt_implied is not None and p_false_positive is not None:
        if p_move_gt_implied >= 0.62 and p_false_positive <= 0.35:
            return TradeVerdict(
                bias="LONG_GAMMA",
                confidence=min(p_move_gt_implied, 1 - p_false_positive),
                reason=f"p_move={p_move_gt_implied:.2f}, p_fp={p_false_positive:.2f}, primary={primary}",
                primary_factor=primary,
            )

    if post_event and p_iv_crush is not None and p_iv_crush >= 0.65:
        return TradeVerdict(
            bias="POST_EVENT_SHORT_VOL",
            confidence=p_iv_crush,
            reason=f"Post-event, p_iv_crush={p_iv_crush:.2f}",
            primary_factor=primary,
        )

    # --- Deterministic fallback rules (pre-training phase) ---

    # LONG_GAMMA: strong event premium + near catalyst + decent quality
    if event_window_flag and hard_catalyst_flag and f_ep >= 0.65 and chain_quality >= 0.5 and s_final >= 0.60:
        return TradeVerdict(
            bias="LONG_GAMMA",
            confidence=min(s_final, chain_quality),
            reason=f"Event window + hard catalyst + EP={f_ep:.2f}, Q={chain_quality:.2f}",
            primary_factor="EP",
        )

    # POST_EVENT_SHORT_VOL: event passed, surface still elevated
    if post_event and f_ep >= 0.50 and f_sr < 0.30:
        return TradeVerdict(
            bias="POST_EVENT_SHORT_VOL",
            confidence=0.5 * chain_quality,
            reason=f"Post-event, EP still elevated={f_ep:.2f}, SR fading={f_sr:.2f}",
            primary_factor="EP",
        )

    # SHORT_PREMIUM_AVOID: high event premium but low quality or high divergence
    if f_ep >= 0.60 and (chain_quality < 0.4 or f_dv >= 0.50):
        return TradeVerdict(
            bias="SHORT_PREMIUM_AVOID",
            confidence=0.4,
            reason=f"High EP={f_ep:.2f} but Q={chain_quality:.2f}, DV={f_dv:.2f}",
            primary_factor=primary,
        )

    # SHORT_PREMIUM_AVOID: strong skew stress on financing catalyst
    if catalyst_class == "financing" and f_sk >= 0.60:
        return TradeVerdict(
            bias="SHORT_PREMIUM_AVOID",
            confidence=0.4,
            reason=f"Financing catalyst + high skew stress={f_sk:.2f}",
            primary_factor="SK",
        )

    return TradeVerdict(
        bias="NO_ACTION",
        confidence=0.0,
        reason="No actionable signal",
        primary_factor=primary,
    )


# ---------------------------------------------------------------------------
# State tracking (new / ongoing / resolved)
# ---------------------------------------------------------------------------


def track_state(
    current_tickers: Dict[str, Dict[str, Any]],
    prior_state_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Track state transitions across daily runs.

    Args:
        current_tickers: {ticker: {om11_* fields}} for tickers with verdict != NONE
        prior_state_path: Path to prior day's state JSON

    Returns:
        {ticker: {state, ...fields}} with state = NEW | ONGOING | RESOLVED
    """
    prior: Dict[str, Dict] = {}
    if prior_state_path and prior_state_path.exists():
        try:
            with open(prior_state_path, encoding="utf-8") as f:
                prior_data = json.load(f)
            for entry in prior_data.get("active", []):
                if entry.get("ticker"):
                    prior[entry["ticker"]] = entry
        except (json.JSONDecodeError, OSError):
            pass

    result: Dict[str, Dict[str, Any]] = {}

    # Current tickers: NEW or ONGOING
    for ticker, fields in current_tickers.items():
        state = "ONGOING" if ticker in prior else "NEW"
        result[ticker] = {**fields, "state": state}

    # Prior tickers not in current: RESOLVED
    for ticker, prev_fields in prior.items():
        if ticker not in current_tickers:
            result[ticker] = {
                "ticker": ticker,
                "state": "RESOLVED",
                "prior_verdict": prev_fields.get("om11_monitor_verdict", ""),
                "prior_score": prev_fields.get("om11_score_final", ""),
            }

    return result


# ---------------------------------------------------------------------------
# Probability model (stub — trained after outcome data)
# ---------------------------------------------------------------------------


class OM11ProbabilityModel:
    """Calibrated probability model for v1.1 outcomes.

    Stub implementation that returns None for all probabilities until
    trained on labeled data. Training uses logistic regression with
    elastic net + isotonic calibration.
    """

    def __init__(self):
        self._trained = False
        self._model_move = None
        self._model_crush = None
        self._model_fp = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        """Predict probabilities for one observation.

        Returns None for all probabilities if model is not trained.
        """
        if not self._trained:
            return {
                "p_move_gt_implied": None,
                "p_post_event_iv_crush": None,
                "p_false_positive": None,
            }

        # TODO: implement after training
        # feature_vec = self._extract_features(features)
        # p_move = self._model_move.predict_proba(feature_vec)
        # p_crush = self._model_crush.predict_proba(feature_vec)
        # p_fp = self._model_fp.predict_proba(feature_vec)
        return {
            "p_move_gt_implied": None,
            "p_post_event_iv_crush": None,
            "p_false_positive": None,
        }

    def train(
        self,
        labels_path: Path,
        min_observations: int = 100,
    ) -> bool:
        """Train probability models from labeled backtest data.

        Requires: tools/backtest_options_monitor_v11.py output with
        enough labeled observations.

        Returns True if training succeeded.
        """
        if not labels_path.exists():
            logger.warning("Labels file not found: %s", labels_path)
            return False

        try:
            with open(labels_path) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load labels: %s", exc)
            return False

        n = report.get("n_observations", 0)
        if n < min_observations:
            logger.warning(
                "Insufficient observations for training: %d < %d",
                n,
                min_observations,
            )
            return False

        # TODO: implement logistic + isotonic calibration
        # This requires scipy (already in requirements.txt) and
        # will be implemented when enough outcome data accumulates.
        logger.info("Training stub: %d observations available, need implementation", n)
        return False

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "trained": self._trained,
            "schema": "om11_model.v1",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def load(self, path: Path) -> bool:
        """Load trained model from disk."""
        if not path.exists():
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            self._trained = state.get("trained", False)
            return self._trained
        except (json.JSONDecodeError, OSError):
            return False


# ---------------------------------------------------------------------------
# Full v1.1 verdict computation (entry point)
# ---------------------------------------------------------------------------


def compute_full_verdict(
    features: Dict[str, Any],
    *,
    model: Optional[OM11ProbabilityModel] = None,
    post_event: bool = False,
) -> Dict[str, str]:
    """Compute monitoring + trade verdict from v1.1 features.

    Args:
        features: Output from compute_v11_features()
        model: Trained probability model (or None for deterministic rules)
        post_event: Whether the catalyst event has already passed

    Returns:
        Dict with all om11_* fields including trade verdict.
    """
    s_final = float(features.get("om11_score_final", "0") or "0")
    f_ep = float(features.get("om11_factor_event_premium", "0") or "0")
    f_sr = float(features.get("om11_factor_surface_repricing", "0") or "0")
    f_sk = float(features.get("om11_factor_skew_tail", "0") or "0")
    f_dv = float(features.get("om11_factor_divergence", "0") or "0")
    quality = float(features.get("om11_chain_quality", "0") or "0")
    event_window = features.get("om11_event_window_flag", "0") == "1"
    catalyst_class = features.get("om11_catalyst_class", "other")

    # Get probabilities from model if available
    probs: dict[str, float | None] = {
        "p_move_gt_implied": None,
        "p_post_event_iv_crush": None,
        "p_false_positive": None,
    }
    if model and model.is_trained:
        probs = model.predict(features)

    tv = compute_trade_verdict(
        s_final=s_final,
        f_ep=f_ep,
        f_sr=f_sr,
        f_sk=f_sk,
        f_dv=f_dv,
        chain_quality=quality,
        event_window_flag=event_window,
        hard_catalyst_flag=features.get("hard_catalyst_flag", False),
        catalyst_class=catalyst_class,
        post_event=post_event,
        p_move_gt_implied=probs.get("p_move_gt_implied"),
        p_iv_crush=probs.get("p_post_event_iv_crush"),
        p_false_positive=probs.get("p_false_positive"),
    )

    result = dict(features)
    result["om11_trade_bias"] = tv.bias
    result["om11_trade_confidence"] = str(round(tv.confidence, 4))
    result["om11_trade_reason"] = tv.reason
    result["om11_p_move_gt_implied"] = str(probs["p_move_gt_implied"]) if probs["p_move_gt_implied"] is not None else ""
    result["om11_p_post_event_iv_crush"] = (
        str(probs["p_post_event_iv_crush"]) if probs["p_post_event_iv_crush"] is not None else ""
    )
    result["om11_p_false_positive"] = str(probs["p_false_positive"]) if probs["p_false_positive"] is not None else ""

    return result
