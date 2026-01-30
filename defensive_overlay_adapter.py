"""
defensive_overlay_adapter.py - Production Defensive Overlay System

Features:
1. Configurable thresholds via DefensiveConfig
2. Multi-factor multiplier: correlation, volatility, momentum, RSI, drawdown
3. Dynamic position floor that scales with universe size
4. Aggressive inverse-volatility weighting

v2.0.0 (2026-01-29): Configuration-driven, utilizes all 9 defensive features
v1.0.0: Original version with hardcoded thresholds
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, List, Optional, Tuple

WQ = Decimal("0.0001")  # Weight quantization

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DefensiveConfig:
    """
    Configuration for defensive overlay calculations.

    All thresholds are configurable for backtesting and regime adaptation.
    Default values are conservative starting points for biotech universe.
    """
    # Config identity (for provenance)
    config_id: str = "default"
    config_version: str = "2.0.0"

    # Multiplier bounds (clamp to prevent extreme values from stacking)
    mult_floor: Decimal = Decimal("0.75")                # Minimum multiplier
    mult_ceiling: Decimal = Decimal("1.60")              # Maximum multiplier

    # Correlation thresholds
    corr_elite_threshold: Decimal = Decimal("0.30")      # Below = elite diversifier
    corr_good_threshold: Decimal = Decimal("0.40")       # Below = good diversifier
    corr_high_threshold: Decimal = Decimal("0.80")       # Above = penalty

    # Volatility thresholds
    vol_elite_threshold: Decimal = Decimal("0.40")       # Below = elite (40% ann)
    vol_good_threshold: Decimal = Decimal("0.50")        # Below = good (50% ann)
    vol_high_threshold: Decimal = Decimal("0.80")        # Above = high vol penalty

    # Multiplier values
    mult_elite: Decimal = Decimal("1.40")                # Elite diversifier bonus
    mult_good: Decimal = Decimal("1.10")                 # Good diversifier bonus
    mult_high_corr_penalty: Decimal = Decimal("0.95")    # High correlation penalty
    mult_high_vol_penalty: Decimal = Decimal("0.97")     # High volatility penalty

    # Momentum thresholds (ret_21d)
    momentum_bonus_threshold: Decimal = Decimal("0.10")  # >10% 21d return = bonus
    momentum_penalty_threshold: Decimal = Decimal("-0.20")  # <-20% = penalty
    mult_momentum_bonus: Decimal = Decimal("1.05")       # Momentum bonus
    mult_momentum_penalty: Decimal = Decimal("0.95")     # Momentum penalty
    enable_momentum: bool = True                         # Feature flag

    # RSI thresholds (regime detection)
    rsi_oversold_threshold: Decimal = Decimal("30")      # RSI < 30 = oversold
    rsi_overbought_threshold: Decimal = Decimal("70")    # RSI > 70 = overbought
    mult_rsi_oversold_bonus: Decimal = Decimal("1.03")   # Mean reversion opportunity
    mult_rsi_overbought_penalty: Decimal = Decimal("0.98")  # Crowded trade
    enable_rsi: bool = True                              # Feature flag

    # Drawdown thresholds
    drawdown_warning_threshold: Decimal = Decimal("-0.30")  # -30% = warning
    drawdown_penalty_threshold: Decimal = Decimal("-0.40")  # -40% = penalty
    mult_drawdown_penalty: Decimal = Decimal("0.92")     # Deep drawdown penalty
    enable_drawdown_penalty: bool = True                 # Feature flag

    # Vol ratio (regime indicator: vol_20d / vol_60d)
    vol_ratio_expanding_threshold: Decimal = Decimal("1.30")  # >1.3 = expanding vol
    mult_vol_expanding_penalty: Decimal = Decimal("0.97")  # Expanding vol penalty
    enable_vol_ratio: bool = False                       # Disabled by default (experimental)

    # Position sizing
    max_position: Decimal = Decimal("0.07")              # 7% max position
    inv_vol_power: Decimal = Decimal("2.0")              # Inverse vol exponent

    def config_hash(self) -> str:
        """Compute short hash of config for provenance tracking."""
        import hashlib
        # Hash key threshold values (not metadata like id/version)
        key_values = (
            str(self.corr_elite_threshold),
            str(self.corr_good_threshold),
            str(self.mult_elite),
            str(self.mult_good),
            str(self.mult_floor),
            str(self.mult_ceiling),
            str(self.enable_momentum),
            str(self.enable_rsi),
            str(self.enable_drawdown_penalty),
            str(self.enable_vol_ratio),
        )
        return hashlib.sha256("|".join(key_values).encode()).hexdigest()[:8]

    def to_provenance(self) -> Dict:
        """Return config provenance for audit trail."""
        return {
            "config_id": self.config_id,
            "config_version": self.config_version,
            "config_hash": self.config_hash(),
            "mult_bounds": [str(self.mult_floor), str(self.mult_ceiling)],
            "enabled_factors": {
                "correlation": True,  # Always enabled
                "volatility": True,   # Always enabled
                "momentum": self.enable_momentum,
                "rsi": self.enable_rsi,
                "drawdown": self.enable_drawdown_penalty,
                "vol_ratio": self.enable_vol_ratio,
            },
        }


# Default configuration (conservative)
DEFAULT_DEFENSIVE_CONFIG = DefensiveConfig()

# Aggressive configuration (larger bonuses/penalties)
AGGRESSIVE_DEFENSIVE_CONFIG = DefensiveConfig(
    config_id="aggressive",
    mult_elite=Decimal("1.50"),
    mult_good=Decimal("1.15"),
    mult_high_corr_penalty=Decimal("0.90"),
    mult_momentum_bonus=Decimal("1.08"),
    mult_momentum_penalty=Decimal("0.90"),
    enable_vol_ratio=True,
    mult_ceiling=Decimal("1.80"),  # Higher ceiling for aggressive config
)


# Null-equivalent values that should be treated as missing data
_NULL_EQUIVALENTS = frozenset({"", "N/A", "Unknown", "NaN", "-", "None", "null", "nan"})


def _is_valid_value(value) -> bool:
    """Check if value is non-null and non-placeholder for coverage counting."""
    if value is None:
        return False
    s = str(value).strip()
    if not s or s in _NULL_EQUIVALENTS:
        return False
    return True


def _safe_decimal(value, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely convert string to Decimal, returning default on failure."""
    if value is None:
        return default
    s = str(value).strip()
    if not s or s in _NULL_EQUIVALENTS:
        return default
    try:
        d = Decimal(s)
        if not d.is_finite():
            return default
        return d
    except (ValueError, TypeError, InvalidOperation):
        return default

def _q(x: Decimal) -> Decimal:
    """Quantize weight to 4 decimal places."""
    return x.quantize(WQ, rounding=ROUND_HALF_UP)

def sanitize_corr(defensive_features: Dict[str, str]) -> Tuple[Optional[Decimal], List[str]]:
    """
    Sanitize correlation data, treating placeholders and invalid values as missing.
    
    Returns: (correlation_value or None, list_of_flags)
    
    Common issues:
    - Placeholder 0.50 when correlation calculation failed
    - Missing field entirely
    - NaN or Infinity values
    - Out of valid range [-1, 1]
    """
    flags: List[str] = []
    PLACEHOLDER_CORR = Decimal("0.50")
    
    # Try both field names
    corr_s = defensive_features.get("corr_xbi") or defensive_features.get("corr_xbi_120d")
    
    if not corr_s:
        flags.append("def_corr_missing")
        return None, flags
    
    try:
        corr = Decimal(str(corr_s))
    except Exception:
        flags.append("def_corr_parse_fail")
        return None, flags
    
    # CRITICAL: Check if Decimal is NaN or Inf BEFORE doing comparisons
    # This prevents InvalidOperation errors
    if not corr.is_finite():
        flags.append("def_corr_not_finite")
        return None, flags
    
    # Treat placeholder as missing
    if corr == PLACEHOLDER_CORR:
        flags.append("def_corr_placeholder_0.50")
        return None, flags
    
    # Validate range (safe now that we know it's finite)
    if corr < Decimal("-1") or corr > Decimal("1"):
        flags.append("def_corr_out_of_range")
        return None, flags
    
    return corr, flags


def _extract_defensive_tags(notes: List[str]) -> List[str]:
    """
    Extract machine-safe tags from defensive notes.

    Converts string notes like 'def_mult_elite_1.40' to semantic tags like 'elite'.
    This enables set-membership checks instead of error-prone substring matching.

    Returns:
        List of semantic tags (e.g., ['elite', 'momentum_bonus'])
    """
    tags = []

    # Tag mappings: note substring -> semantic tag
    tag_patterns = [
        ("def_not_applied", "multiplier_disabled"),
        ("def_mult_elite_", "elite"),
        ("def_mult_good_", "good_diversifier"),
        ("def_mult_high_corr_penalty", "high_corr_penalty"),
        ("def_mult_high_vol_", "high_vol_penalty"),
        ("def_mult_momentum_bonus", "momentum_bonus"),
        ("def_mult_momentum_penalty", "momentum_penalty"),
        ("def_mult_rsi_oversold", "rsi_oversold"),
        ("def_mult_rsi_overbought", "rsi_overbought"),
        ("def_mult_drawdown_penalty", "drawdown_penalty"),
        ("def_mult_vol_expanding", "vol_expanding_penalty"),
        ("def_warn_drawdown", "drawdown_warning"),
        ("def_corr_missing", "corr_missing"),
        ("def_corr_placeholder", "corr_placeholder"),
        ("def_skip_not_elite", "not_elite"),
        ("def_skip_vol_too_high", "vol_too_high"),
    ]

    for note in notes:
        for pattern, tag in tag_patterns:
            if pattern in note and tag not in tags:
                tags.append(tag)
                break  # Only one tag per note

    return tags


def _derive_defensive_bucket(mult: Decimal) -> str:
    """Derive defensive bucket from multiplier value for easy filtering."""
    if mult > Decimal("1.20"):
        return "elite"
    elif mult > Decimal("1.05"):
        return "good"
    elif mult < Decimal("0.98"):
        return "penalty"
    return "neutral"


def _extract_audit_features(defensive_features: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Extract key defensive features for audit output.

    Returns a subset of features used in multiplier calculation,
    making "why did this move?" questions trivial without re-joining to source.

    Returns:
        Dict with corr, vol, rsi, momentum, drawdown (string values or None)
    """
    # Get values with alias handling (same as defensive_multiplier)
    corr = defensive_features.get("corr_xbi") or defensive_features.get("corr_xbi_120d")
    vol = defensive_features.get("vol_60d")
    rsi = defensive_features.get("rsi_14d")
    momentum = defensive_features.get("ret_21d")
    drawdown = defensive_features.get("drawdown_current") or defensive_features.get("drawdown_60d")

    return {
        "corr_xbi": corr,
        "vol_60d": vol,
        "rsi_14d": rsi,
        "ret_21d": momentum,
        "drawdown": drawdown,
    }


def defensive_multiplier(
    defensive_features: Dict[str, str],
    config: Optional[DefensiveConfig] = None,
) -> Tuple[Decimal, List[str]]:
    """
    Calculate defensive multiplier using all available features.

    Multi-factor approach utilizing:
    - Correlation (corr_xbi_120d): diversification benefit/penalty
    - Volatility (vol_60d): risk-adjusted sizing
    - Momentum (ret_21d): trend confirmation
    - RSI (rsi_14d): regime detection
    - Drawdown (drawdown_current): distress penalty
    - Vol ratio (vol_ratio): volatility regime

    Args:
        defensive_features: Dict with feature values as strings
        config: DefensiveConfig with thresholds (uses DEFAULT if None)

    Returns:
        (multiplier, notes) where multiplier is in ~[0.80, 1.50] range
    """
    cfg = config or DEFAULT_DEFENSIVE_CONFIG
    m = Decimal("1.00")
    notes: List[str] = []

    # Extract features safely
    vol = _safe_decimal(defensive_features.get("vol_60d"))
    momentum = _safe_decimal(defensive_features.get("ret_21d"))
    rsi = _safe_decimal(defensive_features.get("rsi_14d"))
    drawdown = _safe_decimal(
        defensive_features.get("drawdown_current") or
        defensive_features.get("drawdown_60d")
    )
    vol_ratio = _safe_decimal(defensive_features.get("vol_ratio"))

    # Sanitize correlation (handle placeholders)
    corr, corr_flags = sanitize_corr(defensive_features or {})
    notes.extend(corr_flags)

    # -------------------------------------------------------------------------
    # FACTOR 1: Correlation (diversification)
    # -------------------------------------------------------------------------
    if corr is not None:
        # High correlation penalty (always applies)
        if corr > cfg.corr_high_threshold:
            m *= cfg.mult_high_corr_penalty
            notes.append(f"def_mult_high_corr_{cfg.mult_high_corr_penalty}")

        # Elite diversifier bonus (VERY selective)
        # Requires: low corr AND low vol AND real correlation data
        elif corr < cfg.corr_elite_threshold:
            if vol and vol < cfg.vol_elite_threshold:
                m *= cfg.mult_elite
                notes.append(f"def_mult_elite_{cfg.mult_elite}")
            else:
                notes.append("def_skip_not_elite_vol")

        # Good diversifier bonus (less selective)
        elif corr < cfg.corr_good_threshold:
            if vol and vol < cfg.vol_good_threshold:
                m *= cfg.mult_good
                notes.append(f"def_mult_good_{cfg.mult_good}")
            else:
                notes.append("def_skip_vol_too_high")

    # -------------------------------------------------------------------------
    # FACTOR 2: High volatility penalty (independent of correlation)
    # -------------------------------------------------------------------------
    if vol is not None and vol > cfg.vol_high_threshold:
        m *= cfg.mult_high_vol_penalty
        notes.append(f"def_mult_high_vol_{cfg.mult_high_vol_penalty}")

    # -------------------------------------------------------------------------
    # FACTOR 3: Momentum (trend confirmation)
    # -------------------------------------------------------------------------
    if cfg.enable_momentum and momentum is not None:
        if momentum > cfg.momentum_bonus_threshold:
            m *= cfg.mult_momentum_bonus
            notes.append(f"def_mult_momentum_bonus_{cfg.mult_momentum_bonus}")
        elif momentum < cfg.momentum_penalty_threshold:
            m *= cfg.mult_momentum_penalty
            notes.append(f"def_mult_momentum_penalty_{cfg.mult_momentum_penalty}")

    # -------------------------------------------------------------------------
    # FACTOR 4: RSI (regime detection)
    # -------------------------------------------------------------------------
    if cfg.enable_rsi and rsi is not None:
        if rsi < cfg.rsi_oversold_threshold:
            m *= cfg.mult_rsi_oversold_bonus
            notes.append(f"def_mult_rsi_oversold_{cfg.mult_rsi_oversold_bonus}")
        elif rsi > cfg.rsi_overbought_threshold:
            m *= cfg.mult_rsi_overbought_penalty
            notes.append(f"def_mult_rsi_overbought_{cfg.mult_rsi_overbought_penalty}")

    # -------------------------------------------------------------------------
    # FACTOR 5: Drawdown (distress detection)
    # -------------------------------------------------------------------------
    if drawdown is not None:
        if cfg.enable_drawdown_penalty and drawdown < cfg.drawdown_penalty_threshold:
            m *= cfg.mult_drawdown_penalty
            notes.append(f"def_mult_drawdown_penalty_{cfg.mult_drawdown_penalty}")
        elif drawdown < cfg.drawdown_warning_threshold:
            notes.append("def_warn_drawdown_gt_30pct")

    # -------------------------------------------------------------------------
    # FACTOR 6: Vol ratio (volatility regime - experimental)
    # -------------------------------------------------------------------------
    if cfg.enable_vol_ratio and vol_ratio is not None:
        if vol_ratio > cfg.vol_ratio_expanding_threshold:
            m *= cfg.mult_vol_expanding_penalty
            notes.append(f"def_mult_vol_expanding_{cfg.mult_vol_expanding_penalty}")

    # -------------------------------------------------------------------------
    # CLAMP: Prevent extreme multipliers from factor stacking
    # -------------------------------------------------------------------------
    m_unclamped = m
    m = max(cfg.mult_floor, min(cfg.mult_ceiling, m))
    if m != m_unclamped:
        notes.append(f"def_mult_clamped_{m_unclamped:.4f}_to_{m}")

    return m, notes


# Backward-compatible wrapper (no config argument)
def defensive_multiplier_legacy(defensive_features: Dict[str, str]) -> Tuple[Decimal, List[str]]:
    """Legacy wrapper for backward compatibility. Uses default config."""
    return defensive_multiplier(defensive_features, config=None)


def raw_inv_vol_weight(defensive_features: Dict[str, str], power: Decimal = Decimal("2.0")) -> Optional[Decimal]:
    """
    Calculate raw inverse-volatility weight with exponential scaling.
    
    ENHANCED: Uses vol^2.0 (default) for maximum differentiation.
    This creates very strong spread between low-vol and high-vol stocks.
    
    Examples (vol^2.0):
    - 20% vol: 1/(0.20^2.0) = 25.0  (very large weight)
    - 25% vol: 1/(0.25^2.0) = 16.0  (large weight)
    - 40% vol: 1/(0.40^2.0) = 6.25  (medium weight)
    - 100% vol: 1/(1.00^2.0) = 1.0  (small weight)
    - 200% vol: 1/(2.00^2.0) = 0.25 (tiny weight)
    
    Args:
        power: Exponent for volatility (2.0 = maximum, 1.8 = aggressive, 1.5 = moderate)
    """
    vol_s = defensive_features.get("vol_60d")
    if not vol_s:
        return None
    try:
        vol = Decimal(vol_s)
        if vol <= 0:
            return None
        return Decimal("1") / (vol ** power)
    except (ValueError, TypeError, InvalidOperation, ZeroDivisionError):
        return None  # Invalid volatility - cannot compute inverse weight


def calculate_dynamic_floor(n_securities: int) -> Decimal:
    """
    Calculate dynamic position floor based on universe size.
    
    Logic:
    - 20-50 stocks: 1.0% floor (traditional focused portfolio)
    - 51-100 stocks: 0.5% floor (allows proper differentiation)
    - 101-200 stocks: 0.3% floor (maximum diversification)
    - 201+ stocks: 0.2% floor (ultra-diversified)
    
    This ensures the floor is always well below the average weight,
    allowing inverse-volatility weighting to work properly.
    """
    if n_securities <= 50:
        return Decimal("0.01")  # 1.0%
    elif n_securities <= 100:
        return Decimal("0.005")  # 0.5%
    elif n_securities <= 200:
        return Decimal("0.003")  # 0.3%
    else:
        return Decimal("0.002")  # 0.2%


def apply_caps_and_renormalize(
    records: List[Dict],
    cash_target: Decimal = Decimal("0.00"),  # No cash reserve - fully invested
    max_pos: Decimal = Decimal("0.07"),  # 7% max
    min_pos: Optional[Decimal] = None,  # Dynamic - calculated based on universe
    top_n: Optional[int] = None,  # NEW: If set, only invest in top N includable names
) -> None:
    """
    Apply position caps and renormalize weights.
    Mutates records in-place, setting record["position_weight"].
    
    ENHANCED: 
    - Automatically calculates appropriate floor based on universe size
    - Max position reduced to 7% for better risk management at scale
    - NEW: Top-N selection for conviction-based portfolios
    
    Args:
        top_n: If provided, only size the top N includable names post-ranking.
               All others get zero weight and "NOT_IN_TOP_N" flag.
               This increases max weights naturally without changing math.
               Recommended: 60 for balanced, 40 for high conviction.
    """
    investable = Decimal("1.0") - cash_target

    # Get all potentially includable securities (before top-N cut)
    included_all = [r for r in records if r.get("rankable", True)]
    
    # Apply top-N selection if specified
    if top_n is not None and len(included_all) > top_n:
        # Records should already be sorted by composite_rank
        # (or if not, sort them now by rank ascending)
        included_all_sorted = sorted(included_all, key=lambda x: x.get("composite_rank", 999))
        
        # Select top N
        included = included_all_sorted[:top_n]
        excluded_by_topn = included_all_sorted[top_n:]
        
        # Mark excluded securities
        for r in excluded_by_topn:
            r["rankable"] = False  # Mark as non-rankable
            position_flags = r.get("position_flags", [])
            if isinstance(position_flags, list):
                position_flags.append("NOT_IN_TOP_N")
            else:
                position_flags = ["NOT_IN_TOP_N"]
            r["position_flags"] = position_flags
            r["position_weight"] = "0.0000"
    else:
        included = included_all
    
    # Excluded get zero weight
    for r in records:
        if not r.get("rankable", True):
            r["position_weight"] = "0.0000"

    if not included:
        return

    # Calculate dynamic floor if not provided
    if min_pos is None:
        min_pos = calculate_dynamic_floor(len(included))

    # Collect raw weights
    raw = []
    for r in included:
        w = r.get("_position_weight_raw")
        raw.append(w if isinstance(w, Decimal) else Decimal("0"))

    total_raw = sum(raw)
    
    # Fallback to equal-weight if no valid weights
    if total_raw <= 0:
        n = Decimal(str(max(1, len(included))))
        ew = investable / n
        for r in included:
            r["position_weight"] = str(_q(max(min_pos, min(max_pos, ew))))
        return

    # Initial normalize to investable capital
    weights = [(w / total_raw) * investable for w in raw]

    # Apply caps and floors
    capped = [max(min_pos, min(max_pos, w)) for w in weights]

    # Renormalize after capping, then re-enforce floor
    # (renormalization can push weights below floor)
    total_capped = sum(capped)
    if total_capped > 0:
        scale = investable / total_capped
        capped = [w * scale for w in capped]

        # Re-enforce floor after scaling (iterative approach)
        for _ in range(3):  # Max 3 iterations to converge
            floor_violations = sum(1 for w in capped if w < min_pos)
            if floor_violations == 0:
                break
            # Bring floor violations up to floor, reduce others proportionally
            floored = []
            excess_needed = Decimal("0")
            for w in capped:
                if w < min_pos:
                    excess_needed += min_pos - w
                    floored.append(min_pos)
                else:
                    floored.append(w)
            # Reduce weights above floor proportionally
            above_floor = [(i, w) for i, w in enumerate(floored) if w > min_pos]
            if above_floor and excess_needed > 0:
                total_above = sum(w - min_pos for i, w in above_floor)
                if total_above > 0:
                    for i, w in above_floor:
                        reduction = (w - min_pos) / total_above * excess_needed
                        floored[i] = max(min_pos, w - reduction)
            capped = floored

    # Quantize all weights first
    quantized = [_q(w) for w in capped]

    # Calculate residual and allocate to anchor (highest weight) for exact sum
    quantized_sum = sum(quantized)
    residual = investable - quantized_sum

    if residual != Decimal("0") and quantized:
        # Find anchor: highest weight position (first by index if tied)
        anchor_idx = max(range(len(quantized)), key=lambda i: (quantized[i], -i))
        # Add residual to anchor and re-quantize
        quantized[anchor_idx] = _q(quantized[anchor_idx] + residual)

    # Assign final weights
    for r, w in zip(included, quantized):
        r["position_weight"] = str(w)


def enrich_with_defensive_overlays(
    output: Dict,
    scores_by_ticker: Dict[str, Dict],
    apply_multiplier: bool = True,
    apply_position_sizing: bool = False,  # Deprecated: position sizing separate from scoring
    top_n: Optional[int] = None,  # Top-N selection for conviction portfolios
    include_position_weight: bool = False,  # Output position_weight (risk-budget, not alpha)
    defensive_config: Optional[DefensiveConfig] = None,  # Configurable thresholds
) -> Dict:
    """
    Enrich Module 5 output with defensive overlays.

    This function:
    1. Applies multi-factor defensive multiplier to composite scores
    2. Optionally calculates position weights using inverse-volatility (deprecated)
    3. Adds defensive_notes field to each security

    Multi-factor multiplier utilizes:
    - Correlation (corr_xbi_120d): diversification benefit/penalty
    - Volatility (vol_60d): risk-adjusted sizing
    - Momentum (ret_21d): trend confirmation
    - RSI (rsi_14d): regime detection
    - Drawdown (drawdown_current): distress penalty

    NOTE: Position sizing (risk-budget weights) is now separate from alpha research.
    Expected Returns (score_z * lambda) are computed in module_5_composite_v3.py.

    Args:
        output: Output dict from rank_securities()
        scores_by_ticker: Dict with defensive_features per ticker
        apply_multiplier: If True, apply multi-factor score multiplier
        apply_position_sizing: If True, calculate position weights (deprecated)
        top_n: If provided, only invest in top N names (deprecated)
        include_position_weight: If True, include position_weight in output (deprecated)
        defensive_config: DefensiveConfig with thresholds (uses DEFAULT if None)

    Returns:
        Modified output dict (mutated in-place, also returned for convenience)
    """
    cfg = defensive_config or DEFAULT_DEFENSIVE_CONFIG
    ranked = output.get("ranked_securities", [])

    if not ranked:
        return output

    # Initialize diagnostics
    if "diagnostic_counts" not in output:
        output["diagnostic_counts"] = {}

    # Track defensive feature coverage (all 9 features + aliases)
    total_securities = len(ranked)
    with_def_features = 0
    n_with_sufficient = 0  # Has corr+vol OR any enabled factor
    feature_coverage = {
        "vol_60d": 0,
        "vol_20d": 0,
        "corr_xbi_120d": 0,
        "beta_xbi_60d": 0,
        "drawdown_current": 0,
        "rsi_14d": 0,
        "ret_21d": 0,
        "skew_60d": 0,
        "vol_ratio": 0,
    }
    # Track alias fields separately for diagnostics
    alias_coverage = {
        "corr_xbi": 0,       # Alias for corr_xbi_120d
        "drawdown_60d": 0,   # Alias for drawdown_current
    }

    for rec in ranked:
        ticker = rec["ticker"]
        ticker_data = scores_by_ticker.get(ticker, {})
        def_features = ticker_data.get("defensive_features", {})

        if def_features:
            with_def_features += 1

            # Track each feature using _is_valid_value for consistent null detection
            for feature in feature_coverage:
                raw_val = def_features.get(feature)
                # Handle field aliases (must match defensive_multiplier logic)
                if feature == "corr_xbi_120d" and not _is_valid_value(raw_val):
                    raw_val = def_features.get("corr_xbi")
                if feature == "drawdown_current" and not _is_valid_value(raw_val):
                    raw_val = def_features.get("drawdown_60d")
                if _is_valid_value(raw_val) and _safe_decimal(raw_val) is not None:
                    feature_coverage[feature] += 1

            # Track alias fields separately
            for alias in alias_coverage:
                if _is_valid_value(def_features.get(alias)) and _safe_decimal(def_features.get(alias)) is not None:
                    alias_coverage[alias] += 1

            # Check if sufficient for multiplier (corr+vol present OR any factor)
            has_corr = _safe_decimal(def_features.get("corr_xbi") or def_features.get("corr_xbi_120d")) is not None
            has_vol = _safe_decimal(def_features.get("vol_60d")) is not None
            has_any_factor = has_corr or has_vol or any(
                _safe_decimal(def_features.get(f)) is not None
                for f in ["ret_21d", "rsi_14d", "drawdown_current", "drawdown_60d"]
            )
            if has_any_factor:
                n_with_sufficient += 1

    # Add coverage diagnostics with per-feature breakdown
    coverage_pct = round(100 * with_def_features / total_securities, 1) if total_securities > 0 else 0
    output["diagnostic_counts"]["defensive_features_coverage"] = {
        "total_securities": total_securities,
        "with_defensive_features": with_def_features,
        "n_with_sufficient_features_for_multiplier": n_with_sufficient,
        "coverage_pct": coverage_pct,
        "by_feature": {
            k: {
                "count": v,
                "pct": round(100 * v / total_securities, 1) if total_securities > 0 else 0,
            }
            for k, v in feature_coverage.items()
        },
        "alias_coverage": {
            k: {
                "count": v,
                "pct": round(100 * v / total_securities, 1) if total_securities > 0 else 0,
            }
            for k, v in alias_coverage.items()
        },
    }

    # Add config provenance for audit trail
    provenance = cfg.to_provenance()
    provenance["defensive_overlay_applied"] = apply_multiplier
    provenance["portfolio_weights_computed"] = apply_position_sizing
    # Legacy alias for backwards compatibility
    provenance["position_sizing_enabled"] = apply_position_sizing
    output["defensive_overlay_config"] = provenance

    # Step 1: Calculate raw weights for position sizing (needed for Step 3)
    # This must run BEFORE multiplier application if position sizing is enabled
    if apply_position_sizing:
        for rec in ranked:
            ticker = rec["ticker"]
            ticker_data = scores_by_ticker.get(ticker, {})
            defensive_features = ticker_data.get("defensive_features", {})
            rec["_position_weight_raw"] = raw_inv_vol_weight(defensive_features or {})

    # Step 2: Compute defensive multiplier for all securities
    # Always add fields; only modify composite_score and re-rank when apply_multiplier=True
    multiplier_stats = {"elite": 0, "good": 0, "penalty": 0, "neutral": 0}

    for rec in ranked:
        ticker = rec["ticker"]
        ticker_data = scores_by_ticker.get(ticker, {})
        defensive_features = ticker_data.get("defensive_features", {})

        # Get current composite score
        current_score = Decimal(rec["composite_score"])

        # Always compute multiplier (for risk_adjusted_score and diagnostics)
        mult, notes = defensive_multiplier(defensive_features or {}, config=cfg)

        # Track multiplier distribution
        bucket = _derive_defensive_bucket(mult)
        multiplier_stats[bucket] += 1

        # Compute risk-adjusted score (always, for downstream use)
        risk_adjusted = current_score * mult
        risk_adjusted = min(Decimal("100"), max(Decimal("0"), risk_adjusted))

        if apply_multiplier:
            # Score modification: update composite_score, track before value
            rec["composite_score_before_defensive"] = str(current_score)
            rec["composite_score"] = str(risk_adjusted.quantize(Decimal("0.01")))
            rec["defensive_multiplier"] = str(mult)
            rec["defensive_notes"] = notes
        else:
            # No score modification: multiplier=1.00, note that it's not applied
            rec["defensive_multiplier"] = "1.00"
            rec["defensive_notes"] = ["def_not_applied"] + notes

        # Always add these fields for audit/analysis
        rec["risk_adjusted_score"] = str(risk_adjusted.quantize(Decimal("0.01")))
        rec["defensive_bucket"] = bucket
        rec["defensive_tags"] = _extract_defensive_tags(rec["defensive_notes"])  # Use final notes
        rec["defensive_features"] = _extract_audit_features(defensive_features or {})

        # Surface cache skip reason if present (e.g., IPO with <120 rows)
        cache_skip = (defensive_features or {}).get("cache_skip_reason")
        if cache_skip and f"def_cache_{cache_skip}" not in rec["defensive_notes"]:
            rec["defensive_notes"].append(f"def_cache_{cache_skip}")

    # Add multiplier distribution to diagnostics
    output["diagnostic_counts"]["multiplier_distribution"] = multiplier_stats
    output["diagnostic_counts"]["apply_multiplier_enabled"] = apply_multiplier

    # Step 3: Re-rank after multiplier application
    if apply_multiplier:
        # Re-sort by adjusted composite score
        ranked.sort(key=lambda x: (-Decimal(x["composite_score"]), x["ticker"]))
        # Re-assign ranks
        for i, rec in enumerate(ranked):
            rec["composite_rank"] = i + 1

    # Step 4: Apply position sizing with dynamic floor and optional top-N selection
    if apply_position_sizing:
        apply_caps_and_renormalize(ranked, top_n=top_n)  # Pass top_n parameter
        
        # Add position sizing diagnostics
        total_weight = sum(Decimal(r["position_weight"]) for r in ranked)
        nonzero = sum(1 for r in ranked if Decimal(r["position_weight"]) > 0)
        
        if "diagnostic_counts" not in output:
            output["diagnostic_counts"] = {}
        
        output["diagnostic_counts"]["with_nonzero_weight"] = nonzero
        output["diagnostic_counts"]["total_allocated_weight"] = str(total_weight.quantize(Decimal("0.0001")))
        
        # Add top-N diagnostic if applied
        if top_n is not None:
            output["diagnostic_counts"]["top_n_cutoff"] = top_n
            excluded_by_topn = sum(1 for r in ranked if "NOT_IN_TOP_N" in r.get("position_flags", []))
            output["diagnostic_counts"]["excluded_by_top_n"] = excluded_by_topn

    # Convert any remaining Decimal objects to strings for JSON serialization
    for rec in ranked:
        # Remove internal _position_weight_raw field (not needed in output)
        if "_position_weight_raw" in rec:
            del rec["_position_weight_raw"]
    
    return output


def validate_defensive_integration(output: Dict) -> None:
    """
    Validate defensive overlay integration.
    Prints validation results to console.
    """
    ranked = output.get("ranked_securities", [])

    print("\n" + "="*60)
    print("DEFENSIVE OVERLAY VALIDATION")
    print("="*60)

    # 1. Check weights sum (only if position sizing is enabled)
    def_config = output.get("defensive_overlay_config", {})
    position_sizing = def_config.get("position_sizing_enabled", False)

    if position_sizing:
        total_weight = sum(Decimal(r.get("position_weight", "0")) for r in ranked)
        expected = Decimal("1.0000")
        tolerance = Decimal("0.0001")
        if abs(total_weight - expected) >= tolerance:
            print(f"[WARN] Weights sum: {total_weight} (expected {expected}, diff: {abs(total_weight - expected)})")
        else:
            print(f"[OK] Weights sum: {total_weight} (target: {expected})")
    else:
        print("[OK] Position sizing disabled (weights not computed)")

    # 2. Check excluded have zero weight
    excluded_with_weight = [
        r["ticker"] for r in ranked
        if not r.get("rankable", True) and Decimal(r.get("position_weight", "0")) != 0
    ]
    if excluded_with_weight:
        print(f"[WARN] {len(excluded_with_weight)} excluded securities have non-zero weight: {excluded_with_weight}")
    else:
        print("[OK] All excluded securities have zero weight")

    # 3. Check defensive feature coverage (from diagnostics)
    diag = output.get("diagnostic_counts", {})
    coverage = diag.get("defensive_features_coverage", {})

    if coverage:
        total = coverage.get("total_securities", 0)
        with_features = coverage.get("with_defensive_features", 0)
        with_corr = coverage.get("with_correlation", 0)
        with_vol = coverage.get("with_volatility", 0)
        pct = coverage.get("coverage_pct", 0)

        if with_features == 0:
            print(f"[WARN] Defensive features coverage: 0/{total} securities (no defensive data loaded)")
        elif with_features < total:
            print(f"[INFO] Defensive features coverage: {with_features}/{total} ({pct}%)")
            print(f"       Correlation data: {with_corr}, Volatility data: {with_vol}")
        else:
            print(f"[OK] Defensive features coverage: {with_features}/{total} ({pct}%)")
            print(f"     Correlation data: {with_corr}, Volatility data: {with_vol}")

    # 4. Check defensive multiplier status (score adjustment)
    has_multiplier_field = any(r.get("defensive_multiplier") for r in ranked)

    if has_multiplier_field:
        # Multiplier was applied - count meaningful adjustments
        with_def_notes = sum(1 for r in ranked if r.get("defensive_notes"))
        non_neutral = sum(
            1 for r in ranked
            if r.get("defensive_multiplier") and Decimal(r["defensive_multiplier"]) != Decimal("1.00")
        )
        elite = sum(1 for r in ranked if "def_mult_elite_1.40" in (r.get("defensive_notes") or []))
        good_div = sum(1 for r in ranked if "def_mult_good_diversifier_1.10" in (r.get("defensive_notes") or []))
        high_corr = sum(1 for r in ranked if "def_mult_high_corr_0.95" in (r.get("defensive_notes") or []))
        corr_missing = sum(1 for r in ranked if "def_corr_missing" in (r.get("defensive_notes") or []))
        corr_placeholder = sum(1 for r in ranked if "def_corr_placeholder_0.50" in (r.get("defensive_notes") or []))

        print(f"[OK] Defensive multiplier: ENABLED")
        print(f"[OK] {with_def_notes}/{len(ranked)} securities have defensive notes")
        print(f"[OK] {non_neutral}/{len(ranked)} securities have non-neutral multiplier")
        if elite > 0 or good_div > 0 or high_corr > 0:
            print(f"     Breakdown: {elite} elite (1.40x), {good_div} good diversifier (1.10x), {high_corr} high-corr penalty (0.95x)")
        if corr_missing > 0 or corr_placeholder > 0:
            print(f"     Data gaps: {corr_missing} missing corr, {corr_placeholder} placeholder corr")
    else:
        # Multiplier was not applied
        print(f"[INFO] Defensive multiplier: DISABLED (apply_defensive_multiplier=False)")
        print(f"[OK] Position sizing applied without score adjustment")
    
    # 5. Weight distribution
    nonzero_weights = [r for r in ranked if Decimal(r.get("position_weight", "0")) > 0]
    if nonzero_weights:
        weights = [Decimal(r["position_weight"]) for r in nonzero_weights]
        print("\nPosition sizing:")
        print(f"  - {len(nonzero_weights)} positions")
        print(f"  - Max weight: {max(weights):.4f} ({max(weights)*100:.2f}%)")
        print(f"  - Min weight: {min(weights):.4f} ({min(weights)*100:.2f}%)")
        print(f"  - Avg weight: {sum(weights)/len(weights):.4f}")
        print(f"  - Range: {max(weights)/min(weights):.1f}:1")

        # Calculate and show dynamic floor used
        n = len(nonzero_weights)
        floor = calculate_dynamic_floor(n)
        print(f"  - Dynamic floor: {floor:.4f} ({floor*100:.2f}%) for {n} securities")
    
    # 6. Top 10 holdings
    print("\nTop 10 holdings:")
    print(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Def Notes'}")
    print("-" * 60)
    for r in ranked[:10]:
        notes_str = ", ".join(r.get("defensive_notes", [])) if r.get("defensive_notes") else "-"
        print(f"{r['composite_rank']:<6}{r['ticker']:<8}{r['composite_score']:<10}{r.get('position_weight', '0.0000'):<10}{notes_str}")
    
    print("="*60)


# =============================================================================
# OUTPUT SCHEMA EXTENSION
# =============================================================================

OUTPUT_SCHEMA_VERSION = "2.1.0"  # Guaranteed columns: composite_score, z_score, expected_excess_return, volatility, drawdown, cluster_id

# Required output columns (always present, null if unavailable)
REQUIRED_OUTPUT_COLUMNS = ["composite_score", "z_score", "expected_excess_return", "volatility", "drawdown", "cluster_id"]


def attach_output_schema_columns(output: Dict) -> Dict[str, int]:
    """
    Attach standardized output schema columns to each record.

    GUARANTEED columns (always present, null if unavailable):
    - composite_score: from existing composite_score field
    - z_score: alias for score_z (standardized score)
    - expected_excess_return: alias for expected_excess_return_annual
    - volatility: from defensive_features.vol_60d
    - drawdown: from defensive_features.drawdown_current
    - cluster_id: from clustering (null if clustering disabled)

    Does NOT recompute values - uses canonical values from Module 5.
    Adds def_note diagnostic tags when values are unavailable.

    Returns:
        field_coverage dict with counts for each column
    """
    ranked = output.get("ranked_securities", [])
    coverage = {col: 0 for col in REQUIRED_OUTPUT_COLUMNS}
    coverage["module_scores"] = 0

    for rec in ranked:
        notes = rec.setdefault("defensive_notes", [])

        # composite_score: guaranteed from Module 5
        if rec.get("composite_score") is not None:
            coverage["composite_score"] += 1
        else:
            rec["composite_score"] = None

        # z_score: alias for score_z
        sz = rec.get("score_z")
        if sz is not None:
            rec["z_score"] = sz
            coverage["z_score"] += 1
        else:
            rec["z_score"] = None
            if "def_missing_z_score" not in notes:
                notes.append("def_missing_z_score")

        # expected_excess_return: alias for expected_excess_return_annual
        er_annual = rec.get("expected_excess_return_annual")
        if er_annual is not None:
            rec["expected_excess_return"] = er_annual
            coverage["expected_excess_return"] += 1
        else:
            rec["expected_excess_return"] = None
            if "def_missing_expected_return" not in notes:
                notes.append("def_missing_expected_return")

        # volatility: from defensive_features.vol_60d
        def_feats = rec.get("defensive_features") or {}
        vol = def_feats.get("vol_60d")
        if vol is not None:
            rec["volatility"] = vol
            coverage["volatility"] += 1
        else:
            rec["volatility"] = None
            if "def_missing_volatility" not in notes:
                notes.append("def_missing_volatility")

        # drawdown: canonical alias chain
        dd = def_feats.get("drawdown_current") or def_feats.get("drawdown_60d") or def_feats.get("drawdown")
        if dd is not None:
            rec["drawdown"] = dd
            coverage["drawdown"] += 1
        else:
            rec["drawdown"] = None
            if "def_missing_drawdown" not in notes:
                notes.append("def_missing_drawdown")

        # cluster_id: null if clustering disabled
        if rec.get("cluster_id") is not None:
            coverage["cluster_id"] += 1
        else:
            rec["cluster_id"] = None  # Explicitly null if clustering disabled

        # module_scores: from component_scores if present (scalars only)
        comp_scores = rec.get("component_scores")
        if comp_scores and isinstance(comp_scores, dict):
            lean_scores = {k: v for k, v in comp_scores.items()
                          if v is None or isinstance(v, (str, int, float, Decimal))}
            if lean_scores:
                rec["module_scores"] = lean_scores
                coverage["module_scores"] += 1

    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    return coverage


# =============================================================================
# CACHE MERGE HELPERS
# =============================================================================

def load_defensive_cache(cache_path: str) -> Dict[str, Dict[str, str]]:
    """Load defensive features from cache file. Returns ticker -> features dict."""
    import json
    from pathlib import Path
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    with open(path) as f:
        cache_data = json.load(f)
    data = cache_data.get("data", cache_data)  # Handle wrapped or raw format
    return data.get("features_by_ticker", {})


def merge_cache_into_scores(
    scores_by_ticker: Dict[str, Dict],
    cache_features: Dict[str, Dict[str, str]],
    overwrite: bool = False,
) -> int:
    """Merge cached features into scores_by_ticker. Returns count of merged fields."""
    merged = 0
    for ticker, features in cache_features.items():
        if ticker not in scores_by_ticker:
            continue
        ticker_data = scores_by_ticker[ticker]
        if "defensive_features" not in ticker_data:
            ticker_data["defensive_features"] = {}
        existing = ticker_data["defensive_features"]
        for key, value in features.items():
            if overwrite or not existing.get(key):
                existing[key] = value
                merged += 1
    return merged


__all__ = [
    # Configuration
    "DefensiveConfig",
    "DEFAULT_DEFENSIVE_CONFIG",
    "AGGRESSIVE_DEFENSIVE_CONFIG",
    # Core functions
    "defensive_multiplier",
    "defensive_multiplier_legacy",
    "sanitize_corr",
    "raw_inv_vol_weight",
    "calculate_dynamic_floor",
    "apply_caps_and_renormalize",
    # Integration
    "enrich_with_defensive_overlays",
    "validate_defensive_integration",
    # Output schema extension
    "attach_output_schema_columns",
    "OUTPUT_SCHEMA_VERSION",
    "REQUIRED_OUTPUT_COLUMNS",
    # Cache helpers
    "load_defensive_cache",
    "merge_cache_into_scores",
    # Utilities
    "_safe_decimal",
    "_is_valid_value",
    "_derive_defensive_bucket",
]


if __name__ == "__main__":
    print("Testing defensive_overlay_adapter v2.0...")
    print()

    # Test multi-factor defensive multiplier
    print("=== Multi-Factor Multiplier Test ===")
    test_cases = [
        # (description, features)
        ("Elite diversifier", {"corr_xbi_120d": "0.25", "vol_60d": "0.35", "ret_21d": "0.15", "rsi_14d": "45"}),
        ("Good diversifier", {"corr_xbi_120d": "0.38", "vol_60d": "0.48", "ret_21d": "0.05", "rsi_14d": "55"}),
        ("High correlation penalty", {"corr_xbi_120d": "0.85", "vol_60d": "0.60", "ret_21d": "0.02", "rsi_14d": "50"}),
        ("Momentum penalty", {"corr_xbi_120d": "0.50", "vol_60d": "0.50", "ret_21d": "-0.25", "rsi_14d": "30"}),
        ("RSI overbought", {"corr_xbi_120d": "0.50", "vol_60d": "0.50", "ret_21d": "0.05", "rsi_14d": "75"}),
        ("Deep drawdown", {"corr_xbi_120d": "0.50", "vol_60d": "0.50", "drawdown_current": "-0.45", "rsi_14d": "25"}),
        ("Stacking factors", {"corr_xbi_120d": "0.25", "vol_60d": "0.35", "ret_21d": "0.15", "rsi_14d": "28"}),
    ]

    for desc, features in test_cases:
        mult, notes = defensive_multiplier(features)
        print(f"\n{desc}:")
        print(f"  Multiplier: {mult}")
        print(f"  Notes: {notes}")

    # Test dynamic floor calculation
    print("\n\n=== Dynamic Floor Test ===")
    for n in [20, 44, 50, 80, 100, 150, 200, 300]:
        floor = calculate_dynamic_floor(n)
        avg = Decimal("1.00") / Decimal(str(n))
        ratio = floor / avg
        print(f"  {n:3} securities: floor={floor:.4f} ({floor*100:.2f}%), avg={avg:.4f}, floor/avg={ratio:.2f}x")

    # Show config
    print("\n\n=== Default Config ===")
    cfg = DEFAULT_DEFENSIVE_CONFIG
    print(f"  Enabled features: momentum={cfg.enable_momentum}, rsi={cfg.enable_rsi}, drawdown={cfg.enable_drawdown_penalty}, vol_ratio={cfg.enable_vol_ratio}")
    print(f"  Elite thresholds: corr<{cfg.corr_elite_threshold}, vol<{cfg.vol_elite_threshold} → {cfg.mult_elite}x")
    print(f"  Momentum: >{cfg.momentum_bonus_threshold} → {cfg.mult_momentum_bonus}x, <{cfg.momentum_penalty_threshold} → {cfg.mult_momentum_penalty}x")

    print("\n[OK] Test complete!")
