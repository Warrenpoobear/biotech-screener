#!/usr/bin/env python3
"""PnL attribution sidecar for daily snapshot monitoring.

Computes position-level PnL contribution between D0 (prior) and D1 (current)
snapshots, with attribution by action type, tier, catalyst bucket, coinvest
tag, and institutional delta bucket.

Outputs: pnl_attribution.json + pnl_attribution.md (schema pnl_attribution.v1)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "pnl_attribution.v1"
DEFAULT_COST_BPS = 30


# ---------------------------------------------------------------------------
# Price loader (lightweight)
# ---------------------------------------------------------------------------

def load_price_map(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv → {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            close_str = (row.get("close") or "").strip()
            date_str = (row.get("date") or "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            prices.setdefault(ticker, {})[date_str] = close
    return prices


def get_price(prices: Dict[str, Dict[str, float]], ticker: str, dt: str) -> Optional[float]:
    """Get close price for a ticker on a given date."""
    return prices.get(ticker, {}).get(dt)


# ---------------------------------------------------------------------------
# Snapshot loaders
# ---------------------------------------------------------------------------

def load_rankings_map(snapshot_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load rankings.csv → {ticker: row_dict}."""
    csv_path = snapshot_dir / "rankings.csv"
    result: Dict[str, Dict[str, str]] = {}
    if not csv_path.exists():
        return result
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "").strip()
            if ticker:
                result[ticker] = row
    return result


def load_positions_list(snapshot_dir: Path) -> List[Dict[str, str]]:
    """Load portfolio_positions.csv."""
    csv_path = snapshot_dir / "portfolio_positions.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metadata(snapshot_dir: Path) -> Dict[str, Any]:
    """Load metadata.json."""
    path = snapshot_dir / "metadata.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Portfolio builder
# ---------------------------------------------------------------------------

def build_portfolio(
    snapshot_dir: Path,
    rankings: Dict[str, Dict[str, str]],
    top_k: int = 20,
    use_positions: bool = False,
) -> Dict[str, float]:
    """Build {ticker: weight} from positions or top-K equal weight.

    Returns normalised weights summing to 1.0.
    """
    if use_positions:
        positions = load_positions_list(snapshot_dir)
        if positions:
            weights: Dict[str, float] = {}
            for row in positions:
                ticker = row.get("ticker", "").strip()
                wt_str = row.get("target_weight_pct", "0")
                try:
                    wt = float(wt_str) / 100.0
                except (ValueError, TypeError):
                    wt = 0.0
                if ticker and wt > 0:
                    weights[ticker] = wt
            total = sum(weights.values())
            if total > 0:
                return {t: w / total for t, w in weights.items()}

    # Fallback: top-K equal weight from rankings
    ranked = sorted(
        rankings.items(),
        key=lambda x: _safe_int(x[1].get("actionable_rank", "9999")),
    )
    eligible = [(t, r) for t, r in ranked if r.get("eligible", "1") == "1"]
    top = eligible[:top_k]
    if not top:
        return {}
    w = 1.0 / len(top)
    return {t: w for t, _ in top}


def _safe_int(v: str) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 9999


# ---------------------------------------------------------------------------
# PnL attribution core
# ---------------------------------------------------------------------------

@dataclass
class PositionPnL:
    ticker: str
    weight_d0: float
    weight_d1: float
    price_d0: Optional[float]
    price_d1: Optional[float]
    return_pct: Optional[float]
    pnl_contribution: Optional[float]
    action: str  # "hold", "entry", "exit"
    tier: str
    catalyst_bucket: str
    coinvest_tag: str
    inst_delta_bucket: str


@dataclass
class AttributionResult:
    schema: str = SCHEMA_VERSION
    as_of_date: str = ""
    prior_date: str = ""
    n_positions_d0: int = 0
    n_positions_d1: int = 0
    n_priced: int = 0
    coverage_pct: float = 0.0
    gross_return: Optional[float] = None
    turnover: float = 0.0
    cost_bps: float = DEFAULT_COST_BPS
    net_return: Optional[float] = None
    by_action: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_tier: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_catalyst: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_coinvest: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_inst_delta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    positions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bucket_inst_delta(z_str: str) -> str:
    """Bucket institutional delta z-score."""
    if not z_str or z_str.strip().lower() in ("nan", "none", ""):
        return "none"
    try:
        z = float(z_str)
    except (ValueError, TypeError):
        return "none"
    if math.isnan(z):
        return "none"
    if z > 0.5:
        return "positive"
    elif z < -0.5:
        return "negative"
    return "neutral"


def _bucket_catalyst(mode: str) -> str:
    """Simplify catalyst_mode into bucket."""
    if not mode or mode in ("missing", ""):
        return "none"
    return mode


def compute_attribution(
    d0_dir: Path,
    d1_dir: Path,
    price_csv: Path,
    cost_bps: float = DEFAULT_COST_BPS,
    use_positions: bool = False,
    top_k: int = 20,
) -> AttributionResult:
    """Compute PnL attribution between two snapshot dates."""
    d0_meta = load_metadata(d0_dir)
    d1_meta = load_metadata(d1_dir)

    d0_date = d0_meta.get("as_of_date", d0_dir.name)
    d1_date = d1_meta.get("as_of_date", d1_dir.name)

    d0_rankings = load_rankings_map(d0_dir)
    d1_rankings = load_rankings_map(d1_dir)

    # Build portfolios
    port_d0 = build_portfolio(d0_dir, d0_rankings, top_k, use_positions)
    port_d1 = build_portfolio(d1_dir, d1_rankings, top_k, use_positions)

    # Load prices
    prices = load_price_map(price_csv)

    # All tickers in either portfolio
    all_tickers = set(port_d0) | set(port_d1)

    # Position-level PnL
    position_pnls: List[PositionPnL] = []
    priced_count = 0

    for ticker in sorted(all_tickers):
        w0 = port_d0.get(ticker, 0.0)
        w1 = port_d1.get(ticker, 0.0)

        p0 = get_price(prices, ticker, d0_date)
        p1 = get_price(prices, ticker, d1_date)

        ret = None
        contrib = None
        if p0 is not None and p1 is not None and p0 > 0:
            ret = p1 / p0 - 1.0
            contrib = w0 * ret
            priced_count += 1

        # Action classification
        if w0 > 0 and w1 > 0:
            action = "hold"
        elif w0 == 0 and w1 > 0:
            action = "entry"
        else:
            action = "exit"

        # Attribution fields from D1 rankings (or D0 if exiting)
        rec = d1_rankings.get(ticker) or d0_rankings.get(ticker) or {}
        tier = rec.get("tier_any", "") or rec.get("tier_dev", "") or ""
        cat_bucket = _bucket_catalyst(rec.get("catalyst_mode", ""))
        coinvest = rec.get("coinvest_tag", "") or "none"
        inst_bucket = _bucket_inst_delta(rec.get("inst_delta_z", ""))

        position_pnls.append(PositionPnL(
            ticker=ticker, weight_d0=w0, weight_d1=w1,
            price_d0=p0, price_d1=p1,
            return_pct=_round_opt(ret, 6),
            pnl_contribution=_round_opt(contrib, 8),
            action=action, tier=tier,
            catalyst_bucket=cat_bucket, coinvest_tag=coinvest,
            inst_delta_bucket=inst_bucket,
        ))

    # Portfolio-level metrics
    coverage = priced_count / max(len(port_d0), 1)
    gross = sum(p.pnl_contribution for p in position_pnls
                if p.pnl_contribution is not None)

    # Turnover
    turnover = compute_portfolio_turnover(port_d0, port_d1)
    net = gross - turnover * cost_bps / 10_000

    # Attribution breakdowns
    result = AttributionResult(
        as_of_date=d1_date, prior_date=d0_date,
        n_positions_d0=len(port_d0), n_positions_d1=len(port_d1),
        n_priced=priced_count, coverage_pct=round(coverage, 4),
        gross_return=round(gross, 8),
        turnover=round(turnover, 4),
        cost_bps=cost_bps,
        net_return=round(net, 8),
        positions=[asdict(p) for p in position_pnls],
    )

    # Group breakdowns
    result.by_action = _group_breakdown(position_pnls, "action")
    result.by_tier = _group_breakdown(position_pnls, "tier")
    result.by_catalyst = _group_breakdown(position_pnls, "catalyst_bucket")
    result.by_coinvest = _group_breakdown(position_pnls, "coinvest_tag")
    result.by_inst_delta = _group_breakdown(position_pnls, "inst_delta_bucket")

    return result


def compute_portfolio_turnover(
    w0: Dict[str, float],
    w1: Dict[str, float],
) -> float:
    """Turnover = 0.5 * sum(|w1_i - w0_i|) across all tickers."""
    all_tickers = set(w0) | set(w1)
    if not all_tickers:
        return 0.0
    diff_sum = sum(abs(w1.get(t, 0.0) - w0.get(t, 0.0)) for t in all_tickers)
    return 0.5 * diff_sum


def _group_breakdown(
    positions: List[PositionPnL],
    attr: str,
) -> Dict[str, Dict[str, Any]]:
    """Group positions by a categorical attribute and sum contributions."""
    groups: Dict[str, List[PositionPnL]] = {}
    for p in positions:
        key = getattr(p, attr, "") or "unknown"
        groups.setdefault(key, []).append(p)

    result: Dict[str, Dict[str, Any]] = {}
    for key, pnls in sorted(groups.items()):
        contribs = [p.pnl_contribution for p in pnls if p.pnl_contribution is not None]
        result[key] = {
            "count": len(pnls),
            "total_contribution": round(sum(contribs), 8) if contribs else 0.0,
            "mean_return": round(statistics.mean(
                [p.return_pct for p in pnls if p.return_pct is not None]
            ), 6) if any(p.return_pct is not None for p in pnls) else None,
        }
    return result


def _round_opt(v: Optional[float], d: int) -> Optional[float]:
    return round(v, d) if v is not None else None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_attribution_json(result: AttributionResult, out_path: Path) -> Path:
    """Write pnl_attribution.json."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return out_path


def write_attribution_md(result: AttributionResult, out_path: Path) -> Path:
    """Write pnl_attribution.md summary."""
    lines = [
        "# PnL Attribution",
        "",
        f"- **Date**: {result.as_of_date} (vs {result.prior_date})",
        f"- **Positions**: {result.n_positions_d0} → {result.n_positions_d1}",
        f"- **Coverage**: {result.coverage_pct:.1%}",
        f"- **Gross Return**: {result.gross_return:.4%}" if result.gross_return is not None else "- **Gross Return**: —",
        f"- **Turnover**: {result.turnover:.2%}",
        f"- **Net Return**: {result.net_return:.4%}" if result.net_return is not None else "- **Net Return**: —",
        f"- **Cost**: {result.cost_bps} bps",
        "",
        "## By Action",
        "",
        "| Action | Count | Total Contribution | Mean Return |",
        "|--------|-------|--------------------|-------------|",
    ]
    for key, val in result.by_action.items():
        mr = f"{val['mean_return']:.4%}" if val.get("mean_return") is not None else "—"
        lines.append(
            f"| {key} | {val['count']} | {val['total_contribution']:.6f} | {mr} |"
        )
    lines.extend(["", "## By Tier", "",
                   "| Tier | Count | Total Contribution |",
                   "|------|-------|--------------------|"])
    for key, val in result.by_tier.items():
        lines.append(f"| {key or '(none)'} | {val['count']} | {val['total_contribution']:.6f} |")

    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Gate check function (for integration with run_daily_production.py)
# ---------------------------------------------------------------------------

def check_pnl_attribution_file(
    snapshot_dir: Path,
    min_coverage_pct: float = 80.0,
) -> Tuple[str, str, Optional[float], Optional[float]]:
    """Validate pnl_attribution.json sidecar. Returns (status, detail, value, threshold).

    WARN-only — never FAIL.
    """
    attr_path = snapshot_dir / "pnl_attribution.json"
    if not attr_path.exists():
        return "PASS", "pnl_attribution.json not present (cold-start OK)", None, None

    try:
        with open(attr_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return "WARN", f"Cannot read pnl_attribution.json: {e}", None, None

    schema = data.get("schema", "")
    if schema != SCHEMA_VERSION:
        return "WARN", f"Unexpected schema: {schema}", None, None

    coverage = data.get("coverage_pct", 0)
    if isinstance(coverage, (int, float)) and coverage * 100 < min_coverage_pct:
        return (
            "WARN",
            f"PnL coverage {coverage:.1%} below {min_coverage_pct}%",
            round(coverage * 100, 1),
            min_coverage_pct,
        )

    gross = data.get("gross_return")
    detail = (
        f"PnL OK: {data.get('prior_date','?')}→{data.get('as_of_date','?')}, "
        f"gross={_fmt_pct_safe(gross)}, turnover={data.get('turnover', '?')}"
    )
    return "PASS", detail, None, None


def _fmt_pct_safe(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4%}"
    return "—"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PnL attribution between two snapshot dates",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument("--as-of-date", required=True,
                        help="Current snapshot date (D1)")
    parser.add_argument("--prior-date", default=None,
                        help="Prior snapshot date (D0). Auto-detect if omitted.")
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
    )
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--use-positions", action="store_true", default=False)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--out-snapshot-sidecar", action="store_true", default=False,
        help="Write output into the snapshot dir instead of --out-dir",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: output/pnl_attribution/)",
    )
    args = parser.parse_args()

    d1_dir = args.snapshot_root / args.as_of_date
    if not d1_dir.exists():
        print(f"ERROR: Snapshot not found: {d1_dir}")
        sys.exit(1)

    # Auto-detect prior date
    if args.prior_date:
        prior_date = args.prior_date
    else:
        prior_date = find_prior_date(args.snapshot_root, args.as_of_date)
        if prior_date is None:
            print("ERROR: No prior snapshot found")
            sys.exit(1)

    d0_dir = args.snapshot_root / prior_date
    if not d0_dir.exists():
        print(f"ERROR: Prior snapshot not found: {d0_dir}")
        sys.exit(1)

    print(f"PnL attribution: {prior_date} → {args.as_of_date}")

    result = compute_attribution(
        d0_dir=d0_dir,
        d1_dir=d1_dir,
        price_csv=args.price_csv,
        cost_bps=args.cost_bps,
        use_positions=args.use_positions,
        top_k=args.top_k,
    )

    # Determine output location
    if args.out_snapshot_sidecar:
        out_dir = d1_dir
    elif args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = PROJECT_ROOT / "output" / "pnl_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_attribution_json(result, out_dir / "pnl_attribution.json")
    write_attribution_md(result, out_dir / "pnl_attribution.md")

    print(f"  Gross: {result.gross_return:.4%}" if result.gross_return is not None else "  Gross: —")
    print(f"  Turnover: {result.turnover:.2%}")
    print(f"  Net: {result.net_return:.4%}" if result.net_return is not None else "  Net: —")
    print(f"  Coverage: {result.coverage_pct:.1%}")
    print(f"  Output → {out_dir}")


def find_prior_date(snapshot_root: Path, current_date: str) -> Optional[str]:
    """Find the most recent snapshot date before current_date."""
    if not snapshot_root.exists():
        return None
    dates: List[str] = []
    for d in sorted(snapshot_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 10 or name[4] != "-" or name[7] != "-":
            continue
        try:
            date.fromisoformat(name)
        except ValueError:
            continue
        if name < current_date and (d / "rankings.csv").exists():
            dates.append(name)
    return dates[-1] if dates else None


if __name__ == "__main__":
    main()
