#!/usr/bin/env python3
"""
run_forward_validation.py — Daily DEM Top-30 EW forward validation capture.

Reads today's frozen snapshot, records the Top-30 equal-weight basket,
runs data-quality checks, seeds adversarial controls, and appends an
immutable row to artifacts/forward_validation/captures.jsonl.
Generates a truth card at artifacts/forward_validation/{date}/TRUTH_CARD.md.

Usage:
    python3 tools/run_forward_validation.py --as-of-date 2026-06-27
    python3 tools/run_forward_validation.py --as-of-date 2026-06-27 --register-candidate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_ROOT = REPO_ROOT / "data" / "snapshots"
# Long-format (date, ticker, close — split-adjusted). Includes XBI.
PRICE_HISTORY = REPO_ROOT / "production_data" / "price_history.csv"
ARTIFACTS = REPO_ROOT / "artifacts" / "forward_validation"
CAPTURES_LEDGER = ARTIFACTS / "captures.jsonl"
CANDIDATE_FILE = ARTIFACTS / "CANDIDATE.json"

TOP_N = 30
SCHEMA_VERSION = "fv_capture.v1"

# Core model files that define candidate identity
MODEL_FILES = [
    REPO_ROOT / "ranker_engine.py",
    REPO_ROOT / "selector_engine.py",
    REPO_ROOT / "decision_engine.py",
]


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def compute_model_hash() -> str:
    h = hashlib.sha256()
    for p in MODEL_FILES:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Candidate registration
# ---------------------------------------------------------------------------


def load_candidate() -> dict | None:
    if not CANDIDATE_FILE.exists():
        return None
    return json.loads(CANDIDATE_FILE.read_text())


def register_candidate(model_hash: str, ruleset_hash: str, as_of_date: str) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    candidate = {
        "registered": as_of_date,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_hash": model_hash,
        "ruleset_hash": ruleset_hash,
        "top_n": TOP_N,
        "weighting": "equal-weight",
        "benchmark": "XBI",
        "price_source": "production_data/price_history.csv (close, split-adjusted)",
        "xbi_source": "production_data/price_history.csv (close, split-adjusted, ticker=XBI)",
        "protocol": "docs/FORWARD_VALIDATION_PROTOCOL.md",
        "status": "active",
    }
    CANDIDATE_FILE.write_text(json.dumps(candidate, indent=2) + "\n")
    return candidate


# ---------------------------------------------------------------------------
# Snapshot readers
# ---------------------------------------------------------------------------


def load_snapshot_manifest(snap_dir: Path) -> dict:
    # run_manifest.json has ruleset.ruleset_hash
    path = snap_dir / "run_manifest.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d.get("ruleset") or {}
    return {}


def load_rankings(snap_dir: Path) -> list[dict]:
    path = snap_dir / "rankings.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_top30(rows: list[dict]) -> list[dict]:
    eligible = []
    for r in rows:
        try:
            rank = int(float(r.get("actionable_rank", 9999) or 9999))
            eligible.append({"ticker": r["ticker"], "rank": rank, "target_weight_pct": r.get("target_weight_pct", "")})
        except (ValueError, TypeError):
            continue
    eligible.sort(key=lambda x: x["rank"])
    return eligible[:TOP_N]


def get_bottom30(rows: list[dict]) -> list[dict]:
    eligible = []
    for r in rows:
        try:
            rank = int(float(r.get("actionable_rank", 9999) or 9999))
            eligible.append({"ticker": r["ticker"], "rank": rank})
        except (ValueError, TypeError):
            continue
    eligible.sort(key=lambda x: x["rank"])
    return eligible[-TOP_N:] if len(eligible) >= TOP_N else eligible


# ---------------------------------------------------------------------------
# Rank-depth shadow cohorts
# ---------------------------------------------------------------------------
# VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.
# Top-30 = primary basket; ranks 31-60 = shadow reserve bench; Top-60 = depth
# cohort. Annotation only — these baskets are never tradable by default.

RANK_BAND_LO = 31
RANK_BAND_HI = 60
TOP60_N = 60


def _ranked_eligible(rows: list[dict]) -> list[dict]:
    """All rows with a numeric actionable_rank, sorted ascending by rank."""
    eligible = []
    for r in rows:
        try:
            rank = int(float(r.get("actionable_rank", 9999) or 9999))
        except (ValueError, TypeError):
            continue
        eligible.append({"ticker": r["ticker"], "rank": rank})
    eligible.sort(key=lambda x: x["rank"])
    return eligible


def get_rank_band(rows: list[dict], lo: int, hi: int) -> list[dict]:
    """Names whose actionable_rank falls in [lo, hi], sorted by rank."""
    return [e for e in _ranked_eligible(rows) if lo <= e["rank"] <= hi]


def build_cohort_baskets(rows: list[dict]) -> dict[str, list[str]]:
    """Return {cohort: [tickers]} for top30, rank31_60, top60 (rank-ordered)."""
    ranked = _ranked_eligible(rows)
    return {
        "top30": [e["ticker"] for e in ranked if 1 <= e["rank"] <= TOP_N],
        "rank31_60": [e["ticker"] for e in ranked if RANK_BAND_LO <= e["rank"] <= RANK_BAND_HI],
        "top60": [e["ticker"] for e in ranked if 1 <= e["rank"] <= TOP60_N],
    }


# ---------------------------------------------------------------------------
# Price readers — production_data/price_history.csv (long: date, ticker, close)
# ---------------------------------------------------------------------------

_PH_BY_DATE: dict[str, dict[str, float]] = {}
_PH_DATES_SORTED: list[str] = []


def _load_price_history() -> None:
    """Load price_history.csv once into memory, keyed by date→ticker→close."""
    global _PH_DATES_SORTED
    if _PH_BY_DATE:
        return
    if not PRICE_HISTORY.exists():
        return
    with open(PRICE_HISTORY, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("date", "")
            t = row.get("ticker", "")
            try:
                c = float(row["close"])
            except (KeyError, ValueError, TypeError):
                continue
            if d not in _PH_BY_DATE:
                _PH_BY_DATE[d] = {}
            _PH_BY_DATE[d][t] = c
    _PH_DATES_SORTED = sorted(_PH_BY_DATE)


def load_price_row(date: str) -> dict[str, float]:
    """Return ticker→close for the given date."""
    _load_price_history()
    return _PH_BY_DATE.get(date, {})


def load_xbi_price(date: str) -> float | None:
    """Return XBI close for the given date."""
    _load_price_history()
    return _PH_BY_DATE.get(date, {}).get("XBI")


def nearest_price_date_before(date: str) -> str | None:
    """Most recent date <= date in price_history.csv."""
    _load_price_history()
    best = None
    for d in _PH_DATES_SORTED:
        if d <= date:
            best = d
        else:
            break
    return best


def nearest_xbi_date_before(date: str) -> str | None:
    """Same source — XBI is a ticker in price_history.csv."""
    return nearest_price_date_before(date)


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------


def run_dq_checks(
    date: str,
    top30: list[dict],
    snap_dir: Path,
    model_hash: str,
    candidate: dict | None,
) -> tuple[str, list[dict]]:
    checks = []

    def chk(name: str, passed: bool, detail: str, hard: bool = False) -> dict:
        return {"check": name, "pass": passed, "hard": hard, "detail": detail}

    # 1. Snapshot completeness
    rankings_ok = (snap_dir / "rankings.csv").exists()
    checks.append(
        chk("snapshot_completeness", rankings_ok, f"rankings.csv {'found' if rankings_ok else 'MISSING'}", hard=True)
    )

    # 2. Top-30 populated
    checks.append(chk("top30_populated", len(top30) == TOP_N, f"top30 has {len(top30)}/{TOP_N} names", hard=True))

    # 3. Price coverage
    effective_date = nearest_price_date_before(date)
    prices_ok = effective_date is not None
    price_lag = 0
    if effective_date and effective_date < date:
        from datetime import date as ddate

        d0 = ddate.fromisoformat(effective_date)
        d1 = ddate.fromisoformat(date)
        price_lag = (d1 - d0).days
    checks.append(
        chk(
            "price_coverage",
            prices_ok and price_lag <= 3,
            f"universe_prices effective_date={effective_date} lag={price_lag}d",
            hard=True,
        )
    )

    # 4. XBI price available (same file as basket prices — endpoint parity guaranteed)
    xbi_price_check = effective_date is not None and load_xbi_price(effective_date) is not None
    checks.append(
        chk(
            "xbi_price_available",
            xbi_price_check,
            f"XBI in price_history.csv at {effective_date}: {xbi_price_check}",
            hard=True,
        )
    )

    # 5. Endpoint parity — guaranteed (basket + XBI both from price_history.csv same date)
    checks.append(
        chk("endpoint_parity", effective_date is not None, f"price_history.csv unified source, date={effective_date}")
    )

    # 6. All top-30 tickers have prices
    prices = load_price_row(effective_date) if effective_date else {}
    missing = [t["ticker"] for t in top30 if t["ticker"] not in prices]
    checks.append(
        chk("basket_price_coverage", len(missing) == 0, f"{len(missing)} missing: {missing[:5] if missing else 'none'}")
    )

    # 7. Model hash matches candidate
    if candidate is not None:
        match = candidate.get("model_hash") == model_hash
        checks.append(
            chk("model_hash_match", match, f"current={model_hash} candidate={candidate.get('model_hash')}", hard=True)
        )
    else:
        checks.append(
            chk("model_hash_match", False, "CANDIDATE.json not found — run with --register-candidate first", hard=True)
        )

    # 8. Snapshot freshness
    from datetime import date as ddate

    snap_date = ddate.fromisoformat(date)
    today = ddate.today()
    staleness = (today - snap_date).days
    checks.append(chk("snapshot_freshness", staleness <= 2, f"snapshot is {staleness}d old"))

    hard_fails = [c for c in checks if c["hard"] and not c["pass"]]
    soft_warns = [c for c in checks if not c["hard"] and not c["pass"]]

    if hard_fails:
        quality = "FAIL"
    elif soft_warns:
        quality = "DEGRADED"
    else:
        quality = "PASS"

    return quality, checks


# ---------------------------------------------------------------------------
# Adversarial control setup
# ---------------------------------------------------------------------------


def build_adversarial_seeds(
    rows: list[dict],
    top30: list[dict],
    bottom30: list[dict],
    seed: int,
) -> dict:
    top30_tickers = {t["ticker"] for t in top30}
    universe = [r["ticker"] for r in rows if r.get("ticker") not in top30_tickers]

    rng = random.Random(seed)
    n_bootstraps = 1000
    bootstraps = []
    for _ in range(n_bootstraps):
        sample = rng.sample(universe, min(TOP_N, len(universe)))
        bootstraps.append(sorted(sample))

    # Size-matched control: placeholder (size data not in this script)
    size_matched_seed = rng.randint(0, 2**31)

    return {
        "seed": seed,
        "n_bootstraps": n_bootstraps,
        "random_universe_size": len(universe),
        "bootstrap_samples": bootstraps,
        "bottom30_tickers": [t["ticker"] for t in bottom30],
        "rank_shuffled_tickers": top30_tickers and sorted(top30_tickers),
        "size_matched_seed": size_matched_seed,
    }


# ---------------------------------------------------------------------------
# Capture record
# ---------------------------------------------------------------------------


def load_existing_captures() -> set[str]:
    existing = set()
    if not CAPTURES_LEDGER.exists():
        return existing
    with open(CAPTURES_LEDGER) as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    existing.add(rec["date"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return existing


def append_capture(record: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(CAPTURES_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Truth card
# ---------------------------------------------------------------------------


def generate_truth_card(capture: dict, fills: list[dict] | None = None) -> str:
    date = capture["date"]
    top30 = capture["top30"]
    quality = capture["data_quality"]
    checks = capture["dq_checks"]

    lines = [
        "# DEM Top-30 EW — Daily Truth Card",
        "",
        f"**Date:** {date}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}",
        f"**Model hash:** `{capture['model_hash']}`",
        f"**Ruleset hash:** `{capture['ruleset_hash']}`",
        f"**Data quality:** `{quality}`",
        "",
        "## Top-30 Selection (Equal-Weight)",
        "",
        "| Rank | Ticker |",
        "|------|--------|",
    ]
    for t in top30:
        lines.append(f"| {t['rank']} | {t['ticker']} |")

    lines += [
        "",
        "## Data-Quality Checks",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for c in checks:
        status = "PASS" if c["pass"] else ("FAIL" if c["hard"] else "WARN")
        lines.append(f"| {c['check']} | {status} | {c['detail']} |")

    lines += ["", "## Forward Returns", ""]

    if fills:
        fill = fills[-1]
        lines += [
            "| Horizon | Basket | XBI | Excess |",
            "|---------|--------|-----|--------|",
        ]
        for hz in ["1d", "5d", "20d"]:
            b = fill.get(f"basket_{hz}")
            x = fill.get(f"xbi_{hz}")
            xs = fill.get(f"xs_{hz}")
            b_s = f"{b:+.2%}" if b is not None else "PENDING"
            x_s = f"{x:+.2%}" if x is not None else "PENDING"
            xs_s = f"{xs:+.2%}" if xs is not None else "PENDING"
            lines.append(f"| {hz} | {b_s} | {x_s} | {xs_s} |")
    else:
        lines += [
            "| Horizon | Basket | XBI | Excess |",
            "|---------|--------|-----|--------|",
            "| 1d | PENDING | PENDING | PENDING |",
            "| 5d | PENDING | PENDING | PENDING |",
            "| 20d | PENDING | PENDING | PENDING |",
        ]

    lines += [
        "",
        "## XBI Baseline",
        "",
        f"- Capture date: {date}",
        f"- Effective price date: {capture['effective_price_date']}",
        f"- XBI price at capture: {capture['xbi_price_at_capture']}",
        "- Price source: `universe_prices.csv` (adj_close)",
        "- XBI source: `indices_prices.csv` (adj_close)",
        "",
        "## Adversarial Controls",
        "",
        f"- Random bootstrap: {capture['adversarial']['n_bootstraps']} samples "
        f"from {capture['adversarial']['random_universe_size']}-name universe",
        f"- Bottom-30: {len(capture['adversarial']['bottom30_tickers'])} names recorded",
        "- Evaluation: once ≥20 non-overlapping 5d windows complete",
        f"- Seed: `{capture['adversarial']['seed']}`",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="DEM Top-30 EW daily forward validation capture")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD snapshot date")
    parser.add_argument(
        "--register-candidate", action="store_true", help="Register current model as the active candidate"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing capture for this date")
    args = parser.parse_args()

    date = args.as_of_date
    snap_dir = SNAPSHOTS_ROOT / date

    if not snap_dir.exists():
        print(f"ERROR: snapshot not found at {snap_dir}", file=sys.stderr)
        return 1

    # --- Already captured? ---
    existing = load_existing_captures()
    if date in existing and not args.force:
        print(f"Capture for {date} already exists. Use --force to overwrite.")
        return 0

    # --- Load snapshot ---
    rows = load_rankings(snap_dir)
    if not rows:
        print(f"ERROR: could not load rankings from {snap_dir}", file=sys.stderr)
        return 1

    manifest = load_snapshot_manifest(snap_dir)
    ruleset_hash = manifest.get("ruleset_hash", "unknown")

    top30 = get_top30(rows)
    bottom30 = get_bottom30(rows)
    # Rank-depth shadow cohorts (annotation only — never tradable by default)
    cohort_baskets = build_cohort_baskets(rows)
    rank31_60 = get_rank_band(rows, RANK_BAND_LO, RANK_BAND_HI)

    # --- Model hash ---
    model_hash = compute_model_hash()

    # --- Candidate registration ---
    candidate = load_candidate()
    if args.register_candidate:
        if candidate is not None:
            print(f"WARNING: Candidate already registered ({candidate['registered']}). " "Overwriting.")
        candidate = register_candidate(model_hash, ruleset_hash, date)
        print(f"Candidate registered: model_hash={model_hash} ruleset={ruleset_hash}")

    # --- Data quality ---
    quality, dq_checks = run_dq_checks(date, top30, snap_dir, model_hash, candidate)

    # --- Prices ---
    effective_price_date = nearest_price_date_before(date)
    xbi_price = load_xbi_price(effective_price_date) if effective_price_date else None

    # --- Adversarial seeds ---
    seed = int(hashlib.sha256(date.encode()).hexdigest(), 16) % (2**31)
    adversarial = build_adversarial_seeds(rows, top30, bottom30, seed)

    # --- Build capture record ---
    capture = {
        "schema": SCHEMA_VERSION,
        "date": date,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model_hash": model_hash,
        "ruleset_hash": ruleset_hash,
        "data_quality": quality,
        "dq_checks": dq_checks,
        "top30": top30,
        # Rank-depth shadow cohorts: ticker lists for forward-return measurement.
        # top30 remains the primary basket; rank31_60 + top60 are shadow only.
        "cohorts": cohort_baskets,
        "rank31_60": [{"ticker": e["ticker"], "rank": e["rank"]} for e in rank31_60],
        "effective_price_date": effective_price_date,
        "xbi_price_at_capture": xbi_price,
        "adversarial": adversarial,
        "n_universe": len(rows),
    }

    # --- Append to ledger ---
    if date in existing and args.force:
        # Re-write ledger without the old entry then re-append
        lines = []
        with open(CAPTURES_LEDGER) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec["date"] != date:
                            lines.append(line)
                    except (json.JSONDecodeError, KeyError):
                        lines.append(line)
        with open(CAPTURES_LEDGER, "w", encoding="utf-8") as f:
            f.writelines(lines)

    append_capture(capture)

    # --- Truth card ---
    card_dir = ARTIFACTS / date
    card_dir.mkdir(parents=True, exist_ok=True)
    card = generate_truth_card(capture)
    (card_dir / "TRUTH_CARD.md").write_text(card, encoding="utf-8")

    # --- Report ---
    print(f"Captured {date}: top30={len(top30)} quality={quality} model={model_hash} ruleset={ruleset_hash}")
    for c in dq_checks:
        if not c["pass"]:
            label = "FAIL" if c["hard"] else "WARN"
            print(f"  [{label}] {c['check']}: {c['detail']}")
    print(f"  Truth card: {card_dir / 'TRUTH_CARD.md'}")
    print(f"  Ledger: {CAPTURES_LEDGER}")

    return 0 if quality != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
