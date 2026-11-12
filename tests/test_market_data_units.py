"""market_data.json market_cap / enterprise_value must be raw USD.

2026-11-04: scripts/refresh_from_sharadar.py copied Sharadar DAILY marketcap/ev
(reported in $ MILLIONS) straight into market_cap/enterprise_value. Every type
check passed, but run_screen computes market_cap_mm = market_cap / 1e6, so each
company shrank ~1e6x. Size bands, liquidity and cost estimates collapsed and
the 2026-11-11 run churned 11/30 of the actionable top-30 (36.7% turnover,
Phase-2 health FAIL) until the file was restored.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from run_daily_production import MARKET_DATA_MIN_MEDIAN_CAP_USD, check_market_data_schema  # noqa: E402


def _load_sharadar():
    spec = importlib.util.spec_from_file_location(
        "refresh_from_sharadar", REPO / "scripts" / "refresh_from_sharadar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(ticker: str, cap: float, ev: float = 0.0) -> dict:
    return {
        "ticker": ticker,
        "price": 10.0,
        "market_cap": cap,
        "enterprise_value": ev,
        "collected_at": "2026-11-11",
    }


def _write(tmp_path: Path, records: list) -> Path:
    (tmp_path / "market_data.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


# --- producer ------------------------------------------------------------------
def test_sharadar_daily_millions_are_scaled_to_usd():
    mod = _load_sharadar()
    rec = {"ticker": "ABVX"}
    mod._apply_daily_fields(rec, {"marketcap": 7065.6, "ev": 6622.0})
    assert rec["market_cap"] == 7_065_600_000.0
    assert rec["enterprise_value"] == 6_622_000_000.0


def test_sharadar_missing_fields_leave_record_untouched():
    mod = _load_sharadar()
    rec = {"ticker": "X", "market_cap": 5e8}
    mod._apply_daily_fields(rec, {"marketcap": None})
    assert rec == {"ticker": "X", "market_cap": 5e8}


# --- gate ----------------------------------------------------------------------
def test_gate_fails_on_millions_scale_market_cap(tmp_path):
    recs = [_record(f"T{i}", 500.0 + i) for i in range(50)]  # $M-scale
    gate = check_market_data_schema(_write(tmp_path, recs))
    assert gate.status == "FAIL"
    assert "millions" in gate.detail


def test_gate_passes_on_usd_market_cap(tmp_path):
    recs = [_record(f"T{i}", 5e8 + i * 1e7) for i in range(50)]
    gate = check_market_data_schema(_write(tmp_path, recs))
    assert gate.status == "PASS"


def test_gate_tolerates_a_few_genuinely_tiny_caps(tmp_path):
    recs = [_record(f"T{i}", 5e8) for i in range(48)] + [_record("NANO1", 3e6), _record("NANO2", 4e6)]
    assert check_market_data_schema(_write(tmp_path, recs)).status == "PASS"


# --- committed data --------------------------------------------------------------
def test_committed_market_data_is_usd():
    """Regression guard on the repo's own production_data/market_data.json."""
    records = json.loads((REPO / "production_data" / "market_data.json").read_text(encoding="utf-8"))
    caps = sorted(
        r["market_cap"] for r in records if isinstance(r.get("market_cap"), (int, float)) and r["market_cap"] > 0
    )
    assert caps, "no market caps"
    assert caps[len(caps) // 2] >= MARKET_DATA_MIN_MEDIAN_CAP_USD
    assert sum(1 for c in caps if c < 1e6) == 0
