"""Tests for the PIT market snapshot backfill.

Classification: PIT_MARKET_SNAPSHOT_BACKFILL_INFRA_NO_MODEL_CHANGE

Acceptance tests covered:
  1. look-ahead: rows dated after as_of_date cannot change the snapshot
  2. every input date recorded in provenance is <= as_of_date
  3. weekend as_of_date uses the prior trading day's close
  4. known fixture -> expected field values
  5. HY OAS uses the latest valid FRED observation <= as_of_date
  6. fed_rate_change_3m is null (never carried) when IRX history is short
  7. writing through a hard link does not touch the other linked path
  8. identical inputs -> byte-identical output
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from backfill_market_snapshots import build_pit_snapshot, serialize_snapshot, write_frozen  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _weekdays(start: str, end: str) -> list[str]:
    d, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= stop:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _weekdays("2026-06-01", "2026-10-30")


def _series(base: float, step: float) -> list[list]:
    return [[d, round(base + step * i, 4)] for i, d in enumerate(DAYS)]


def _volume_series(base: float, step: float) -> list[list]:
    return [[d, round(base + step * i, 4), 1_000_000.0] for i, d in enumerate(DAYS)]


def _inputs() -> dict:
    return {
        "closes": {
            "^VIX": _series(15.0, 0.02),
            "SPY": _series(500.0, 0.5),
            "XBI": _series(100.0, 0.2),
            "HYG": _series(80.0, 0.01),
            "^TNX": _series(4.0, 0.002),
            "^IRX": _series(3.5, 0.001),
        },
        "volumes": {
            "XBI": _volume_series(100.0, 0.2),
            "IBB": _volume_series(150.0, 0.1),
        },
        "hy_oas": [[d, "3.10"] for d in DAYS],
    }


def _with_future_spike(inputs: dict, as_of: str) -> dict:
    """Corrupt every row after as_of_date with absurd values."""
    spiked = json.loads(json.dumps(inputs))
    for rows in spiked["closes"].values():
        for r in rows:
            if r[0] > as_of:
                r[1] = 9999.0
    for rows in spiked["volumes"].values():
        for r in rows:
            if r[0] > as_of:
                r[1], r[2] = 9999.0, 9e12
    for r in spiked["hy_oas"]:
        if r[0] > as_of:
            r[1] = "99.99"
    return spiked


# ---------------------------------------------------------------------------
# Look-ahead
# ---------------------------------------------------------------------------


def test_future_rows_do_not_change_snapshot():
    as_of = "2026-09-15"
    clean = {
        "closes": {k: [r for r in v if r[0] <= as_of] for k, v in _inputs()["closes"].items()},
        "volumes": {k: [r for r in v if r[0] <= as_of] for k, v in _inputs()["volumes"].items()},
        "hy_oas": [r for r in _inputs()["hy_oas"] if r[0] <= as_of],
    }
    assert serialize_snapshot(build_pit_snapshot(as_of, _with_future_spike(_inputs(), as_of))) == serialize_snapshot(
        build_pit_snapshot(as_of, clean)
    )


def test_provenance_input_dates_never_exceed_as_of():
    as_of = "2026-09-15"
    snap = build_pit_snapshot(as_of, _with_future_spike(_inputs(), as_of))
    prov = snap["provenance"]
    assert prov["as_of_date"] == as_of
    assert prov["vix_date"] <= as_of
    assert prov["data_available_timestamp"] <= as_of
    for series, last in prov["input_last_dates"].items():
        assert last is None or last <= as_of, series


def test_weekend_as_of_uses_prior_trading_day():
    snap = build_pit_snapshot("2026-09-13", _inputs())  # Sunday
    assert snap["provenance"]["vix_date"] == "2026-09-11"
    assert snap["provenance"]["as_of_date"] == "2026-09-13"


# ---------------------------------------------------------------------------
# Field values
# ---------------------------------------------------------------------------


def test_known_fixture_values():
    snap = build_pit_snapshot("2026-09-15", _inputs())
    vix_rows = [r for r in _inputs()["closes"]["^VIX"] if r[0] <= "2026-09-15"]
    assert snap["vix"] == f"{vix_rows[-1][1]:.2f}"
    assert snap["hy_credit_spread"] == "310.00"
    # Linear series rise every day, so both momentum fields must be positive
    assert float(snap["xbi_momentum_10d"]) > 0
    assert float(snap["spy_momentum_10d"]) > 0
    # HYG rising => spreads tightening => negative change proxy
    assert float(snap["credit_spread_change"]) < 0
    assert snap["feeds"]["biotech_fund_flows"] == "pit_volume_proxy"
    assert snap["feeds"]["vix"] == "pit"


def test_hy_oas_uses_latest_valid_observation_on_or_before_as_of():
    inputs = _inputs()
    inputs["hy_oas"] = [
        ["2026-09-10", "3.00"],
        ["2026-09-11", "."],
        ["2026-09-16", "4.00"],
    ]
    snap = build_pit_snapshot("2026-09-15", inputs)
    assert snap["hy_credit_spread"] == "300.00"
    assert snap["provenance"]["input_last_dates"]["hy_oas_fred"] == "2026-09-10"


def test_fed_rate_null_not_carried_when_irx_history_short():
    inputs = _inputs()
    inputs["closes"]["^IRX"] = [r for r in inputs["closes"]["^IRX"] if r[0] >= "2026-09-01"]
    snap = build_pit_snapshot("2026-09-15", inputs)
    assert snap["fed_rate_change_3m"] is None
    assert snap["feeds"]["fed_rate"] == "failed"


def test_windows_are_trading_rows_like_yfinance_period():
    # yfinance period="70d" returns the last 70 rows, not 70 calendar days
    as_of = "2026-10-30"
    snap = build_pit_snapshot(as_of, _inputs())
    irx = [r for r in _inputs()["closes"]["^IRX"] if r[0] <= as_of]
    assert snap["feeds"]["fed_rate"] == "pit"
    assert snap["fed_rate_change_3m"] == f"{irx[-1][1] - irx[-70][1]:.2f}"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_frozen_breaks_hard_link(tmp_path):
    prod = tmp_path / "production_market_snapshot.json"
    staged = tmp_path / "staged" / "market_snapshot.json"
    staged.parent.mkdir()
    prod.write_text('{"provenance": {"as_of_date": "2026-10-30"}}\n', encoding="utf-8")
    os.link(prod, staged)

    write_frozen(staged, build_pit_snapshot("2026-09-15", _inputs()))

    assert json.loads(prod.read_text(encoding="utf-8"))["provenance"]["as_of_date"] == "2026-10-30"
    assert json.loads(staged.read_text(encoding="utf-8"))["provenance"]["as_of_date"] == "2026-09-15"
    assert os.stat(prod).st_nlink == 1
    assert os.stat(staged).st_nlink == 1


def test_identical_inputs_byte_identical_output(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    sha_a = write_frozen(a, build_pit_snapshot("2026-09-15", _inputs()))
    sha_b = write_frozen(b, build_pit_snapshot("2026-09-15", _inputs()))
    assert a.read_bytes() == b.read_bytes()
    assert sha_a == sha_b
    assert b"\r\n" not in a.read_bytes()


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        build_pit_snapshot("2026-06-02", _inputs())
