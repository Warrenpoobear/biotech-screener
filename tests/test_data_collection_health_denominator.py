"""price_coverage must not count retired / benchmark universe rows.

2026-11-10: 14 retired names (delisted + excluded_acquired) plus the
_XBI_BENCHMARK_ row sat in the denominator, producing a false
price_coverage=94.4% FAIL although every tradable name had a price.
"""

from __future__ import annotations

import json

from tools.build_data_collection_health import _check_market_data


def _setup(tmp_path, universe, priced):
    (tmp_path / "universe.json").write_text(json.dumps(universe), encoding="utf-8")
    rows = ["date,ticker,close"] + [f"2026-11-10,{t},1.0" for t in priced]
    (tmp_path / "price_history.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_retired_and_benchmark_rows_excluded_from_denominator(tmp_path):
    universe = [{"ticker": f"A{i}", "status": "active"} for i in range(19)] + [
        {"ticker": "DEAD1", "status": "delisted"},
        {"ticker": "DEAD2", "status": "excluded_acquired"},
        {"ticker": "_XBI_BENCHMARK_", "status": "benchmark"},
    ]
    _setup(tmp_path, universe, [f"A{i}" for i in range(19)])
    out = _check_market_data(tmp_path, tmp_path, "2026-11-10", {"market_data_min_coverage_pct": 0.95})
    assert out["universe_ticker_count"] == 19
    assert out["price_coverage_pct"] == 100.0
    assert out["status"] == "PASS"


def test_missing_active_price_still_fails(tmp_path):
    universe = [{"ticker": f"A{i}", "status": "active"} for i in range(20)]
    _setup(tmp_path, universe, [f"A{i}" for i in range(18)])
    out = _check_market_data(tmp_path, tmp_path, "2026-11-10", {"market_data_min_coverage_pct": 0.95})
    assert out["price_coverage_pct"] == 90.0
    assert out["status"] == "FAIL"
