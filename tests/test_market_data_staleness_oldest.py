"""market_data_staleness gate must judge market_data.json by its OLDEST row.

2026-11-12 regression: a commit replaced production_data/market_data.json with a
file whose rows mostly dated 2026-02-19. The gate took max(collected_at), so one
fresh row (or the price patch re-stamping rows) let it PASS and the
collect_market_data.py auto-refresh never ran, leaving avg_volume, beta,
volatility_90d and short_percent months old.

Kept out of test_phase2_daily.py because that module is marked ``network`` and
is deselected by the default ``-m 'not network'`` run and in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from run_daily_production import check_market_data_staleness  # noqa: E402


def _write(data_dir: Path, dates: list) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"ticker": f"T{i}", "price": 10.0, "collected_at": d} for i, d in enumerate(dates)]
    (data_dir / "market_data.json").write_text(json.dumps(rows), encoding="utf-8")


def test_one_fresh_row_does_not_mask_stale_rows(tmp_path):
    _write(tmp_path, ["2026-11-12", "2026-02-19", "2026-02-19"])
    result = check_market_data_staleness(tmp_path, "2026-11-12", max_age_days=5)
    assert result.status == "FAIL"
    assert result.value == 266
    assert "2026-02-19" in result.detail
    assert "2/3" in result.detail  # rows past the limit, so the operator sees the extent


def test_all_rows_within_limit_passes_and_reports_oldest(tmp_path):
    _write(tmp_path, ["2026-11-12", "2026-11-10", "2026-11-11"])
    result = check_market_data_staleness(tmp_path, "2026-11-12", max_age_days=5)
    assert result.status == "PASS"
    assert result.value == 2
    assert "2026-11-10" in result.detail


def test_uniform_stale_file_still_fails(tmp_path):
    _write(tmp_path, ["2026-11-03"] * 3)
    result = check_market_data_staleness(tmp_path, "2026-11-12", max_age_days=5)
    assert result.status == "FAIL"
    assert result.value == 9
    assert "collect_market_data.py" in result.detail
