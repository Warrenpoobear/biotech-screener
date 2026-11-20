"""LVB research backtests: month-end rebalance calendar and annualisation by actual span.

The snapshot set holds extra mid-month dates (e.g. 2026-02-27 next to 2026-02-28, and
2026-04-16). Using every snapshot as a rebalance date made 1-day and half-month periods
that were annualised as full months. Research code only (Tier 2).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

from lvb_universe_backtest import month_end_dates, span_years, stats  # noqa: E402


def test_month_end_dates_keeps_last_snapshot_per_month():
    dates = ["2026-01-30", "2026-02-27", "2026-02-28", "2026-03-31", "2026-04-16", "2026-04-30", "2026-05-29"]
    assert month_end_dates(dates) == ["2026-01-30", "2026-02-28", "2026-03-31", "2026-04-30", "2026-05-29"]


def test_month_end_dates_keeps_final_date_and_is_order_independent():
    dates = ["2026-11-16", "2026-10-30", "2026-10-15"]
    assert month_end_dates(dates) == ["2026-10-30", "2026-11-16"]


def test_span_years_runs_from_first_start_to_last_end():
    rows = [{"d0": "2025-01-31", "d1": "2025-02-28"}, {"d0": "2025-02-28", "d1": "2026-01-31"}]
    assert math.isclose(span_years(rows), 365 / 365.25)


def test_stats_annualises_by_actual_years_when_given():
    rs = [0.02, 0.0] * 12
    xs = [0.01, -0.005, 0.0] * 8
    growth = 1.02**12
    s_count = stats(rs, xs)
    assert math.isclose(s_count["ann_return"], growth ** (12 / 24) - 1)  # default: periods are months
    s_span = stats(rs, xs, years=1.5)
    assert math.isclose(s_span["ann_return"], growth ** (1 / 1.5) - 1)
    assert s_span["sharpe"] == s_count["sharpe"]  # only the annual return depends on the span
