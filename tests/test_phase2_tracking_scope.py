"""
Test to verify Phase 2 tracks all 30 canonical decision-portfolio holdings.

Governance requirement: Phase 2 Day 1 uses the 30-holding canonical decision portfolio,
not a brevity-truncated 10-holding subset.
"""

import json
from pathlib import Path


def test_phase2_day1_tracks_all_30_holdings():
    """Verify that Phase 2 Day 1 artifacts contain all 30 holdings from canonical snapshot."""
    artifacts_dir = Path(__file__).parent / "fixtures" / "portfolio_policy_forward_test" / "2026-06-01"

    assert artifacts_dir.exists(), f"Day 1 artifacts not found: {artifacts_dir}"

    holdings_file = artifacts_dir / "holdings.json"
    assert holdings_file.exists(), f"Holdings file not found: {holdings_file}"

    with open(holdings_file) as f:
        data = json.load(f)

    # Governance requirement: 30 holdings
    holdings_count = data.get("holdings_count", 0)
    holdings = data.get("holdings", [])

    assert holdings_count == 30, (
        f"Expected 30 tracked holdings, got {holdings_count}. "
        f"This violates governance decision to use canonical 30-holding decision portfolio."
    )

    assert len(holdings) == 30, (
        f"Expected 30 holdings in array, got {len(holdings)}. " f"Array count must match holdings_count."
    )

    # Verify no truncation: all tickers present
    expected_tickers = [
        "DNTH",
        "NRIX",
        "URGN",
        "ARWR",
        "CELC",
        "RCUS",
        "PHVS",
        "MLTX",
        "SYRE",
        "PRAX",
        "ABVX",
        "XENE",
        "MIRM",
        "ALKS",
        "TYRA",
        "TNGX",
        "CMPS",
        "EWTX",
        "BCRX",
        "DRUG",
        "STOK",
        "RVMD",
        "ERAS",
        "ORKA",
        "NBIX",
        "MBX",
        "COGT",
        "ALMS",
        "TRVI",
        "RYTM",
    ]

    actual_tickers = [h.get("ticker") for h in holdings]

    for ticker in expected_tickers:
        assert ticker in actual_tickers, (
            f"Missing ticker: {ticker}. " f"The 20-ticker exclusion (for brevity) must be removed."
        )

    # Verify all tickers are unique
    assert len(set(actual_tickers)) == 30, "Duplicate tickers found. Expected 30 unique holdings."

    print("✓ Phase 2 Day 1 tracks all 30 canonical holdings")
    print(f"  Tickers: {', '.join(actual_tickers[:5])}...{', '.join(actual_tickers[-5:])}")


if __name__ == "__main__":
    test_phase2_day1_tracks_all_30_holdings()
    print("\nTest PASSED: Phase 2 governance scope verified.")
