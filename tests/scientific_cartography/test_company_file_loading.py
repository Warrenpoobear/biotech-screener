"""Regression tests for the diagnostic tool's --company-file ticker wiring.

Covers the _load_company_records precedence logic that feeds SponsorResolver and
resolves the root cause of 0% ticker coverage: companies was always empty because
ticker data only loaded from the (intentionally absent) snapshot_dir/rankings.csv.
"""

import argparse
import importlib.util
from pathlib import Path

from scientific_cartography.normalize.sponsor_resolver import SponsorResolver

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "run_scientific_cartography_diagnostics.py"
_spec = importlib.util.spec_from_file_location("run_scientific_cartography_diagnostics", _TOOL_PATH)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def _args(**overrides):
    ns = argparse.Namespace(company_file=None)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _status():
    return {"warnings": []}


def test_company_file_csv_populates_tickers(tmp_path):
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("ticker,company\nVRTX,Vertex Pharmaceuticals\nALNY,Alnylam Pharmaceuticals\n")
    companies, source = diag._load_company_records(_args(company_file=str(csv_path)), tmp_path, "2026-06-27", _status())
    assert source == "universe.csv"
    assert {c.ticker for c in companies} == {"VRTX", "ALNY"}
    resolved = SponsorResolver(company_records=companies).resolve("Vertex Pharmaceuticals")
    assert resolved["ticker"] == "VRTX"
    assert resolved["is_public"] is True


def test_company_file_takes_precedence_over_rankings(tmp_path):
    (tmp_path / "rankings.csv").write_text("ticker,company\nRANK,Should Not Load\n")
    company_file = tmp_path / "company_universe.csv"
    company_file.write_text("ticker,company\nVRTX,Vertex Pharmaceuticals\n")
    companies, source = diag._load_company_records(
        _args(company_file=str(company_file)), tmp_path, "2026-06-27", _status()
    )
    assert source == "company_universe.csv"
    assert {c.ticker for c in companies} == {"VRTX"}


def test_missing_company_file_warns_loudly_no_fallback(tmp_path):
    # rankings.csv exists, but an explicitly requested company-file that is
    # missing must NOT silently fall back to it (forbidden-source guardrail).
    (tmp_path / "rankings.csv").write_text("ticker,company\nRANK,Should Not Load\n")
    status = _status()
    companies, source = diag._load_company_records(
        _args(company_file=str(tmp_path / "absent.csv")), tmp_path, "2026-06-27", status
    )
    assert companies == []
    assert source is None
    assert any("not found" in w for w in status["warnings"])


def test_default_falls_back_to_rankings(tmp_path):
    (tmp_path / "rankings.csv").write_text("ticker,company\nVRTX,Vertex Pharmaceuticals\n")
    companies, source = diag._load_company_records(_args(company_file=None), tmp_path, "2026-06-27", _status())
    assert source == "rankings.csv"
    assert companies[0].ticker == "VRTX"


def test_no_sources_returns_empty(tmp_path):
    status = _status()
    companies, source = diag._load_company_records(_args(company_file=None), tmp_path, "2026-06-27", status)
    assert companies == []
    assert source is None
    assert any("rankings.csv not found" in w for w in status["warnings"])
