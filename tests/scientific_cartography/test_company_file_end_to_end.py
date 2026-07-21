"""End-to-end regression: --company-file -> diagnostic -> generator -> map.json tickers.

Proves the full chain that resolves the 0% ticker-coverage root cause: an authorized
company/universe snapshot (--company-file) is resolved to tickers by the diagnostic and
surfaced by the map generator. Trials deliberately carry NO ticker, so resolution must
flow through the sponsor->ticker path.
"""

import argparse
import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load("run_scientific_cartography_diagnostics", "tools/run_scientific_cartography_diagnostics.py")
gen = _load("generate_scientific_cartography_map", "tools/generate_scientific_cartography_map.py")


def _diag_args(tmp_path, company_file, trials_file, out):
    return argparse.Namespace(
        as_of_date="2026-06-27",
        snapshot_dir=str(tmp_path / "snap"),
        ctgov_cache=str(tmp_path / "cache"),
        trials_file=str(trials_file),
        company_file=str(company_file),
        mechanism_aliases=None,
        output_dir=str(out),
        strict=False,
        created_at_utc="2026-06-27T00:00:00Z",
        quiet=True,
    )


def _trials():
    return [
        {
            "nct_id": "NCT90000001",
            "brief_title": "Tirzepatide in T2D",
            "sponsor": "Eli Lilly and Company",
            "conditions": ["Type 2 Diabetes Mellitus"],
            "interventions": ["Tirzepatide"],
            "phases": ["Phase 3"],
            "overall_status": "Recruiting",
            "study_type": "Interventional",
        },
        {
            "nct_id": "NCT90000002",
            "brief_title": "VX-880 in T2D",
            "sponsor": "Vertex Pharmaceuticals",
            "conditions": ["Type 2 Diabetes Mellitus"],
            "interventions": ["VX-880"],
            "phases": ["Phase 2"],
            "overall_status": "Recruiting",
            "study_type": "Interventional",
        },
        {
            "nct_id": "NCT90000003",
            "brief_title": "Private agent in T2D",
            "sponsor": "Tiny Private Biotech LLC",
            "conditions": ["Type 2 Diabetes Mellitus"],
            "interventions": ["XYZ-123"],
            "phases": ["Phase 1"],
            "overall_status": "Recruiting",
            "study_type": "Interventional",
        },
    ]


def _walk_nodes(obj):
    if isinstance(obj, dict):
        if "ticker" in obj and "asset_name" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_nodes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_nodes(v)


def test_company_file_to_map_ticker_flow(tmp_path):
    (tmp_path / "snap").mkdir()
    (tmp_path / "cache").mkdir()
    company = tmp_path / "universe.csv"
    company.write_text("ticker,company\nVRTX,Vertex Pharmaceuticals\nLLY,Eli Lilly and Company\n")
    trials = tmp_path / "trials.json"
    trials.write_text(json.dumps(_trials()))
    diag_out = tmp_path / "diagout"

    rc = diag.run_diagnostics(_diag_args(tmp_path, company, trials, diag_out))
    assert rc == 0

    # Diagnostic resolved tickers via the sponsor path into program records.
    progs = [json.loads(l) for l in (diag_out / "program_records.jsonl").read_text().splitlines() if l.strip()]
    by_company = {p["company_name"]: p["ticker"] for p in progs}
    assert by_company.get("Eli Lilly and Company") == "LLY"
    assert by_company.get("Vertex Pharmaceuticals") == "VRTX"
    assert by_company.get("Tiny Private Biotech LLC") in (None, "")

    # Status records the company source and mechanism alias source.
    status = json.loads((diag_out / "scientific_cartography_status.json").read_text())
    assert status["company_source"] == "universe.csv"
    assert status["mechanism_alias_source"] in ("mechanism_aliases_v0_1.csv", "builtin")

    # Generator surfaces those tickers into map.json.
    map_out = tmp_path / "mapout"
    gen.generate_map(input_dir=diag_out, disease="diabetes", output_dir=map_out, quiet=True)
    m = json.loads((map_out / "map.json").read_text())
    assert m["summary"]["ticker_coverage_pct"] > 0
    node_tickers = {n["company_name"]: n["ticker"] for n in _walk_nodes(m)}
    assert node_tickers.get("Eli Lilly and Company") == "LLY"
    assert node_tickers.get("Vertex Pharmaceuticals") == "VRTX"


def test_missing_mechanism_alias_override_warns(tmp_path):
    (tmp_path / "snap").mkdir()
    (tmp_path / "cache").mkdir()
    company = tmp_path / "universe.csv"
    company.write_text("ticker,company\nLLY,Eli Lilly and Company\n")
    trials = tmp_path / "trials.json"
    trials.write_text(json.dumps(_trials()[:1]))
    diag_out = tmp_path / "diagout"
    args = _diag_args(tmp_path, company, trials, diag_out)
    args.mechanism_aliases = str(tmp_path / "absent_aliases.csv")  # explicit but missing

    rc = diag.run_diagnostics(args)
    assert rc == 0
    status = json.loads((diag_out / "scientific_cartography_status.json").read_text())
    assert status["mechanism_alias_source"] == "builtin"
    assert any("--mechanism-aliases not found" in w for w in status["warnings"])
