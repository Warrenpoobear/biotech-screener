"""Ticker-mapping robustness: company-file aliases resolve CT.gov sponsor-name variants.

The dominant real-world sponsor->ticker failure is name-variant mismatch (CT.gov lists
"Lilly USA" or a subsidiary while the universe carries "Eli Lilly"). The SponsorResolver
already indexes CompanyRecord.aliases, but the CSV company-file loader did not populate
them. These tests prove operator-curated aliases (CSV column / JSON field) flow through
ingest -> resolver -> map, deterministically and without fuzzy false-positives.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from scientific_cartography.normalize.sponsor_resolver import SponsorResolver

_REPO = Path(__file__).resolve().parents[2]


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load("run_scientific_cartography_diagnostics", "tools/run_scientific_cartography_diagnostics.py")
gen = _load("generate_scientific_cartography_map", "tools/generate_scientific_cartography_map.py")


def _args(**overrides):
    ns = argparse.Namespace(company_file=None)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _status():
    return {"warnings": []}


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


def _walk_nodes(obj):
    if isinstance(obj, dict):
        if "ticker" in obj and "asset_name" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_nodes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_nodes(v)


def test_csv_aliases_populate_and_resolve_variants(tmp_path):
    csv_path = tmp_path / "universe.csv"
    # Mixed ';' and '|' delimiters; aliases are sponsor-name variants for one ticker.
    csv_path.write_text("ticker,company,aliases\nLLY,Eli Lilly and Company,Lilly USA;Eli Lilly|Lilly Research\n")
    companies, source = diag._load_company_records(_args(company_file=str(csv_path)), tmp_path, "2026-06-27", _status())
    assert source == "universe.csv"
    (company,) = companies
    assert company.ticker == "LLY"
    assert set(company.aliases) == {"Lilly USA", "Eli Lilly", "Lilly Research"}
    resolver = SponsorResolver(company_records=companies)
    for variant in ("Lilly USA", "Eli Lilly", "Lilly Research"):
        assert resolver.resolve(variant)["ticker"] == "LLY", variant


def test_json_explicit_aliases_resolve(tmp_path):
    json_path = tmp_path / "universe.json"
    json_path.write_text(
        json.dumps(
            [{"ticker": "VRTX", "company": "Vertex Pharmaceuticals", "aliases": ["Vertex Pharma", "Vertex Inc"]}]
        )
    )
    companies, source = diag._load_company_records(
        _args(company_file=str(json_path)), tmp_path, "2026-06-27", _status()
    )
    resolver = SponsorResolver(company_records=companies)
    assert resolver.resolve("Vertex Pharma")["ticker"] == "VRTX"
    assert resolver.resolve("Vertex Inc")["ticker"] == "VRTX"


def test_no_aliases_column_is_backward_compatible(tmp_path):
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("ticker,company\nLLY,Eli Lilly and Company\n")
    companies, _ = diag._load_company_records(_args(company_file=str(csv_path)), tmp_path, "2026-06-27", _status())
    assert companies[0].aliases == []
    assert SponsorResolver(company_records=companies).resolve("Eli Lilly and Company")["ticker"] == "LLY"


def test_alias_sponsor_resolves_end_to_end(tmp_path):
    (tmp_path / "snap").mkdir()
    (tmp_path / "cache").mkdir()
    company = tmp_path / "universe.csv"
    # Trial sponsor below ("Lilly USA") is ONLY present as an alias, never the canonical name.
    company.write_text("ticker,company,aliases\nLLY,Eli Lilly and Company,Lilly USA;Lilly Research\n")
    trials = tmp_path / "trials.json"
    trials.write_text(
        json.dumps(
            [
                {
                    "nct_id": "NCT93000001",
                    "brief_title": "Variant sponsor T2D",
                    "sponsor": "Lilly USA",
                    "conditions": ["Type 2 Diabetes Mellitus"],
                    "interventions": ["DrugAlias"],
                    "phases": ["Phase 2"],
                    "overall_status": "Recruiting",
                    "study_type": "Interventional",
                },
                {
                    "nct_id": "NCT93000002",
                    "brief_title": "Private T2D",
                    "sponsor": "Tiny Private Biotech LLC",
                    "conditions": ["Type 2 Diabetes Mellitus"],
                    "interventions": ["PRV-1"],
                    "phases": ["Phase 1"],
                    "overall_status": "Recruiting",
                    "study_type": "Interventional",
                },
            ]
        )
    )
    diag_out = tmp_path / "diagout"
    assert diag.run_diagnostics(_diag_args(tmp_path, company, trials, diag_out)) == 0

    progs = [json.loads(l) for l in (diag_out / "program_records.jsonl").read_text().splitlines() if l.strip()]
    tickers = {p["ticker"] for p in progs}
    assert "LLY" in tickers  # resolved purely through the 'Lilly USA' alias
    assert any(p["ticker"] in (None, "") for p in progs)  # private sponsor stays unresolved (no false map)

    map_out = tmp_path / "mapout"
    gen.generate_map(input_dir=diag_out, disease="diabetes", output_dir=map_out, quiet=True)
    m = json.loads((map_out / "map.json").read_text())
    assert "LLY" in {n["ticker"] for n in _walk_nodes(m)}
