"""Pipeline fixes 2026-11-11: Purple Book ingest, universe-maintenance source,
options-enrichment as_of discipline."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ingest_purple_book as pb  # noqa: E402

# --- Purple Book -----------------------------------------------------------

PAGE = """
<a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2025/purplebook-search-december-data-download.csv">x</a>
<a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-january-data-download.csv">x</a>
<a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-October-data-download.csv">x</a>
<a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-September-data-download.csv">x</a>
"""


def test_latest_link_handles_mixed_month_case():
    url, y, m = pb.find_latest_csv_url(PAGE)
    assert (y, m) == (2026, 10)
    assert url.endswith("2026/purplebook-search-October-data-download.csv")


def test_no_links_returns_none():
    assert pb.find_latest_csv_url("<html>nothing</html>") is None


HEADER = (
    "N/R/U,Applicant,BLA Number,Proprietary Name,Proper Name,License Type,"
    "Ref. Product Proper Name,Ref. Product Proprietary Name,Approval Date\n"
)


def _csv(tmp_path, rows):
    p = tmp_path / "pb.csv"
    p.write_text("Purple Book Monthly Report\n,,,\n" + HEADER + "".join(rows))
    return p


def test_license_type_column_maps_biosimilars(tmp_path):
    p = _csv(
        tmp_path,
        [
            'U,Amgen,125000,REFX,drugmab,351(a),,,"January 1, 2020"\n',
            'N,Sandoz,761000,BIOX,drugmab-abcd,351(k) Biosimilar,drugmab,REFX,"May 1, 2026"\n',
        ],
    )
    result = pb.ingest_csv(p, "2026-11-11")
    assert result["stats"]["n_biosimilars"] == 1


def test_main_refuses_zero_biosimilars(tmp_path):
    p = _csv(tmp_path, ['U,Amgen,125000,REFX,drugmab,351(a),,,"January 1, 2020"\n'])
    out = tmp_path / "out.json"
    out.write_text('{"keep": true}')
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ingest_purple_book.py"), str(p), "--output", str(out)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 2
    assert json.loads(out.read_text()) == {"keep": True}  # previous file untouched


# --- Universe maintenance --------------------------------------------------


def test_universe_maintenance_reads_daily_snapshots():
    sys.path.insert(0, str(REPO_ROOT))
    from tools.build_universe_maintenance import build_universe_maintenance

    default = inspect.signature(build_universe_maintenance).parameters["snapshots_dir"].default
    assert Path(default).parts[-2:] == ("data", "snapshots")


# --- Options enrichment ----------------------------------------------------


def _enrich_env(tmp_path, monkeypatch, latest_as_of, rank_dates):
    import tools.enrich_options_features as eof

    (tmp_path / "production_data").mkdir()
    snap = {"metadata": {"as_of_date": latest_as_of}, "tickers": {}}
    (tmp_path / "production_data" / "options_snapshot_latest.json").write_text(json.dumps(snap))
    for d in rank_dates:
        (tmp_path / "data" / "snapshots" / d).mkdir(parents=True)
        (tmp_path / "data" / "snapshots" / d / "rankings.csv").write_text("ticker\nAAA\n")
    monkeypatch.setattr(eof, "REPO_ROOT", tmp_path)
    return eof


def test_enrichment_refuses_snapshot_from_another_date(tmp_path, monkeypatch):
    eof = _enrich_env(tmp_path, monkeypatch, "2026-11-11", ["2026-11-10"])
    with pytest.raises(FileNotFoundError, match="labelled 2026-11-05"):
        eof.run("2026-11-05", tmp_path / "out", True, False, False)


def test_enrichment_never_reads_later_rankings(tmp_path, monkeypatch):
    eof = _enrich_env(tmp_path, monkeypatch, "2026-11-05", ["2026-11-10"])
    with pytest.raises(FileNotFoundError, match="on or before 2026-11-05"):
        eof.run("2026-11-05", tmp_path / "out", True, False, False)
