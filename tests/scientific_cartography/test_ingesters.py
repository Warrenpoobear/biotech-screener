"""Tests for Phase 2 ingest modules."""

from pathlib import Path

import pytest

from scientific_cartography.ingest.ctgov_ingest import CTGovIngest
from scientific_cartography.ingest.existing_universe_ingest import ExistingUniverseIngest


class TestExistingUniverseIngest:
    """Test universe ingest from CSV."""

    @pytest.fixture
    def ingest(self):
        return ExistingUniverseIngest(as_of_date="2026-06-16")

    @pytest.fixture
    def universe_csv(self):
        return Path(__file__).parent.parent / "fixtures" / "scientific_cartography" / "universe" / "sample_universe.csv"

    def test_ingest_from_csv(self, ingest, universe_csv):
        """Should ingest companies from CSV."""
        records = ingest.ingest_from_csv(universe_csv)

        assert len(records) == 5
        assert records[0].ticker == "COGT"
        assert records[0].company_name == "Cognito Therapeutics"
        assert records[0].is_public is True
        assert records[0].as_of_date == "2026-06-16"
        assert universe_csv.name in records[0].source_refs[0]

    def test_ingest_deduplicates(self, ingest, tmp_path):
        """Should deduplicate by ticker."""
        csv_path = tmp_path / "dup.csv"
        csv_path.write_text("ticker,company,cik\nCOGT,Company1,111\nCOGT,Company2,111\n")

        records = ingest.ingest_from_csv(csv_path)

        # Should only have 1 record (deduplicated)
        assert len(records) == 1

    def test_ingest_missing_file(self, ingest):
        """Should handle missing file gracefully."""
        records = ingest.ingest_from_csv(Path("/nonexistent/file.csv"))

        assert len(records) == 0

    def test_ingest_from_csv_accepts_name_column(self, ingest, tmp_path):
        """Canonical universe CSVs use name rather than company for issuer name."""
        csv_path = tmp_path / "universe.csv"
        csv_path.write_text("ticker,name,sector\nMRNA,Moderna Inc,Biotechnology\n")

        records = ingest.ingest_from_csv(csv_path)

        assert len(records) == 1
        assert records[0].ticker == "MRNA"
        assert records[0].company_name == "Moderna Inc"

    def test_ingest_from_rankings(self, ingest, tmp_path):
        """Should ingest from rankings CSV format."""
        csv_path = tmp_path / "rankings.csv"
        csv_path.write_text("ticker,rank,score\nCOGT,1,0.95\nDNTH,2,0.90\n")

        records = ingest.ingest_from_rankings_csv(csv_path)

        assert len(records) == 2
        assert records[0].ticker == "COGT"
        assert records[1].ticker == "DNTH"

    def test_ingest_from_production_universe_json_schema(self, ingest, tmp_path):
        """Should use production universe company-name fields for sponsor matching."""
        json_path = tmp_path / "universe.json"
        json_path.write_text("""[
                {
                    "ticker": "AARD",
                    "name": "AARD",
                    "exchange": "NASDAQ",
                    "market_data": {
                        "company_name": "Aardvark Therapeutics, Inc.",
                        "industry": "Biotechnology"
                    },
                    "cik": "0001774857"
                }
            ]""")

        records = ingest.ingest_from_json(json_path)

        assert len(records) == 1
        assert records[0].ticker == "AARD"
        assert records[0].company_name == "Aardvark Therapeutics, Inc."
        assert records[0].aliases == []
        assert records[0].exchange == "NASDAQ"
        assert records[0].confidence == 0.95

    def test_ingest_from_json_skips_generic_market_company_name(self, ingest, tmp_path):
        """Generic sector labels should not be used as company names."""
        json_path = tmp_path / "universe.json"
        json_path.write_text("""[
                {
                    "ticker": "ABEO",
                    "market_data": {
                        "company_name": "Healthcare",
                        "industry": "Biotechnology"
                    }
                }
            ]""")

        records = ingest.ingest_from_json(json_path)

        assert len(records) == 1
        assert records[0].ticker == "ABEO"
        assert records[0].company_name == "ABEO"
        assert records[0].aliases == []
        assert records[0].confidence == 0.85

    def test_company_records_have_as_of_date(self, ingest, universe_csv):
        """All company records should include as_of_date."""
        records = ingest.ingest_from_csv(universe_csv)

        for record in records:
            assert record.as_of_date == "2026-06-16"
            assert record.source_refs


class TestCTGovIngest:
    """Test CTGov trial ingest."""

    @pytest.fixture
    def ingest(self):
        return CTGovIngest(as_of_date="2026-06-16")

    @pytest.fixture
    def ctgov_jsonl(self):
        return Path(__file__).parent.parent / "fixtures" / "scientific_cartography" / "ctgov" / "sample_trials.jsonl"

    def test_ingest_from_jsonl(self, ingest, ctgov_jsonl):
        """Should ingest trials from JSONL."""
        records = ingest.ingest_from_jsonl_file(ctgov_jsonl)

        assert len(records) == 3
        assert records[0].nct_id == "NCT03456789"
        assert records[0].brief_title == "Study of Asset A in Atopic Dermatitis"
        assert records[0].sponsor == "Cognito Therapeutics"
        assert "Asset A" in records[0].interventions
        assert "Atopic Dermatitis" in records[0].conditions

    def test_trial_records_have_as_of_date(self, ingest, ctgov_jsonl):
        """All trial records should include as_of_date."""
        records = ingest.ingest_from_jsonl_file(ctgov_jsonl)

        for record in records:
            assert record.as_of_date == "2026-06-16"

    def test_parse_simplified_format(self, ingest):
        """Should parse simplified fixture format."""
        data = {
            "nct_id": "NCT12345678",
            "brief_title": "Test Trial",
            "sponsor": "Test Sponsor",
            "ticker": "TEST",
            "conditions": ["Disease A"],
            "interventions": ["Drug A"],
            "phases": ["Phase 2"],
            "overall_status": "Active",
        }

        record = ingest._parse_simplified_format(data)

        assert record.nct_id == "NCT12345678"
        assert record.sponsor == "Test Sponsor"
        assert record.ticker == "TEST"
        assert record.conditions == ["Disease A"]

    def test_ingest_handles_missing_nct_id(self, ingest):
        """Should skip records without NCT ID."""
        data = {"brief_title": "No NCT", "sponsor": "Test"}
        record = ingest._parse_simplified_format(data)

        assert record is None

    def test_ingest_from_json_file(self, ingest, tmp_path):
        """Should ingest from JSON array file."""
        json_path = tmp_path / "trials.json"
        json_path.write_text("""[
            {"nct_id": "NCT11111111", "brief_title": "Trial 1", "sponsor": "Sponsor 1", "conditions": ["Disease A"], "interventions": ["Drug A"], "phases": ["Phase 1"]},
            {"nct_id": "NCT22222222", "brief_title": "Trial 2", "sponsor": "Sponsor 2", "conditions": ["Disease B"], "interventions": ["Drug B"], "phases": ["Phase 2"]}
        ]""")

        records = ingest.ingest_from_json_file(json_path)

        assert len(records) == 2
        assert records[0].nct_id == "NCT11111111"

    def test_ingest_missing_ctgov_file(self, ingest):
        """Should handle missing file gracefully."""
        records = ingest.ingest_from_jsonl_file(Path("/nonexistent/trials.jsonl"))

        assert len(records) == 0

    def test_ensure_list_helper(self, ingest):
        """_ensure_list should handle various input types."""
        assert ingest._ensure_list(["a", "b"]) == ["a", "b"]
        assert ingest._ensure_list("single") == ["single"]
        assert ingest._ensure_list("") == []
        assert ingest._ensure_list(None) == []
        assert ingest._ensure_list([]) == []
