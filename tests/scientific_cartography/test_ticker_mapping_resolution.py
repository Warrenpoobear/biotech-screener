"""Tests for Scientific Cartography ticker mapping resolution via sponsor matching."""

from scientific_cartography.build.asset_indication_builder import AssetIndicationBuilder
from scientific_cartography.build.competitive_cluster_builder import CompetitiveClusterBuilder
from scientific_cartography.normalize.asset_alias_resolver import AssetAliasResolver
from scientific_cartography.normalize.disease_normalizer import DiseaseNormalizer
from scientific_cartography.normalize.sponsor_resolver import SponsorResolver
from scientific_cartography.normalize.stage_normalizer import StageNormalizer
from scientific_cartography.schemas.company_schema import CompanyRecord
from scientific_cartography.schemas.program_schema import ProgramRecord


class TestSponsorResolverSuffixStripping:
    """Tests for corporate suffix stripping in sponsor resolution."""

    def test_exact_company_name_match(self):
        """Exact company name should match sponsor to ticker."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Vertex Pharmaceuticals Inc")
        assert result is not None
        assert result["ticker"] == "VRTX"
        assert result["confidence"] == 0.90

    def test_sponsor_with_incorporated_vs_inc(self):
        """'Vertex Pharmaceuticals Incorporated' should match 'Vertex Pharmaceuticals Inc'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Vertex Pharmaceuticals Incorporated")
        assert result is not None
        assert result["ticker"] == "VRTX"
        assert result["confidence"] == 0.85  # Slightly lower for normalized match
        assert "suffix stripped" in result["warnings"][0].lower()

    def test_sponsor_with_comma_suffix(self):
        """'Arcellx, Inc.' should match 'Arcellx Inc'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_ACLX",
                ticker="ACLX",
                company_name="Arcellx Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Arcellx, Inc.")
        assert result is not None
        assert result["ticker"] == "ACLX"
        assert result["confidence"] == 0.85

    def test_sponsor_with_corp_vs_corporation(self):
        """'Cogent Corp' should match 'Cogent Corporation'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_COGT",
                ticker="COGT",
                company_name="Cogent Biosciences Corporation",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Cogent Biosciences Corp")
        assert result is not None
        assert result["ticker"] == "COGT"
        assert result["confidence"] == 0.85

    def test_sponsor_with_ltd_suffix(self):
        """'Company Ltd' should match 'Company Limited'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_TEST",
                ticker="TEST",
                company_name="Test Company Limited",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Test Company Ltd")
        assert result is not None
        assert result["ticker"] == "TEST"

    def test_multi_suffix_company_name_match(self):
        """'Alnylam Pharmaceuticals' should match 'Alnylam Pharmaceuticals, Inc.'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_ALNY",
                ticker="ALNY",
                company_name="Alnylam Pharmaceuticals, Inc.",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Alnylam Pharmaceuticals")
        assert result is not None
        assert result["ticker"] == "ALNY"
        assert result["confidence"] == 0.85

    def test_international_suffix_punctuation_match(self):
        """'Abivax S.A.' should match universe company name 'Abivax SA'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_ABVX",
                ticker="ABVX",
                company_name="Abivax SA",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Abivax S.A.")
        assert result is not None
        assert result["ticker"] == "ABVX"

    def test_normalized_exact_stem_match(self):
        """'argenx' should match universe company name 'argenx SE'."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_ARGX",
                ticker="ARGX",
                company_name="argenx SE",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("argenx")
        assert result is not None
        assert result["ticker"] == "ARGX"

    def test_unknown_sponsor_remains_null(self):
        """Unknown sponsor should result in null ticker."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("Unknown Therapeutics Inc")
        assert result is not None
        assert result["ticker"] is None
        assert result["confidence"] == 0.0
        assert result["resolution_status"] == "unknown"

    def test_ambiguous_sponsor_conservatively_unmatched(self):
        """Ambiguous suffix stripping should not match to avoid false positives."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_THER",
                ticker="THER",
                company_name="Therapeutics Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        # Just "Therapeutics" by itself should NOT match
        result = resolver.resolve("Therapeutics")
        assert result is not None
        # Should not match due to conservative approach
        assert result["ticker"] is None or result["confidence"] < 0.5

    def test_manual_sponsor_alias_highest_priority(self):
        """Manual sponsor alias should have highest priority."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        sponsor_aliases = {"vertex pharma": "COMPANY_TICKER_VRTX"}
        resolver = SponsorResolver(company_records=companies, sponsor_aliases=sponsor_aliases)
        result = resolver.resolve("Vertex Pharma")
        assert result is not None
        assert result["ticker"] == "VRTX"
        assert result["confidence"] == 0.95  # Alias match has highest confidence

    def test_exact_ticker_match_highest_priority(self):
        """Exact ticker match should have high confidence."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]
        resolver = SponsorResolver(company_records=companies)
        result = resolver.resolve("VRTX")
        assert result is not None
        assert result["ticker"] == "VRTX"
        assert result["confidence"] == 0.95


class TestAssetIndicationBuilderTickerMapping:
    """Tests for ProgramRecord ticker mapping through asset indication builder."""

    def test_program_record_includes_ticker(self):
        """ProgramRecord should include ticker from resolved sponsor."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]

        disease_normalizer = DiseaseNormalizer()
        stage_normalizer = StageNormalizer()
        asset_alias_resolver = AssetAliasResolver()
        sponsor_resolver = SponsorResolver(company_records=companies)

        builder = AssetIndicationBuilder(
            disease_normalizer=disease_normalizer,
            stage_normalizer=stage_normalizer,
            asset_alias_resolver=asset_alias_resolver,
            sponsor_resolver=sponsor_resolver,
            as_of_date="2026-06-17",
        )

        # Create mock trial
        from scientific_cartography.schemas.trial_schema import TrialRecord

        trial = TrialRecord(
            nct_id="NCT12345678",
            brief_title="Test Study",
            sponsor="Vertex Pharmaceuticals Incorporated",
            conditions=["Hepatitis C"],
            interventions=["VX-123"],
            phases=["Phase 2"],
            overall_status="Recruiting",
            source_ref="nct_12345678",
        )

        programs, _ = builder.build_from_trials([trial], companies)
        assert len(programs) > 0
        assert programs[0].ticker == "VRTX"
        assert programs[0].ticker_resolution_source == "sponsor_resolver"
        assert programs[0].ticker_resolution_confidence == 0.85

    def test_program_record_uses_trial_ticker_when_sponsor_unmatched(self):
        """Ticker-bearing local trial records should map even with institution sponsor text."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_ABSI",
                ticker="ABSI",
                company_name="Absci Corporation",
                cik=None,
                is_public=True,
                aliases=[],
                as_of_date="2026-06-17",
            )
        ]

        builder = AssetIndicationBuilder(
            disease_normalizer=DiseaseNormalizer(),
            stage_normalizer=StageNormalizer(),
            asset_alias_resolver=AssetAliasResolver(),
            sponsor_resolver=SponsorResolver(company_records=companies),
            as_of_date="2026-06-17",
        )

        from scientific_cartography.schemas.trial_schema import TrialRecord

        trial = TrialRecord(
            nct_id="NCT87654321",
            brief_title="Institution-sponsored mapped study",
            sponsor="Peking University People's Hospital",
            ticker="ABSI",
            conditions=["Cancer"],
            interventions=["ABSI-101"],
            phases=["Phase 1"],
            overall_status="Recruiting",
            source_ref="nct_87654321",
        )

        programs, diagnostics = builder.build_from_trials([trial], companies)

        assert len(programs) == 1
        assert programs[0].ticker == "ABSI"
        assert programs[0].company_id == "COMPANY_TICKER_ABSI"
        assert programs[0].company_name == "Peking University People's Hospital"
        assert programs[0].ticker_resolution_source == "trial_ticker"
        assert programs[0].ticker_resolution_confidence == 0.95
        assert diagnostics["programs_with_unknown_sponsor"] == 0

    def test_program_record_prefers_trial_ticker_over_different_public_sponsor(self):
        """Local trial ticker should prevent sponsor-text cross-ticker attribution."""
        companies = [
            CompanyRecord(
                company_id="COMPANY_TICKER_AMGN",
                ticker="AMGN",
                company_name="Amgen Inc.",
                cik=None,
                is_public=True,
                aliases=["Amgen"],
                as_of_date="2026-06-17",
            ),
            CompanyRecord(
                company_id="COMPANY_TICKER_AZN",
                ticker="AZN",
                company_name="AstraZeneca PLC",
                cik=None,
                is_public=True,
                aliases=["AstraZeneca"],
                as_of_date="2026-06-17",
            ),
        ]

        builder = AssetIndicationBuilder(
            disease_normalizer=DiseaseNormalizer(),
            stage_normalizer=StageNormalizer(),
            asset_alias_resolver=AssetAliasResolver(),
            sponsor_resolver=SponsorResolver(company_records=companies),
            as_of_date="2026-06-17",
        )

        from scientific_cartography.schemas.trial_schema import TrialRecord

        trial = TrialRecord(
            nct_id="NCT11112222",
            brief_title="Mapped local trial",
            sponsor="Amgen",
            ticker="AZN",
            conditions=["Cancer"],
            interventions=["AZN-101"],
            phases=["Phase 2"],
            overall_status="Recruiting",
        )

        programs, _ = builder.build_from_trials([trial], companies)

        assert len(programs) == 1
        assert programs[0].ticker == "AZN"
        assert programs[0].company_id == "COMPANY_TICKER_AZN"
        assert programs[0].company_name == "Amgen"
        assert programs[0].ticker_resolution_source == "trial_ticker"


class TestCompetitiveClusterTickerAggregation:
    """Tests for CompetitiveClusterRecord public_tickers aggregation."""

    def test_cluster_aggregates_program_tickers(self):
        """Cluster should aggregate tickers from member programs."""
        programs = [
            ProgramRecord(
                program_id="PROGRAM_1",
                asset_id="ASSET_1",
                asset_name="Asset A",
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                disease_id="DISEASE_123",
                disease_name="Hepatitis C",
                mondo_id=None,
                therapeutic_area="Virology",
                indication_detail="Hepatitis C",
                mechanism_class="NS5A inhibitor",
                modality="Small molecule",
                target="NS5A",
                clinical_stage="Phase 3",
                trial_ids=["NCT12345"],
                regulatory_status=None,
                source_priority="ctgov",
                source_refs=["NCT12345"],
                confidence=0.85,
                as_of_date="2026-06-17",
            ),
            ProgramRecord(
                program_id="PROGRAM_2",
                asset_id="ASSET_2",
                asset_name="Asset B",
                company_id="COMPANY_TICKER_ACLX",
                ticker="ACLX",
                company_name="Arcellx Inc",
                disease_id="DISEASE_123",
                disease_name="Hepatitis C",
                mondo_id=None,
                therapeutic_area="Virology",
                indication_detail="Hepatitis C",
                mechanism_class="NS5A inhibitor",
                modality="Small molecule",
                target="NS5A",
                clinical_stage="Phase 2",
                trial_ids=["NCT67890"],
                regulatory_status=None,
                source_priority="ctgov",
                source_refs=["NCT67890"],
                confidence=0.80,
                as_of_date="2026-06-17",
            ),
        ]

        builder = CompetitiveClusterBuilder(as_of_date="2026-06-17")
        clusters, _ = builder.build_from_programs(programs)

        assert len(clusters) > 0
        cluster = clusters[0]
        assert len(cluster.public_tickers) >= 2
        assert "VRTX" in cluster.public_tickers
        assert "ACLX" in cluster.public_tickers

    def test_cluster_counts_public_programs(self):
        """Cluster should count programs with public tickers separately."""
        programs = [
            ProgramRecord(
                program_id="PROGRAM_1",
                asset_id="ASSET_1",
                asset_name="Asset A",
                company_id="COMPANY_TICKER_VRTX",
                ticker="VRTX",
                company_name="Vertex Pharmaceuticals Inc",
                disease_id="DISEASE_123",
                disease_name="Test Disease",
                mondo_id=None,
                therapeutic_area="Test Area",
                indication_detail="Test Indication",
                mechanism_class="Test Mechanism",
                modality="Small molecule",
                target="Test Target",
                clinical_stage="Phase 2",
                trial_ids=["NCT1"],
                regulatory_status=None,
                source_priority="ctgov",
                source_refs=["NCT1"],
                confidence=0.85,
                as_of_date="2026-06-17",
            ),
            ProgramRecord(
                program_id="PROGRAM_2",
                asset_id="ASSET_2",
                asset_name="Asset B",
                company_id=None,
                ticker=None,
                company_name="Unknown Company",
                disease_id="DISEASE_123",
                disease_name="Test Disease",
                mondo_id=None,
                therapeutic_area="Test Area",
                indication_detail="Test Indication",
                mechanism_class="Test Mechanism",
                modality="Small molecule",
                target="Test Target",
                clinical_stage="Phase 1",
                trial_ids=["NCT2"],
                regulatory_status=None,
                source_priority="ctgov",
                source_refs=["NCT2"],
                confidence=0.70,
                as_of_date="2026-06-17",
            ),
        ]

        builder = CompetitiveClusterBuilder(as_of_date="2026-06-17")
        clusters, _ = builder.build_from_programs(programs)

        assert len(clusters) > 0
        cluster = clusters[0]
        assert cluster.public_program_count == 1
        assert cluster.private_or_unknown_program_count == 1
