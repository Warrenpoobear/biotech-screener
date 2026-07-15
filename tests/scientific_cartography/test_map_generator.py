"""Tests for Scientific Cartography Map UX v0.2c static generator."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.generate_scientific_cartography_map import (
    ForbiddenSourceError,
    _canonical_asset_name,
    _check_forbidden,
    _deduplicate_programs,
    _filter_non_drug_programs,
    build_map_data,
    generate_map,
    render_html,
    render_svg,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_KNOWN_DISEASE = "type 2 diabetes mellitus"
_MONDO = "MONDO:0005148"


def _make_program(
    asset_name="DrugA",
    disease_name=_KNOWN_DISEASE,
    clinical_stage="phase2",
    mechanism_class=None,
    modality=None,
    ticker=None,
    confidence=0.75,
    mondo_id=_MONDO,
    therapeutic_area="Metabolic",
    program_id=None,
):
    return {
        "program_id": program_id or f"PROG_{asset_name}",
        "asset_name": asset_name,
        "company_name": "ACME Bio",
        "ticker": ticker,
        "disease_name": disease_name,
        "disease_id": mondo_id,
        "mondo_id": mondo_id,
        "therapeutic_area": therapeutic_area,
        "mechanism_class": mechanism_class,
        "target": None,
        "modality": modality,
        "clinical_stage": clinical_stage,
        "confidence": confidence,
        "source_refs": ["NCT12345678"],
    }


def _fixture_dir_with_programs(programs: list[dict]) -> Path:
    """Write a minimal artifact dir with program_records.jsonl + manifest."""
    tmp = Path(tempfile.mkdtemp())
    with open(tmp / "program_records.jsonl", "w") as f:
        for p in programs:
            f.write(json.dumps(p) + "\n")
    manifest = {
        "artifact_type": "scientific_cartography_export_manifest",
        "as_of_date": "2026-06-23",
        "governance": {"read_only_diagnostic": True},
    }
    (tmp / "artifact_manifest.json").write_text(json.dumps(manifest))
    return tmp


# ---------------------------------------------------------------------------
# 1. Forbidden-source guard
# ---------------------------------------------------------------------------


class TestForbiddenSourceGuard:
    def test_rankings_csv_blocked(self):
        with pytest.raises(ForbiddenSourceError, match="rankings.csv"):
            _check_forbidden(Path("/data/rankings.csv"))

    def test_portfolio_positions_blocked(self):
        with pytest.raises(ForbiddenSourceError, match="portfolio_positions"):
            _check_forbidden(Path("/data/portfolio_positions.csv"))

    def test_screen_output_blocked(self):
        with pytest.raises(ForbiddenSourceError, match="screen_output"):
            _check_forbidden(Path("/data/screen_output.json"))

    def test_selector_blocked(self):
        with pytest.raises(ForbiddenSourceError, match="selector"):
            _check_forbidden(Path("/data/selector_output.json"))

    def test_sizing_blocked(self):
        with pytest.raises(ForbiddenSourceError, match="sizing"):
            _check_forbidden(Path("/data/sizing_result.json"))

    def test_final_score_blocked(self):
        with pytest.raises(ForbiddenSourceError, match="final_score"):
            _check_forbidden(Path("/data/final_score.json"))

    def test_allowed_path_passes(self):
        # Allowed (non-scoring) artifact paths must pass the guard without raising;
        # _check_forbidden returns None on success.
        assert _check_forbidden(Path("/data/program_records.jsonl")) is None
        assert _check_forbidden(Path("/data/competitive_clusters.jsonl")) is None
        assert _check_forbidden(Path("/data/map_index.json")) is None

    def test_generate_map_aborts_on_forbidden_input_dir(self, tmp_path):
        forbidden_dir = tmp_path / "rankings.csv_dir"
        forbidden_dir.mkdir()
        with pytest.raises(ForbiddenSourceError):
            generate_map(
                input_dir=forbidden_dir,
                disease="test disease",
                output_dir=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# 2. Disease filtering
# ---------------------------------------------------------------------------


class TestDiseaseFilter:
    def test_exact_match_found(self, tmp_path):
        programs = [
            _make_program("DrugA", disease_name="type 2 diabetes mellitus"),
            _make_program("DrugB", disease_name="non-small cell lung cancer"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        result = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out",
            quiet=True,
        )
        assert result["programs"] == 1

    def test_substring_match(self, tmp_path):
        programs = [
            _make_program("DrugA", disease_name="type 2 diabetes mellitus"),
            _make_program("DrugB", disease_name="Type 2 Diabetes Mellitus, severe"),
            _make_program("DrugC", disease_name="other disease"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        result = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes",
            output_dir=tmp_path / "out",
            quiet=True,
        )
        assert result["programs"] == 2

    def test_no_match_raises(self, tmp_path):
        programs = [_make_program("DrugA", disease_name="pancreatic cancer")]
        artifact_dir = _fixture_dir_with_programs(programs)
        with pytest.raises(ValueError, match="No programs found"):
            generate_map(
                input_dir=artifact_dir,
                disease="type 2 diabetes mellitus",
                output_dir=tmp_path / "out",
                quiet=True,
            )


# ---------------------------------------------------------------------------
# 3. Map data structure
# ---------------------------------------------------------------------------


class TestBuildMapData:
    def test_lanes_sorted_by_count_descending(self):
        programs = (
            [_make_program(f"A{i}", mechanism_class="KRAS inhibitor") for i in range(3)]
            + [_make_program(f"B{i}", mechanism_class="PARP inhibitor") for i in range(5)]
            + [_make_program(f"C{i}", mechanism_class=None) for i in range(10)]
        )
        md = build_map_data(programs, "test disease", {})
        assert md["lanes"][0] == "PARP inhibitor"
        assert md["lanes"][1] == "KRAS inhibitor"
        assert md["lanes"][-1] == "Unknown Mechanism"

    def test_unknown_mechanism_lane_always_last(self):
        programs = [_make_program("A", mechanism_class="JAK inhibitor")] + [
            _make_program(f"U{i}", mechanism_class=None) for i in range(5)
        ]
        md = build_map_data(programs, "test disease", {})
        assert md["lanes"][-1] == "Unknown Mechanism"

    def test_unknown_mechanism_lane_present_when_all_unknown(self):
        programs = [_make_program(f"X{i}", mechanism_class=None) for i in range(4)]
        md = build_map_data(programs, "test disease", {})
        assert "Unknown Mechanism" in md["lanes"]

    def test_columns_include_active_stages_only(self):
        programs = [
            _make_program("A", clinical_stage="phase2"),
            _make_program("B", clinical_stage="phase3"),
        ]
        md = build_map_data(programs, "test disease", {})
        assert "phase2" in md["columns"]
        assert "phase3" in md["columns"]
        assert "phase1" not in md["columns"]

    def test_unknown_stage_column_included(self):
        programs = [
            _make_program("A", clinical_stage="phase2"),
            _make_program("B", clinical_stage=None),
        ]
        md = build_map_data(programs, "test disease", {})
        assert "unknown" in md["columns"]

    def test_cells_sorted_by_confidence(self):
        programs = [
            _make_program("LowConf", clinical_stage="phase2", confidence=0.1),
            _make_program("HighConf", clinical_stage="phase2", confidence=0.9),
        ]
        md = build_map_data(programs, "test disease", {})
        unknown_cell = md["cells"]["Unknown Mechanism"]["phase2"]
        assert unknown_cell[0]["asset_name"] == "HighConf"

    def test_stage_order_preserved_in_columns(self):
        programs = [
            _make_program("A", clinical_stage="phase3"),
            _make_program("B", clinical_stage="phase1"),
            _make_program("C", clinical_stage="phase2"),
        ]
        md = build_map_data(programs, "test disease", {})
        cols = md["columns"]
        assert cols.index("phase1") < cols.index("phase2") < cols.index("phase3")

    def test_metadata_fields_present(self):
        programs = [_make_program("A")]
        md = build_map_data(programs, "type 2 diabetes mellitus", {"as_of_date": "2026-06-23"})
        meta = md["metadata"]
        assert meta["disease_name"] == "type 2 diabetes mellitus"
        assert meta["disease_slug"] == "type-2-diabetes-mellitus"
        assert "governance" in meta
        assert meta["governance"]["read_only_diagnostic"] is True
        assert "NOT AN INVESTMENT RECOMMENDATION" in meta["governance"]["disclaimer"]

    def test_summary_coverage_stats(self):
        programs = [
            _make_program("A", clinical_stage="phase2", mechanism_class="JAK inhibitor"),
            _make_program("B", clinical_stage=None, mechanism_class=None),
            _make_program("C", clinical_stage="phase3", mechanism_class=None, ticker="BIOX"),
        ]
        md = build_map_data(programs, "test disease", {})
        summ = md["summary"]
        assert summ["total_programs"] == 3
        assert summ["mechanism_coverage_pct"] == pytest.approx(33.3, 0.1)
        assert summ["ticker_coverage_pct"] == pytest.approx(33.3, 0.1)

    def test_sparse_mechanism_warning_emitted(self):
        programs = [_make_program(f"X{i}", mechanism_class=None) for i in range(20)]
        md = build_map_data(programs, "test disease", {})
        assert any("Mechanism coverage is sparse" in w for w in md["warnings"])

    def test_deterministic_output_same_inputs(self):
        programs = [
            _make_program("C", clinical_stage="phase2"),
            _make_program("A", clinical_stage="phase3"),
            _make_program("B", clinical_stage="phase1"),
        ]
        md1 = build_map_data(programs, "test disease", {})
        md2 = build_map_data(programs, "test disease", {})
        assert md1["columns"] == md2["columns"]
        assert md1["lanes"] == md2["lanes"]

    def test_therapeutic_area_in_metadata(self):
        programs = [_make_program("A", therapeutic_area="Oncology")]
        md = build_map_data(programs, "test disease", {})
        assert md["metadata"]["therapeutic_area"] == "Oncology"

    def test_therapeutic_area_none_when_absent(self):
        programs = [_make_program("A", therapeutic_area=None)]
        md = build_map_data(programs, "test disease", {})
        assert md["metadata"]["therapeutic_area"] is None


# ---------------------------------------------------------------------------
# 4. SVG output
# ---------------------------------------------------------------------------


class TestRenderSVG:
    def test_svg_produced(self):
        programs = [_make_program("DrugA"), _make_program("DrugB", clinical_stage="phase3")]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        assert svg.startswith("<svg")
        assert svg.strip().endswith("</svg>")

    def test_unknown_mechanism_lane_in_svg(self):
        programs = [_make_program("DrugA", mechanism_class=None)]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        assert "Unknown Mechanism" in svg

    def test_unknown_stage_column_in_svg_when_present(self):
        programs = [_make_program("DrugA", clinical_stage=None)]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        assert "UNKNOWN STAGE" in svg

    def test_overflow_label_present(self):
        programs = [_make_program(f"Drug{i}", clinical_stage="phase2") for i in range(8)]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        assert "more" in svg

    def test_no_recommendation_language_in_svg(self):
        programs = [_make_program("DrugA")]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        for bad_word in ["ranking", "score", "buy", "sell", "recommendation"]:
            assert bad_word.lower() not in svg.lower()


# ---------------------------------------------------------------------------
# 5. HTML output
# ---------------------------------------------------------------------------


class TestRenderHTML:
    def test_governance_header_present(self):
        programs = [_make_program("DrugA")]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        html = render_html(md, svg)
        assert "DIAGNOSTIC ONLY" in html
        assert "NOT AN INVESTMENT RECOMMENDATION" in html

    def test_no_action_language_in_html(self):
        """HTML must not contain trade-action or production-model language."""
        programs = [_make_program("DrugA")]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        html = render_html(md, svg)
        for bad_phrase in ["buy ", "sell ", "final_score", " sizing", "trade now"]:
            assert bad_phrase.lower() not in html.lower(), f"Forbidden phrase {bad_phrase!r} found in HTML"

    def test_therapeutic_area_appears_if_present(self):
        programs = [_make_program("DrugA", therapeutic_area="Oncology")]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        html = render_html(md, svg)
        assert "Oncology" in html

    def test_warning_banner_appears_for_sparse_mechanism(self):
        programs = [_make_program(f"X{i}", mechanism_class=None) for i in range(20)]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        html = render_html(md, svg)
        assert "warning" in html.lower()

    def test_html_self_contained_no_external_cdn(self):
        programs = [_make_program("DrugA")]
        md = build_map_data(programs, "test disease", {})
        svg = render_svg(md)
        html = render_html(md, svg)
        # No external URLs
        for external in ["cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"]:
            assert external not in html


# ---------------------------------------------------------------------------
# 5b. Poster layout (v0.3)
# ---------------------------------------------------------------------------


class TestRenderHTMLV03Poster:
    """Tests for v0.3 poster-style layout sections."""

    def _make_html(self, programs=None, disease="test disease"):
        if programs is None:
            programs = [
                _make_program("DrugA", clinical_stage="phase2", mechanism_class="SGLT2 inhibitor"),
                _make_program("DrugB", clinical_stage="phase3", mechanism_class=None),
                _make_program("DrugC", clinical_stage="approved", therapeutic_area="Metabolic"),
            ]
        md = build_map_data(programs, disease, {})
        svg = render_svg(md)
        return render_html(md, svg)

    def test_poster_header_present(self):
        assert "poster-header" in self._make_html()

    def test_disease_name_in_header(self):
        assert "Type 2 Diabetes Mellitus" in self._make_html(disease="Type 2 Diabetes Mellitus")

    def test_left_rail_present(self):
        assert "left-rail" in self._make_html()

    def test_center_panel_present(self):
        assert "center-panel" in self._make_html()

    def test_right_rail_present(self):
        assert "right-rail" in self._make_html()

    def test_poster_footer_present(self):
        assert "poster-footer" in self._make_html()

    def test_stage_distribution_in_right_rail(self):
        programs = [
            _make_program("A", clinical_stage="phase3"),
            _make_program("B", clinical_stage="approved"),
        ]
        html = self._make_html(programs=programs)
        assert "Phase 3" in html
        assert "Approved" in html

    def test_mechanism_lanes_in_right_rail(self):
        programs = [
            _make_program("A", mechanism_class="SGLT2 inhibitor"),
            _make_program("B", mechanism_class="Biguanide"),
        ]
        html = self._make_html(programs=programs)
        assert "SGLT2 inhibitor" in html
        assert "Biguanide" in html

    def test_coverage_stats_in_left_rail(self):
        html = self._make_html()
        assert "Coverage" in html
        assert "Stage" in html
        assert "Mechanism" in html

    def test_pipeline_stats_in_left_rail(self):
        assert "Pipeline" in self._make_html()

    def test_caveats_section_in_left_rail(self):
        assert "Caveats" in self._make_html()

    def test_data_provenance_in_left_rail(self):
        html = self._make_html()
        assert "Provenance" in html or "provenance" in html

    def test_governance_banner_still_present(self):
        html = self._make_html()
        assert "gov-banner" in html
        assert "DIAGNOSTIC ONLY" in html
        assert "NOT AN INVESTMENT RECOMMENDATION" in html

    def test_no_cdn_in_poster(self):
        html = self._make_html()
        for cdn in ["cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"]:
            assert cdn not in html

    def test_no_action_language_in_poster(self):
        html = self._make_html()
        for phrase in ["buy ", "sell ", "final_score", " sizing", "trade now"]:
            assert phrase.lower() not in html.lower()

    def test_svg_embedded_in_center_panel(self):
        html = self._make_html()
        assert "map-viewport" in html
        assert "<svg" in html

    def test_legend_in_center_panel(self):
        assert "legend-strip" in self._make_html()

    def test_warnings_appear_when_sparse_mechanism(self):
        programs = [_make_program(f"X{i}") for i in range(20)]
        html = self._make_html(programs=programs)
        assert "warning-block" in html or "warning" in html.lower()


# ---------------------------------------------------------------------------
# 6. Full generation
# ---------------------------------------------------------------------------


class TestGenerateMap:
    def test_all_four_files_created(self, tmp_path):
        programs = [
            _make_program("DrugA", clinical_stage="phase2"),
            _make_program("DrugB", clinical_stage="phase3"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        assert (out_dir / "index.html").exists()
        assert (out_dir / "map.svg").exists()
        assert (out_dir / "map.json").exists()
        assert (out_dir / "README.md").exists()

    def test_map_json_valid_schema(self, tmp_path):
        programs = [_make_program("DrugA"), _make_program("DrugB")]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        with open(out_dir / "map.json") as f:
            md = json.load(f)
        assert "metadata" in md
        assert "summary" in md
        assert "lanes" in md
        assert "columns" in md
        assert "cells" in md
        assert "warnings" in md

    def test_svg_not_empty(self, tmp_path):
        programs = [_make_program("DrugA", clinical_stage="phase2")]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        svg_content = (out_dir / "map.svg").read_text()
        assert "<svg" in svg_content
        assert len(svg_content) > 200

    def test_result_dict_shape(self, tmp_path):
        programs = [_make_program("DrugA"), _make_program("DrugB")]
        artifact_dir = _fixture_dir_with_programs(programs)
        result = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out",
            quiet=True,
        )
        assert result["programs"] == 2
        assert "lanes" in result
        assert "columns" in result
        assert "warnings" in result


# ---------------------------------------------------------------------------
# 7. D3: Non-drug filter
# ---------------------------------------------------------------------------


class TestNonDrugFilter:
    def test_exercise_filtered(self):
        programs = [_make_program("Exercise")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0
        assert meta["filtered_count"] == 1

    def test_diet_filtered(self):
        programs = [_make_program("Diet")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_aerobic_training_filtered(self):
        programs = [_make_program("aerobic training, tobacco cessation")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_tobacco_cessation_filtered(self):
        programs = [_make_program("tobacco cessation and nutritional advice")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_no_treatment_filtered(self):
        programs = [_make_program("No treatment given")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_comparison_of_eating_filtered(self):
        programs = [_make_program("Comparison of eating windows intervention")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_watchful_waiting_filtered(self):
        programs = [_make_program("Watchful waiting")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_metformin_not_filtered(self):
        programs = [_make_program("Metformin")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 1
        assert meta["filtered_count"] == 0

    def test_dapagliflozin_not_filtered(self):
        programs = [_make_program("Dapagliflozin")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 1

    def test_mixed_input_correct_counts(self):
        programs = [
            _make_program("Exercise"),
            _make_program("Metformin"),
            _make_program("Diet"),
            _make_program("Saxagliptin"),
        ]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 2
        assert meta["filtered_count"] == 2

    def test_examples_in_meta(self):
        programs = [_make_program("aerobic training"), _make_program("Metformin")]
        kept, meta = _filter_non_drug_programs(programs)
        assert "aerobic training" in meta["examples"]

    def test_case_insensitive(self):
        programs = [_make_program("EXERCISE"), _make_program("Aerobic Training")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0
        assert meta["filtered_count"] == 2


# ---------------------------------------------------------------------------
# 8. D1: Asset deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_asset_company_keeps_highest_stage(self):
        programs = [
            _make_program("DrugA", clinical_stage="phase1"),
            _make_program("DrugA", clinical_stage="phase3"),
        ]
        result = _deduplicate_programs(programs)
        assert len(result) == 1
        assert result[0]["clinical_stage"] == "phase3"

    def test_different_assets_not_merged(self):
        programs = [
            _make_program("DrugA", clinical_stage="phase2"),
            _make_program("DrugB", clinical_stage="phase2"),
        ]
        result = _deduplicate_programs(programs)
        assert len(result) == 2

    def test_approved_beats_phase3(self):
        programs = [
            _make_program("DrugA", clinical_stage="phase3"),
            _make_program("DrugA", clinical_stage="approved"),
            _make_program("DrugA", clinical_stage="phase1"),
        ]
        result = _deduplicate_programs(programs)
        assert len(result) == 1
        assert result[0]["clinical_stage"] == "approved"

    def test_trial_count_recorded(self):
        programs = [
            _make_program("DrugA", clinical_stage="phase1"),
            _make_program("DrugA", clinical_stage="phase2"),
            _make_program("DrugA", clinical_stage="phase3"),
        ]
        result = _deduplicate_programs(programs)
        assert result[0]["trial_count"] == 3

    def test_source_refs_merged(self):
        p1 = _make_program("DrugA", clinical_stage="phase1")
        p2 = _make_program("DrugA", clinical_stage="phase2")
        p1["source_refs"] = ["NCT001"]
        p2["source_refs"] = ["NCT002", "NCT003"]
        result = _deduplicate_programs([p1, p2])
        assert len(result) == 1
        refs = result[0]["source_refs"]
        assert "NCT001" in refs
        assert "NCT002" in refs
        assert "NCT003" in refs

    def test_unknown_stage_loses_to_phase1(self):
        programs = [
            _make_program("DrugA", clinical_stage=None),
            _make_program("DrugA", clinical_stage="phase1"),
        ]
        result = _deduplicate_programs(programs)
        assert result[0]["clinical_stage"] == "phase1"

    def test_different_companies_not_merged(self):
        p1 = _make_program("DrugA", clinical_stage="phase2")
        p2 = dict(_make_program("DrugA", clinical_stage="phase3"))
        p2["company_name"] = "Different Bio"
        result = _deduplicate_programs([p1, p2])
        assert len(result) == 2

    def test_ticker_preserved_from_best_record(self):
        p1 = _make_program("DrugA", clinical_stage="phase2", ticker="ACME")
        p2 = _make_program("DrugA", clinical_stage="phase1")
        result = _deduplicate_programs([p1, p2])
        assert len(result) == 1
        assert result[0]["ticker"] == "ACME"


# ---------------------------------------------------------------------------
# 9. v0.2c integration: D1+D3 in generate_map
# ---------------------------------------------------------------------------


class TestGenerateMapV02c:
    def test_non_drug_filtered_out(self, tmp_path):
        programs = [
            _make_program("Metformin", clinical_stage="phase3"),
            _make_program("Exercise"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        result = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out",
            quiet=True,
        )
        assert result["programs"] == 1

    def test_duplicates_deduped(self, tmp_path):
        programs = [
            _make_program("Metformin", clinical_stage="phase1"),
            _make_program("Metformin", clinical_stage="phase3"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        result = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out",
            quiet=True,
        )
        assert result["programs"] == 1

    def test_dedup_keeps_highest_stage_in_map_json(self, tmp_path):
        programs = [
            _make_program("Metformin", clinical_stage="phase1"),
            _make_program("Metformin", clinical_stage="phase3"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        with open(out_dir / "map.json") as f:
            md = json.load(f)
        assert "phase3" in md["columns"]
        assert "phase1" not in md["columns"]

    def test_non_drug_warning_in_map_json(self, tmp_path):
        programs = [
            _make_program("Metformin", clinical_stage="phase3"),
            _make_program("Exercise"),
            _make_program("Diet"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        with open(out_dir / "map.json") as f:
            md = json.load(f)
        warnings_text = " ".join(md["warnings"]).lower()
        assert "non-pharmaceutical" in warnings_text or "filtered" in warnings_text

    def test_preprocessing_counts_in_summary(self, tmp_path):
        programs = [
            _make_program("Metformin", clinical_stage="phase1"),
            _make_program("Metformin", clinical_stage="phase3"),
            _make_program("Exercise"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        with open(out_dir / "map.json") as f:
            md = json.load(f)
        summ = md["summary"]
        assert summ["raw_program_count"] == 3
        assert summ["non_drug_filtered_count"] == 1
        assert summ["deduped_program_count"] == 1
        assert summ["total_programs"] == 1


# ---------------------------------------------------------------------------
# 10. v0.2d: Expanded D3 filter (devices, instruments, diagnostics)
# ---------------------------------------------------------------------------


class TestNonDrugFilterV02d:
    def test_questionnaire_filtered(self):
        programs = [_make_program("Questionnaire: patient outcomes")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_withings_device_filtered(self):
        programs = [_make_program("Withings BPM Connect")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_withings_body_filtered(self):
        programs = [_make_program("Withings Body+")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_mems_cap_filtered(self):
        programs = [_make_program("MEMS (Medication Electronic Monitoring System) Cap")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_ogtt_exact_filtered(self):
        programs = [_make_program("OGTT")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_oral_glucose_tolerance_filtered(self):
        programs = [_make_program("Oral Glucose Tolerance Test")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_blood_glucose_meter_filtered(self):
        programs = [_make_program("Comparison of different Blood Glucose Meters")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_dose_regimen_filtered(self):
        programs = [_make_program("0.5 units/kg daily insulin")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_dose_regimen_integer_filtered(self):
        programs = [_make_program("1 unit/kg basal insulin")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_saxagliptin_not_filtered(self):
        programs = [_make_program("Saxagliptin")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 1

    def test_hem_col_device_filtered(self):
        programs = [_make_program("Hem-Col Capillary Blood Collection Device")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0

    def test_basal_bolus_filtered(self):
        programs = [_make_program("Basal Bolus")]
        kept, meta = _filter_non_drug_programs(programs)
        assert len(kept) == 0


# ---------------------------------------------------------------------------
# 11. v0.2d: Asset-name canonicalization
# ---------------------------------------------------------------------------


class TestAssetCanonicalization:
    def test_insulin_prefix_stripped(self):
        assert _canonical_asset_name("insulin glargine") == "glargine"

    def test_insulin_prefix_case_insensitive(self):
        assert _canonical_asset_name("Insulin Glargine") == "glargine"

    def test_glargine_without_prefix_unchanged(self):
        assert _canonical_asset_name("Glargine") == "glargine"

    def test_insulin_alone_unchanged(self):
        assert _canonical_asset_name("insulin") == "insulin"

    def test_via_pen_suffix_stripped(self):
        assert _canonical_asset_name("glargine via insulin pen") == "glargine"

    def test_dose_suffix_stripped(self):
        assert _canonical_asset_name("Glargine 300 U/mL") == "glargine"

    def test_insulin_glargine_dose_collapses(self):
        assert _canonical_asset_name("insulin glargine 300 U/mL") == _canonical_asset_name("insulin glargine")

    def test_metformin_unchanged(self):
        assert _canonical_asset_name("Metformin") == "metformin"

    def test_dapagliflozin_10mg_tab_collapses(self):
        assert _canonical_asset_name("Dapagliflozin 10mg Tab") == "dapagliflozin"

    def test_dapagliflozin_10mg_collapses(self):
        assert _canonical_asset_name("dapagliflozin 10 mg") == "dapagliflozin"

    def test_canonical_dedup_merges_insulin_variants(self):
        programs = [
            _make_program("insulin glargine", clinical_stage="phase1"),
            _make_program("Insulin Glargine", clinical_stage="phase3"),
            _make_program("glargine", clinical_stage="phase2"),
        ]
        result = _deduplicate_programs(programs)
        assert len(result) == 1
        assert result[0]["clinical_stage"] == "phase3"
        assert result[0]["trial_count"] == 3

    def test_display_name_preserved_from_best_record(self):
        """Canonical key collapses but the original asset_name from best record is kept."""
        programs = [
            _make_program("insulin glargine", clinical_stage="phase1"),
            _make_program("Insulin Glargine", clinical_stage="phase3"),
        ]
        result = _deduplicate_programs(programs)
        assert len(result) == 1
        assert result[0]["asset_name"] in ("Insulin Glargine", "insulin glargine")

    def test_combination_product_stays_distinct(self):
        """Glargine/lixisenatide is a different drug from glargine alone."""
        programs = [
            _make_program("insulin glargine/lixisenatide", clinical_stage="phase3"),
            _make_program("insulin glargine", clinical_stage="phase3"),
        ]
        result = _deduplicate_programs(programs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 12. v0.2d integration: full pipeline in generate_map
# ---------------------------------------------------------------------------


class TestGenerateMapV02d:
    def test_canonicalization_merge_count_in_summary(self, tmp_path):
        # "insulin glargine" and "glargine" have different raw-lowercase keys
        # but the same canonical key → canonicalization merge detected.
        programs = [
            _make_program("insulin glargine", clinical_stage="phase1"),
            _make_program("glargine", clinical_stage="phase3"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        out_dir = tmp_path / "out"
        generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=out_dir,
            quiet=True,
        )
        with open(out_dir / "map.json") as f:
            md = json.load(f)
        assert md["summary"]["canonicalization_merge_count"] >= 1
        assert md["summary"]["total_programs"] == 1

    def test_device_and_behavioral_removed(self, tmp_path):
        programs = [
            _make_program("Metformin", clinical_stage="phase3"),
            _make_program("Withings BPM Connect"),
            _make_program("Exercise"),
            _make_program("Questionnaire: patient outcomes"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        result = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out",
            quiet=True,
        )
        assert result["programs"] == 1

    def test_deterministic_output(self, tmp_path):
        programs = [
            _make_program("insulin glargine", clinical_stage="phase1"),
            _make_program("Insulin Glargine", clinical_stage="phase3"),
            _make_program("Metformin", clinical_stage="phase2"),
        ]
        artifact_dir = _fixture_dir_with_programs(programs)
        result1 = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out1",
            quiet=True,
        )
        result2 = generate_map(
            input_dir=artifact_dir,
            disease="type 2 diabetes mellitus",
            output_dir=tmp_path / "out2",
            quiet=True,
        )
        assert result1["programs"] == result2["programs"]
        assert result1["lanes"] == result2["lanes"]
        assert result1["columns"] == result2["columns"]
