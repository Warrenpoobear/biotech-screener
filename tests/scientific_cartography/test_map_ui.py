"""UI tests for the cartography dashboard templates and the poster/map emitter.

Covers the P1-P3 fixes: HTML/SVG escaping, nav active-state, Blocking column,
real ticker-linked count, SVG accessibility attributes + node tooltips, and
acronym-preserving disease title-casing.
"""

import importlib.util
from pathlib import Path

from scientific_cartography.dashboard_static import templates

_REPO = Path(__file__).resolve().parents[2]

NAV = [
    ("Index", "index.html"),
    ("Review Runs", "review_runs.html"),
    ("Disease Maps", "disease_maps.html"),
    ("Human Decisions", "human_decisions.html"),
    ("Scheduled Review", "scheduled_review_health.html"),
    ("Governance", "governance.html"),
]


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gen = _load("generate_scientific_cartography_map", "tools/generate_scientific_cartography_map.py")


def _programs():
    return [
        {
            "asset_name": "DrugA",
            "ticker": "LLY",
            "modality": "small molecule",
            "mechanism_class": "GLP-1",
            "clinical_stage": "phase3",
            "confidence": 0.8,
            "disease_name": "type 2 diabetes",
            "mondo_id": "MONDO:0005148",
            "therapeutic_area": "Metabolic",
        },
        {
            "asset_name": "DrugB",
            "ticker": "",
            "modality": "monoclonal antibody",
            "mechanism_class": "",
            "clinical_stage": "phase1",
            "confidence": 0.0,
            "disease_name": "type 2 diabetes",
        },
        {
            "asset_name": "DrugC",
            "ticker": "VRTX",
            "modality": "small molecule",
            "mechanism_class": "GLP-1",
            "clinical_stage": "phase2",
            "confidence": 0.5,
            "disease_name": "type 2 diabetes",
        },
    ]


# ----- Dashboard (P1 escaping + P2 nav + P2/P3 blocking) -----


def test_human_decisions_escapes_markup():
    decisions = [
        {
            "created_at_utc": "2026-06-28T00:00:00Z",
            "decision_state": "s<1",
            "decision_actor": "a&b",
            "decision_reason": "<script>alert(1)</script>",
            "review_continuation_approved": True,
            "automation_approval": False,
        }
    ]
    out = templates.human_decisions_template(decisions, NAV)
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "a&amp;b" in out
    assert "s&lt;1" in out


def test_review_runs_escapes_values():
    out = templates.review_runs_template("/a<b", {"decision": "<b>x</b>"}, NAV)
    assert "&lt;b&gt;x&lt;/b&gt;" in out
    assert "/a&lt;b" in out


def test_scheduled_review_nav_active_and_blocking():
    execs = [
        {
            "executed_at_utc": "2026-06-28T00:00:00Z",
            "outcome": "failure",
            "duration_seconds": 1.2,
            "error_message": "E" * 250,
            "governance": {"non_blocking": True},
        }
    ]
    out = templates.scheduled_review_health_template(execs, NAV)
    # nav active-state now matches (was "Scheduled Review Health")
    assert 'class="active">Scheduled Review</a>' in out
    # Exit Code column replaced by honest Blocking column
    assert "<th>Blocking</th>" in out
    assert "0 (always)" not in out
    assert "No (non-blocking)" in out
    # long error gets an ellipsis
    assert "\u2026" in out


def test_scheduled_review_short_error_no_ellipsis():
    execs = [
        {
            "executed_at_utc": "t",
            "outcome": "success",
            "duration_seconds": 0.1,
            "error_message": "short",
            "governance": {"non_blocking": False},
        }
    ]
    out = templates.scheduled_review_health_template(execs, NAV)
    assert "short" in out
    assert "short\u2026" not in out
    assert "Yes" in out  # non_blocking False -> blocking Yes


def test_index_template_has_nav():
    out = templates.index_template("/art", "2026-06-28", [("Index", "index.html", "overview")], [], [], NAV)
    assert "<nav>" in out
    assert 'class="active">Index</a>' in out


# ----- Poster / map emitter (P2 count + P2 a11y + P2 title-case) -----


def test_summary_exports_ticker_linked_count():
    md = gen.build_map_data(_programs(), "type 2 diabetes", {"as_of_date": "2026-06-28"})
    assert md["summary"]["ticker_linked_count"] == 2
    html = gen.render_html(md, gen.render_svg(md))
    # Unlinked tickers now uses the real count (3 - 2 = 1), not a rounded %
    assert "<span>Unlinked tickers</span><b>1</b>" in html


def test_svg_a11y_and_node_titles():
    md = gen.build_map_data(_programs(), "type 2 diabetes", {"as_of_date": "2026-06-28"})
    svg = gen.render_svg(md)
    assert 'role="img"' in svg
    assert 'aria-label="Disease landscape map: type 2 diabetes"' in svg
    assert "<title>Disease landscape map: type 2 diabetes</title>" in svg
    # node tooltip carries the full (untruncated) asset name + ticker
    assert "<title>DrugA (LLY)" in svg


def test_display_disease_preserves_acronyms():
    assert gen._display_disease("NSCLC trial") == "NSCLC Trial"
    assert gen._display_disease("type 2 diabetes mellitus") == "Type 2 Diabetes Mellitus"
    assert gen._display_disease("HER2 positive") == "HER2 Positive"
    assert gen._display_disease("COPD") == "COPD"
