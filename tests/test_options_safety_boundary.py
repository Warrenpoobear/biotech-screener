"""Stage 1 safety-boundary tests — options diagnostics must stay shadow-only.

Spec: options diagnostics (opt_*, straddle_price, priced_move_pct, and the
options_quality.py / options_diagnostics.py / massive_chain_analytics.py /
options_history_massive.py / options_snapshot.py overlay) are POLICY-CLOSED
for alpha/selector/ranker/sizing/portfolio use. This file proves (and pins)
the current boundary so future edits cannot silently reopen it.

IMPORTANT — verified discrepancy vs. the "zero leakage" framing of this task:
    `ranker_engine.py` (the clinical_50 ranker) and `ranker_v2_pairwise.py`
    (the pairwise ranker module) BOTH contain live references to options
    diagnostic fields (`opt_rr_25d`, `opt_event_premium`, `opt_iv_regime`,
    `opt_has_data`, `cheap_vol_score`, `ovf_composite`/`ovf11_score`):

      - ranker_engine.py: `RankerConfig.options_signals` is a *live, non-zero*
        5% block (`options_weight=0.05`) in `DEFAULT_RANKER_CONFIG`. The
        function `compute_ranker_adjustments()` actively computes this block
        whenever it is called.
      - ranker_v2_pairwise.py: `BLOCK_OPTIONS` references the same opt_*
        fields, and is included in `get_feature_specs()` for the "expanded"
        feature set and for any "ablation_drop_X" set other than "options".

    However, in `run_screen.py`'s actual production wiring (`ranker_mode`
    defaults to "pairwise_minimal", confirmed at run_screen.py:4020,4469):
      - The clinical_50 path (`compute_ranker_adjustments`, called at
        run_screen.py:5844) is relegated to *shadow-only* logging
        (`_shadow_clinical50`, run_screen.py:5916-5923) and its output is
        NEVER written to `final_score` — final_score instead comes from the
        pairwise v2 path (run_screen.py:5895).
      - The pairwise v2 path uses `PRODUCTION_RANKER_V2_CONFIG`
        (run_screen.py:168), which pins `feature_set="minimal_v2"`. Per
        `get_feature_specs()`, "minimal_v2" resolves to
        `FEATURES_MINIMAL_V2 = (coinvest_score_z, financial_score)` only —
        zero options fields. This is also confirmed by the deployed model
        artifact `production_data/ranker_v2_model.json`
        (`config.feature_set == "minimal_v2"`,
        `model.feature_names == ["coinvest_score_z", "financial_score"]`).

    Net: in the CURRENT DEFAULT production configuration, no options field
    reaches `final_score`. But this is a *configuration* fact, not a
    *structural* one — the options-consuming code paths exist live in the
    repo and would activate if `ranker_mode` were switched to "clinical_50"
    (legacy fallback) or if `ranker_v2_pairwise` were reconfigured to
    "expanded"/non-options-ablation feature sets. Tests below pin BOTH:
      (a) the structural fact (which modules/functions reference opt_*), and
      (b) the production-configuration fact (what the deployed config
          actually selects) — so a future change to either is caught.

selector_engine.py, portfolio_risk_layer.py, portfolio_vol_corr_layer.py,
and event_ev/portfolio_sizing.py have ZERO references to any options
diagnostic field at the source level (verified by grep below) — this part
of the audit's "zero leakage" framing IS accurate.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Options diagnostic field name patterns. Matches opt_* columns,
# straddle_price, priced_move_pct, ovf_composite/ovf11_score (options
# verdict features), cheap_vol_score / vol_classification (straddle
# mispricing overlay), and the options_quality.py / options_diagnostics.py
# manifest field prefixes (options_data_state etc).
_OPTIONS_FIELD_PATTERN = re.compile(
    r"\bopt_[a-zA-Z_]+\b"
    r"|\bstraddle_price\b"
    r"|\bpriced_move_pct\b"
    r"|\bovf_composite\b"
    r"|\bovf11_score\b"
    r"|\bcheap_vol_score\b"
    r"|\bvol_classification\b"
    r"|\boptions_data_state\b"
    r"|\boptions_chain_quality_score\b"
    r"|\boptions_quality_composite\b"
)

# Files that are genuinely "production decision" surfaces per the task scope.
# These must NEVER reference an options diagnostic field.
_CLEAN_PRODUCTION_FILES = [
    REPO_ROOT / "selector_engine.py",
    REPO_ROOT / "portfolio_risk_layer.py",
    REPO_ROOT / "portfolio_vol_corr_layer.py",
    REPO_ROOT / "event_ev" / "portfolio_sizing.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. selector_engine / portfolio_* must be options-clean (structural)
# ---------------------------------------------------------------------------


class TestCleanProductionFilesHaveNoOptionsFields:
    """selector_engine.py, portfolio_risk_layer.py, portfolio_vol_corr_layer.py,
    and event_ev/portfolio_sizing.py must contain zero references to any
    options diagnostic field, at the source-text level.
    """

    @pytest.mark.parametrize("path", _CLEAN_PRODUCTION_FILES, ids=lambda p: p.name)
    def test_no_options_field_reference(self, path: Path):
        assert path.exists(), f"expected file not found: {path}"
        text = _read(path)
        matches = sorted(set(_OPTIONS_FIELD_PATTERN.findall(text)))
        assert matches == [], (
            f"{path.relative_to(REPO_ROOT)} references options diagnostic field(s) "
            f"{matches} — options data is policy-closed for this module"
        )


# ---------------------------------------------------------------------------
# 2. ranker_engine.py / ranker_v2_pairwise.py — pin the KNOWN exception,
#    don't let it silently grow or silently disappear unnoticed.
# ---------------------------------------------------------------------------


class TestRankerModulesOptionsReferenceIsPinned:
    """ranker_engine.py and ranker_v2_pairwise.py DO reference options
    fields (see module docstring above). Pin the known reference set so any
    *new* options field finding its way into either ranker raises a flag for
    review, rather than silently expanding the surface.
    """

    _KNOWN_RANKER_ENGINE_OPTIONS_FIELDS = {
        "opt_rr_25d",
        "opt_event_premium",
        "opt_iv_regime",
        "opt_has_data",
        "ovf_composite",
        "cheap_vol_score",
        # Local variable names derived from opt_has_data (not separate
        # field references): `opt_raw = row.get("opt_has_data", "")` and
        # `opt_has = float(opt_raw) ...` in _check_activation_gate().
        "opt_has",
        "opt_raw",
    }

    _KNOWN_RANKER_V2_OPTIONS_FIELDS = {
        "opt_rr_25d",
        "opt_event_premium",
        "opt_iv_regime",
        "opt_term_slope",
        "ovf_composite",
        "ovf11_score",
        "cheap_vol_score",
    }

    def test_ranker_engine_options_fields_match_known_set(self):
        text = _read(REPO_ROOT / "ranker_engine.py")
        found = set(_OPTIONS_FIELD_PATTERN.findall(text))
        unexpected = found - self._KNOWN_RANKER_ENGINE_OPTIONS_FIELDS
        assert unexpected == set(), (
            f"ranker_engine.py references NEW options field(s) not in the pinned "
            f"known set: {unexpected}. This expands the options-in-ranker surface "
            f"and is out of scope for this diagnostic-repair task — review required."
        )

    def test_ranker_v2_pairwise_options_fields_match_known_set(self):
        text = _read(REPO_ROOT / "ranker_v2_pairwise.py")
        found = set(_OPTIONS_FIELD_PATTERN.findall(text))
        unexpected = found - self._KNOWN_RANKER_V2_OPTIONS_FIELDS
        assert unexpected == set(), (
            f"ranker_v2_pairwise.py references NEW options field(s) not in the "
            f"pinned known set: {unexpected}. Review required."
        )

    def test_ranker_engine_options_block_weight_is_unchanged_nonzero(self):
        """Pin the known (non-zero, shadow-irrelevant under prod config)
        options_weight in RankerConfig — a change here changes how much
        clinical_50 (shadow) leans on options; if clinical_50 is ever
        promoted to production this number matters."""
        from ranker_engine import DEFAULT_RANKER_CONFIG

        assert DEFAULT_RANKER_CONFIG.options_weight == pytest.approx(0.05)

    def test_ranker_v2_minimal_v2_feature_set_excludes_options(self):
        """Pin: the production feature_set ('minimal_v2') used by
        PRODUCTION_RANKER_V2_CONFIG must resolve to a feature list that
        contains zero options fields."""
        from ranker_v2_pairwise import FEATURES_MINIMAL_V2, RankerV2Config, get_feature_specs

        cfg = RankerV2Config(feature_set="minimal_v2")
        specs = get_feature_specs(cfg)
        names = {s.name for s in specs}
        options_named = {n for n in names if _OPTIONS_FIELD_PATTERN.search(n)}
        assert options_named == set(), (
            f"minimal_v2 feature set now includes options field(s) {options_named} "
            f"— production ranker would start consuming options data"
        )
        assert specs == list(FEATURES_MINIMAL_V2)


# ---------------------------------------------------------------------------
# 3. Production wiring fact: ranker_mode defaults to pairwise_minimal, and
#    PRODUCTION_RANKER_V2_CONFIG pins feature_set=minimal_v2.
# ---------------------------------------------------------------------------


class TestProductionRankerConfigExcludesOptions:
    """Pin the run_screen.py production configuration constants that
    determine whether options data reaches final_score. These are read
    via source-text parsing (not import — run_screen.py has heavy
    module-level side effects) to avoid accidentally executing the
    production pipeline as an import side-effect.
    """

    def _run_screen_source(self) -> str:
        return _read(REPO_ROOT / "run_screen.py")

    def test_default_ranker_mode_keeps_options_out_of_final_score(self):
        """The default mode must not route clinical_50 (the options-consuming ranker)
        into final_score. Since 2026-11-13 the default is selector_only: final_score is
        the selector score and no ranker output reaches it."""
        from common.ranker_mode import DEFAULT_RANKER_MODE

        assert DEFAULT_RANKER_MODE in ("selector_only", "pairwise_minimal"), (
            "Default ranker_mode is no longer an options-free mode — clinical_50 consumes "
            "live options fields. Changing this default requires re-evaluating the options "
            "policy boundary."
        )
        src = self._run_screen_source()
        assert "ranker_mode: str = DEFAULT_RANKER_MODE" in src
        assert "default=DEFAULT_RANKER_MODE" in src
        assert 'ranker_mode="pairwise_minimal"' not in src and 'ranker_mode="clinical_50"' not in src

    def test_selector_only_branch_never_reads_clinical50_results(self):
        src = self._run_screen_source()
        start = src.index('if ranker_mode == "selector_only":')
        end = src.index('elif ranker_mode == "pairwise_minimal" and _rv2_ok:', start)
        body = src[start:end]
        assert "apply_selector_only_ranking(" in body
        assert "_rnk_results" not in body and "_rr." not in body

    def test_production_ranker_v2_config_pins_minimal_v2(self):
        src = self._run_screen_source()
        # The PRODUCTION_RANKER_V2_CONFIG = RankerV2Config(...) block must
        # set feature_set="minimal_v2" (the options-free feature set).
        m = re.search(
            r"PRODUCTION_RANKER_V2_CONFIG\s*=\s*RankerV2Config\((.*?)\)",
            src,
            re.DOTALL,
        )
        assert m is not None, "Could not locate PRODUCTION_RANKER_V2_CONFIG definition in run_screen.py"
        block = m.group(1)
        assert 'feature_set="minimal_v2"' in block, (
            "PRODUCTION_RANKER_V2_CONFIG.feature_set is no longer 'minimal_v2' — "
            "this is the config that keeps options fields out of the deployed "
            "ranker_v2 score path."
        )

    def test_clinical50_path_is_shadow_only_in_pairwise_minimal_mode(self):
        """Pin that, in the pairwise_minimal branch, _rnk_results (the
        clinical_50/options-consuming ranker output) is only used to build
        the `_shadow_clinical50` logging dict, never written into
        `final_score` for that branch."""
        src = self._run_screen_source()
        tree = ast.parse(src)

        # Find the function containing both `_rnk_results = compute_ranker_adjustments(`
        # and the `if ranker_mode == "pairwise_minimal" and _rv2_ok:` branch, and
        # confirm `_shadow_clinical50` exists as the only place `_rnk_results` is
        # consumed inside that specific branch's lexical region.
        assert "_shadow_clinical50" in src, "Expected shadow-logging dict _shadow_clinical50 not found"
        assert 'ranker_mode == "pairwise_minimal" and _rv2_ok:' in src

        # Within the pairwise_minimal production branch body, final_score is
        # assigned from _rv2 (ranker_v2_score), never from _rr (RankerResult/
        # clinical_50). We check the specific span between the branch's `if`
        # and its matching top-level `else:` for ranker_mode dispatch.
        start = src.index('ranker_mode == "pairwise_minimal" and _rv2_ok:')
        # The dispatch's else branch begins the clinical_50 PRODUCTION fallback;
        # bound our search window there.
        else_marker = "# --- PRODUCTION: clinical_50 (legacy / fallback) ---"
        end = src.index(else_marker, start)
        branch_body = src[start:end]

        assert '_row["final_score"] = _rv2["ranker_v2_score"]' in branch_body
        # No assignment of final_score from a clinical_50 RankerResult (`_rr`)
        # inside the pairwise_minimal production branch body.
        assert '_row["final_score"] = _rr.final_score' not in branch_body


# ---------------------------------------------------------------------------
# 4. priced_move_pct / straddle_price remain diagnostic-only (not consumed
#    by selector/ranker scoring functions as an input signal).
# ---------------------------------------------------------------------------


class TestPricedMoveAndStraddleAreShadowOnly:
    def test_not_referenced_by_selector_engine(self):
        text = _read(REPO_ROOT / "selector_engine.py")
        assert "priced_move_pct" not in text
        assert "straddle_price" not in text

    def test_not_referenced_by_clean_ranker_v2_minimal_features(self):
        from ranker_v2_pairwise import FEATURES_MINIMAL_V2

        names = {s.name for s in FEATURES_MINIMAL_V2}
        assert "priced_move_pct" not in names
        assert "straddle_price" not in names

    def test_not_referenced_by_portfolio_modules(self):
        for path in [
            REPO_ROOT / "portfolio_risk_layer.py",
            REPO_ROOT / "portfolio_vol_corr_layer.py",
            REPO_ROOT / "event_ev" / "portfolio_sizing.py",
        ]:
            text = _read(path)
            assert "priced_move_pct" not in text, f"{path.name} references priced_move_pct"
            assert "straddle_price" not in text, f"{path.name} references straddle_price"


# ---------------------------------------------------------------------------
# 5. run_screen.py code-ordering protection: selector/ranker scoring calls
#    must execute (lexically, in source order — run_screen.py's pipeline
#    function is a long linear sequence of statements, so source order ==
#    execution order here) BEFORE straddle_price / priced_move_pct
#    population. Pure source-inspection; does not import or execute
#    run_screen.py, and does not modify it.
# ---------------------------------------------------------------------------


class TestRunScreenOrderingProtection:
    """Guards the P0-fix invariant described in run_screen.py's own
    `_finalize_priced_move` docstring: "Must run AFTER all options scoring
    ... and BEFORE CSV write" / "This was the source of a P0 execution-order
    bug when it ran too early." If a future edit moves the options-field
    population (straddle_price assignment or the _finalize_priced_move call)
    to before selector/ranker scoring, this test fails loudly.
    """

    @staticmethod
    def _first_line_containing(lines: List[str], needle: str) -> int:
        for i, line in enumerate(lines):
            if needle in line:
                return i
        raise AssertionError(f"Could not find {needle!r} in run_screen.py")

    def test_selector_scoring_precedes_straddle_price_population(self):
        src = _read(REPO_ROOT / "run_screen.py")
        lines = src.splitlines()

        selector_line = self._first_line_containing(lines, "compute_selector_scores(_eligible_for_selector")
        straddle_line = self._first_line_containing(lines, 'row["straddle_price"] =')

        assert selector_line < straddle_line, (
            f"selector scoring call (line {selector_line + 1}) must precede "
            f"straddle_price population (line {straddle_line + 1})"
        )

    def test_ranker_scoring_precedes_straddle_price_population(self):
        src = _read(REPO_ROOT / "run_screen.py")
        lines = src.splitlines()

        ranker_line = self._first_line_containing(lines, "compute_ranker_adjustments(_eligible_for_selector")
        straddle_line = self._first_line_containing(lines, 'row["straddle_price"] =')

        assert ranker_line < straddle_line, (
            f"ranker scoring call (line {ranker_line + 1}) must precede "
            f"straddle_price population (line {straddle_line + 1})"
        )

    def test_selector_and_ranker_scoring_precede_finalize_priced_move_call(self):
        src = _read(REPO_ROOT / "run_screen.py")
        lines = src.splitlines()

        selector_line = self._first_line_containing(lines, "compute_selector_scores(_eligible_for_selector")
        ranker_line = self._first_line_containing(lines, "compute_ranker_adjustments(_eligible_for_selector")
        finalize_call_line = self._first_line_containing(lines, "_finalize_priced_move(csv_rows)")

        assert selector_line < finalize_call_line, (
            f"selector scoring call (line {selector_line + 1}) must precede "
            f"the _finalize_priced_move(csv_rows) call (line {finalize_call_line + 1})"
        )
        assert ranker_line < finalize_call_line, (
            f"ranker scoring call (line {ranker_line + 1}) must precede "
            f"the _finalize_priced_move(csv_rows) call (line {finalize_call_line + 1})"
        )

    def test_finalize_priced_move_call_is_the_first_call_after_definition(self):
        """Sanity: there is exactly one call site for _finalize_priced_move,
        and it is not the function definition line itself."""
        src = _read(REPO_ROOT / "run_screen.py")
        def_count = src.count("def _finalize_priced_move(")
        call_count = src.count("_finalize_priced_move(csv_rows)")
        assert def_count == 1
        assert call_count == 1

    def test_finalize_priced_move_uses_ast_call_order_within_module(self):
        """AST-based cross-check (independent of the line-number heuristic
        above): walk run_screen.py's module-level statement list and confirm
        that, among top-level statements, any statement containing a call to
        compute_selector_scores or compute_ranker_adjustments appears before
        any top-level statement containing a call to _finalize_priced_move.

        run_screen.py's pipeline logic lives inside one large function body
        rather than at module level, so we walk all statements in document
        order via ast.walk + lineno, which is robust to nesting (if/try/for
        blocks) without relying on exact textual line matching.
        """
        src = _read(REPO_ROOT / "run_screen.py")
        tree = ast.parse(src)

        selector_calls: List[int] = []
        ranker_calls: List[int] = []
        finalize_calls: List[int] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name == "compute_selector_scores":
                    selector_calls.append(node.lineno)
                elif name == "compute_ranker_adjustments":
                    ranker_calls.append(node.lineno)
                elif name == "_finalize_priced_move":
                    finalize_calls.append(node.lineno)

        assert selector_calls, "No AST Call node found for compute_selector_scores"
        assert ranker_calls, "No AST Call node found for compute_ranker_adjustments"
        assert finalize_calls, "No AST Call node found for _finalize_priced_move"

        assert max(selector_calls) < min(finalize_calls), (
            f"compute_selector_scores call(s) at line(s) {selector_calls} must all "
            f"precede _finalize_priced_move call(s) at line(s) {finalize_calls}"
        )
        assert max(ranker_calls) < min(finalize_calls), (
            f"compute_ranker_adjustments call(s) at line(s) {ranker_calls} must all "
            f"precede _finalize_priced_move call(s) at line(s) {finalize_calls}"
        )


# ---------------------------------------------------------------------------
# 6. Behavioral pin: selector/ranker output is identical with vs. without
#    options diagnostics present on the input rows.
# ---------------------------------------------------------------------------


class TestSelectorRankerOutputUnaffectedByOptionsDiagnostics:
    """Build a small synthetic eligible-row cohort, run selector + the
    PRODUCTION ranker_v2 path (minimal_v2) twice — once with options fields
    populated, once with them stripped/absent — and assert identical output.

    This directly tests requirement #4 from the Stage 1 spec: "No options
    diagnostic output affects production ranking behavior."
    """

    @staticmethod
    def _make_row(ticker: str, seed: int, with_options: bool) -> dict:
        row = {
            "ticker": ticker,
            "eligible": "1",
            "actionable_rank": seed,
            "selector_rank_bucket": "top30",
            "catalyst_days": 30,
            "catalyst_family": "CLINICAL",
            "catalyst_type_tier": "T2",
            "endpoint_strength_score": 0.5 + (seed % 5) * 0.05,
            "design_quality_score": 0.6,
            "clinical_optionality_pct_dev": 50,
            "program_diversification": 0.4,
            "single_asset_risk": 0.3,
            "inst_delta_z": 0.1 * (seed % 3),
            "coinvest_filing_age_days": 10,
            "coinvest_conviction": 0.5,
            "inst_delta_net": 0.0,
            "coinvest_score_z": 0.2 * ((seed % 7) - 3),
            "financial_score": 0.4 + (seed % 4) * 0.1,
            "runway_bucket": "adequate",
            "severity": "NONE",
        }
        if with_options:
            row.update(
                {
                    "opt_has_data": "1",
                    "opt_atm_iv": "0.85",
                    "opt_front_iv": "1.10",
                    "opt_back_iv": "0.65",
                    "opt_term_slope": "-0.45",
                    "opt_put_call_skew": "0.05",
                    "opt_rr_25d": "-0.03",
                    "opt_event_premium": "YES",
                    "opt_iv_regime": "ELEVATED",
                    "opt_liquidity_ok": "1",
                    "opt_liquidity_state": "liquid",
                    "opt_use_for_judgment": "YES",
                    "ovf_composite": "0.7",
                    "ovf11_score": "0.6",
                    "cheap_vol_score": "0.42",
                    "vol_classification": "CHEAP",
                    "straddle_price": "0.18",
                    "priced_move_pct": "18.0",
                }
            )
        else:
            # Explicitly absent (not even empty-string placeholders) to
            # simulate a ticker with zero options coverage.
            pass
        return row

    def _make_cohort(self, with_options: bool) -> list:
        return [self._make_row(f"TIC{i:02d}", i, with_options) for i in range(1, 21)]

    def test_selector_scores_identical_with_and_without_options_fields(self):
        from selector_engine import DEFAULT_SELECTOR_CONFIG, compute_selector_scores

        rows_with = self._make_cohort(with_options=True)
        rows_without = self._make_cohort(with_options=False)

        results_with = compute_selector_scores(rows_with, config=DEFAULT_SELECTOR_CONFIG)
        results_without = compute_selector_scores(rows_without, config=DEFAULT_SELECTOR_CONFIG)

        scores_with = [r.selector_score for r in results_with]
        scores_without = [r.selector_score for r in results_without]

        assert scores_with == scores_without, (
            "selector_score changed when options diagnostic fields were added to "
            "input rows — selector_engine.py must be fully options-blind"
        )

        buckets_with = [r.selector_rank_bucket for r in results_with]
        buckets_without = [r.selector_rank_bucket for r in results_without]
        assert buckets_with == buckets_without

    def test_production_ranker_v2_scores_identical_with_and_without_options_fields(self):
        """Uses the PRODUCTION config (minimal_v2, the actually-deployed
        feature set) — not the 'expanded' set, which would legitimately
        differ since it includes BLOCK_OPTIONS by design (that block is
        for research, not deployed)."""
        from ranker_v2_pairwise import PairwiseLogisticModel, RankerV2Config, score_snapshot

        prod_config = RankerV2Config(
            feature_set="minimal_v2",
            cohort_top_n=60,
            require_catalyst_window=False,
            n_epochs=200,
            max_pairs_per_date=400,
            train_window=36,
        )

        model = PairwiseLogisticModel(
            weights=[0.02, -0.0533],
            bias=0.5019,
            n_features=2,
            feature_names=["coinvest_score_z", "financial_score"],
            trained=True,
        )

        rows_with = self._make_cohort(with_options=True)
        rows_without = self._make_cohort(with_options=False)

        results_with = score_snapshot(rows_with, model, prod_config)
        results_without = score_snapshot(rows_without, model, prod_config)

        scores_with = {r["ticker"]: r["ranker_v2_score"] for r in results_with}
        scores_without = {r["ticker"]: r["ranker_v2_score"] for r in results_without}

        assert scores_with == scores_without, (
            "ranker_v2 (minimal_v2/production feature set) score changed when "
            "options diagnostic fields were added to input rows"
        )

        ranks_with = {r["ticker"]: r["ranker_v2_rank"] for r in results_with}
        ranks_without = {r["ticker"]: r["ranker_v2_rank"] for r in results_without}
        assert ranks_with == ranks_without
