"""Production selector ruleset contract (Phase 2 pinned ruleset).

Pins the active/pinned decision ruleset (8887576e / v1.14.0) as the
coinvest-only selector. The deployed selector zeroes out inst_delta_z and
runs coinvest-only (run_screen.A4_SELECTOR_CONFIG / v1.14.0 ruleset).

TEST_CONTRACT_ONLY — no production code, model, or behavior changes.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_engine import DecisionRuleset

RULESET_PATH = (
    Path(__file__).resolve().parents[1]
    / "production_data"
    / "decision_rulesets"
    / "v1.14.0_coinvest_only_selector.json"
)


class TestPinnedRulesetIdentity:
    """The pinned ruleset id and its file hash must agree."""

    def test_pinned_id_constant(self):
        from run_screen_columns import PHASE2_PINNED_RULESET_ID

        assert PHASE2_PINNED_RULESET_ID == "8887576e"

    def test_ruleset_file_hashes_to_pinned_id(self):
        rs = DecisionRuleset.from_json(str(RULESET_PATH))
        assert rs.ruleset_id == "8887576e"

    def test_sort_anchor_is_selector_score(self):
        rs = DecisionRuleset.from_json(str(RULESET_PATH))
        assert rs.sort_anchor == "selector_score"


class TestCoinvestOnlySelectorMetadata:
    """Coinvest-only selector metadata documented in the ruleset JSON.

    These fields live in the raw ruleset file (they are not normalized onto
    the DecisionRuleset dataclass), so they are asserted against the file.
    """

    def _raw(self):
        return json.loads(RULESET_PATH.read_text(encoding="utf-8"))

    def test_version_label(self):
        assert self._raw()["version_label"] == "v1.14.0"

    def test_selector_config_is_coinvest_only(self):
        assert self._raw()["selector_config"] == "coinvest_only"

    def test_coinvest_selector_weight_is_one(self):
        assert self._raw()["coinvest_score_z_selector_weight"] == pytest.approx(1.0)

    def test_inst_delta_selector_weight_is_zero(self):
        assert self._raw()["inst_delta_z_selector_weight"] == pytest.approx(0.0)


class TestLiveSelectorConfig:
    """Tie the contract to the live A4_SELECTOR_CONFIG used by run_screen."""

    def test_a4_selector_is_coinvest_only(self):
        from run_screen import A4_SELECTOR_CONFIG

        weights = {s.name: s.weight for s in A4_SELECTOR_CONFIG.institutional_signals}
        assert weights["coinvest_score_z"] == pytest.approx(1.0)
        assert weights["inst_delta_z"] == pytest.approx(0.0)

    def test_a4_block_weights_are_the_live_blend(self):
        """Institutional signals are coinvest-only. The selector is not.

        Catalyst, survivability, and market structure still enter the cohort.
        final_score is then the pairwise ranker, not this blend.
        """
        from run_screen import A4_SELECTOR_CONFIG

        blocks = {bw.name: bw.weight for bw in A4_SELECTOR_CONFIG.block_weights}
        assert blocks == {
            "clinical": pytest.approx(0.0),
            "catalyst": pytest.approx(0.15),
            "survivability": pytest.approx(0.10),
            "institutional": pytest.approx(0.65),
            "market_structure": pytest.approx(0.10),
        }
