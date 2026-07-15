"""Tests for OpenClaw agent workspace integrity and configuration.

Validates that agent workspaces (ops, sentinel, qa, calibration) have:
- All required documentation files with correct sections
- Consistent active ruleset references across all agents
- Valid boundary definitions (no agent claims write access outside its memory dir)
- Referenced tools/scripts exist in the repo
- Memory directories can be created
- Cross-agent consistency (same model, same repo path, same ruleset ID)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AGENTS_DIR = PROJECT_ROOT / "agents"
# Fully compliant agents: all 4 docs + well-structured SOUL.md that passes
# all integrity checks (TestSoulIntegrity, TestToolsIntegrity, etc.)
AGENT_NAMES = [
    "calibration",
    "catalyst_delta",
    "ops",
    "options_watch",
    "qa",
    "sentinel",
]
# Workspaces with all 4 docs but non-conforming content (AGENTS.md missing
# "startup"/"SOUL.md" references, HEARTBEAT.md missing OK token, TOOLS.md
# referencing no python script, etc.)
# Tested for SOUL.md structure only via TestPartialAgents.
PARTIAL_AGENTS = [
    "earnings_calendar_sync",
    "fleet_steward",
    "herald",
    "postmortem",
    "review_queue_steward",
]
# Known incomplete workspaces — missing docs, major SOUL.md gaps, or
# intentionally different model (haiku monitoring class).
# Tested for basic existence only via TestIncompleteAgents.
INCOMPLETE_AGENTS = [
    "aact_trial_ingest",
    "crt_resolution_watcher",
    "ctgov_poller",
    "data_auditor",
    "ees_v3_veto_monitor",
    "event_analyst",
    "ic_health_monitor",
    "intraday_mover_watch",
    "price_action_watch",
    "production_qa",
    "shadow_monitor",
    "universe_maintenance",
]
# Lane A Hermes governance jobs: deterministic run_job.py only (llm_policy: none).
# HEARTBEAT.md is optional — added per commit 0b4b4427 for documentation purposes
# but not required for the Lane A heartbeat contract.
LANE_A_HERMES_AGENTS = [
    "hermes-contradiction-detector",
    "hermes-first-fire-validator",
    "hermes-held-spec-ledger",
    "hermes-ruleset-integrity",
]
# Retired agent workspaces — merged into other agents, dirs kept for history.
RETIRED_AGENTS = [
    "bioshort_watch",  # LLM consumer suppressed; deterministic producer retained
    "calibration_evidence",  # merged into calibration (dir retained)
    "shadow_watch",  # placeholder retired; shadow_monitor is canonical
]
# Stub / not-yet-built workspaces — dirs exist for AGENT_REGISTRY or placeholder
# reasons but have not been built out to INCOMPLETE_AGENTS standards.
# No doc requirements; tracked here only so test_total_agent_count stays clean.
STUB_AGENTS = [
    "allocation_lens",  # .gitkeep; AGENT_REGISTRY entry requires dir
    "biotech_news_digest",  # SOUL.md only; not yet wired
    "company_news_ingest",  # SOUL.md only; not yet wired
    "hermes-skill-sync-agent",  # HEARTBEAT.md only; active but not fully built
    "policy_shadow_watch",  # SOUL.md only; not yet wired
]
# Terminal supervisor workspace: intentionally active but unsupervised and
# does not carry a HEARTBEAT.md because tools/agent_supervisor_sentinel.py is
# the final watchdog above it.
TERMINAL_AGENTS = ["ops_supervisor"]
REQUIRED_DOCS = ["SOUL.md", "TOOLS.md", "HEARTBEAT.md", "AGENTS.md"]
EXPECTED_RULESET_ID = "8887576e"
EXPECTED_RULESET_VERSION = "v1.14.0"
EXPECTED_MODEL = "deepseek/deepseek-v4-flash:free"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=AGENT_NAMES)
def agent_name(request):
    """Parametrize over all agent names."""
    return request.param


@pytest.fixture
def agent_dir(agent_name):
    return AGENTS_DIR / agent_name


@pytest.fixture
def soul_text(agent_dir):
    return (agent_dir / "SOUL.md").read_text(encoding="utf-8")


@pytest.fixture
def tools_text(agent_dir):
    return (agent_dir / "TOOLS.md").read_text(encoding="utf-8")


@pytest.fixture
def heartbeat_text(agent_dir):
    return (agent_dir / "HEARTBEAT.md").read_text(encoding="utf-8")


@pytest.fixture
def agents_text(agent_dir):
    return (agent_dir / "AGENTS.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Workspace structure
# ---------------------------------------------------------------------------


class TestWorkspaceStructure:
    """Every agent workspace must have the four required docs."""

    def test_agents_directory_exists(self):
        assert AGENTS_DIR.is_dir(), f"agents/ directory missing at {AGENTS_DIR}"

    def test_all_agent_dirs_exist(self):
        for name in AGENT_NAMES:
            d = AGENTS_DIR / name
            assert d.is_dir(), f"Agent directory missing: {d}"

    def test_required_docs_present(self, agent_dir):
        for doc in REQUIRED_DOCS:
            path = agent_dir / doc
            assert path.is_file(), f"Missing {doc} in {agent_dir.name}"
            assert path.stat().st_size > 50, f"{doc} in {agent_dir.name} is suspiciously small"

    def test_no_unexpected_python_files(self, agent_dir):
        """Agent workspaces should not contain .py files — agents are doc-driven."""
        py_files = list(agent_dir.glob("*.py"))
        assert py_files == [], f"Unexpected .py files in {agent_dir.name}/: {[f.name for f in py_files]}"


# ---------------------------------------------------------------------------
# SOUL.md validation
# ---------------------------------------------------------------------------


class TestSoulIntegrity:
    """SOUL.md must define identity, principles, boundaries, and ruleset."""

    def test_has_identity_section(self, soul_text, agent_name):
        assert "## Identity" in soul_text, f"{agent_name} SOUL.md missing ## Identity"

    def test_has_name_field(self, soul_text, agent_name):
        assert f"**Name**: {agent_name}" in soul_text, f"{agent_name} SOUL.md should declare Name: {agent_name}"

    def test_has_role_field(self, soul_text, agent_name):
        assert "**Role**:" in soul_text, f"{agent_name} SOUL.md missing Role field"

    def test_has_model_field(self, soul_text, agent_name):
        assert (
            f"**Model**: {EXPECTED_MODEL}" in soul_text
        ), f"{agent_name} SOUL.md should reference model {EXPECTED_MODEL}"

    def test_has_core_principles(self, soul_text, agent_name):
        assert "## Core principles" in soul_text, f"{agent_name} SOUL.md missing ## Core principles"

    def test_has_boundaries(self, soul_text, agent_name):
        assert "## Boundaries" in soul_text, f"{agent_name} SOUL.md missing ## Boundaries"

    def test_has_active_ruleset(self, soul_text, agent_name):
        assert "## Active ruleset" in soul_text, f"{agent_name} SOUL.md missing ## Active ruleset"

    def test_ruleset_id_matches(self, soul_text, agent_name):
        assert (
            EXPECTED_RULESET_ID in soul_text
        ), f"{agent_name} SOUL.md does not reference ruleset ID {EXPECTED_RULESET_ID}"

    def test_ruleset_version_matches(self, soul_text, agent_name):
        assert (
            EXPECTED_RULESET_VERSION in soul_text
        ), f"{agent_name} SOUL.md does not reference ruleset version {EXPECTED_RULESET_VERSION}"

    def test_never_clause_present(self, soul_text, agent_name):
        """Every agent must have at least one Never boundary."""
        assert "**Never**:" in soul_text, f"{agent_name} SOUL.md missing **Never**: boundary clause"

    def test_write_restricted_to_memory(self, soul_text, agent_name):
        """Write permissions should only target the agent's own memory dir."""
        write_match = re.search(r"\*\*Write\*\*:\s*only to\s+`([^`]+)`", soul_text)
        assert write_match, f"{agent_name} SOUL.md should have a **Write**: only to `...` clause"
        write_target = write_match.group(1)
        assert (
            f"agents/{agent_name}/memory/" in write_target
        ), f"{agent_name} write target {write_target!r} should include agents/{agent_name}/memory/"


# ---------------------------------------------------------------------------
# TOOLS.md validation
# ---------------------------------------------------------------------------


class TestToolsIntegrity:
    """TOOLS.md must reference real scripts and paths."""

    def test_references_python_scripts(self, tools_text, agent_name):
        """Each agent's TOOLS.md should reference at least one python script."""
        scripts = re.findall(r"python3?\s+([\w/._-]+\.py)", tools_text)
        assert len(scripts) >= 1, f"{agent_name} TOOLS.md should reference at least one python script"

    def test_referenced_scripts_exist(self, tools_text, agent_name):
        """Python scripts mentioned in TOOLS.md should exist in the repo."""
        scripts = re.findall(r"python3?\s+([\w/._-]+\.py)", tools_text)
        for script in scripts:
            path = PROJECT_ROOT / script
            assert path.is_file(), f"{agent_name} TOOLS.md references {script} but it doesn't exist"


# ---------------------------------------------------------------------------
# HEARTBEAT.md validation
# ---------------------------------------------------------------------------


class TestHeartbeatIntegrity:
    """HEARTBEAT.md must define a checklist and OK response."""

    def test_has_checklist(self, heartbeat_text, agent_name):
        # Should have numbered items (1. 2. etc.) or bullet checklist
        has_numbered = bool(re.search(r"^\d+\.", heartbeat_text, re.MULTILINE))
        has_bullets = bool(re.search(r"^[-*]\s", heartbeat_text, re.MULTILINE))
        assert has_numbered or has_bullets, f"{agent_name} HEARTBEAT.md should have a numbered or bulleted checklist"

    def test_defines_ok_response(self, heartbeat_text, agent_name):
        """Should define a clear 'all OK' response token."""
        ok_patterns = ["HEARTBEAT_OK", "OK", "CLEAR", "PASS"]
        found = any(p in heartbeat_text for p in ok_patterns)
        assert found, f"{agent_name} HEARTBEAT.md should define an OK/CLEAR response token"


# ---------------------------------------------------------------------------
# AGENTS.md validation
# ---------------------------------------------------------------------------


class TestAgentsFileIntegrity:
    """AGENTS.md must define session startup, daily workflow, and red lines."""

    def test_has_session_startup(self, agents_text, agent_name):
        assert (
            "startup" in agents_text.lower() or "session" in agents_text.lower()
        ), f"{agent_name} AGENTS.md should describe session startup"

    def test_references_soul_and_tools(self, agents_text, agent_name):
        assert "SOUL.md" in agents_text, f"{agent_name} AGENTS.md should reference SOUL.md"
        assert "TOOLS.md" in agents_text, f"{agent_name} AGENTS.md should reference TOOLS.md"

    def test_has_memory_section(self, agents_text, agent_name):
        assert "memory" in agents_text.lower(), f"{agent_name} AGENTS.md should describe memory protocol"


# ---------------------------------------------------------------------------
# Cross-agent consistency
# ---------------------------------------------------------------------------


class TestCrossAgentConsistency:
    """All agents must agree on ruleset ID, model, and key constraints."""

    def test_all_agents_share_ruleset_id(self):
        """Every agent's SOUL.md must reference the same active ruleset ID."""
        for name in AGENT_NAMES:
            text = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
            assert EXPECTED_RULESET_ID in text, f"Agent {name} does not reference ruleset {EXPECTED_RULESET_ID}"

    def test_all_agents_share_model(self):
        """Every agent's SOUL.md must reference the same model."""
        for name in AGENT_NAMES:
            text = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
            assert EXPECTED_MODEL in text, f"Agent {name} does not reference model {EXPECTED_MODEL}"

    def test_no_agent_claims_git_write(self):
        """No agent should have git push/commit in its allowed boundaries."""
        for name in AGENT_NAMES:
            soul = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
            boundaries = soul[soul.find("## Boundaries") :] if "## Boundaries" in soul else ""
            # The boundaries should mention "never" with git operations
            # but should NOT list git push/commit as allowed actions
            run_section = re.search(r"\*\*Run\*\*:\s*(.+?)(?:\n-|\n##|\Z)", boundaries, re.DOTALL)
            if run_section:
                run_text = run_section.group(1).lower()
                assert "git push" not in run_text, f"Agent {name} should not have git push in Run permissions"
                assert "git commit" not in run_text, f"Agent {name} should not have git commit in Run permissions"

    def test_no_agent_writes_outside_own_memory(self):
        """Each agent's write scope is limited to its own memory directory."""
        for name in AGENT_NAMES:
            soul = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
            write_match = re.search(r"\*\*Write\*\*:\s*only to\s+`([^`]+)`", soul)
            assert write_match, f"Agent {name} missing Write boundary"
            write_targets = write_match.group(1)
            # Should not write to another agent's workspace
            other_agents = [a for a in AGENT_NAMES if a != name]
            for other in other_agents:
                assert f"agents/{other}/" not in write_targets, f"Agent {name} should not write to agents/{other}/"

    def test_ruleset_id_matches_claude_md(self):
        """Active ruleset in CLAUDE.md must match what agents reference."""
        claude_md = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        assert EXPECTED_RULESET_ID in claude_md, f"CLAUDE.md does not contain expected ruleset ID {EXPECTED_RULESET_ID}"

    def test_ops_agent_has_extra_write_for_digest(self):
        """Ops agent uniquely has write access to artifacts/ops_digest/."""
        ops_soul = (AGENTS_DIR / "ops" / "SOUL.md").read_text(encoding="utf-8")
        assert "artifacts/ops_digest/" in ops_soul, "Ops agent should have write access to artifacts/ops_digest/"


# ---------------------------------------------------------------------------
# Referenced tool scripts exist
# ---------------------------------------------------------------------------


class TestReferencedToolsExist:
    """Scripts and tools referenced across all agent docs must exist."""

    CRITICAL_SCRIPTS = [
        "tools/run_daily_production.py",
        "tools/build_ops_digest.py",
        "tools/ruleset_health_monitor.py",
        "tools/live_shadow_portfolio.py",
        "tools/build_portfolio_report.py",
        "tools/weekly_readiness_scorecard.py",
        "scripts/promote_ruleset.py",
        "scripts/research/run_promotion_battery.py",
        "scripts/run_signal_evidence.py",
        "run_screen.py",
        "decision_engine.py",
    ]

    @pytest.mark.parametrize("script_path", CRITICAL_SCRIPTS)
    def test_critical_script_exists(self, script_path):
        full_path = PROJECT_ROOT / script_path
        assert full_path.is_file(), f"Critical script missing: {script_path}"

    def test_qa_referenced_test_files_exist(self):
        """Test files listed in QA agent's TOOLS.md must exist."""
        qa_tools = (AGENTS_DIR / "qa" / "TOOLS.md").read_text(encoding="utf-8")
        test_files = re.findall(r"tests/(test_\w+\.py)", qa_tools)
        for tf in test_files:
            path = PROJECT_ROOT / "tests" / tf
            assert path.is_file(), f"QA agent references tests/{tf} but it doesn't exist"

    def test_active_ruleset_file_exists(self):
        """The active ruleset JSON file should exist."""
        ruleset_pattern = list(
            (PROJECT_ROOT / "production_data" / "decision_rulesets").glob(f"{EXPECTED_RULESET_VERSION}_*")
        )
        assert len(ruleset_pattern) >= 1, (
            f"No ruleset file matching {EXPECTED_RULESET_VERSION}_* found in " "production_data/decision_rulesets/"
        )

    def test_manifest_exists(self):
        manifest = PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
        assert manifest.is_file(), "Ruleset manifest.json missing"


# ---------------------------------------------------------------------------
# Incomplete agent workspaces — basic structure checks only
# ---------------------------------------------------------------------------


class TestPartialAgents:
    """Agents with all 4 docs but non-conforming AGENTS.md/HEARTBEAT.md content.

    SOUL.md structure passes, but AGENTS.md may lack "startup"/"SOUL.md" refs,
    HEARTBEAT.md may lack OK token, etc. Test SOUL.md structure to catch drift.
    """

    @pytest.mark.parametrize("name", PARTIAL_AGENTS)
    def test_has_required_docs(self, name):
        for doc in REQUIRED_DOCS:
            path = AGENTS_DIR / name / doc
            assert path.is_file(), f"Partial agent {name} missing {doc}"

    @pytest.mark.parametrize("name", PARTIAL_AGENTS)
    def test_soul_has_identity(self, name):
        soul = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
        assert "## Identity" in soul or "## What you do" in soul
        assert "**Model**:" in soul or "**Role**:" in soul
        assert EXPECTED_RULESET_ID in soul

    @pytest.mark.parametrize("name", PARTIAL_AGENTS)
    def test_soul_has_boundaries(self, name):
        soul = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
        assert "## Boundaries" in soul
        assert "**Never**:" in soul

    @pytest.mark.parametrize("name", PARTIAL_AGENTS)
    def test_soul_has_ruleset(self, name):
        soul = (AGENTS_DIR / name / "SOUL.md").read_text(encoding="utf-8")
        assert EXPECTED_RULESET_ID in soul


class TestLaneAHermesAgents:
    """Hermes governance jobs (Lane A): explicit run_job.py, no LLM heartbeat contract."""

    @pytest.mark.parametrize("name", LANE_A_HERMES_AGENTS)
    def test_workspace_exists(self, name):
        assert (AGENTS_DIR / name).is_dir(), f"Lane A Hermes workspace missing: {name}"

    @pytest.mark.parametrize("name", LANE_A_HERMES_AGENTS)
    def test_run_job_exists(self, name):
        run_job = AGENTS_DIR / name / "run_job.py"
        assert run_job.is_file(), f"Lane A Hermes agent {name} missing run_job.py"

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    @pytest.mark.parametrize("name", LANE_A_HERMES_AGENTS)
    def test_no_heartbeat_by_design(self, name):
        """HEARTBEAT.md is optional for Lane A agents (added for doc purposes per 0b4b4427).
        Contract is run_job.py; HEARTBEAT.md presence is not a violation."""
        pass  # not required, not forbidden — no assertion


class TestIncompleteAgents:
    """Agents known to have incomplete SOUL.md structure.

    These need SOUL.md rework to add missing sections (Identity/Name,
    Core principles, Boundaries, Active ruleset, Never clause, Write scope).
    This test class ensures they at least exist and have a SOUL.md, and
    tracks them so they don't silently disappear.
    """

    @pytest.mark.parametrize("name", INCOMPLETE_AGENTS)
    def test_workspace_exists(self, name):
        assert (AGENTS_DIR / name).is_dir(), f"Incomplete agent workspace missing: {name}"

    @pytest.mark.parametrize("name", INCOMPLETE_AGENTS)
    def test_soul_exists(self, name):
        soul = AGENTS_DIR / name / "SOUL.md"
        assert soul.is_file(), f"Incomplete agent {name} missing SOUL.md"
        assert soul.stat().st_size > 30, f"Incomplete agent {name} SOUL.md is too small"

    @pytest.mark.parametrize("name", INCOMPLETE_AGENTS)
    def test_heartbeat_exists(self, name):
        assert (AGENTS_DIR / name / "HEARTBEAT.md").is_file(), f"Incomplete agent {name} missing HEARTBEAT.md"

    def test_total_agent_count(self):
        """All agent workspaces are accounted for (compliant + partial + incomplete + retired)."""
        all_agents = (
            set(AGENT_NAMES)
            | set(PARTIAL_AGENTS)
            | set(INCOMPLETE_AGENTS)
            | set(LANE_A_HERMES_AGENTS)
            | set(RETIRED_AGENTS)
            | set(STUB_AGENTS)
            | set(TERMINAL_AGENTS)
        )
        actual = {d.name for d in AGENTS_DIR.iterdir() if d.is_dir()}
        untracked = actual - all_agents
        assert not untracked, f"Agent workspace(s) not in any agent list: {untracked}"
