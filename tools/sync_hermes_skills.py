#!/usr/bin/env python3
"""Sync Cursor skills/ into docs/hermes_skills/ mirrors and maintain _meta.json registry.

Preserves YAML frontmatter on existing Hermes skill files. Injects Hermes-only
sections (e.g. Path C block in screener-ops) when the mirror has content absent
from the Cursor skill source.

Usage:
    python3 tools/audit_hermes_skills.py
    python3 tools/sync_hermes_skills.py
    python3 tools/sync_hermes_skills.py --register-meta
    python3 tools/sync_hermes_skills.py --dry-run --only dossier_generation,excel-xlsx
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "skills"
HERMES = REPO / "docs" / "hermes_skills"
META = HERMES / "_meta.json"

# skills/<dir>/SKILL.md -> docs/hermes_skills/<file>
SKILL_MAP: dict[str, str] = {
    "screener_ops": "screener-ops.md",
    "codegraph": "codegraph.md",
    "firecrawl_research": "firecrawl-research-discovery.md",
    "catalyst_resolution": "catalyst-resolution.md",
    "clinical_scoring": "clinical-scoring.md",
    "financial_health": "financial-health.md",
    "institutional_signal": "institutional-signal.md",
    "selector_ranker": "selector-ranker.md",
    "ic_evaluation": "ic-evaluation.md",
    "memory_steward": "memory-steward.md",
    "validation": "validation.md",
    "spending_liquidity": "spending-liquidity.md",
    "sfo_liquidity_architecture": "sfo-liquidity-architecture.md",
    "self-improving": "self-improving.md",
    "operational_health_baselines": "operational-health-baselines.md",
    "pe_pacing": "pe-pacing.md",
    "openclaw-agent-optimize": "openclaw-agent-optimize.md",
}

# skills/<dir>/REFERENCE.md -> docs/hermes_skills/<file>
REFERENCE_MAP: dict[str, str] = {
    "dossier_generation": "dossier-generation.md",
    "excel-xlsx": "excel-xlsx.md",
    "word-docx": "word-docx.md",
    "self-improving": "self-improving-reference.md",
}

# Hermes-only docs (no skills/ source) — registered in _meta only
HERMES_NATIVE: dict[str, str] = {
    "browser-automation": "Browser Automation (OpenClaw)",
    "governance-spec-enforcement": "Spec Enforcement & Governance",
    "hermeslink-state-capture": "Hermes Knowledge Layer State Capture",
    "openclaw-agent-scope-audit": "OpenClaw Agent Scope Audit",
    "openclaw-cron-scheduler-debug": "OpenClaw Cron Scheduler Debug",
    "openclaw-data-pipeline-debug": "OpenClaw Data Pipeline Debug",
    "openclaw-session-routing-debug": "OpenClaw Session Routing Debug",
    "phase-2-step-4-readiness": "Phase 2 Step 4 Readiness",
    "town-operator-bridge": "Town-Hermes Bridge (Spec 090)",
    "13f-validation-coordinator": "13F Cohort Validation",
    "path-c-governance-monitoring": "Path C Governance Monitoring",
    "path-c-operational-runbook": "Path C Operational Runbook",
}

HERMES_ONLY_SECTIONS: dict[str, list[str]] = {
    "screener-ops.md": [
        r"(## Path C Governance Monitoring[\s\S]*?)(?=\n\n### Operator WSL acceptance gate|\n---\n\n## Town-Hermes Bridge)",
    ],
}

HERMES_AUTHORITATIVE: set[str] = {
    "memory_steward",
}

# Values stored in _meta.json `source_authority` (audit + operator runbook).
SOURCE_AUTHORITY_HERMES_NATIVE = "hermes_native"
SOURCE_AUTHORITY_HERMES_AUTHORITATIVE = "hermes_authoritative"

DISPLAY_NAMES: dict[str, str] = {
    "screener_ops": "Screener Ops & Governance",
    "codegraph": "Codegraph Repo Intelligence",
    "firecrawl_research": "Firecrawl Research Discovery",
    "catalyst_resolution": "Catalyst Resolution & Tracking",
    "clinical_scoring": "Clinical Trial Scoring",
    "financial_health": "Financial Health Assessment",
    "institutional_signal": "Institutional Holdings Signal",
    "selector_ranker": "Selector & Ranker Architecture",
    "ic_evaluation": "IC Evaluation",
    "validation": "Validation & Export Contracts",
    "spending_liquidity": "Spending Liquidity",
    "sfo_liquidity_architecture": "SFO Liquidity Architecture",
    "self-improving": "Self-Improving Agent Loop",
    "operational_health_baselines": "Operational Health Baselines",
    "pe_pacing": "PE Pacing",
    "openclaw-agent-optimize": "OpenClaw Agent Optimize",
    "dossier_generation": "Dossier Generation",
    "excel-xlsx": "Excel / XLSX",
    "word-docx": "Word / DOCX",
}


def unescape_md(text: str) -> str:
    return text.replace("\\(", "(").replace("\\)", ")").replace("\\-", "-")


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[: end + 4], text[end + 4 :].lstrip("\n")


def extract_hermes_only(hermes_body: str, filename: str) -> list[str]:
    blocks: list[str] = []
    for pattern in HERMES_ONLY_SECTIONS.get(filename, []):
        match = re.search(pattern, hermes_body)
        if match:
            blocks.append(match.group(1).strip())
    return blocks


def inject_after_host_authority(skill_body: str, blocks: list[str]) -> str:
    if not blocks:
        return skill_body
    marker = "**Standing rule:**"
    idx = skill_body.find(marker)
    if idx == -1:
        return skill_body + "\n\n---\n\n" + "\n\n---\n\n".join(blocks) + "\n"
    line_end = skill_body.find("\n", idx)
    insert_at = line_end + 1 if line_end != -1 else len(skill_body)
    injection = "\n---\n\n" + "\n\n---\n\n".join(blocks) + "\n\n"
    return skill_body[:insert_at] + injection + skill_body[insert_at:]


def _source_path(skill_key: str, *, hermes_name: str | None = None) -> Path | None:
    d = SKILLS / skill_key
    reference_names = set(REFERENCE_MAP.values())
    if hermes_name in reference_names:
        p = d / "REFERENCE.md"
        return p if p.exists() else None
    for name in ("SKILL.md", "REFERENCE.md"):
        p = d / name
        if p.exists():
            return p
    return None


def sync_pair(skill_key: str, hermes_name: str, dry_run: bool) -> str:
    if skill_key in HERMES_AUTHORITATIVE:
        return f"SKIP {skill_key}: Hermes mirror authoritative"
    skill_path = _source_path(skill_key, hermes_name=hermes_name)
    hermes_path = HERMES / hermes_name
    if skill_path is None:
        return f"SKIP {skill_key}: no SKILL.md or REFERENCE.md"
    skill_body = unescape_md(skill_path.read_text())
    frontmatter, old_body = split_frontmatter(hermes_path.read_text()) if hermes_path.exists() else ("", "")
    if not frontmatter:
        frontmatter = f"---\nname: {hermes_path.stem}\n---\n"
    only_blocks = extract_hermes_only(old_body, hermes_name) if old_body else []
    if only_blocks:
        skill_body = inject_after_host_authority(skill_body, only_blocks)
    new_text = frontmatter.rstrip() + "\n\n" + skill_body.lstrip()
    if hermes_path.exists() and hermes_path.read_text() == new_text:
        return f"OK   {skill_key}: already in sync"
    if dry_run:
        return f"DRY  {skill_key}: would update {hermes_path.name}"
    hermes_path.write_text(new_text)
    return f"SYNC {skill_key} -> {hermes_path.name}"


def source_authority_for(meta_key: str, fname: str) -> str:
    """Resolve canonical edit path for a Hermes skill mirror (_meta.json source_authority)."""
    for skill_key, hermes_name in SKILL_MAP.items():
        if hermes_name == fname:
            if skill_key in HERMES_AUTHORITATIVE:
                return SOURCE_AUTHORITY_HERMES_AUTHORITATIVE
            return f"skills/{skill_key}/SKILL.md"
    for skill_key, hermes_name in REFERENCE_MAP.items():
        if hermes_name == fname:
            return f"skills/{skill_key}/REFERENCE.md"
    if meta_key in HERMES_NATIVE or Path(fname).stem in HERMES_NATIVE:
        return SOURCE_AUTHORITY_HERMES_NATIVE
    return "unknown"


def register_meta(dry_run: bool) -> list[str]:
    if not META.exists():
        return ["SKIP meta: _meta.json missing"]
    data = json.loads(META.read_text())
    skills: dict = data.setdefault("skills", {})
    lines: list[str] = []

    def add(key: str, fname: str, name: str, source: str) -> None:
        authority = source_authority_for(key, fname)
        entry = {
            "name": name,
            "file": fname,
            "status": skills.get(key, {}).get("status", "Active"),
            "source": source,
            "source_authority": authority,
        }
        is_new = key not in skills
        prior_auth = skills.get(key, {}).get("source_authority")
        if is_new:
            lines.append(f"ADD  {key} ({source}, authority={authority})")
        elif prior_auth != authority:
            lines.append(f"PATCH {key} source_authority={authority}")
        elif skills[key].get("file") != fname:
            lines.append(f"PATCH {key} file={fname}")
        if not dry_run:
            merged = {**skills.get(key, {}), **entry}
            skills[key] = merged

    for key, fname in {**SKILL_MAP, **REFERENCE_MAP}.items():
        meta_key = Path(fname).stem
        name = DISPLAY_NAMES.get(key, meta_key.replace("-", " ").title())
        src = "cursor_reference" if key in REFERENCE_MAP else "cursor_skill"
        if (HERMES / fname).exists():
            add(meta_key, fname, name, src)

    for meta_key, name in HERMES_NATIVE.items():
        fname = f"{meta_key}.md"
        if (HERMES / fname).exists():
            add(meta_key, fname, name, SOURCE_AUTHORITY_HERMES_NATIVE)

    # Backfill source_authority on entries not touched above (e.g. town-operator-bridge extras).
    for meta_key, entry in skills.items():
        fname = entry.get("file", f"{meta_key}.md")
        authority = source_authority_for(meta_key, fname)
        if entry.get("source_authority") != authority:
            if not dry_run:
                entry["source_authority"] = authority
            lines.append(f"AUTH {meta_key} -> {authority}")

    data["last_updated"] = date.today().isoformat()
    data["source_authority_legend"] = {
        "skills/SKILL.md": "Edit skills/<dir>/SKILL.md then sync",
        "skills/<dir>/REFERENCE.md": "Edit skills/<dir>/REFERENCE.md then sync",
        SOURCE_AUTHORITY_HERMES_NATIVE: "Edit docs/hermes_skills/<file>.md directly",
        SOURCE_AUTHORITY_HERMES_AUTHORITATIVE: "Edit docs/hermes_skills mirror only (sync skips)",
    }
    if not dry_run:
        META.write_text(json.dumps(data, indent=2) + "\n")
    return lines


def all_sync_keys() -> dict[str, str]:
    merged = {**SKILL_MAP, **REFERENCE_MAP}
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync skills/ to docs/hermes_skills/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="Comma-separated skill dir keys")
    parser.add_argument("--register-meta", action="store_true", help="Register all skills in _meta.json")
    args = parser.parse_args()

    if args.register_meta:
        for line in register_meta(args.dry_run):
            print(line)
        print()

    mapping = all_sync_keys()
    keys = list(mapping.keys())
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
    results = [sync_pair(k, mapping[k], args.dry_run) for k in keys if k in mapping]
    if not args.register_meta:
        register_meta(args.dry_run)  # still bump last_updated
    elif not args.dry_run:
        pass
    else:
        register_meta(True)

    for line in results:
        print(line)
    synced = sum(1 for r in results if r.startswith("SYNC"))
    print(f"\nSync: {synced} updated, {len(results) - synced} unchanged/skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
