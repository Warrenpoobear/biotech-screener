#!/usr/bin/env python3
"""Scan .learnings/LEARNINGS.md for promotion-ready patterns and DRAFT skill patches.

STAGED — writes drafts only; never edits skill files directly.

Usage:
    SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py
    SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py --min-recurrence 3 --out artifacts/skill_patch_drafts
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# FENCE: set SELFIMPROVE_GATES_MET=1 to enable (selfimprove_audit_2026-06-24).
# Rule 12: MUST refuse Promotion-lane: spec (see skills/self-improving/SKILL.md).

FROZEN_SKILL_TARGETS = {
    "selector-ranker",
    "selector_ranker",
    "clinical-scoring",
    "ic-evaluation",
    "financial-health",
    "institutional-signal",
    "catalyst-resolution",
}

SPEC_LANE_AREAS = {"research", "portfolio"}
ENTRY_RE = re.compile(r"^## \[(LRN-\d{8}-\d+)\]\s+(.+?)\s*$", re.MULTILINE)


def infer_promotion_lane(area: str | None, explicit: str | None) -> str:
    """Return promotion lane: skill | spec | none."""
    if explicit:
        return explicit.strip().lower()
    if area and area.lower() in SPEC_LANE_AREAS:
        return "spec"
    return "skill"


def parse_learnings(text: str):
    """Yield dicts for each LRN entry with metadata fields."""
    matches = list(ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        rec = re.search(r"Recurrence-Count:\s*(\d+)", body)
        pkey = re.search(r"Pattern-Key:\s*([\w./-]+)", body)
        action = re.search(r"### Suggested Action\s*(.+?)(?=\n###|\Z)", body, re.DOTALL)
        summary = re.search(r"### Summary\s*(.+?)(?=\n###|\Z)", body, re.DOTALL)
        skillc = re.search(r"SKILL-CANDIDATE:\s*([\w./|-]+)", body)
        skillp = re.search(r"Skill-Path:\s*([\w./,-]+)", body)
        area_m = re.search(r"\*\*Area\*\*:\s*(\w+)", body)
        lane_m = re.search(r"Promotion-lane:\s*(\w+)", body)
        status_m = re.search(r"\*\*Status\*\*:\s*(\w+)", body)
        area = area_m.group(1) if area_m else None
        lane = infer_promotion_lane(area, lane_m.group(1) if lane_m else None)
        skill_path = None
        if skillp:
            skill_path = skillp.group(1).split(",")[0].strip()
        elif skillc:
            skill_path = skillc.group(1).split("|")[0].strip()
        yield {
            "id": m.group(1),
            "title": m.group(2),
            "recurrence": int(rec.group(1)) if rec else 0,
            "pattern_key": pkey.group(1) if pkey else None,
            "summary": (summary.group(1).strip() if summary else "")[:500],
            "action": (action.group(1).strip() if action else "")[:500],
            "skill_path": skill_path,
            "skill_candidate": skill_path or (skillc.group(1) if skillc else None),
            "area": area,
            "promotion_lane": lane,
            "status": (status_m.group(1).lower() if status_m else "unknown"),
        }


def refuse_spec_lane_entries(entries: list[dict]) -> list[dict]:
    """Rule 12 lane gate: return spec-lane entries that must not become skill patches."""
    return [e for e in entries if e.get("promotion_lane") == "spec"]


def draft_patch(learning: dict) -> str:
    """Produce a human-readable proposed skill-doc addition (NOT a git diff)."""
    if learning.get("promotion_lane") == "spec":
        return (
            f"## BLOCKED (spec lane) — {learning['id']}\n\n"
            f"- **Pattern-Key:** `{learning['pattern_key']}`  \n"
            f"- **Area:** {learning.get('area') or 'unknown'}  \n"
            f"- **Promotion-lane:** spec — Rule 12: route to governance Spec / "
            f"`projects/biotech_screener.md`, not a skill patch.\n\n"
            f"**Summary:** {learning['summary'] or '(none)'}\n\n"
            f"**Operator action:** open or update a Spec under `specs/changes/`; "
            f"do not patch `skills/` from this LRN.\n"
        )
    if learning.get("promotion_lane") == "none":
        return (
            f"## SKIP (none lane) — {learning['id']}\n\n"
            f"- **Pattern-Key:** `{learning['pattern_key']}` — log only, no promotion.\n"
        )

    target = learning.get("skill_path") or learning.get("skill_candidate") or "screener_ops"
    note = ""
    if target in FROZEN_SKILL_TARGETS:
        note = (
            f"\n> ⚠ BLOCKED: target `{target}` encodes production behavior. "
            f"Requires a governance Spec, not a learnings patch. Re-route to a "
            f"docs/plumbing skill or open a Spec.\n"
        )
        target = "(needs operator re-route)"
    return (
        f"## Proposed skill patch — {learning['id']}\n\n"
        f"- **Pattern-Key:** `{learning['pattern_key']}`  \n"
        f"- **Recurrence-Count:** {learning['recurrence']}  \n"
        f"- **Promotion-lane:** skill  \n"
        f"- **Suggested target skill:** `{target}`\n"
        f"{note}\n"
        f"**Why now:** recurred {learning['recurrence']}× — meets promotion threshold.\n\n"
        f"**Summary:** {learning['summary'] or '(none)'}\n\n"
        f"**Proposed addition to skill doc (operator to review/edit):**\n\n"
        f"```markdown\n"
        f"### {learning['title'].replace('_', ' ').title()}\n"
        f"{learning['action'] or learning['summary'] or '(fill in from LRN entry)'}\n"
        f"(source: {learning['id']}, Pattern-Key {learning['pattern_key']})\n"
        f"```\n\n"
        f"**Apply path:** edit `skills/{target}/SKILL.md` -> "
        f"`python3 tools/sync_hermes_skills.py` -> `python3 tools/audit_hermes_skills.py` "
        f"-> log to `docs/hermes_skills/harvest_log.md` -> commit on a branch.\n\n"
        f"**Efficacy back-check:** harvest_log verification block 2 weeks post-merge (Rule 12).\n"
    )


def main() -> int:
    if os.getenv("SELFIMPROVE_GATES_MET") != "1":
        print(
            "pattern_to_skillpatch: SELFIMPROVE_GATES_MET not set — tool is staged, not active. "
            "Set SELFIMPROVE_GATES_MET=1 to run.",
            file=sys.stderr,
        )
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--learnings", default=".learnings/LEARNINGS.md")
    ap.add_argument("--min-recurrence", type=int, default=3)
    ap.add_argument("--out", default="artifacts/skill_patch_drafts")
    args = ap.parse_args()

    src = Path(args.learnings)
    if not src.exists():
        print(f"LEARNINGS file not found: {src}")
        return 1

    entries = list(parse_learnings(src.read_text(encoding="utf-8")))
    eligible = [
        e
        for e in entries
        if e["recurrence"] >= args.min_recurrence
        and e["pattern_key"]
        and e["status"] in ("pending", "unknown", "promoted", "resolved")
    ]

    refused = refuse_spec_lane_entries(eligible)
    if refused:
        ids = ", ".join(e["id"] for e in refused)
        print(f"Rule 12 lane gate: {len(refused)} spec-lane entry refused ({ids})")

    print(
        f"Scanned {len(entries)} LRN entries; "
        f"{len(eligible)} meet recurrence >= {args.min_recurrence} with a Pattern-Key."
    )
    if not eligible:
        print("Nothing to promote. (This is the expected steady state most weeks.)")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = [
        f"# Skill-patch drafts — {stamp}",
        "",
        f"{len(eligible)} pattern(s) reviewed. DRAFTS ONLY — operator approves before editing skills.",
        "",
    ]
    blocked = 0
    for e in eligible:
        patch = draft_patch(e)
        if "BLOCKED" in patch or "SKIP" in patch:
            blocked += 1
        report.append(patch)
        report.append("---\n")

    out_file = out_dir / f"skill_patch_drafts_{stamp}.md"
    out_file.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {len(eligible)} draft(s) -> {out_file}")
    if blocked:
        print(f"  ({blocked} blocked/skipped: spec lane or frozen production skill)")
    if refused:
        print(f"  ({len(refused)} spec-lane entry refused — governance Spec only)")
    print("Operator action required: review drafts, then manually apply eligible ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
