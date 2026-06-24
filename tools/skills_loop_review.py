#!/usr/bin/env python3
"""Shared helpers for the self-improvement loop review (trim, efficacy, contradiction).

Advisory-only — no auto-delete, no auto-merge. Used by monthly report, weekly digest,
and pattern_to_skillpatch pre-draft gate.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
META_JSON = REPO / "docs" / "hermes_skills" / "_meta.json"
HARVEST_LOG = REPO / "docs" / "hermes_skills" / "harvest_log.md"
MEMORY_MD = REPO / ".learnings" / "memory.md"
LOGS_DIR = REPO / "artifacts" / "skills_learning"

TRIM_UNUSED_DAYS = 30
EFFICACY_GRACE_DAYS = 14

PROHIBITION_RE = re.compile(
    r"(?i)(?:never|do\s+not|must\s+not|don't)\s+([^.!\n]{8,120})",
)
HARVEST_DATE_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})", re.MULTILINE)
SKILL_PATCH_RE = re.compile(
    r"\*\*([a-z][\w-]+)\*\*|`([a-z][\w-]+)`|skills/([\w-]+)/SKILL\.md",
    re.IGNORECASE,
)
VERIFY_MARKERS = re.compile(
    r"(?i)verification|efficacy back-check|0 recurrence|verify \(2 weeks",
)
STALLED_ID_RE = re.compile(r"\|\s*(F-2026-00[56])\s*\|")


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_registered_skill_keys() -> list[str]:
    """Registered Hermes skill keys from _meta.json."""
    if not META_JSON.exists():
        return []
    data = json.loads(META_JSON.read_text(encoding="utf-8"))
    return sorted(data.get("skills", {}).keys())


def load_executions_in_window(
    days: int = TRIM_UNUSED_DAYS,
    environment: str = "prod",
    as_of: date | None = None,
    logs_dir: Path | None = None,
) -> dict[str, int]:
    """Return skill_name -> execution count in the last `days` days."""
    as_of = as_of or date.today()
    cutoff = datetime.combine(as_of - timedelta(days=days), datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    logs_dir = logs_dir or LOGS_DIR
    counts: dict[str, int] = {}

    if not logs_dir.exists():
        return counts

    legacy_re = re.compile(r"^execution_log_\d{4}-\d{2}\.jsonl$")

    for path in sorted(logs_dir.glob("execution_log*.jsonl")):
        name = path.name
        if f"_{environment}_" in name:
            pass
        elif environment == "prod" and legacy_re.match(name):
            pass
        else:
            continue

        for row in load_jsonl(path):
            ts_raw = row.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts < cutoff:
                continue
            row_env = row.get("environment")
            if row_env is not None and row_env != environment:
                continue
            skill = row.get("skill_name", "unknown")
            counts[skill] = counts.get(skill, 0) + 1
    return counts


def trim_candidates(
    days: int = TRIM_UNUSED_DAYS,
    environment: str = "prod",
    as_of: date | None = None,
    logs_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Skills registered in Hermes with 0 telemetry loads in the last N days."""
    as_of = as_of or date.today()
    registered = load_registered_skill_keys()
    loads = load_executions_in_window(days=days, environment=environment, as_of=as_of, logs_dir=logs_dir)

    # Normalize load keys for matching (underscore vs hyphen)
    load_keys = set(loads.keys())
    normalized_loads: set[str] = set()
    for k in load_keys:
        normalized_loads.add(k)
        normalized_loads.add(k.replace("_", "-"))
        normalized_loads.add(k.replace("-", "_"))

    candidates: list[dict[str, Any]] = []
    for skill in registered:
        variants = {skill, skill.replace("-", "_"), skill.replace("_", "-")}
        if variants & normalized_loads:
            continue
        candidates.append(
            {
                "skill": skill,
                "days": days,
                "action": "demote/archive candidate — 0 loads; review Hermes mirror + skills/ source",
            }
        )
    return candidates


def resolve_skill_file(skill_slug: str, repo: Path | None = None) -> Path | None:
    repo = repo or REPO
    slug = skill_slug.strip()
    for candidate in (
        repo / "skills" / slug / "SKILL.md",
        repo / "skills" / slug.replace("-", "_") / "SKILL.md",
        repo / "skills" / slug.replace("_", "-") / "SKILL.md",
    ):
        if candidate.exists():
            return candidate
    return None


def check_skill_contradiction(skill_text: str, lesson_text: str) -> list[str]:
    """Return human-readable contradiction flags between skill body and lesson text."""
    conflicts: list[str] = []
    lesson_lower = lesson_text.lower()

    for match in PROHIBITION_RE.finditer(skill_text):
        prohibition = _normalize_phrase(match.group(1))
        if len(prohibition) < 8:
            continue
        needle = prohibition[: min(60, len(prohibition))]
        idx = lesson_lower.find(needle)
        if idx == -1:
            continue
        window = lesson_lower[max(0, idx - 40) : idx]
        if re.search(r"(?:never|do not|must not|don't)\s", window):
            continue
        conflicts.append(
            f"Lesson may contradict skill prohibition: “{prohibition[:80]}”"
        )

    for match in re.finditer(r"(?i)(?:always|must)\s+([^.!\n]{8,120})", skill_text):
        requirement = _normalize_phrase(match.group(1))
        if len(requirement) < 8:
            continue
        needle = requirement[: min(60, len(requirement))]
        idx = lesson_lower.find(needle)
        if idx == -1:
            continue
        window = lesson_lower[max(0, idx - 40) : idx + len(needle) + 10]
        if re.search(r"(?:never|do not|must not|don't|avoid)\s", window):
            conflicts.append(
                f"Lesson may contradict skill requirement: “{requirement[:80]}”"
            )

    return conflicts


def check_learning_contradiction(learning: dict, repo: Path | None = None) -> list[str]:
    """Check an LRN entry against its target skill file."""
    target = learning.get("skill_path") or learning.get("skill_candidate")
    if not target or learning.get("promotion_lane") != "skill":
        return []
    skill_file = resolve_skill_file(target, repo)
    if not skill_file:
        return []
    lesson = "\n".join(
        filter(
            None,
            [
                learning.get("summary") or "",
                learning.get("action") or "",
                learning.get("title") or "",
            ],
        )
    )
    return check_skill_contradiction(skill_file.read_text(encoding="utf-8"), lesson)


def _parse_harvest_sections(text: str) -> list[dict[str, Any]]:
    """Split harvest_log into dated sections."""
    sections: list[dict[str, Any]] = []
    matches = list(HARVEST_DATE_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        sections.append(
            {
                "date": date.fromisoformat(m.group(1)),
                "body": body,
                "is_patch": "### Skill patches" in body or "### Skill patch" in body,
                "is_verify": bool(VERIFY_MARKERS.search(body)),
            }
        )
    return sections


def _skills_in_patch_section(body: str) -> set[str]:
    skills: set[str] = set()
    if "### Skill patch" not in body:
        return skills
    patch_block = body.split("### Skill patches", 1)[-1]
    patch_block = patch_block.split("###", 1)[0]
    for m in SKILL_PATCH_RE.finditer(patch_block):
        for g in m.groups():
            if g:
                skills.add(g.replace("_", "-"))
    return skills


def efficacy_overdue(
    grace_days: int = EFFICACY_GRACE_DAYS,
    as_of: date | None = None,
    harvest_path: Path | None = None,
    stalled_open: bool = True,
) -> list[dict[str, Any]]:
    """Patches in harvest_log older than grace_days without a verification block."""
    if stalled_open:
        return [
            {
                "reason": "stalled-loop",
                "detail": "F-2026-005/F-2026-006 OPEN — efficacy checks blocked until host confirms recovery",
            }
        ]

    path = harvest_path or HARVEST_LOG
    if not path.exists():
        return []

    as_of = as_of or date.today()
    sections = _parse_harvest_sections(path.read_text(encoding="utf-8"))
    overdue: list[dict[str, Any]] = []

    for sec in sections:
        if not sec["is_patch"]:
            continue
        merge_date: date = sec["date"]
        if (as_of - merge_date).days < grace_days:
            continue
        skills = _skills_in_patch_section(sec["body"])
        if not skills:
            continue
        verified: set[str] = set()
        for later in sections:
            if later["date"] <= merge_date:
                continue
            if not later["is_verify"]:
                continue
            for skill in skills:
                if skill in later["body"] or skill.replace("-", "_") in later["body"]:
                    verified.add(skill)
        for skill in sorted(skills - verified):
            overdue.append(
                {
                    "skill": skill,
                    "merged": merge_date.isoformat(),
                    "days_since_merge": (as_of - merge_date).days,
                    "action": "append harvest_log verification block (Rule 12)",
                }
            )
    return overdue


def stalled_loop_entries(memory_path: Path | None = None) -> list[dict[str, Any]]:
    """Parse OPEN stalled-loop rows from memory.md."""
    path = memory_path or MEMORY_MD
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if "Stalled-loop verdicts" not in text:
        return []
    rows: list[dict[str, Any]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| F-2026-"):
            in_table = True
        if not in_table or not line.startswith("|"):
            continue
        if line.startswith("| ---"):
            continue
        if "ID" in line and "System" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        fid, system, status = parts[0], parts[1], parts[2].replace("*", "").strip()
        if fid.startswith("F-2026-"):
            rows.append({"id": fid, "system": system, "status": status})
    return rows


def stalled_loops_open(memory_path: Path | None = None) -> bool:
    entries = stalled_loop_entries(memory_path)
    return any(e.get("status", "").upper() == "OPEN" for e in entries)


def format_loop_review_sections(
    *,
    environment: str = "prod",
    as_of: date | None = None,
    logs_dir: Path | None = None,
    harvest_path: Path | None = None,
    memory_path: Path | None = None,
) -> list[str]:
    """Markdown sections for trim list, efficacy overdue, stalled loops."""
    as_of = as_of or date.today()
    lines: list[str] = []

    lines.extend(["## Trim candidates (0 loads in 30 days)", ""])
    trim = trim_candidates(environment=environment, as_of=as_of, logs_dir=logs_dir)
    if not trim:
        lines.append("None — all registered skills had telemetry in the last 30 days (or no registry).")
    else:
        lines.append(
            f"**{len(trim)} skill(s)** with 0 prod loads in {TRIM_UNUSED_DAYS}d — "
            "demote/archive candidates (operator review; no auto-delete):"
        )
        lines.append("")
        for item in trim[:25]:
            lines.append(f"- `{item['skill']}` — {item['action']}")
        if len(trim) > 25:
            lines.append(f"- ... and {len(trim) - 25} more")
    lines.append("")

    lines.extend(["## Efficacy overdue (patches without 14-day verification)", ""])
    stalled = stalled_loop_entries(memory_path)
    open_stalled = [s for s in stalled if s.get("status", "").upper() == "OPEN"]
    if open_stalled:
        lines.append("**Blocked** — stalled-loop verdicts still OPEN:")
        for s in open_stalled:
            lines.append(f"- `{s['id']}` ({s['system']}): confirm host recovery before efficacy tracking")
        lines.append("")
        lines.append("Rule 12: cannot measure recurrence-after-fix on unconfirmed outages.")
    else:
        overdue = efficacy_overdue(as_of=as_of, harvest_path=harvest_path, stalled_open=False)
        if not overdue:
            lines.append("None — all merges past grace period have verification blocks (or no patch merges).")
        else:
            for item in overdue:
                lines.append(
                    f"- `{item['skill']}` merged {item['merged']} "
                    f"({item['days_since_merge']}d ago) — {item['action']}"
                )
    lines.append("")

    lines.extend(["## Stalled-loop PENDINGs", ""])
    if not stalled:
        lines.append("None in memory.md stalled-loop table.")
    else:
        for s in stalled:
            lines.append(f"- `{s['id']}` **{s['system']}**: {s['status']}")
    lines.append("")

    return lines
