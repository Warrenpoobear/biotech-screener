#!/usr/bin/env python3
"""Read-only audit of .learnings/ for recursive self-improvement hygiene.

Reports tier line counts, Pattern-Key recurrence, promotion candidates, and
simple stale hints. Does not modify any file.

Usage:
    python3 tools/audit_learnings.py
    python3 tools/audit_learnings.py --json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEARNINGS = REPO / ".learnings"
ENV_JSON = REPO / ".cursor" / "environment.json"

TIER_LIMITS = {
    "memory.md": 100,
    "projects/biotech_screener.md": 200,
}

DOMAIN_LIMIT = 200
BOOTSTRAP_MARKER = "## Bootstrap (read first)"
PATTERN_IN_MEMORY = re.compile(r"\*\*([a-z][a-z0-9_]+)\*\*:|Pattern-Key:\s*(\S+)", re.IGNORECASE)

LRN_HEADING = re.compile(r"^## \[(LRN-\d{8}-\d{3})\]", re.MULTILINE)
META_LINE = re.compile(
    r"^- (Pattern-Key|Recurrence-Count|Skill-Path|Status|Promotion-lane|Area):\s*(.+)$",
    re.MULTILINE,
)
AREA_LINE = re.compile(r"^\*\*Area\*\*:\s*(\w+)", re.MULTILINE)
STATUS_LINE = re.compile(r"^\*\*Status\*\*:\s*(\w+)", re.MULTILINE)


from tools.pattern_to_skillpatch import infer_promotion_lane


@dataclass
class LrnEntry:
    lrn_id: str
    status: str = "unknown"
    pattern_key: str | None = None
    recurrence_count: int = 0
    skill_path: str | None = None
    area: str | None = None
    promotion_lane: str | None = None


@dataclass
class AuditReport:
    tier_lines: dict[str, dict] = field(default_factory=dict)
    lrn_total: int = 0
    pattern_groups: dict[str, list[str]] = field(default_factory=dict)
    promotion_candidates: list[dict] = field(default_factory=list)
    skill_candidates: list[dict] = field(default_factory=list)
    spec_lane_blocked: list[dict] = field(default_factory=list)
    stale_hints: list[str] = field(default_factory=list)
    domain_files: list[str] = field(default_factory=list)
    memory_bootstrap_lines: int = 0
    memory_patterns_hot: list[str] = field(default_factory=list)
    lrn_pending_but_in_hot: list[dict] = field(default_factory=list)
    compaction_hints: list[str] = field(default_factory=list)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def parse_learnings(text: str) -> list[LrnEntry]:
    entries: list[LrnEntry] = []
    parts = LRN_HEADING.split(text)
    # parts[0] is preamble; then alternating id, body
    i = 1
    while i + 1 < len(parts):
        lrn_id = parts[i].strip()
        body = parts[i + 1]
        entry = LrnEntry(lrn_id=lrn_id)
        status_m = STATUS_LINE.search(body)
        if status_m:
            entry.status = status_m.group(1).lower()
        area_m = AREA_LINE.search(body)
        if area_m:
            entry.area = area_m.group(1)
        for key, val in META_LINE.findall(body):
            val = val.strip()
            if key == "Pattern-Key":
                entry.pattern_key = val
            elif key == "Recurrence-Count":
                try:
                    entry.recurrence_count = int(val)
                except ValueError:
                    pass
            elif key == "Skill-Path":
                entry.skill_path = val
            elif key == "Status":
                entry.status = val.lower()
            elif key == "Promotion-lane":
                entry.promotion_lane = val.lower()
            elif key == "Area":
                entry.area = val
        entry.promotion_lane = infer_promotion_lane(entry.area, entry.promotion_lane)
        entries.append(entry)
        i += 2
    return entries


def _bootstrap_line_count(memory_text: str) -> int:
    if BOOTSTRAP_MARKER not in memory_text:
        return 0
    start = memory_text.index(BOOTSTRAP_MARKER)
    rest = memory_text[start:]
    end = rest.find("\n---\n")
    block = rest[:end] if end != -1 else rest
    return len(block.splitlines())


def _hot_pattern_keys(memory_text: str) -> set[str]:
    keys: set[str] = set()
    for m in PATTERN_IN_MEMORY.finditer(memory_text):
        keys.add((m.group(1) or m.group(2) or "").lower())
    if "raw_count_size_confound" in memory_text.lower():
        keys.add("raw_count_size_confound")
    return {k for k in keys if k}


def _compaction_hints(memory_text: str, domain_text: str) -> list[str]:
    hints: list[str] = []
    if memory_text.count("CodeGraph") > 2 and "domains/agent_ops" not in memory_text:
        hints.append("memory.md repeats CodeGraph — prefer pointer to domains/agent_ops.md")
    if memory_text.count("WSL") > 2 and "domains/agent_ops" not in memory_text:
        hints.append("memory.md repeats WSL authority — consolidate in Bootstrap table + domains/")
    if _line_count_from_text(memory_text) > 70:
        hints.append("memory.md >70 lines — demote detail to WARM tiers before next promotion")
    if domain_text and "Skill ↔ knowledge recursion" in domain_text and "Recursion" in memory_text:
        pass  # intentional overlap via bootstrap pointer
    return hints


def _line_count_from_text(text: str) -> int:
    return len(text.splitlines())


def _codegraph_pin() -> str | None:
    if not ENV_JSON.exists():
        return None
    m = re.search(r"codegraph@([\d.]+)", ENV_JSON.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def build_report() -> AuditReport:
    report = AuditReport()

    for rel, limit in TIER_LIMITS.items():
        path = LEARNINGS / rel
        n = _line_count(path)
        report.tier_lines[rel] = {
            "lines": n,
            "limit": limit,
            "over": max(0, n - limit),
            "exists": path.exists(),
        }

    domains = LEARNINGS / "domains"
    domain_text = ""
    if domains.is_dir():
        report.domain_files = sorted(p.name for p in domains.glob("*.md"))
        for p in domains.glob("*.md"):
            n = _line_count(p)
            report.tier_lines[f"domains/{p.name}"] = {
                "lines": n,
                "limit": DOMAIN_LIMIT,
                "over": max(0, n - DOMAIN_LIMIT),
                "exists": True,
            }
            domain_text += p.read_text(encoding="utf-8") + "\n"

    learnings_path = LEARNINGS / "LEARNINGS.md"
    if not learnings_path.exists():
        report.stale_hints.append("Missing .learnings/LEARNINGS.md")
        return report

    entries = parse_learnings(learnings_path.read_text(encoding="utf-8"))
    report.lrn_total = len(entries)

    by_pattern: dict[str, list[LrnEntry]] = defaultdict(list)
    for e in entries:
        if e.pattern_key:
            by_pattern[e.pattern_key].append(e)
            report.pattern_groups.setdefault(e.pattern_key, []).append(e.lrn_id)

    for pattern, group in sorted(by_pattern.items()):
        total_rec = sum(e.recurrence_count for e in group)
        pending = [e for e in group if e.status in ("pending", "unknown")]
        if total_rec >= 3 and pending:
            report.promotion_candidates.append(
                {
                    "pattern_key": pattern,
                    "total_recurrence": total_rec,
                    "pending_lrns": [e.lrn_id for e in pending],
                    "action": "promote to memory.md or projects/biotech_screener.md",
                }
            )
        for e in group:
            if e.promotion_lane == "spec" and e.recurrence_count >= 2:
                report.spec_lane_blocked.append(
                    {
                        "lrn_id": e.lrn_id,
                        "pattern_key": pattern,
                        "area": e.area,
                        "recurrence": e.recurrence_count,
                    }
                )
            if (
                e.skill_path
                and e.status in ("pending", "resolved")
                and e.recurrence_count >= 2
                and e.promotion_lane == "skill"
            ):
                report.skill_candidates.append(
                    {
                        "lrn_id": e.lrn_id,
                        "pattern_key": pattern,
                        "skill_path": e.skill_path,
                        "status": e.status,
                        "promotion_lane": e.promotion_lane,
                    }
                )

    memory_path = LEARNINGS / "memory.md"
    memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    report.memory_bootstrap_lines = _bootstrap_line_count(memory_text)
    report.memory_patterns_hot = sorted(_hot_pattern_keys(memory_text))
    report.compaction_hints = _compaction_hints(memory_text, domain_text)

    hot_patterns = _hot_pattern_keys(memory_text)
    for pattern, group in by_pattern.items():
        if pattern.lower() in hot_patterns or pattern.replace("-", "_") in hot_patterns:
            for e in group:
                if e.status in ("pending", "unknown"):
                    report.lrn_pending_but_in_hot.append(
                        {
                            "lrn_id": e.lrn_id,
                            "pattern_key": pattern,
                            "action": "set LEARNINGS Status to promoted",
                        }
                    )

    pin = _codegraph_pin()
    if pin and pin not in memory_text and "codegraph" in memory_text.lower():
        report.stale_hints.append(
            f"memory.md may stale-pin codegraph; environment.json has @{pin}"
        )
    if "pytest-xdist" in memory_text.lower() and "not required" not in memory_text.lower():
        report.stale_hints.append("memory.md: verify pytest-xdist guidance (see LRN-20260528-002)")

    return report


def print_report(report: AuditReport) -> None:
    print("# Learnings Audit\n")
    print("## Tier line counts\n")
    for rel, info in report.tier_lines.items():
        flag = " OVER LIMIT" if info.get("over", 0) > 0 else ""
        print(f"- `{rel}`: {info['lines']}/{info['limit']} lines{flag}")
    if report.domain_files:
        print(f"\nDomain files: {', '.join(report.domain_files)}")
    print(f"\n## HOT memory\n- Bootstrap block: {report.memory_bootstrap_lines} lines")
    if report.memory_patterns_hot:
        print(f"- Pattern-Keys in HOT: {', '.join(report.memory_patterns_hot)}")
    print(f"\n## LEARNINGS.md\n- Entries: {report.lrn_total}")
    print(f"- Distinct Pattern-Keys: {len(report.pattern_groups)}")

    print("\n## Promotion candidates (Pattern-Key ≥3 recurrence, still pending)\n")
    if not report.promotion_candidates:
        print("None.")
    else:
        for c in report.promotion_candidates:
            print(f"- `{c['pattern_key']}` (rec={c['total_recurrence']}): {', '.join(c['pending_lrns'])}")

    print("\n## Skill-patch candidates (Skill-Path + recurrence ≥2)\n")
    if not report.skill_candidates:
        print("None.")
    else:
        for c in report.skill_candidates[:15]:
            print(f"- {c['lrn_id']} → skills/{c['skill_path']}/ (lane={c.get('promotion_lane', 'skill')}, {c['status']})")
        if len(report.skill_candidates) > 15:
            print(f"  ... and {len(report.skill_candidates) - 15} more")

    print("\n## Spec-lane blocked (require governance Spec, not skill patch)\n")
    if not report.spec_lane_blocked:
        print("None.")
    else:
        for c in report.spec_lane_blocked[:15]:
            print(f"- {c['lrn_id']}: `{c['pattern_key']}` (area={c.get('area')}, rec={c['recurrence']})")
        if len(report.spec_lane_blocked) > 15:
            print(f"  ... and {len(report.spec_lane_blocked) - 15} more")

    print("\n## LRN pending but already in HOT memory\n")
    if not report.lrn_pending_but_in_hot:
        print("None.")
    else:
        for item in report.lrn_pending_but_in_hot:
            print(f"- {item['lrn_id']} (`{item['pattern_key']}`): {item['action']}")

    print("\n## Memory compaction hints\n")
    if not report.compaction_hints:
        print("None.")
    else:
        for h in report.compaction_hints:
            print(f"- {h}")

    print("\n## Stale hints\n")
    if not report.stale_hints:
        print("None.")
    else:
        for h in report.stale_hints:
            print(f"- {h}")

    print("\n## Promotion checklist (Rule 12 — shared with Town)\n")
    print("| Gate | Threshold | Action |")
    print("| --- | --- | --- |")
    print("| Recurrence | Pattern-Key >=3 (7d behavioral; all-time failure modes) | HOT memory.md or domains/ |")
    print("| Skill-path + recurrence | Skill-Path + rec >=2 | Draft patch (no auto-merge) |")
    print("| Operator verdict | >=3 helpful on same skill | Eligible for skill merge |")
    print("| Observation | 7+ days true-PIT telemetry | Eligible for routing changes |")
    print("")
    print("Feeds: LEARNINGS.md + failure-patterns (Hermes); Town Correction Ledger (rec>=3).")
    print("Lane: Promotion-lane spec -> governance Spec only (F-2026-001: do not fork thresholds).")

    print("\n## Commands")
    print("  SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py --min-recurrence 3")
    print("  python3 tools/build_hermes_knowledge_layer.py  # ops ledgers")
    print("  See skills/self-improving/SKILL.md Rule 12 and .learnings/README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit .learnings/ for recursion hygiene")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print_report(report)

    over = any(t.get("over", 0) > 0 for t in report.tier_lines.values())
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
