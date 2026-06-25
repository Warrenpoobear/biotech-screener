#!/usr/bin/env python3
"""Sci-Cart Phase 13.2 (R4): stratified normalization sample review.

Draws a deterministic sample of CT.gov condition strings grouped by top
normalized disease targets. Emits a markdown worksheet for manual TRUE_POSITIVE /
FALSE_POSITIVE / AMBIGUOUS verdicts.

Usage:
    python3 tools/sciart_normalization_sample_review.py
    python3 tools/sciart_normalization_sample_review.py --write
    python3 tools/sciart_normalization_sample_review.py --summarize docs/governance/SCIART_PHASE13_2_*.md
    python3 tools/sciart_normalization_sample_review.py --trials-file production_data/trial_records.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "governance"
DEFAULT_TRIALS = REPO / "production_data" / "trial_records.json"
SAMPLE_PER_DISEASE = 10
TOP_DISEASES = 5
SEED = 42

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TOP_DISEASE_TARGETS = [
    "lymphoma",
    "breast cancer",
    "non-small cell lung cancer",
    "colorectal cancer",
    "melanoma",
]

VALID_VERDICTS = frozenset({"TRUE_POSITIVE", "FALSE_POSITIVE", "AMBIGUOUS"})


def _match_tier(source: str) -> str:
    mapping = {
        "manual_override": "exact",
        "mondo": "exact",
        "mondo_synonym": "synonym",
        "mondo_substring": "substring",
    }
    return mapping.get(source, source or "unknown")


def collect_condition_rows(trials_path: Path, as_of: str) -> list[dict[str, object]]:
    from scientific_cartography.ingest.ctgov_ingest import CTGovIngest
    from scientific_cartography.normalize.disease_normalizer import DiseaseNormalizer

    ingest = CTGovIngest(as_of_date=as_of)
    trials = ingest.ingest_from_json_file(trials_path)
    normalizer = DiseaseNormalizer(as_of_date=as_of)

    rows: list[dict[str, object]] = []
    for trial in trials:
        for condition in trial.conditions or []:
            if not condition or not isinstance(condition, str):
                continue
            record = normalizer.normalize(condition)
            rows.append(
                {
                    "raw_condition": condition,
                    "normalized_name": record.normalized_name,
                    "mondo_id": record.mondo_id,
                    "match_tier": _match_tier(record.source),
                    "confidence": record.confidence,
                    "therapeutic_area": record.therapeutic_area,
                    "nct_id": trial.nct_id,
                }
            )
    return rows


def build_sample(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        norm = str(row["normalized_name"]).lower()
        for target in TOP_DISEASE_TARGETS:
            if target in norm or norm == target:
                by_target[target].append(row)
                break

    rng = random.Random(SEED)
    sample: list[dict[str, object]] = []
    for target in TOP_DISEASE_TARGETS:
        pool = by_target.get(target, [])
        if not pool:
            continue
        rng.shuffle(pool)
        for row in pool[:SAMPLE_PER_DISEASE]:
            item = dict(row)
            item["target_bucket"] = target
            item["manual_verdict"] = ""
            sample.append(item)
    return sample


def render_markdown(sample: list[dict[str, object]], *, trials_path: Path, as_of: str) -> str:
    counts = Counter(str(r["target_bucket"]) for r in sample)
    try:
        trials_label = str(trials_path.relative_to(REPO))
    except ValueError:
        trials_label = str(trials_path)
    lines = [
        "# Sci-Cart Phase 13.2 — Normalization Sample Review",
        "",
        f"**Generated:** {as_of}T00:00:00Z",
        f"**Trials source:** `{trials_label}`",
        f"**As-of:** {as_of}",
        f"**Sample:** {len(sample)} records ({SAMPLE_PER_DISEASE} per top disease, seed={SEED})",
        "",
        "Manual verdict per row: `TRUE_POSITIVE` | `FALSE_POSITIVE` | `AMBIGUOUS`",
        "",
        "## Sample counts",
        "",
    ]
    for target in TOP_DISEASE_TARGETS:
        lines.append(f"- {target}: {counts.get(target, 0)}")
    lines.extend(["", "## Annotated sample", ""])
    lines.append(
        "| target | nct_id | raw_condition | normalized | mondo_id | tier | confidence | verdict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sample:
        raw = str(row["raw_condition"]).replace("|", "\\|")[:80]
        lines.append(
            f"| {row['target_bucket']} | {row.get('nct_id', '')} | {raw} | "
            f"{row['normalized_name']} | {row.get('mondo_id') or ''} | {row['match_tier']} | "
            f"{row['confidence']} | {row['manual_verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict guidance",
            "",
            "- **TRUE_POSITIVE**: MONDO parent/synonym is clinically valid",
            "- **FALSE_POSITIVE**: match would mislead the disease map",
            "- **AMBIGUOUS**: defensible but imprecise subtype→parent mapping",
            "",
        ]
    )
    return "\n".join(lines)


def parse_worksheet_verdicts(worksheet_path: Path) -> list[dict[str, str]]:
    """Parse annotated sample table from a filled R4 worksheet."""
    lines = worksheet_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    in_table = False
    for line in lines:
        if line.startswith("| target | nct_id |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("| ") or line.startswith("| ---"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        rows.append(
            {
                "target": cells[0],
                "nct_id": cells[1],
                "match_tier": cells[5],
                "verdict": cells[7].strip().upper().replace(" ", "_"),
            }
        )
    return rows


def summarize_verdicts(parsed_rows: list[dict[str, str]], *, worksheet: Path) -> dict[str, object]:
    verdict_counts = Counter(r["verdict"] for r in parsed_rows if r["verdict"])
    pending = sum(1 for r in parsed_rows if not r["verdict"])
    labeled = len(parsed_rows) - pending
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for row in parsed_rows:
        if row["verdict"]:
            by_target[row["target"]][row["verdict"]] += 1

    tp = verdict_counts.get("TRUE_POSITIVE", 0)
    fp = verdict_counts.get("FALSE_POSITIVE", 0)
    amb = verdict_counts.get("AMBIGUOUS", 0)
    precision_denom = tp + fp
    precision = round(tp / precision_denom, 4) if precision_denom else None

    invalid = sorted({v for v in verdict_counts if v not in VALID_VERDICTS})
    complete = pending == 0 and not invalid

    return {
        "schema": "sciart_r4_verdict_summary.v1",
        "worksheet": str(worksheet),
        "total_rows": len(parsed_rows),
        "labeled_rows": labeled,
        "pending_rows": pending,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "by_target": {k: dict(sorted(v.items())) for k, v in sorted(by_target.items())},
        "precision_true_positive": precision,
        "complete": complete,
        "invalid_verdicts": invalid,
        "r4_pass_threshold": "precision_true_positive >= 0.85 and pending_rows == 0",
        "r4_pass": bool(complete and precision is not None and precision >= 0.85),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sci-Cart R4 normalization sample review")
    ap.add_argument("--as-of-date", help="YYYY-MM-DD for normalizer (default: today)")
    ap.add_argument("--trials-file", type=Path, default=DEFAULT_TRIALS)
    ap.add_argument("--worksheet", type=Path, help="Filled R4 worksheet for --summarize")
    ap.add_argument("--write", action="store_true", help="Write governance markdown worksheet")
    ap.add_argument("--summarize", action="store_true", help="Summarize verdicts from --worksheet")
    ap.add_argument("--json", action="store_true", help="Print sample JSON")
    args = ap.parse_args()

    if args.summarize:
        worksheet = args.worksheet
        if worksheet is None:
            print("--summarize requires --worksheet PATH", file=sys.stderr)
            return 2
        if not worksheet.is_file():
            print(f"worksheet not found: {worksheet}", file=sys.stderr)
            return 2
        parsed = parse_worksheet_verdicts(worksheet)
        if not parsed:
            print("No verdict rows parsed from worksheet", file=sys.stderr)
            return 1
        summary = summarize_verdicts(parsed, worksheet=worksheet)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    as_of = args.as_of_date or date.today().isoformat()
    trials_path = args.trials_file
    if not trials_path.is_file():
        print(f"trials file not found: {trials_path}", file=sys.stderr)
        return 2

    rows = collect_condition_rows(trials_path, as_of)
    sample = build_sample(rows)
    if not sample:
        print("No sample rows drawn — check trials file and disease targets", file=sys.stderr)
        return 1

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"SCIART_PHASE13_2_NORMALIZATION_SAMPLE_REVIEW_{as_of.replace('-', '_')}.md"
        out_path.write_text(render_markdown(sample, trials_path=trials_path, as_of=as_of), encoding="utf-8")
        if not args.json:
            print(f"Wrote {out_path}")

    if args.json:
        print(json.dumps(sample, indent=2, sort_keys=True))
    elif not args.write:
        print(render_markdown(sample, trials_path=trials_path, as_of=as_of))

    return 0


if __name__ == "__main__":
    sys.exit(main())
