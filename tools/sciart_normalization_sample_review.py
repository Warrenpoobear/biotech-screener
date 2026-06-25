#!/usr/bin/env python3
"""Sci-Cart Phase 13.2 (R4): stratified normalization sample review.

Draws a deterministic sample of CT.gov condition strings grouped by top
normalized disease targets. Emits a markdown worksheet for manual TRUE_POSITIVE /
FALSE_POSITIVE / AMBIGUOUS verdicts.

Usage:
    python3 tools/sciart_normalization_sample_review.py
    python3 tools/sciart_normalization_sample_review.py --write
    python3 tools/sciart_normalization_sample_review.py --trials-file production_data/trial_records.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
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
        f"**Generated:** {datetime.now().isoformat()}",
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Sci-Cart R4 normalization sample review")
    ap.add_argument("--as-of-date", help="YYYY-MM-DD for normalizer (default: today)")
    ap.add_argument("--trials-file", type=Path, default=DEFAULT_TRIALS)
    ap.add_argument("--write", action="store_true", help="Write governance markdown worksheet")
    ap.add_argument("--json", action="store_true", help="Print sample JSON")
    args = ap.parse_args()

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
