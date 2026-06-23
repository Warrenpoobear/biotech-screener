"""Scientific cartography MCP tools.

Read-only diagnostic access to scientific_cartography package metadata and
latest generated artifacts. This module deliberately does not write files,
start jobs, call the network, or feed cartography fields into ranking/selector
logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_server.app import mcp
from mcp_server.config import PROJECT_ROOT

CARTOGRAPHY_DIR = PROJECT_ROOT / "scientific_cartography"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "scientific_cartography"


def _json_default(value: Any) -> str:
    return str(value)


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _latest_artifact_dir() -> Path | None:
    if not ARTIFACTS_ROOT.exists():
        return None
    candidates = sorted((p for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    return candidates[0] if candidates else None


def _list_python_modules(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.py") if path.name != "__init__.py")


def _line_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _base_payload(category: str) -> dict[str, Any]:
    latest = _latest_artifact_dir()
    return {
        "module": "scientific_cartography",
        "category": category or "overview",
        "path": str(CARTOGRAPHY_DIR),
        "artifact_root": str(ARTIFACTS_ROOT),
        "latest_artifact_date": latest.name if latest else None,
        "governance": {
            "read_only_diagnostic": True,
            "production_wiring": False,
            "ranker_change": False,
            "selector_change": False,
            "sizing_change": False,
            "final_score_change": False,
            "alpha_promotion": False,
        },
    }


@mcp.tool()
def get_atlas_data(category: str = "") -> str:
    """Get scientific cartography diagnostic data.

    Args:
        category: Optional filter: overview, schemas, normalizers, diseases,
            programs, review, artifacts.

    Returns:
        JSON string with read-only scientific cartography metadata and compact
        artifact summaries. Large JSONL artifacts are summarized, not streamed.
    """
    category_key = (category or "overview").strip().lower()
    payload = _base_payload(category_key)

    if not CARTOGRAPHY_DIR.exists():
        payload["error"] = "Scientific cartography package not found"
        return json.dumps(payload, default=_json_default)

    latest = _latest_artifact_dir()

    if category_key in {"overview", "", "all"}:
        payload.update(
            {
                "subdirectories": sorted(path.name for path in CARTOGRAPHY_DIR.iterdir() if path.is_dir()),
                "schemas": _list_python_modules(CARTOGRAPHY_DIR / "schemas"),
                "normalizers": _list_python_modules(CARTOGRAPHY_DIR / "normalize"),
                "artifacts": (
                    sorted((path.name for path in ARTIFACTS_ROOT.iterdir() if path.is_dir()), reverse=True)[:20]
                    if ARTIFACTS_ROOT.exists()
                    else []
                ),
            }
        )
    elif category_key == "schemas":
        payload["schemas"] = _list_python_modules(CARTOGRAPHY_DIR / "schemas")
    elif category_key == "normalizers":
        payload["normalizers"] = _list_python_modules(CARTOGRAPHY_DIR / "normalize")
    elif category_key in {"artifacts", "review"}:
        payload["artifacts"] = (
            sorted((path.name for path in ARTIFACTS_ROOT.iterdir() if path.is_dir()), reverse=True)[:50]
            if ARTIFACTS_ROOT.exists()
            else []
        )
        if latest:
            payload["latest_status"] = _read_json(latest / "scientific_cartography_status.json")
            payload["latest_manifest"] = _read_json(latest / "artifact_manifest.json")
    elif category_key in {"diseases", "disease_maps"}:
        if latest:
            summary = _read_json(latest / "disease_map_summary.json")
            index = _read_json(latest / "map_index.json")
            payload["disease_map_summary"] = summary
            if isinstance(index, dict):
                payload["map_index_counts"] = index.get("counts", {})
                diseases = index.get("diseases", [])
                payload["disease_count"] = len(diseases) if isinstance(diseases, list) else 0
                payload["disease_sample"] = diseases[:10] if isinstance(diseases, list) else []
    elif category_key in {"programs", "program_records"}:
        if latest:
            programs_path = latest / "program_records.jsonl"
            clusters_path = latest / "competitive_clusters.jsonl"
            features_path = latest / "landscape_features.jsonl"
            payload["record_counts"] = {
                "program_records": _line_count(programs_path),
                "competitive_clusters": _line_count(clusters_path),
                "landscape_features": _line_count(features_path),
            }
            payload["cluster_coverage"] = _read_json(latest / "cluster_coverage_report.json")
            payload["landscape_feature_coverage"] = _read_json(latest / "landscape_feature_coverage_report.json")
    else:
        payload["error"] = f"Unknown category: {category}"
        payload["valid_categories"] = [
            "overview",
            "schemas",
            "normalizers",
            "diseases",
            "programs",
            "review",
            "artifacts",
        ]

    return json.dumps(payload, default=_json_default)
