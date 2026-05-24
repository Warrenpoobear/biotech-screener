#!/usr/bin/env python3
"""
Build the Spec 089 Knowledge Graph (KG pilot, Phase 1.5A).

Extracts governance state, specs, policies, artifacts, and code files;
normalizes into nodes and edges; detects contradictions; emits JSONL.

Read-only operation. Does NOT modify production code, cron, or git state.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ops" / "knowledge_graph"
OUTPUT_NODES = ARTIFACTS_DIR / "nodes.jsonl"
OUTPUT_EDGES = ARTIFACTS_DIR / "edges.jsonl"
OUTPUT_SUMMARY = ARTIFACTS_DIR / "summary.md"
BUILD_AS_OF_DATE = "2026-05-19"

# Node type registry
NODE_TYPES = {
    "Spec",
    "Policy",
    "Commit",
    "Artifact",
    "CodeFile",
    "Signal",
    "ModelComponent",
    "Blocker",
    "ValidationGate",
    "Snapshot",
    "Review",
    "Action",
}

# Edge type registry
EDGE_TYPES = {
    "IMPLEMENTS",
    "DOCUMENTS",
    "BLOCKS",
    "DEPENDS_ON",
    "GOVERNS",
    "VALIDATES",
    "INVALIDATES",
    "TOUCHES",
    "PRODUCES",
    "CONSUMES",
    "REFERENCES",
    "SUPERSEDES",
    "PENDING_ON",
    "CLEARS",
    "CONTRADICTS",
}


# ============================================================================
# HELPERS
# ============================================================================


def extract_heading(content: str) -> str:
    """Extract title from markdown (first # heading or document start)."""
    for line in content.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_status_from_content(content: str) -> str:
    """Extract status from markdown content (Status: field or context)."""
    for line in content.split("\n"):
        if "status:" in line.lower():
            # Try to extract status value
            parts = line.split(":", 1)
            if len(parts) == 2:
                status = parts[1].strip().upper()
                # Normalize to known statuses
                if any(s in status for s in ["COMPLETE", "DONE", "FINISHED"]):
                    return "COMPLETE"
                elif any(s in status for s in ["PENDING", "IN_PROGRESS"]):
                    return "PENDING"
                elif any(s in status for s in ["HOLD", "BLOCK"]):
                    return "BLOCKED"
    return "PENDING"


# ============================================================================
# LAYER 1: EXTRACT
# ============================================================================


def extract_specs() -> Dict[str, Dict[str, Any]]:
    """Extract Spec nodes from specs/changes/*.md."""
    specs = {}
    specs_dir = REPO_ROOT / "specs" / "changes"

    if not specs_dir.exists():
        return specs

    for spec_file in specs_dir.glob("spec_*.md"):
        spec_id = spec_file.stem  # e.g., "spec_089_hermes_knowledge_layer"
        with open(spec_file) as f:
            content = f.read()

        # Extract metadata from markdown frontmatter/heading
        title = extract_heading(content)
        status = extract_status_from_content(content)

        specs[spec_id] = {
            "id": spec_id,
            "type": "Spec",
            "title": title or spec_id,
            "status": status,
            "related_artifacts": [str(spec_file)],
        }

    # Edges use stable short IDs for high-traffic specs. Keep those aliases
    # explicit so graph validation and query output reference real nodes.
    short_aliases = {
        "spec_089": ["spec_089_hermes_knowledge_layer"],
        "spec_100": [
            "spec_100_ranker_ic_tooling_correction_2026_05_13",
            "spec_100_investigation_memo_2026_05_16",
        ],
    }
    for alias, source_ids in short_aliases.items():
        if alias in specs:
            continue
        source_nodes = [specs[source_id] for source_id in source_ids if source_id in specs]
        if not source_nodes:
            continue
        primary = source_nodes[0]
        related_artifacts = []
        for node in source_nodes:
            related_artifacts.extend(node.get("related_artifacts", []))
        specs[alias] = {
            **primary,
            "id": alias,
            "alias_of": primary["id"],
            "related_artifacts": sorted(set(related_artifacts)),
        }

    return specs


def extract_policies() -> Dict[str, Dict[str, Any]]:
    """Extract Policy nodes from governance docs and memory."""
    policies = {}

    # Known policies from current governance state
    policies["spec_096"] = {
        "id": "spec_096",
        "type": "Policy",
        "title": "Ranker Governance Doctrine",
        "status": "ACTIVE",
        "related_artifacts": [],
    }

    policies["ranker_freeze"] = {
        "id": "ranker_freeze",
        "type": "Policy",
        "title": "Architecture Freeze (Ranker)",
        "status": "ACTIVE",
        "since": "2026-04-19",
        "related_artifacts": [],
    }

    policies["checklist_v2"] = {
        "id": "checklist_v2",
        "type": "Policy",
        "title": "Checklist v2 Promotion Gate",
        "status": "ACTIVE",
        "requirements": ["signal_card", "fm_incremental", "bootstrap", "bh_fdr", "loso"],
        "related_artifacts": [],
    }

    policies["alpha_freeze"] = {
        "id": "alpha_freeze",
        "type": "Policy",
        "title": "Alpha Freeze (No Promotions Without Checklist v2)",
        "status": "ACTIVE",
        "since": "2026-04-04",
        "related_artifacts": [],
    }

    return policies


def extract_validation_gates() -> Dict[str, Dict[str, Any]]:
    """Extract ValidationGate nodes (13F, Phase 2 steps, etc.)."""
    gates = {}

    gates["13f_clearance"] = {
        "id": "13f_clearance",
        "type": "ValidationGate",
        "title": "13F Cohort Quarantine Clearance",
        "status": "PENDING",
        "description": "Q1 2026 13F filing refresh and cohort validation",
        "expected_date": "2026-05-21",
        "related_artifacts": ["artifacts/13f_validation_verdict_template_2026_05_19.md"],
    }

    gates["phase_2_step_3_complete"] = {
        "id": "phase_2_step_3_complete",
        "type": "ValidationGate",
        "title": "Phase 2 Step 3: Evening Reliability Audit Complete",
        "status": "COMPLETE",
        "description": "Watchdog + preflight integration verified",
        "related_artifacts": [
            "artifacts/phase_2_step_3_evening_reliability_complete_2026_05_15.md",
            "artifacts/phase_2_step_3b_preflight_integration_complete_2026_05_15.md",
        ],
    }

    gates["phase_2_step_5_kg_validation"] = {
        "id": "phase_2_step_5_kg_validation",
        "type": "ValidationGate",
        "title": "Phase 2 Step 5: KG Validation Gate",
        "status": "PENDING",
        "description": "KG schema built, validation framework defined",
        "expected_date": "2026-05-24",
        "related_artifacts": [],
    }

    gates["h20d_freeze_decision"] = {
        "id": "h20d_freeze_decision",
        "type": "ValidationGate",
        "title": "H20D Hard Decision (May 26): Freeze Lift Decision",
        "status": "PENDING",
        "decision_date": "2026-05-26",
        "related_artifacts": [],
    }

    return gates


def extract_artifacts() -> Dict[str, Dict[str, Any]]:
    """Extract Artifact nodes from artifacts/audit/*.md and other key artifacts."""
    artifacts = {}
    audit_dir = REPO_ROOT / "artifacts" / "audit"

    if audit_dir.exists():
        for audit_file in audit_dir.glob("*.md"):
            artifact_id = audit_file.stem
            artifacts[artifact_id] = {
                "id": artifact_id,
                "type": "Artifact",
                "title": audit_file.stem,
                "status": "COMPLETE",
                "related_artifacts": [str(audit_file)],
            }

    # Add known key artifacts
    key_artifacts = [
        ("13f_validation_verdict_template", "13F Validation Verdict Template"),
        ("spec_100_ic_baseline", "Spec 100 IC Baseline (Corrected final_score)"),
        ("backtest_harness_integrity_audit", "Backtest Harness Integrity Audit"),
    ]

    for artifact_id, title in key_artifacts:
        if artifact_id not in artifacts:
            artifacts[artifact_id] = {
                "id": artifact_id,
                "type": "Artifact",
                "title": title,
                "status": "COMPLETE",
                "related_artifacts": [],
            }

    return artifacts


def extract_code_files() -> Dict[str, Dict[str, Any]]:
    """Extract CodeFile nodes (particularly stubs and test files)."""
    files = {}

    # Known stubs/critical files
    critical_files = [
        ("run_true_ranker_ic_stub", "scripts/research/run_true_ranker_ic.py", "STUB_PLACEHOLDER"),
        ("run_screen", "run_screen.py", "COMPLETE"),
        ("selector_engine", "selector_engine.py", "COMPLETE"),
        ("ranker_v2_pairwise", "ranker_v2_pairwise.py", "COMPLETE"),
    ]

    for file_id, path, status in critical_files:
        files[file_id] = {
            "id": file_id,
            "type": "CodeFile",
            "title": Path(path).name,
            "path": path,
            "status": status,
            "related_artifacts": [path],
        }

    return files


def extract_snapshots() -> Dict[str, Dict[str, Any]]:
    """Extract Snapshot nodes from data/snapshots/."""
    snapshots = {}
    snapshots_dir = REPO_ROOT / "data" / "snapshots"

    if snapshots_dir.exists():
        for snapshot_dir in sorted(snapshots_dir.iterdir(), reverse=True)[:5]:  # Last 5
            if snapshot_dir.is_dir():
                snapshot_date = snapshot_dir.name
                snapshots[f"snapshot_{snapshot_date}"] = {
                    "id": f"snapshot_{snapshot_date}",
                    "type": "Snapshot",
                    "title": f"Production Snapshot {snapshot_date}",
                    "status": "COMPLETE",
                    "as_of_date": snapshot_date,
                    "related_artifacts": [str(snapshot_dir)],
                }

    return snapshots


def extract_reviews() -> Dict[str, Dict[str, Any]]:
    """Extract Review nodes from governance calendar."""
    reviews = {}

    reviews["review_2026_05_22"] = {
        "id": "review_2026_05_22",
        "type": "Review",
        "title": "2026-05-22 Ranker Review (H20D)",
        "status": "SCHEDULED",
        "scheduled_date": "2026-05-22",
        "related_artifacts": ["specs/changes/spec_072_screener_vnext_2026_05_01.md"],
    }

    return reviews


def extract_actions() -> Dict[str, Dict[str, Any]]:
    """Extract Action nodes (production changes, implementations)."""
    actions = {}

    actions["production_ranker_change"] = {
        "id": "production_ranker_change",
        "type": "Action",
        "title": "Production Ranker Weight/Feature Change",
        "status": "BLOCKED",
        "description": "Any change to ranker weights, features, or evaluation logic",
        "related_artifacts": [],
    }

    actions["spec_089_kg_implementation"] = {
        "id": "spec_089_kg_implementation",
        "type": "Action",
        "title": "Spec 089 KG Pilot Implementation (Phase 2 Step 4)",
        "status": "IN_PROGRESS",
        "description": "Build knowledge graph infrastructure, tests, queries",
        "related_artifacts": [
            "specs/changes/spec_089_hermes_knowledge_layer.md",
            "tests/test_knowledge_graph_spec089.py",
        ],
    }

    return actions


def extract_edges() -> List[Dict[str, Any]]:
    """Extract Edge definitions (relationships between nodes)."""
    edges = []

    # Governance relationships
    edges.extend(
        [
            {"source": "spec_096", "type": "GOVERNS", "target": "production_ranker_change"},
            {"source": "spec_096", "type": "BLOCKS", "target": "production_ranker_change"},
            {"source": "ranker_freeze", "type": "BLOCKS", "target": "production_ranker_change"},
            {"source": "checklist_v2", "type": "BLOCKS", "target": "production_ranker_change"},
            {"source": "13f_clearance", "type": "BLOCKS", "target": "production_ranker_change"},
            {"source": "h20d_freeze_decision", "type": "CLEARS", "target": "ranker_freeze"},
        ]
    )

    # Spec dependencies
    edges.extend(
        [
            {"source": "spec_089", "type": "DOCUMENTS", "target": "spec_089_kg_implementation"},
            {"source": "spec_089_kg_implementation", "type": "DEPENDS_ON", "target": "phase_2_step_3_complete"},
            {"source": "spec_089_kg_implementation", "type": "DEPENDS_ON", "target": "13f_clearance"},
            {"source": "phase_2_step_5_kg_validation", "type": "VALIDATES", "target": "spec_089_kg_implementation"},
        ]
    )

    # Code/artifact references
    edges.extend(
        [
            {"source": "spec_100", "type": "TOUCHES", "target": "run_true_ranker_ic_stub"},
            {"source": "run_true_ranker_ic_stub", "type": "CONTRADICTS", "target": "spec_100"},
        ]
    )

    # Production implications
    edges.extend(
        [
            {"source": "spec_089_kg_implementation", "type": "PRODUCES", "target": "spec_089_kg_implementation"},
        ]
    )

    return edges


# ============================================================================
# LAYER 2: NORMALIZE
# ============================================================================


def normalize_graph(
    specs: Dict,
    policies: Dict,
    gates: Dict,
    artifacts: Dict,
    code_files: Dict,
    snapshots: Dict,
    reviews: Dict,
    actions: Dict,
    edges: List,
) -> Tuple[Dict, List]:
    """Merge all node types into unified graph."""
    nodes = {}
    nodes.update(specs)
    nodes.update(policies)
    nodes.update(gates)
    nodes.update(artifacts)
    nodes.update(code_files)
    nodes.update(snapshots)
    nodes.update(reviews)
    nodes.update(actions)

    return nodes, edges


# ============================================================================
# LAYER 3: VALIDATE (Contradiction Detection)
# ============================================================================


def detect_contradictions(nodes: Dict, edges: List) -> List[Dict[str, Any]]:
    """Detect all 5 contradiction types."""
    contradictions = []
    seen = set()

    def add_contradiction(rule: str, node_id: str, issue: str, severity: str) -> None:
        key = (rule, node_id, issue)
        if key in seen:
            return
        seen.add(key)
        contradictions.append(
            {
                "rule": rule,
                "node_id": node_id,
                "issue": issue,
                "severity": severity,
            }
        )

    # Rule 0: Edge integrity. Query results are misleading if an edge endpoint
    # does not resolve to a node.
    for edge in edges:
        missing_endpoints = [endpoint for endpoint in ("source", "target") if edge.get(endpoint) not in nodes]
        if missing_endpoints:
            add_contradiction(
                "dangling_edge",
                f"{edge.get('source')}->{edge.get('target')}",
                f"Edge {edge.get('type')} has missing endpoint(s): {', '.join(missing_endpoints)}",
                "HIGH",
            )

    # Explicit CONTRADICTS edges are contradictions even when the target status is
    # not COMPLETE. Status-specific rules below add more detail when applicable.
    for edge in edges:
        if edge.get("type") == "CONTRADICTS":
            add_contradiction(
                "explicit_contradiction",
                str(edge.get("target")),
                f"{edge.get('source')} CONTRADICTS {edge.get('target')}",
                "HIGH",
            )

    # Rule 1: Status contradiction (COMPLETE with unresolved PENDING_ON or CONTRADICTS edges)
    for node_id, node in nodes.items():
        if node.get("status") == "COMPLETE":
            # Check for PENDING_ON edges
            pending_edges = [e for e in edges if e["source"] == node_id and e["type"] == "PENDING_ON"]
            if pending_edges:
                add_contradiction(
                    "status_contradiction",
                    node_id,
                    "Status COMPLETE but has unresolved PENDING_ON edges",
                    "MEDIUM",
                )

            # Check for CONTRADICTS edges
            contradicts_edges = [e for e in edges if e["target"] == node_id and e["type"] == "CONTRADICTS"]
            if contradicts_edges:
                add_contradiction(
                    "stub_contradiction",
                    node_id,
                    f"Status COMPLETE but has CONTRADICTS edges from {contradicts_edges[0]['source']}",
                    "HIGH",
                )

    # Rule 2: Stub contradiction (COMPLETE file status with STUB_PLACEHOLDER)
    for node_id, node in nodes.items():
        if node.get("type") == "CodeFile" and node.get("status") == "STUB_PLACEHOLDER":
            # Check if this file is touched by a COMPLETE spec
            touching_specs = [
                e["source"]
                for e in edges
                if e["target"] == node_id
                and e["type"] == "TOUCHES"
                and nodes.get(e["source"], {}).get("status") == "COMPLETE"
            ]
            if touching_specs:
                add_contradiction(
                    "stub_contradiction",
                    node_id,
                    f"Status STUB_PLACEHOLDER but touched by COMPLETE spec {touching_specs[0]}",
                    "HIGH",
                )

    # Rule 3: Scope contradiction (ranker freeze active with recent ranker commits)
    freeze_node = nodes.get("ranker_freeze")
    if freeze_node and freeze_node.get("status") == "ACTIVE":
        # Check for recent commits touching ranker files
        ranker_files = {"ranker_v2_pairwise", "selector_engine", "run_screen"}
        recent_ranker_touches = [e for e in edges if e["source"] in ranker_files and e["type"] == "TOUCHES"]
        if recent_ranker_touches:
            add_contradiction(
                "scope_contradiction",
                "ranker_freeze",
                f"Freeze ACTIVE but ranker files touched by {recent_ranker_touches[0]['source']}",
                "HIGH",
            )

    # Rule 4: Artifact contradiction (claimed but missing)
    # (Simplified: just report if related_artifacts list is empty for key specs)
    key_specs = {"spec_100", "spec_089"}
    for spec_id in key_specs:
        if spec_id in nodes:
            spec = nodes[spec_id]
            artifacts = spec.get("related_artifacts", [])
            if not artifacts:
                add_contradiction(
                    "artifact_contradiction",
                    spec_id,
                    "Spec has no related_artifacts documented",
                    "LOW",
                )

    # Rule 5: Promotion contradiction (blocker active while action marked not-blocked)
    for node_id, node in nodes.items():
        if node.get("type") == "Action" and node.get("status") != "BLOCKED":
            blocks = [e for e in edges if e["type"] == "BLOCKS" and e["target"] == node_id]
            if blocks:
                add_contradiction(
                    "promotion_contradiction",
                    node_id,
                    f"Action status {node.get('status')} but {len(blocks)} BLOCKS edges exist",
                    "HIGH",
                )

    return contradictions


# ============================================================================
# LAYER 4: EMIT
# ============================================================================


def emit_jsonl(nodes: Dict, edges: List, contradictions: List) -> None:
    """Write JSONL outputs and summary."""
    # Ensure output directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write nodes.jsonl
    with open(OUTPUT_NODES, "w") as f:
        for node in nodes.values():
            f.write(json.dumps(node) + "\n")

    # Write edges.jsonl
    with open(OUTPUT_EDGES, "w") as f:
        for edge in edges:
            f.write(json.dumps(edge) + "\n")

    # Write summary.md
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("# Knowledge Graph Summary\n\n")
        f.write(f"**Generated:** {BUILD_AS_OF_DATE}\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Nodes: {len(nodes)}\n")
        f.write(f"- Edges: {len(edges)}\n")
        f.write(f"- Contradictions detected: {len(contradictions)}\n\n")

        if contradictions:
            f.write("## Contradictions\n\n")
            for c in contradictions:
                f.write(f"- [{c['severity']}] {c['rule']}: {c['node_id']} — {c['issue']}\n")
        else:
            f.write("## Contradictions\n\nNone detected.\n\n")

        f.write("\n## Node Types\n\n")
        node_type_counts = {}
        for node in nodes.values():
            nt = node.get("type", "unknown")
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

        for nt in sorted(node_type_counts.keys()):
            f.write(f"- {nt}: {node_type_counts[nt]}\n")

    print(f"✓ Written: {OUTPUT_NODES}")
    print(f"✓ Written: {OUTPUT_EDGES}")
    print(f"✓ Written: {OUTPUT_SUMMARY}")  # noqa: F541


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Build the knowledge graph."""
    print(f"[build_knowledge_graph] {BUILD_AS_OF_DATE}")
    print(f"  repo: {REPO_ROOT}")
    print()

    # Layer 1: Extract
    print("Layer 1: extract...")
    specs = extract_specs()
    policies = extract_policies()
    gates = extract_validation_gates()
    artifacts = extract_artifacts()
    code_files = extract_code_files()
    snapshots = extract_snapshots()
    reviews = extract_reviews()
    actions = extract_actions()
    edges = extract_edges()
    print(f"  specs: {len(specs)}, policies: {len(policies)}, gates: {len(gates)}")
    print(f"  artifacts: {len(artifacts)}, files: {len(code_files)}, snapshots: {len(snapshots)}")
    print(f"  reviews: {len(reviews)}, actions: {len(actions)}, edges: {len(edges)}")
    print()

    # Layer 2: Normalize
    print("Layer 2: normalize...")
    nodes, edges = normalize_graph(specs, policies, gates, artifacts, code_files, snapshots, reviews, actions, edges)
    print(f"  nodes: {len(nodes)}, edges: {len(edges)}")
    print()

    # Layer 3: Validate
    print("Layer 3: validate...")
    contradictions = detect_contradictions(nodes, edges)
    print(f"  contradictions: {len(contradictions)}")
    for c in contradictions:
        print(f"    [{c['severity']}] {c['rule']}: {c['node_id']}")
    print()

    # Layer 4: Emit
    print("Layer 4: emit...")
    emit_jsonl(nodes, edges, contradictions)
    print()

    print("=== Summary ===")
    print(f"  nodes: {len(nodes)}")
    print(f"  edges: {len(edges)}")
    print(f"  contradictions: {len(contradictions)}")
    print()

    if contradictions:
        print("⚠ Contradictions detected (expected during governance transitions):")
        for c in contradictions:
            print(f"  - {c['node_id']}: {c['issue']}")


if __name__ == "__main__":
    main()
