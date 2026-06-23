"""Static HTML dashboard generator for Scientific Cartography artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scientific_cartography.dashboard_static import templates


class DashboardGenerator:
    """Generate static HTML dashboard from artifact files."""

    FORBIDDEN_DATA_SOURCES = {
        "rankings.csv",
        "portfolio_positions.csv",
        "screen_output.json",
        "production_data",
        "selector",
        "sizing",
        "final_score",
    }

    def __init__(self, artifact_dir: str, output_dir: str):
        """Initialize dashboard generator.

        Args:
            artifact_dir: Path to Scientific Cartography artifacts
            output_dir: Output directory for generated HTML
        """
        self.artifact_dir = Path(artifact_dir)
        self.output_dir = Path(output_dir)
        self.artifacts_read = []
        self.artifacts_missing = []
        self.warnings = []

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> Dict[str, Any]:
        """Generate all dashboard pages.

        Returns:
            Manifest dict with generation metadata
        """
        # Extract as_of_date from artifact_dir path
        as_of_date = self._extract_date_from_path()

        # Read artifacts
        review_data = self._load_review_data()
        diseases = self._load_disease_maps()
        human_decisions = self._load_human_decisions()
        scheduled_executions = self._load_scheduled_review_health()

        # Navigation items (shared across all pages)
        nav_items = [
            ("Index", "index.html"),
            ("Review Runs", "review_runs.html"),
            ("Disease Maps", "disease_maps.html"),
            ("Human Decisions", "human_decisions.html"),
            ("Scheduled Review", "scheduled_review_health.html"),
            ("Governance", "governance.html"),
        ]

        # Generate pages
        pages = [
            ("Index", "index.html", "overview"),
            ("Review Runs", "review_runs.html", "LG1/LG3 metadata"),
            ("Disease Maps", "disease_maps.html", "browse disease artifacts"),
            ("Human Decisions", "human_decisions.html", "LG2 audit trail"),
            ("Scheduled Review", "scheduled_review_health.html", "LG3 wrapper health"),
            ("Governance", "governance.html", "governance boundaries"),
        ]

        # Index page
        index_html = templates.index_template(
            str(self.artifact_dir),
            as_of_date,
            pages,
            self.artifacts_missing,
            self.warnings,
        )
        self._write_page("index.html", index_html)

        # Review runs page
        review_runs_html = templates.review_runs_template(
            str(self.artifact_dir),
            review_data,
            nav_items,
        )
        self._write_page("review_runs.html", review_runs_html)

        # Disease maps page
        disease_maps_html = templates.disease_maps_template(diseases, nav_items)
        self._write_page("disease_maps.html", disease_maps_html)

        # Human decisions page
        human_decisions_html = templates.human_decisions_template(human_decisions, nav_items)
        self._write_page("human_decisions.html", human_decisions_html)

        # Scheduled review health page
        scheduled_health_html = templates.scheduled_review_health_template(
            scheduled_executions,
            nav_items,
        )
        self._write_page("scheduled_review_health.html", scheduled_health_html)

        # Governance page
        governance_html = templates.governance_template(nav_items)
        self._write_page("governance.html", governance_html)

        # Generate manifest
        manifest = {
            "artifact_type": "scientific_cartography_lg4a_dashboard_manifest",
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat() + "Z",
            "artifact_dir": str(self.artifact_dir),
            "output_dir": str(self.output_dir),
            "as_of_date": as_of_date,
            "pages_written": [name for name, _, _ in pages],
            "artifacts_read": self.artifacts_read,
            "artifacts_missing": self.artifacts_missing,
            "governance_flags": {
                "read_only_diagnostic": True,
                "production_model_change": False,
                "ranker_change": False,
                "selector_change": False,
                "sizing_change": False,
                "final_score_change": False,
                "trading_or_portfolio_action": False,
                "automation_approval": False,
            },
            "forbidden_data_sources_used": [],
            "runtime_server_started": False,
            "production_hook_enabled": False,
            "automation_approval": False,
        }

        # Write manifest
        manifest_path = self.output_dir / "dashboard_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.artifacts_read.append("dashboard_manifest.json")

        return manifest

    def _extract_date_from_path(self) -> str:
        """Extract as_of_date from artifact directory name."""
        for part in self.artifact_dir.parts:
            if len(part) == 10 and part.count("-") == 2:
                try:
                    # Validate it looks like a date
                    datetime.strptime(part, "%Y-%m-%d")
                    return part
                except ValueError:
                    pass
        return ""

    def _load_review_data(self) -> Dict[str, Any]:
        """Load LG1 review summary metadata."""
        review_dir = self.artifact_dir / "review"
        summary_path = review_dir / "langgraph_review_summary.json"

        if not summary_path.exists():
            self.artifacts_missing.append("langgraph_review_summary.json")
            return {}

        try:
            with open(summary_path) as f:
                summary = json.load(f)
                self.artifacts_read.append("langgraph_review_summary.json")

                # Extract key fields for display
                return {
                    "decision": summary.get("decision", "—"),
                    "governance_scan_passed": summary.get("governance_scan_passed", False),
                    "selected_disease_count": summary.get("selected_disease_count", 0),
                    "forbidden_terms_found": len(summary.get("forbidden_terms_found", [])),
                }
        except Exception as e:
            self.warnings.append(f"Error reading review summary: {str(e)}")
            return {}

    def _load_disease_maps(self) -> List[Dict[str, Any]]:
        """Load disease map index."""
        index_path = self.artifact_dir / "disease_map_index.json"

        if not index_path.exists():
            self.artifacts_missing.append("disease_map_index.json")
            return []

        try:
            with open(index_path) as f:
                index = json.load(f)
                self.artifacts_read.append("disease_map_index.json")
                return index.get("diseases", [])
        except Exception as e:
            self.warnings.append(f"Error reading disease map index: {str(e)}")
            return []

    def _load_human_decisions(self) -> List[Dict[str, Any]]:
        """Load LG2 human decision audit trail."""
        review_dir = self.artifact_dir / "review"
        decisions_path = review_dir / "langgraph_human_decisions.jsonl"

        if not decisions_path.exists():
            self.artifacts_missing.append("langgraph_human_decisions.jsonl")
            return []

        decisions = []
        try:
            with open(decisions_path) as f:
                for line in f:
                    if line.strip():
                        decisions.append(json.loads(line))
                self.artifacts_read.append("langgraph_human_decisions.jsonl")
        except Exception as e:
            self.warnings.append(f"Error reading human decisions: {str(e)}")

        return decisions

    def _load_scheduled_review_health(self) -> List[Dict[str, Any]]:
        """Load LG3 scheduled review audit trail."""
        # Check for both per-run audit trail and global cron audit trail
        review_dir = self.artifact_dir / "review"
        audit_path = review_dir / "scheduled_review_audit.jsonl"

        executions = []
        if audit_path.exists():
            try:
                with open(audit_path) as f:
                    for line in f:
                        if line.strip():
                            executions.append(json.loads(line))
                    self.artifacts_read.append("scheduled_review_audit.jsonl")
            except Exception as e:
                self.warnings.append(f"Error reading scheduled review audit: {str(e)}")

        # Also check global cron audit trail (if available at artifact root level)
        cron_audit_path = self.artifact_dir.parent / "scheduled_review_cron.jsonl"
        if cron_audit_path.exists():
            try:
                with open(cron_audit_path) as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            # Filter to this date if present
                            if entry.get("as_of_date") == self._extract_date_from_path():
                                executions.append(entry)
            except Exception as e:
                pass  # Non-critical, don't warn

        return executions

    def _write_page(self, filename: str, html_content: str) -> None:
        """Write HTML page to output directory."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(html_content)
