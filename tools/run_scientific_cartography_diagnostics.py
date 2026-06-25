#!/usr/bin/env python3
"""Phase 7A: Standalone diagnostic wrapper for scientific cartography.

Orchestrates existing builders (asset_indication, competitive_cluster,
landscape_feature) and exporters to generate diagnostic artifacts from
local snapshot/cache data only.

Cache-only, read-only, non-blocking by default.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is in path for standalone execution from tools/
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scientific_cartography.build.asset_indication_builder import AssetIndicationBuilder  # noqa: E402
from scientific_cartography.build.competitive_cluster_builder import CompetitiveClusterBuilder  # noqa: E402
from scientific_cartography.build.landscape_feature_builder import LandscapeFeatureBuilder  # noqa: E402
from scientific_cartography.export import ArtifactManifestExporter, DiseaseMapExporter, MapIndexExporter  # noqa: E402
from scientific_cartography.ingest.ctgov_ingest import CTGovIngest  # noqa: E402
from scientific_cartography.ingest.existing_universe_ingest import ExistingUniverseIngest  # noqa: E402
from scientific_cartography.io import write_json, write_jsonl  # noqa: E402
from scientific_cartography.normalize.asset_alias_resolver import AssetAliasResolver  # noqa: E402
from scientific_cartography.normalize.disease_normalizer import DiseaseNormalizer  # noqa: E402
from scientific_cartography.normalize.mechanism_normalizer import MechanismNormalizer  # noqa: E402
from scientific_cartography.normalize.sponsor_resolver import SponsorResolver  # noqa: E402
from scientific_cartography.normalize.stage_normalizer import StageNormalizer  # noqa: E402


def run_diagnostics(args: argparse.Namespace) -> int:
    """Run scientific cartography diagnostics.

    Arguments:
        args.as_of_date: Date for artifacts (YYYY-MM-DD).
        args.snapshot_dir: Path to snapshot directory.
        args.ctgov_cache: Path to CTGov cache directory.
        args.output_dir: Output directory for artifacts.
        args.strict: Fail on errors (default: false, non-blocking).
        args.created_at_utc: Optional deterministic timestamp.
        args.quiet: Suppress progress output.
    """
    as_of_date = args.as_of_date
    snapshot_dir = Path(args.snapshot_dir)
    ctgov_cache = Path(args.ctgov_cache)
    output_dir = Path(args.output_dir)
    strict = getattr(args, "strict", False)
    created_at_utc = getattr(args, "created_at_utc", None)
    quiet = getattr(args, "quiet", False)

    status = {
        "as_of_date": as_of_date,
        "status": "failed",
        "strict": strict,
        "cache_only": True,
        "output_dir": str(output_dir),
        "artifacts_written": [],
        "warnings": [],
        "errors": [],
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

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load existing universe data (from snapshot)
        if not quiet:
            print(f"Loading universe data from {snapshot_dir}...", file=sys.stderr)

        universe_ingest = ExistingUniverseIngest(as_of_date=as_of_date)
        companies = []

        # Try loading from standard locations in snapshot directory
        rankings_csv = snapshot_dir / "rankings.csv"
        if rankings_csv.exists():
            try:
                companies = universe_ingest.ingest_from_rankings_csv(rankings_csv)
            except Exception as e:
                status["warnings"].append(f"Failed to load from rankings.csv: {e}")
        else:
            status["warnings"].append("rankings.csv not found in snapshot_dir")

        if not companies:
            if strict:
                raise ValueError("Company data required in strict mode")
            status["warnings"].append("Continuing with empty company data")

        if not quiet:
            print(f"✓ Loaded {len(companies)} companies", file=sys.stderr)

        # Step 2: Load CTGov trial data (from cache)
        if not quiet:
            print(f"Loading trial data from {ctgov_cache}...", file=sys.stderr)

        ctgov_ingest = CTGovIngest(as_of_date=as_of_date)
        trials = []

        # Try loading from standard locations in ctgov_cache directory.
        # Priority order: trials.jsonl → trials.json → trial_records.json
        # Optional --trials-file bypasses directory lookup entirely.
        trials_file = getattr(args, "trials_file", None)
        _trial_source = None
        if trials_file:
            trial_path = Path(trials_file)
            if trial_path.is_file():
                try:
                    if trial_path.suffix == ".jsonl":
                        trials = ctgov_ingest.ingest_from_jsonl_file(trial_path)
                    else:
                        trials = ctgov_ingest.ingest_from_json_file(trial_path)
                    _trial_source = trial_path.name
                except Exception as e:
                    status["warnings"].append(f"Failed to load --trials-file {trial_path}: {e}")
            else:
                status["warnings"].append(f"--trials-file not found: {trial_path}")
        else:
            _trial_candidates = [
                (ctgov_cache / "trials.jsonl", "jsonl"),
                (ctgov_cache / "trials.json", "json"),
                (ctgov_cache / "trial_records.json", "json"),
            ]
            for _candidate_path, _fmt in _trial_candidates:
                if _candidate_path.exists():
                    try:
                        if _fmt == "jsonl":
                            trials = ctgov_ingest.ingest_from_jsonl_file(_candidate_path)
                        else:
                            trials = ctgov_ingest.ingest_from_json_file(_candidate_path)
                        _trial_source = _candidate_path.name
                    except Exception as e:
                        status["warnings"].append(f"Failed to load from {_candidate_path.name}: {e}")
                    break

        if _trial_source is None and not trials:
            status["warnings"].append(
                "No trial data files (trials.jsonl, trials.json, or trial_records.json)" " found in ctgov_cache"
            )

        if not quiet:
            _source_label = _trial_source or "none"
            print(
                f"✓ Loaded {len(trials)} trials from cache (source: {_source_label})",
                file=sys.stderr,
            )

        # Step 3: Initialize normalizers
        disease_normalizer = DiseaseNormalizer(as_of_date=as_of_date)
        stage_normalizer = StageNormalizer()
        asset_alias_resolver = AssetAliasResolver(as_of_date=as_of_date)
        sponsor_resolver = SponsorResolver(company_records=companies)
        # Load mechanism aliases: explicit override → well-known path → built-in only
        _mech_alias_override = getattr(args, "mechanism_aliases", None)
        _mech_alias_default = repo_root / "scientific_cartography" / "data" / "mechanism_aliases_v0_1.csv"
        _mech_alias_path = (
            Path(_mech_alias_override)
            if _mech_alias_override
            else (_mech_alias_default if _mech_alias_default.exists() else None)
        )
        if _mech_alias_path is not None:
            mechanism_normalizer = MechanismNormalizer.from_csv(_mech_alias_path, as_of_date=as_of_date)
        else:
            mechanism_normalizer = MechanismNormalizer(as_of_date=as_of_date)

        # Step 4: Build programs from trials
        if not quiet:
            print("Building program records...", file=sys.stderr)

        asset_builder = AssetIndicationBuilder(
            disease_normalizer=disease_normalizer,
            stage_normalizer=stage_normalizer,
            asset_alias_resolver=asset_alias_resolver,
            sponsor_resolver=sponsor_resolver,
            as_of_date=as_of_date,
        )

        programs, asset_diagnostics = asset_builder.build_from_trials(trials, companies)

        if not programs and not trials:
            status["warnings"].append("No programs or trials available")

        if not quiet:
            print(f"✓ Built {len(programs)} program records", file=sys.stderr)

        # Therapeutic area coverage report (R5)
        if programs:
            ta_counts: dict = {}
            ta_null = 0
            for p in programs:
                if p.therapeutic_area:
                    ta_counts[p.therapeutic_area] = ta_counts.get(p.therapeutic_area, 0) + 1
                else:
                    ta_null += 1
            ta_total = len(programs)
            ta_filled = ta_total - ta_null
            status["therapeutic_area_coverage"] = {
                "total_programs": ta_total,
                "with_therapeutic_area": ta_filled,
                "without_therapeutic_area": ta_null,
                "coverage_pct": round(100.0 * ta_filled / ta_total, 1) if ta_total else 0.0,
                "top_areas": sorted(ta_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            }
            if not quiet:
                print(
                    f"✓ therapeutic_area coverage: {ta_filled}/{ta_total}"
                    f" ({status['therapeutic_area_coverage']['coverage_pct']}%)",
                    file=sys.stderr,
                )

        # Step 5: Enrich programs with mechanism (in-place)
        if not quiet:
            print("Enriching programs with mechanism/modality/target...", file=sys.stderr)

        enriched_count = 0
        for program in programs:
            if program.asset_name and not program.mechanism_class:
                normalized = mechanism_normalizer.normalize(program.asset_name)
                program.mechanism_class = normalized.mechanism_class
                program.modality = normalized.modality
                program.target = normalized.target
                if normalized.mechanism_class:
                    enriched_count += 1

        if not quiet:
            print(f"✓ Enriched {enriched_count} programs with mechanism data", file=sys.stderr)

        # Step 6: Build competitive clusters
        if not quiet:
            print("Building competitive clusters...", file=sys.stderr)

        cluster_builder = CompetitiveClusterBuilder(as_of_date=as_of_date)
        clusters, cluster_coverage = cluster_builder.build_from_programs(programs)

        if not quiet:
            print(f"✓ Built {len(clusters)} competitive clusters", file=sys.stderr)

        # Step 7: Build landscape features
        if not quiet:
            print("Building landscape features...", file=sys.stderr)

        feature_builder = LandscapeFeatureBuilder(as_of_date=as_of_date)
        features, feature_coverage = feature_builder.build_from_programs_and_clusters(programs, clusters)

        if not quiet:
            print(f"✓ Built {len(features)} landscape features", file=sys.stderr)

        # Step 8: Write JSONL artifacts
        if not quiet:
            print("Writing JSONL artifacts...", file=sys.stderr)

        programs_path = output_dir / "program_records.jsonl"
        write_jsonl(programs_path, (program.to_dict() for program in programs))
        status["artifacts_written"].append("program_records.jsonl")

        clusters_path = output_dir / "competitive_clusters.jsonl"
        write_jsonl(clusters_path, (cluster.to_dict() for cluster in clusters))
        status["artifacts_written"].append("competitive_clusters.jsonl")

        features_path = output_dir / "landscape_features.jsonl"
        write_jsonl(features_path, (feature.to_dict() for feature in features))
        status["artifacts_written"].append("landscape_features.jsonl")

        # Step 9: Write coverage reports
        cluster_coverage_path = output_dir / "cluster_coverage_report.json"
        write_json(cluster_coverage_path, cluster_coverage)
        status["artifacts_written"].append("cluster_coverage_report.json")

        feature_coverage_path = output_dir / "landscape_feature_coverage_report.json"
        write_json(feature_coverage_path, feature_coverage)
        status["artifacts_written"].append("landscape_feature_coverage_report.json")

        if not quiet:
            print("✓ Wrote JSONL artifacts", file=sys.stderr)

        # Step 10: Export diagnostic artifacts
        if not quiet:
            print("Exporting diagnostic artifacts...", file=sys.stderr)

        map_exporter = MapIndexExporter(as_of_date=as_of_date)
        map_index = map_exporter.build_index(programs, clusters, features)
        map_index_path = output_dir / "map_index.json"
        map_exporter.write_index(map_index, map_index_path)
        status["artifacts_written"].append("map_index.json")

        disease_exporter = DiseaseMapExporter(as_of_date=as_of_date)
        disease_summary = disease_exporter.build_disease_summary(programs, clusters, features)
        disease_summary_json_path = output_dir / "disease_map_summary.json"
        disease_exporter.write_disease_summary(disease_summary, disease_summary_json_path)
        status["artifacts_written"].append("disease_map_summary.json")

        disease_summary_md_path = output_dir / "disease_map_summary.md"
        disease_exporter.write_disease_summary_markdown(disease_summary, disease_summary_md_path)
        status["artifacts_written"].append("disease_map_summary.md")

        manifest_exporter = ArtifactManifestExporter(
            as_of_date=as_of_date,
            created_at_utc=created_at_utc,
        )
        manifest = manifest_exporter.build_manifest(
            inputs={
                "program_records": "program_records.jsonl",
                "competitive_clusters": "competitive_clusters.jsonl",
                "landscape_features": "landscape_features.jsonl",
            },
            outputs=[
                "map_index.json",
                "disease_map_summary.json",
                "disease_map_summary.md",
                "artifact_manifest.json",
            ],
        )
        manifest_path = output_dir / "artifact_manifest.json"
        manifest_exporter.write_manifest(manifest, manifest_path)
        status["artifacts_written"].append("artifact_manifest.json")

        if not quiet:
            print("✓ Exported diagnostic artifacts", file=sys.stderr)

        # Step 11: Write status file
        status["status"] = "success"
        status_path = output_dir / "scientific_cartography_status.json"
        write_json(status_path, status)

        if not quiet:
            print("", file=sys.stderr)
            print("✓ Scientific cartography diagnostics complete", file=sys.stderr)
            print(f"✓ Output directory: {output_dir}", file=sys.stderr)
            print(f"✓ Status: {status['status']}", file=sys.stderr)
            if status["warnings"]:
                print(f"⚠ Warnings: {len(status['warnings'])}", file=sys.stderr)

        return 0

    except Exception as e:
        status["status"] = "failed"
        status["errors"].append(str(e))

        status_path = output_dir / "scientific_cartography_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(status_path, status)

        if not quiet:
            print(f"✗ Scientific cartography diagnostics failed: {e}", file=sys.stderr)

        return 1 if strict else 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scientific Cartography Phase 7A: Diagnostic wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--as-of-date",
        type=str,
        required=True,
        help="Date for artifacts (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        required=True,
        help="Path to snapshot directory",
    )
    parser.add_argument(
        "--ctgov-cache",
        type=str,
        default=str(repo_root / "production_data"),
        help="Path to CTGov cache directory (default: production_data/)",
    )
    parser.add_argument(
        "--trials-file",
        type=str,
        default=None,
        help="Optional direct path to trials JSON/JSONL (bypasses ctgov-cache lookup)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on errors (default: non-blocking)",
    )
    parser.add_argument(
        "--created-at-utc",
        type=str,
        default=None,
        help="Optional deterministic timestamp for manifest",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output",
    )

    args = parser.parse_args()

    return run_diagnostics(args)


if __name__ == "__main__":
    sys.exit(main())
