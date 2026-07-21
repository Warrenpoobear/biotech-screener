"""Build landscape context features from Phase 9/10 records."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scientific_cartography.schemas.asset_indication_map_schema import AssetIndicationMapRecord
from scientific_cartography.schemas.enhanced_cluster_schema import EnhancedCompetitiveClusterRecord
from scientific_cartography.schemas.landscape_context_schema import (
    LandscapeContextCoverageReport,
    LandscapeContextFeatureRecord,
)


class LandscapeContextFeatureBuilder:
    """Build landscape context features from Phase 9 and Phase 10 records."""

    def __init__(self, as_of_date: str = ""):
        """Initialize builder.

        Args:
            as_of_date: Record date (YYYY-MM-DD).
        """
        self.as_of_date = as_of_date

    def _get_disease_key(self, record: AssetIndicationMapRecord) -> str:
        """Determine primary disease key for context.

        Priority: mondo_id > normalized_disease_name > raw_indication > unknown.

        Args:
            record: AssetIndicationMapRecord.

        Returns:
            Disease key string.
        """
        if record.mondo_id:
            return record.mondo_id
        elif record.normalized_disease_name:
            return record.normalized_disease_name
        elif record.raw_indication:
            return record.raw_indication
        else:
            return "unknown_disease"

    def _get_mechanism_class(self, record: AssetIndicationMapRecord) -> str:
        """Get mechanism class, defaulting to unknown.

        Args:
            record: AssetIndicationMapRecord.

        Returns:
            Mechanism class or "unknown_mechanism".
        """
        return record.mechanism_class or "unknown_mechanism"

    def _get_target(self, record: AssetIndicationMapRecord) -> str:
        """Get target, defaulting to unknown.

        Args:
            record: AssetIndicationMapRecord.

        Returns:
            Target or "unknown_target".
        """
        return record.target or "unknown_target"

    def _get_modality(self, record: AssetIndicationMapRecord) -> str:
        """Get modality, defaulting to unknown.

        Args:
            record: AssetIndicationMapRecord.

        Returns:
            Modality or "unknown_modality".
        """
        return record.modality or "unknown_modality"

    def _make_feature_id(self, source_record_id: str, cluster_id: str | None) -> str:
        """Create deterministic feature ID.

        Args:
            source_record_id: Phase 9 record ID.
            cluster_id: Phase 10 cluster ID (or None).

        Returns:
            Hex digest (first 16 chars of SHA256).
        """
        cluster_part = cluster_id or "no_cluster"
        key = f"{source_record_id}|{cluster_part}|{self.as_of_date}".lower().strip()
        h = hashlib.sha256(key.encode())
        return h.hexdigest()[:16]

    def _categorize_mechanism_novelty(self, same_mechanism_competition_count: int) -> str:
        """Categorize mechanism novelty.

        Args:
            same_mechanism_competition_count: Count in same disease + same mechanism.

        Returns:
            Category string.
        """
        if same_mechanism_competition_count <= 1:
            return "novel_or_sparse"
        elif 2 <= same_mechanism_competition_count <= 4:
            return "moderately_represented"
        elif same_mechanism_competition_count >= 5:
            return "well_represented"
        else:
            return "unknown"

    def _categorize_target_disease_evidence(
        self,
        source_type: str,
        source_type_distribution: dict[str, int],
    ) -> str:
        """Categorize target disease evidence.

        Based on source_type_distribution and presence of curated sources.

        Args:
            source_type: Source type from record.
            source_type_distribution: Source distribution from cluster.

        Returns:
            Category string.
        """
        if not source_type_distribution and source_type == "unknown":
            return "unknown"

        curated_sources = {
            "sec_filing",
            "investor_deck",
            "fda_label",
            "manual_override",
        }

        has_curated = any(s in curated_sources for s in source_type_distribution.keys())
        if has_curated:
            return "curated_or_regulatory_source_present"

        source_count = len(source_type_distribution)
        if source_count > 1:
            return "multi_source"
        elif source_count == 1:
            return "single_source"
        else:
            return "unknown"

    def _categorize_white_space(self, disease_competition_count: int, same_mechanism_competition_count: int) -> str:
        """Categorize white space context.

        Args:
            disease_competition_count: Total programs in disease.
            same_mechanism_competition_count: Programs in same mechanism cluster.

        Returns:
            Category string.
        """
        if disease_competition_count == 0:
            return "unknown"

        if disease_competition_count <= 2 and same_mechanism_competition_count <= 1:
            return "sparse_context"
        elif 3 <= disease_competition_count <= 9:
            return "moderate_context"
        elif disease_competition_count >= 10 or same_mechanism_competition_count >= 5:
            return "crowded_context"
        else:
            return "unknown"

    def _categorize_crowding(
        self,
        same_mechanism_competition_count: int,
        approved_incumbent_count: int,
    ) -> str:
        """Categorize crowding.

        Args:
            same_mechanism_competition_count: Programs in same mechanism cluster.
            approved_incumbent_count: Approved programs in disease.

        Returns:
            Category string.
        """
        if same_mechanism_competition_count == 0 and approved_incumbent_count == 0:
            return "unknown"

        if same_mechanism_competition_count <= 1 and approved_incumbent_count == 0:
            return "low"
        elif 2 <= same_mechanism_competition_count <= 4 or 1 <= approved_incumbent_count <= 2:
            return "moderate"
        elif same_mechanism_competition_count >= 5 or approved_incumbent_count >= 3:
            return "high"
        else:
            return "unknown"

    def build_from_records(
        self,
        asset_indication_records: list[AssetIndicationMapRecord],
        enhanced_clusters: list[EnhancedCompetitiveClusterRecord],
    ) -> tuple[list[LandscapeContextFeatureRecord], LandscapeContextCoverageReport]:
        """Build landscape context features from Phase 9 and Phase 10 records.

        Args:
            asset_indication_records: List of AssetIndicationMapRecord.
            enhanced_clusters: List of EnhancedCompetitiveClusterRecord.

        Returns:
            Tuple of (context_features, coverage_report).
        """
        # Build cluster lookup by cluster_key
        cluster_lookup = {}
        for cluster in enhanced_clusters:
            cluster_lookup[cluster.cluster_key] = cluster

        # Build disease-level index: disease_key -> [records]
        disease_index = defaultdict(list)
        for record in asset_indication_records:
            disease_key = self._get_disease_key(record)
            disease_index[disease_key].append(record)

        # Build context features
        context_features = []

        for record in asset_indication_records:
            disease_key = self._get_disease_key(record)
            mechanism_class = self._get_mechanism_class(record)
            target = self._get_target(record)
            modality = self._get_modality(record)

            # Build cluster key (same as Phase 10)
            cluster_key = f"{disease_key}|{mechanism_class}|{target}|{modality}"

            # Find matching cluster
            matching_cluster = cluster_lookup.get(cluster_key)
            cluster_id = matching_cluster.cluster_id if matching_cluster else None

            # Count disease competition
            disease_competition_count = len(disease_index[disease_key])

            # Count same mechanism competition
            same_mechanism_competition_count = 0
            approved_incumbent_count = 0
            same_stage_competition_count = 0

            if matching_cluster:
                same_mechanism_competition_count = matching_cluster.program_count
                approved_incumbent_count = 0
                same_stage_competition_count = 0

                # Count approved in disease
                for dr in disease_index[disease_key]:
                    if dr.clinical_stage and "approved" in dr.clinical_stage.lower():
                        approved_incumbent_count += 1

                # Count same stage in cluster
                if record.clinical_stage:
                    for cr in asset_indication_records:
                        if (
                            self._get_disease_key(cr) == disease_key
                            and self._get_mechanism_class(cr) == mechanism_class
                            and self._get_target(cr) == target
                            and self._get_modality(cr) == modality
                            and cr.clinical_stage == record.clinical_stage
                        ):
                            same_stage_competition_count += 1

            # Categorize fields
            mechanism_novelty = self._categorize_mechanism_novelty(same_mechanism_competition_count)
            target_disease_evidence = self._categorize_target_disease_evidence(
                record.source_type, matching_cluster.source_type_distribution if matching_cluster else {}
            )
            white_space = self._categorize_white_space(disease_competition_count, same_mechanism_competition_count)
            crowding = self._categorize_crowding(same_mechanism_competition_count, approved_incumbent_count)

            # Build warnings
            warnings = []
            if not matching_cluster:
                warnings.append("cluster_not_found")
            if not record.mondo_id:
                warnings.append("disease_unmapped")

            # Merge source refs from record and cluster
            all_source_refs = set(record.source_refs or [])
            if matching_cluster:
                all_source_refs.update(matching_cluster.source_refs or [])

            # Build feature
            feature = LandscapeContextFeatureRecord(
                feature_id=self._make_feature_id(record.record_id, cluster_id),
                source_record_id=record.record_id,
                cluster_id=cluster_id,
                cluster_key=cluster_key if matching_cluster else None,
                company_id=record.company_id,
                ticker=record.ticker,
                company_name=record.company_name,
                asset_id=record.asset_id,
                asset_name=record.asset_name,
                raw_indication=record.raw_indication,
                normalized_disease_name=record.normalized_disease_name,
                mondo_id=record.mondo_id,
                therapeutic_area=record.therapeutic_area,
                mechanism_class=mechanism_class,
                target=target,
                modality=modality,
                clinical_stage=record.clinical_stage,
                disease_competition_count=disease_competition_count,
                same_mechanism_competition_count=same_mechanism_competition_count,
                same_stage_competition_count=same_stage_competition_count,
                approved_incumbent_count=approved_incumbent_count,
                mechanism_novelty_category=mechanism_novelty,
                target_disease_evidence_category=target_disease_evidence,
                trial_design_strength_category="unknown",
                next_readout_days=None,
                white_space_category=white_space,
                crowding_category=crowding,
                supporting_cluster_program_count=matching_cluster.program_count if matching_cluster else 0,
                supporting_cluster_asset_count=matching_cluster.asset_count if matching_cluster else 0,
                supporting_cluster_company_count=matching_cluster.company_count if matching_cluster else 0,
                supporting_cluster_ticker_count=matching_cluster.ticker_count if matching_cluster else 0,
                source_type_distribution=matching_cluster.source_type_distribution.copy() if matching_cluster else {},
                clinical_stage_distribution=(
                    matching_cluster.clinical_stage_distribution.copy() if matching_cluster else {}
                ),
                source_refs=sorted(all_source_refs),
                as_of_date=self.as_of_date,
                warnings=warnings,
            )

            context_features.append(feature)

        # Sort deterministically
        context_features.sort(key=lambda f: f.feature_id)

        # Build coverage report
        coverage = self._build_coverage_report(context_features, asset_indication_records)

        return context_features, coverage

    def _build_coverage_report(
        self,
        features: list[LandscapeContextFeatureRecord],
        records: list[AssetIndicationMapRecord],
    ) -> LandscapeContextCoverageReport:
        """Build coverage report from features.

        Args:
            features: List of LandscapeContextFeatureRecord.
            records: Original AssetIndicationMapRecord list.

        Returns:
            LandscapeContextCoverageReport.
        """
        report = LandscapeContextCoverageReport(as_of_date=self.as_of_date)

        unique_companies = set()
        unique_tickers = set()
        unique_assets = set()
        unique_diseases = set()
        unique_mondo_ids = set()
        unique_mechanisms = set()
        unique_targets = set()
        unique_modalities = set()

        category_novelty_counts = defaultdict(int)
        category_evidence_counts = defaultdict(int)
        category_trial_counts = defaultdict(int)
        category_white_space_counts = defaultdict(int)
        category_crowding_counts = defaultdict(int)

        aggregate_warnings = set()

        for feature in features:
            report.total_features += 1

            if feature.cluster_id:
                report.records_with_cluster += 1
            else:
                report.records_without_cluster += 1

            if feature.company_id:
                unique_companies.add(feature.company_id)
            if feature.ticker:
                unique_tickers.add(feature.ticker)
            if feature.asset_id:
                unique_assets.add(feature.asset_id)
            if feature.normalized_disease_name:
                unique_diseases.add(feature.normalized_disease_name)
            if feature.mondo_id:
                unique_mondo_ids.add(feature.mondo_id)
            if feature.mechanism_class and feature.mechanism_class != "unknown_mechanism":
                unique_mechanisms.add(feature.mechanism_class)
            if feature.target and feature.target != "unknown_target":
                unique_targets.add(feature.target)
            if feature.modality and feature.modality != "unknown_modality":
                unique_modalities.add(feature.modality)

            category_novelty_counts[feature.mechanism_novelty_category] += 1
            category_evidence_counts[feature.target_disease_evidence_category] += 1
            category_trial_counts[feature.trial_design_strength_category] += 1
            category_white_space_counts[feature.white_space_category] += 1
            category_crowding_counts[feature.crowding_category] += 1

            if feature.next_readout_days is not None:
                report.features_with_next_readout_days += 1
            else:
                report.features_without_next_readout_days += 1

            if feature.warnings:
                report.features_with_warnings += 1
                aggregate_warnings.update(feature.warnings)

        report.unique_companies = len(unique_companies)
        report.unique_tickers = len(unique_tickers)
        report.unique_assets = len(unique_assets)
        report.unique_diseases = len(unique_diseases)
        report.unique_mondo_ids = len(unique_mondo_ids)
        report.unique_mechanisms = len(unique_mechanisms)
        report.unique_targets = len(unique_targets)
        report.unique_modalities = len(unique_modalities)

        report.category_counts_mechanism_novelty = dict(category_novelty_counts)
        report.category_counts_target_disease_evidence = dict(category_evidence_counts)
        report.category_counts_trial_design_strength = dict(category_trial_counts)
        report.category_counts_white_space = dict(category_white_space_counts)
        report.category_counts_crowding = dict(category_crowding_counts)

        report.warnings = sorted(aggregate_warnings)

        return report

    def write_jsonl(
        self,
        features: list[LandscapeContextFeatureRecord],
        path: Path | str,
    ) -> None:
        """Write features to JSONL file.

        Args:
            features: List of LandscapeContextFeatureRecord.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for feature in features:
                f.write(json.dumps(feature.to_dict()) + "\n")

    def write_coverage_report(
        self,
        report: LandscapeContextCoverageReport,
        path: Path | str,
    ) -> None:
        """Write coverage report to JSON file.

        Args:
            report: LandscapeContextCoverageReport.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
