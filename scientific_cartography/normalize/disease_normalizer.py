"""Disease normalization with conservative matching and manual override support."""

import hashlib
from pathlib import Path
from typing import Optional

from scientific_cartography.schemas.disease_schema import DiseaseRecord


class DiseaseNormalizer:
    """Normalize raw disease labels into canonical disease records.

    Uses MONDO (Mondo Disease Ontology) as the primary normalization spine,
    providing semantic harmonization across disease terminology variants.

    Priority order:
    1. Manual override file
    2. MONDO exact match (by ID or preferred term)
    3. MONDO synonym match (looser variant matching)
    4. Case/punctuation-normalized MONDO match
    5. Preserve raw as unmapped with low confidence

    MONDO advantages:
    - Semantic equivalence: "atopic dermatitis" = "eczema" (both map to MONDO:0004980)
    - Cross-resource mapping: UMLS, ICD-10, MeSH, SNOMED CT references
    - Hierarchical: parent-child relationships (AD → skin diseases → diseases)
    - Conservative: distinguishes true equivalence from looser relationships
    """

    def __init__(
        self,
        manual_overrides_csv: Optional[Path] = None,
        mondo_cache: Optional[dict] = None,
        as_of_date: str = "",
    ):
        """Initialize disease normalizer with MONDO spine.

        Args:
            manual_overrides_csv: Path to manual disease aliases CSV.
            mondo_cache: Dict mapping raw disease → MONDO record.
            as_of_date: Date for records (YYYY-MM-DD).
        """
        self.as_of_date = as_of_date
        self.manual_overrides: dict[str, DiseaseRecord] = {}
        self.mondo_cache = mondo_cache or {}
        self._disease_cache: dict[str, DiseaseRecord] = {}
        self._mondo_index = self._build_mondo_index()

        if manual_overrides_csv and manual_overrides_csv.exists():
            self._load_manual_overrides(manual_overrides_csv)

    def _load_manual_overrides(self, csv_path: Path) -> None:
        """Load manual disease aliases from CSV.

        CSV format:
        raw_name,normalized_name,mondo_id,therapeutic_area,confidence,notes
        """
        try:
            import csv

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_id = self._make_disease_id(row["normalized_name"], row.get("mondo_id"))
                    record = DiseaseRecord(
                        disease_id=disease_id,
                        raw_name=row["raw_name"],
                        normalized_name=row["normalized_name"],
                        mondo_id=row.get("mondo_id"),
                        therapeutic_area=row.get("therapeutic_area"),
                        confidence=float(row.get("confidence", 1.0)),
                        source="manual_override",
                        as_of_date=self.as_of_date,
                    )
                    self.manual_overrides[row["raw_name"].lower().strip()] = record

                    # Also index by normalized name for reverse lookup
                    key = row["normalized_name"].lower().strip()
                    if key not in self.manual_overrides:
                        self.manual_overrides[key] = record
        except Exception as e:
            raise ValueError(f"Failed to load manual disease overrides: {e}")

    def _make_disease_id(self, normalized_name: str, mondo_id: Optional[str]) -> str:
        """Create stable disease ID from normalized name and MONDO ID."""
        if mondo_id:
            return mondo_id

        normalized = normalized_name.lower().strip()
        hash_hex = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"DISEASE_{hash_hex}"

    def _normalize_for_lookup(self, raw_disease: str) -> str:
        """Normalize a disease string for case-insensitive lookup."""
        return raw_disease.lower().strip()

    def _build_mondo_index(self) -> dict:
        """Build MONDO disease index from curated biotech disease list.

        Returns dict mapping normalized names → DiseaseRecord with MONDO IDs.
        MONDO harmonizes disease terminology across resources.
        """
        # Curated MONDO diseases covering major biotech therapeutic areas
        # Format: preferred_term → {mondo_id, therapeutic_area, synonyms}
        mondo_diseases = {
            # Oncology
            "non-small cell lung cancer": {
                "mondo_id": "MONDO:0005233",
                "therapeutic_area": "Oncology",
                "synonyms": ["nsclc", "non-small-cell lung cancer"],
            },
            "small cell lung cancer": {
                "mondo_id": "MONDO:0005235",
                "therapeutic_area": "Oncology",
                "synonyms": ["sclc", "small cell lung carcinoma"],
            },
            "melanoma": {
                "mondo_id": "MONDO:0005105",
                "therapeutic_area": "Oncology",
                "synonyms": ["malignant melanoma", "cutaneous melanoma"],
            },
            "pancreatic cancer": {
                "mondo_id": "MONDO:0005147",
                "therapeutic_area": "Oncology",
                "synonyms": ["pancreatic adenocarcinoma"],
            },
            "colorectal cancer": {
                "mondo_id": "MONDO:0005575",
                "therapeutic_area": "Oncology",
                "synonyms": ["colon cancer", "rectal cancer", "crc"],
            },
            "breast cancer": {
                "mondo_id": "MONDO:0007254",
                "therapeutic_area": "Oncology",
                "synonyms": ["mammary cancer"],
            },
            "lymphoma": {
                "mondo_id": "MONDO:0005105",
                "therapeutic_area": "Oncology",
                "synonyms": ["non-hodgkin lymphoma", "nhl", "hodgkin lymphoma"],
            },
            # Immunology/Inflammation
            "atopic dermatitis": {
                "mondo_id": "MONDO:0004980",
                "therapeutic_area": "Dermatology",
                "synonyms": ["eczema", "ad", "atopic eczema", "dermatitis"],
            },
            "psoriasis": {
                "mondo_id": "MONDO:0005083",
                "therapeutic_area": "Dermatology",
                "synonyms": ["psoriatic dermatitis"],
            },
            "rheumatoid arthritis": {
                "mondo_id": "MONDO:0005148",
                "therapeutic_area": "Immunology",
                "synonyms": ["ra", "rheumatoid disease"],
            },
            "systemic lupus erythematosus": {
                "mondo_id": "MONDO:0007617",
                "therapeutic_area": "Immunology",
                "synonyms": ["sle", "lupus"],
            },
            "crohn's disease": {
                "mondo_id": "MONDO:0005011",
                "therapeutic_area": "Gastroenterology",
                "synonyms": ["crohns disease", "regional enteritis"],
            },
            "ulcerative colitis": {
                "mondo_id": "MONDO:0005052",
                "therapeutic_area": "Gastroenterology",
                "synonyms": ["uc", "idiopathic proctocolitis"],
            },
            # Neurology
            "parkinson's disease": {
                "mondo_id": "MONDO:0005180",
                "therapeutic_area": "Neurology",
                "synonyms": ["parkinsons disease", "pd"],
            },
            "alzheimer's disease": {
                "mondo_id": "MONDO:0004975",
                "therapeutic_area": "Neurology",
                "synonyms": ["alzheimers disease", "ad"],
            },
            "multiple sclerosis": {
                "mondo_id": "MONDO:0005301",
                "therapeutic_area": "Neurology",
                "synonyms": ["ms", "multiple sclerosis disease"],
            },
            # Metabolic
            "type 2 diabetes mellitus": {
                "mondo_id": "MONDO:0005148",
                "therapeutic_area": "Metabolic",
                "synonyms": ["type 2 diabetes", "t2dm", "diabetes mellitus type 2"],
            },
            "obesity": {
                "mondo_id": "MONDO:0004994",
                "therapeutic_area": "Metabolic",
                "synonyms": ["adiposity"],
            },
        }

        index = {}
        for normalized_name, attrs in mondo_diseases.items():
            record = DiseaseRecord(
                disease_id=attrs["mondo_id"],
                raw_name=normalized_name,
                normalized_name=normalized_name,
                mondo_id=attrs["mondo_id"],
                therapeutic_area=attrs["therapeutic_area"],
                synonyms=attrs.get("synonyms", []),
                source="mondo",
                confidence=0.95,
                as_of_date=self.as_of_date,
            )
            index[normalized_name.lower()] = record

        return index

    def normalize(self, raw_disease: str) -> DiseaseRecord:
        """Normalize a raw disease label to a canonical DiseaseRecord.

        Uses MONDO ontology as primary spine for semantic harmonization.
        Returns a DiseaseRecord with confidence > 0. If normalization fails,
        returns a low-confidence record preserving the raw label.

        Args:
            raw_disease: Raw disease label from source.

        Returns:
            DiseaseRecord with mondo_id, normalized_name, and confidence.
        """
        # Preserve and return cached result
        cache_key = self._normalize_for_lookup(raw_disease)
        if cache_key in self._disease_cache:
            return self._disease_cache[cache_key]

        # Priority 1: Manual override (user-curated highest confidence)
        if cache_key in self.manual_overrides:
            record = self.manual_overrides[cache_key]
            self._disease_cache[cache_key] = record
            return record

        # Priority 2: Try exact normalized match in MONDO cache first
        if cache_key in self.mondo_cache:
            mondo_data = self.mondo_cache[cache_key]
            record = DiseaseRecord(
                disease_id=mondo_data.get("id", self._make_disease_id(raw_disease, None)),
                raw_name=raw_disease,
                normalized_name=mondo_data.get("name", raw_disease),
                mondo_id=mondo_data.get("id"),
                therapeutic_area=mondo_data.get("therapeutic_area"),
                synonyms=mondo_data.get("synonyms", []),
                source="mondo",
                confidence=0.95,
                as_of_date=self.as_of_date,
            )
            self._disease_cache[cache_key] = record
            return record

        # Priority 3: Try synonym match in MONDO cache
        for syn_raw, mondo_data in self.mondo_cache.items():
            if raw_disease.lower() in [s.lower() for s in mondo_data.get("synonyms", [])]:
                record = DiseaseRecord(
                    disease_id=mondo_data.get("id", self._make_disease_id(raw_disease, None)),
                    raw_name=raw_disease,
                    normalized_name=mondo_data.get("name", raw_disease),
                    mondo_id=mondo_data.get("id"),
                    therapeutic_area=mondo_data.get("therapeutic_area"),
                    synonyms=mondo_data.get("synonyms", []),
                    source="mondo_synonym",
                    confidence=0.90,
                    as_of_date=self.as_of_date,
                )
                self._disease_cache[cache_key] = record
                return record

        # Priority 4: MONDO built-in index exact match (ontology-backed)
        if cache_key in self._mondo_index:
            record = self._mondo_index[cache_key]
            self._disease_cache[cache_key] = record
            return record

        # Priority 5: MONDO built-in exact synonym match. Exact matches are safe
        # for short aliases such as "AD"; substring matches below are not.
        for record in self._mondo_index.values():
            if cache_key in {synonym.lower().strip() for synonym in record.synonyms}:
                matched_record = DiseaseRecord(
                    disease_id=record.disease_id,
                    raw_name=raw_disease,
                    normalized_name=record.normalized_name,
                    mondo_id=record.mondo_id,
                    therapeutic_area=record.therapeutic_area,
                    synonyms=record.synonyms,
                    source="mondo_synonym",
                    confidence=0.90,
                    as_of_date=self.as_of_date,
                )
                self._disease_cache[cache_key] = matched_record
                return matched_record

        # Priority 6: Conservative phrase matching against MONDO index terms
        # (e.g., "moderate-to-severe atopic dermatitis" → MONDO:0004980).
        # Do not substring-match short aliases such as "AD"; that incorrectly
        # maps unrelated diseases like "Prader-Willi Syndrome" because "ad" is
        # embedded inside another word.
        for normalized_name, record in self._mondo_index.items():
            if len(normalized_name) < 4:
                continue
            if normalized_name in cache_key or cache_key in normalized_name:
                # Confidence slightly lower for substring match
                matched_record = DiseaseRecord(
                    disease_id=record.disease_id,
                    raw_name=raw_disease,
                    normalized_name=record.normalized_name,
                    mondo_id=record.mondo_id,
                    therapeutic_area=record.therapeutic_area,
                    synonyms=record.synonyms,
                    source="mondo_substring",
                    confidence=0.80,
                    as_of_date=self.as_of_date,
                )
                self._disease_cache[cache_key] = matched_record
                return matched_record

        # Unmapped: preserve raw disease with low confidence
        disease_id = self._make_disease_id(raw_disease, None)
        record = DiseaseRecord(
            disease_id=disease_id,
            raw_name=raw_disease,
            normalized_name=raw_disease,
            source="unmapped",
            confidence=0.0,
            as_of_date=self.as_of_date,
        )
        self._disease_cache[cache_key] = record
        return record

    def bulk_normalize(self, raw_diseases: list[str]) -> list[DiseaseRecord]:
        """Normalize multiple raw disease labels.

        Args:
            raw_diseases: List of raw disease labels.

        Returns:
            List of DiseaseRecords in same order.
        """
        return [self.normalize(d) for d in raw_diseases]
