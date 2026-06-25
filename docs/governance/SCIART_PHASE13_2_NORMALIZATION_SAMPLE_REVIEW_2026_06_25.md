# Sci-Cart Phase 13.2 — Normalization Sample Review

**Generated:** 2026-06-25T00:54:00.303537
**Trials source:** `production_data/trial_records.json`
**As-of:** 2026-06-25
**Sample:** 50 records (10 per top disease, seed=42)

Manual verdict per row: `TRUE_POSITIVE` | `FALSE_POSITIVE` | `AMBIGUOUS`

## Sample counts

- lymphoma: 10
- breast cancer: 10
- non-small cell lung cancer: 10
- colorectal cancer: 10
- melanoma: 10

## Annotated sample

| target | nct_id | raw_condition | normalized | mondo_id | tier | confidence | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lymphoma | NCT06149286 | Relapsed/Refractory Marginal Zone Lymphoma (R/R MZL) | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT05222555 | Diffuse Large B Cell Lymphoma | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT01513603 | Burkitts Leukemia/Lymphoma | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT02992522 | Refractory Follicular Lymphoma | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT03589469 | Diffuse Large B-cell Lymphoma Recurrent | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT05144009 | Diffuse Large B-cell Lymphoma | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT00918333 | Recurrent Small Lymphocytic Lymphoma | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT04162756 | Relapse/Refractory Mantle Cell Lymphoma | lymphoma | MONDO:0005105 | substring | 0.8 |  |
| lymphoma | NCT00299182 | Lymphoma | lymphoma | MONDO:0005105 | exact | 0.95 |  |
| lymphoma | NCT02568683 | Non-Hodgkin Lymphoma | lymphoma | MONDO:0005105 | synonym | 0.9 |  |
| breast cancer | NCT06767462 | Ophthalmic Safety in Patients With Breast Cancer | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT01223833 | Breast Cancer | breast cancer | MONDO:0007254 | exact | 0.95 |  |
| breast cancer | NCT06750484 | Breast Cancer Metastatic | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT03624543 | Breast Cancer | breast cancer | MONDO:0007254 | exact | 0.95 |  |
| breast cancer | NCT03725436 | Prognostic Stage IIIC Breast Cancer AJCC v8 | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT04616248 | Prognostic Stage IV Breast Cancer AJCC v8 | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT03144648 | Breast Cancer Female | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT03045289 | Breast Cancer Stage IV | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT00777101 | Advanced Breast Cancer | breast cancer | MONDO:0007254 | substring | 0.8 |  |
| breast cancer | NCT05759949 | Breast Cancer | breast cancer | MONDO:0007254 | exact | 0.95 |  |
| non-small cell lung cancer | NCT03215810 | Non-Small Cell Lung Cancer | non-small cell lung cancer | MONDO:0005233 | exact | 0.95 |  |
| non-small cell lung cancer | NCT04083599 | Non-Small Cell Lung Cancer (NSCLC) | non-small cell lung cancer | MONDO:0005233 | substring | 0.8 |  |
| non-small cell lung cancer | NCT00111137 | Cancer | non-small cell lung cancer | MONDO:0005233 | substring | 0.8 |  |
| non-small cell lung cancer | NCT03899155 | Cancer | non-small cell lung cancer | MONDO:0005233 | substring | 0.8 |  |
| non-small cell lung cancer | NCT07216105 | Non-Small Cell Lung Cancer | non-small cell lung cancer | MONDO:0005233 | exact | 0.95 |  |
| non-small cell lung cancer | NCT02544633 | Non-Small Cell Lung Cancer | non-small cell lung cancer | MONDO:0005233 | exact | 0.95 |  |
| non-small cell lung cancer | NCT06449313 | Non-small Cell Lung Cancer Stage III | non-small cell lung cancer | MONDO:0005233 | substring | 0.8 |  |
| non-small cell lung cancer | NCT02515032 | NSCLC | non-small cell lung cancer | MONDO:0005233 | synonym | 0.9 |  |
| non-small cell lung cancer | NCT06216301 | Metastatic Non-small Cell Lung Cancer | non-small cell lung cancer | MONDO:0005233 | substring | 0.8 |  |
| non-small cell lung cancer | NCT07185997 | Non-Small-Cell Lung Cancer | non-small cell lung cancer | MONDO:0005233 | synonym | 0.9 |  |
| colorectal cancer | NCT02953782 | Colorectal Cancer | colorectal cancer | MONDO:0005575 | exact | 0.95 |  |
| colorectal cancer | NCT01508000 | Colorectal Cancer Metastatic | colorectal cancer | MONDO:0005575 | substring | 0.8 |  |
| colorectal cancer | NCT04406714 | Colorectal Cancer | colorectal cancer | MONDO:0005575 | exact | 0.95 |  |
| colorectal cancer | NCT04014530 | Colorectal Cancer | colorectal cancer | MONDO:0005575 | exact | 0.95 |  |
| colorectal cancer | NCT05362344 | Colorectal Cancer | colorectal cancer | MONDO:0005575 | exact | 0.95 |  |
| colorectal cancer | NCT03610490 | Stage IV Colorectal Cancer AJCC v8 | colorectal cancer | MONDO:0005575 | substring | 0.8 |  |
| colorectal cancer | NCT03186326 | Metastatic Colorectal Cancer | colorectal cancer | MONDO:0005575 | substring | 0.8 |  |
| colorectal cancer | NCT02327078 | Colorectal Cancer (CRC) | colorectal cancer | MONDO:0005575 | substring | 0.8 |  |
| colorectal cancer | NCT04456699 | Metastatic Colorectal Cancer | colorectal cancer | MONDO:0005575 | substring | 0.8 |  |
| colorectal cancer | NCT00630786 | Metastatic Colorectal Cancer | colorectal cancer | MONDO:0005575 | substring | 0.8 |  |
| melanoma | NCT02297529 | Unresected Stage IIIB to IVM1c Melanoma | melanoma | MONDO:0005105 | substring | 0.8 |  |
| melanoma | NCT06151847 | Clinical Stage IV Cutaneous Melanoma AJCC v8 | melanoma | MONDO:0005105 | substring | 0.8 |  |
| melanoma | NCT00471133 | Melanoma (Skin) | melanoma | MONDO:0005105 | substring | 0.8 |  |
| melanoma | NCT00365937 | Melanoma | melanoma | MONDO:0005105 | exact | 0.95 |  |
| melanoma | NCT06007690 | Ocular Melanoma | melanoma | MONDO:0005105 | substring | 0.8 |  |
| melanoma | NCT06590480 | Advanced Melanoma | melanoma | MONDO:0005105 | substring | 0.8 |  |
| melanoma | NCT02897765 | Melanoma | melanoma | MONDO:0005105 | exact | 0.95 |  |
| melanoma | NCT02355587 | Cutaneous Melanoma | melanoma | MONDO:0005105 | synonym | 0.9 |  |
| melanoma | NCT07475572 | Melanoma | melanoma | MONDO:0005105 | exact | 0.95 |  |
| melanoma | NCT02147951 | Unresected Stage IIIb to IVM1c Melanoma | melanoma | MONDO:0005105 | substring | 0.8 |  |

## Verdict guidance

- **TRUE_POSITIVE**: MONDO parent/synonym is clinically valid
- **FALSE_POSITIVE**: match would mislead the disease map
- **AMBIGUOUS**: defensible but imprecise subtype→parent mapping
