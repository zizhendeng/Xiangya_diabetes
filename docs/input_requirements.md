# Input requirements

No example records or empirical study outputs are included. Column names and coding rules below describe interfaces, not participant data.

## UK Biobank

Supply five CSV files in one authorized input directory: `base.csv`, `lifestyle.csv`, `bmi.csv`, `cystatin.csv` and `covariates.csv`. Each file must contain a unique, nonmissing `eid` in the same licensed participant identifier system. The exported population must already satisfy the study eligibility rule: the first recorded ICD-10 E11 date (field 130708) is on or before the initial assessment date (field 53, instance 0). The scripts validate this eligibility and do not select participants by outcome change.

The merged files must provide these fields:

- Dates and demographics: `p53_i0`, `p53_i1`, `p130706`, `p130708`, `p21022`, `p34`, `p31`, `p21000_i0`.
- Clinical measures: `p30750_i0`, `p30750_i1`, `p21001_i0`, `p30700_i0`, `p30740_i0`, `p4080_i0_a0`, `p4080_i0_a1`.
- Diet fields at both instances 0 and 1: 1289, 1299, 1309, 1329, 1339, 1359, 1349, 1369, 1379 and 1389. For example, `p1289_i0` and `p1289_i1`.
- Activity fields at both instances 0 and 1: 864, 884, 904, 874, 894 and 914.
- Initial smoking and drinking categories: `p20116_i0` and `p1558_i0`.

The longitudinal implementation expects `p31` labelled `Female` or `Male`, matching the analyzed export. The cross sectional script also accepts numeric sex codes and numeric or labelled ethnicity values. Do not silently mix coding systems or identifier projects. Shared columns other than `eid` should not be duplicated across the five files because the one-to-one merges would then create suffixed field names.

For longitudinal smoking and drinking comparisons, supply a separate CSV with unique `eid`, `p20116_i1` and `p1558_i1`. Only these repeat fields are read. Repeat BMI is not required. UK Biobank HbA1c fields must be in mmol/mol, creatinine in micromoles per litre and glucose in mmol/l.

## Xiangya

Supply the authorized baseline and first follow up workbook in its original column layout. The first worksheet is read. The code uses the exact Chinese questionnaire headers to preserve the analyzed definitions; the relevant keys are in `analysis/models/rebuild_full_cohort.py` and `analysis/models/lifestyle_joint_analysis.py`.

Core clinical columns include `Sex`, `Birth_year`, `Date`, `Height`, `Weight`, `WC`, `HC`, `HBA1c`, `FBG`, `SBP`, `DBP`, `Cr`, `eGFR`, `ALT` and the first follow up HbA1c column. HbA1c values are on the NGSP (%) scale. Baseline questionnaire columns retain their collection-source prefixes because those names are needed for deterministic harmonization; these are field identifiers, not participant records.

Additional inputs are the dietary direction indicators, exercise-change response, baseline medication fields, reported medication change, the management-support column containing `组别`, and the first follow up date column starting with `1st-` and containing `采集日期`. If an export contains several candidate management or visit-date columns, reconcile them with the analyzed workbook before running the scripts. The implementation follows the original export column order.

Direct identifiers, contact details, device identifiers and free-text notes are not required as analysis predictors or exported by the runners. Do not add a workbook, subset, participant list or screenshot to this repository when reporting an issue.
