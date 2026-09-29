# Xiangya diabetes lifestyle analyses

Analysis code for **Prioritizing lifestyle changes for glycaemic control through causal inference and prospective validation**.

This repository contains code, dependency specifications and execution instructions only. It contains no participant records, raw data, derived datasets, aggregate study results, figures, manuscripts, fitted models or notebook outputs. Data access is not granted by access to this repository.

## Analyses

- UK Biobank cross sectional comparisons of diet, physical activity, smoking and drinking frequency, with covariate adjustment and Holm correction.
- UK Biobank longitudinal ANCOVA with baseline adjustment and inverse probability of observation weighting; joint and individual behaviour comparisons, direct contrasts, sensitivity analyses and observation selection diagnostics.
- Xiangya cohort preparation, behavioural change definitions, primary and secondary models, sequential adjustment, medication and other sensitivity analyses, numerical stability checks, dietary component models, clinical strata and global interaction tests.

The public scripts retain the manuscript's measurement definitions, model formulas, observation weight construction and multiplicity families. Study-specific expected values used in local regression checks are not embedded in the public code. Unrelated medication-effect analyses and document-generation scripts are excluded. Figure rendering and aggregate Source Data are distributed separately from this code repository.

## Installation

Use Python 3.12 in an isolated environment and install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
```

## Authorized inputs

Keep all inputs and outputs outside the repository. The scripts never download study data and do not require credentials in configuration files. See [Input requirements](docs/input_requirements.md) for field and file conventions.

Run the commands below from the repository root, replacing the example paths with authorized locations. The directories are examples, not supplied data.

```sh
python -B analysis/run_ukb_cross_sectional.py --ukb-dir /authorized_inputs/ukb --out-dir /analysis_outputs/ukb_cross_sectional

python -B analysis/run_ukb_longitudinal.py --ukb-dir /authorized_inputs/ukb --repeat-file /authorized_inputs/repeat_behaviours.csv --out-dir /analysis_outputs/ukb_longitudinal

python -B analysis/run_xiangya.py --xiangya-workbook /authorized_inputs/xiangya.xlsx --out-dir /analysis_outputs/xiangya
```

The runners require licensed UK Biobank exports and/or an institutionally authorized Xiangya workbook. Without these inputs, study results cannot be reproduced from this repository alone. Each runner writes aggregate analysis outputs to the requested external directory; these generated outputs must not be committed to this repository. Inputs are read without modification. Patient-level tables, individual outcome arrays and fitted-model objects are not exported by the runners.

## Statistical conventions

HbA1c levels use the NGSP (%) scale for reporting. Changes and regression coefficients represent absolute differences on that scale, not relative percentage reductions. UK Biobank model differences are converted from mmol/mol by multiplying by 0.09148; the intercept 2.152 is used only when converting absolute levels.

Observation weights account for measured differences in contributing follow up data; they are not exposure propensity weights. Joint-model categories, individual behaviour models and direct joint-versus-single contrasts answer different questions. The methods and comparison families are described in [Analysis notes](docs/analysis_notes.md).

Institutional authorization is required for Xiangya individual records, and UK Biobank access is subject to its application and data-use requirements. This code release does not authorize redistribution of either dataset or of generated study results. No software licence has been assigned in this release.
