# Analysis notes

## Cohort definitions

The cross sectional UK Biobank sample and the repeated-measure sample have different completeness requirements. The longitudinal sample is not a sequential subset obtained by filtering the cross sectional model frame. All four behaviours are considered at both analytical levels; cross sectional significance is not a selection criterion.

Longitudinal diet and activity models use the same complete-case cohort, baseline covariates and observation weights as the primary joint exposure model. Smoking cessation and reduced drinking frequency use the applicable baseline smoker and drinker subsets. Sparse transitions do not receive a fitted adjusted effect when either comparison category has fewer than ten participants.

## Exposures

UK Biobank dietary change combines eight directional components. Scores of at least two indicate improvement, scores of at most minus two indicate worsening, and intermediate scores indicate no change. The activity index is walking days plus twice moderate-activity days plus three times vigorous-activity days; changes greater than two indicate improvement and changes less than minus two worsening. The activity-dose sensitivity uses MET-minutes and the specified change band.

Xiangya dietary change is the net balance of four favourable and four unfavourable questionnaire indicators. Positive, zero and negative scores define improvement, unchanged/net neutral and worsening. The exercise response is mapped directly to increased, unchanged or decreased activity. Missing responses remain missing. The resulting three-by-three joint exposure contains mutually exclusive categories.

## Outcome estimation and inference

Primary longitudinal models regress follow up HbA1c on joint lifestyle category, baseline HbA1c and its quadratic term, and the specified demographic, clinical and behavioural covariates. Stabilized inverse probability of observation weights address measured selection into follow up. HC3 provides heteroskedasticity-consistent covariance estimation. These procedures do not remove the need for causal identification assumptions.

Separate Xiangya diet and activity terms are mutually adjusted. Models comparing diet only or activity only improvement with both behaviours unchanged instead use the nine-category joint exposure. Direct comparisons between joint and single behaviour changes and departure from additivity are linear contrasts of coefficients in this same joint model.

Holm correction is applied separately to the four cross sectional UK Biobank behavioural hypotheses, the four longitudinal UK Biobank behavioural hypotheses, the two Xiangya improvement hypotheses, and the three direct exploratory joint contrasts in each cohort. For the longitudinal four-test family, an unavailable cessation comparison enters the adjustment calculation as one but is not reported as an observed P value. Exploratory worsening coefficients retain the four-coefficient Benjamini-Hochberg family. Dietary component and global interaction tests use separate Benjamini-Hochberg families.

Clinical strata use the original minimum cell-size rules and nuisance-variable specification. Global interactions, rather than differences in subgroup significance, assess effect modification. Numerical sensitivity fits remove unit-leverage records and linearly dependent columns using pivoted QR. The unchanged-medication confidence interval uses this numerically stable fit; the primary population definition is not replaced by the diagnostic refit population.

## Scope of the code release

The three runners generate aggregate estimates and diagnostics from authorized inputs. No empirical estimates, source-data arrays, sample-size assertions or expected study outputs are embedded as regression fixtures. Clinical thresholds, questionnaire codes, unit conversion constants and algorithm settings are retained because they define the analysis. Local numerical validation records, input hashes and generated results are not part of the repository.
