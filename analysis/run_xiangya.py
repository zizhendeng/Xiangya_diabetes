"""Run Xiangya lifestyle analyses from the authorized workbook.

Only aggregate summaries are written, to a directory outside this repository.
Participant identifiers, individual outcome arrays and model pickles are not saved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import qr
from statsmodels.stats.multitest import multipletests

from models import rebuild_full_cohort as cleaning
from models import lifestyle_joint_analysis as lifestyle
from models import clinical_discrete_analysis as clinical
from safe_io import require_input, prepare_output

REFERENCE = 'unchanged / unchanged'
TARGET = 'improved / improved'
EXPOSURE = 'C(joint_lifestyle, Treatment(reference="unchanged / unchanged"))'
TARGET_TERM = EXPOSURE + '[T.improved / improved]'


def save(folder, name, rows):
    pd.DataFrame(rows).to_csv(folder / (name + '.csv'), index=False)


def contrast(fit, terms, **labels):
    vector = np.array([terms.get(name, 0.) for name in fit.params.index])
    if not np.any(vector):
        raise ValueError('The requested contrast is not represented in the fitted model.')
    test = fit.t_test(vector)
    ci = np.asarray(test.conf_int()).ravel()
    return dict(**labels, estimate=float(np.asarray(test.effect).item()),
                se=float(np.asarray(test.sd).item()), ci_low=float(ci[0]),
                ci_high=float(ci[1]), p_value=float(np.asarray(test.pvalue).item()))


def stable_refit(formula, frame):
    basic = smf.wls(formula, data=frame, weights=frame.ipcw).fit()
    design = np.asarray(basic.model.exog)
    weights = np.asarray(basic.model.weights)
    outcome = np.asarray(basic.model.endog)
    names = np.asarray(basic.model.exog_names)
    weighted = design * np.sqrt(weights)[:, None]
    leverage = np.sum(weighted * np.linalg.pinv(weighted).T, axis=1)
    keep = leverage < 1 - 1e-10
    _, _, pivots = qr(weighted[keep], mode='economic', pivoting=True)
    rank = np.linalg.matrix_rank(weighted[keep])
    retained = np.sort(pivots[:rank])
    reduced = pd.DataFrame(design[keep][:, retained], columns=names[retained].tolist())
    fitted = sm.WLS(outcome[keep], reduced, weights=weights[keep]).fit(cov_type='HC3')
    return fitted, dict(n=int(keep.sum()), saturated_records_excluded=int((~keep).sum()),
                        removed_dependent_columns=design.shape[1]-rank)


def descriptive_tables(raw, frame, out):
    observed = raw.loc[frame.index]
    flow = [('Enrolled', len(raw)), ('Baseline HbA1c', raw.hba1c_baseline.notna().sum()),
            ('Follow up HbA1c', raw.hba1c_followup.notna().sum()),
            ('Paired HbA1c', (raw.hba1c_baseline.notna() & raw.hba1c_followup.notna()).sum()),
            ('Paired HbA1c and joint change', len(frame))]
    save(out, 'cohort_flow', [dict(stage=label, n=int(n)) for label,n in flow])
    columns = lifestyle.NUMERIC_COVS + ['survey_source','baseline_med_status','med_changed']
    save(out, 'covariate_missingness', [dict(variable=c, population_n=len(observed),
         missing_n=int(observed[c].isna().sum()), missing_percent=float(100*observed[c].isna().mean())) for c in columns])
    rows = []
    for col in lifestyle.NUMERIC_COVS + ['hba1c_followup','delta_hba1c']:
        values = pd.to_numeric(observed[col], errors='coerce').dropna()
        rows.append(dict(variable=col, available_n=len(values), mean=values.mean(), sd=values.std(),
                         median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75)))
    save(out, 'cohort_descriptives', rows)
    rows = []
    for (diet, activity), group in frame.groupby(['diet_change3','exercise_change3'], observed=True):
        rows.append(dict(diet=diet, activity=activity, n=len(group),
                         baseline_mean=group.hba1c_baseline.mean(), followup_mean=group.hba1c_followup.mean(),
                         raw_delta_mean=group.delta_hba1c.mean(), raw_delta_sd=group.delta_hba1c.std()))
    save(out, 'joint_cells', rows)
    before = []
    for col in lifestyle.NUMERIC_COVS:
        a = frame.loc[frame.joint_lifestyle.eq(TARGET),col]
        b = frame.loc[frame.joint_lifestyle.eq(REFERENCE),col]
        scale = np.sqrt((a.var()+b.var())/2)
        before.append(dict(variable=col, target_mean=a.mean(), reference_mean=b.mean(), smd=(a.mean()-b.mean())/scale))
    save(out, 'baseline_standardized_differences', before)
    summaries = []
    for label, subset in [('Overall',frame), ('Joint improvement',frame.loc[frame.joint_lifestyle.eq(TARGET)]),
                          ('Both unchanged',frame.loc[frame.joint_lifestyle.eq(REFERENCE)])]:
        w = subset.ipcw.to_numpy()
        summaries.append(dict(group=label, n=len(w), ess=w.sum()**2/(w*w).sum(), minimum=w.min(),
                              p01=np.quantile(w,.01), median=np.median(w), p99=np.quantile(w,.99), maximum=w.max()))
    save(out, 'observation_weights', summaries)


def analyse(workbook, out):
    cleaning.DATA_FILE = workbook
    raw = lifestyle.prepare()
    frame = lifestyle.model_frame(raw)
    if frame.empty:
        raise ValueError('No evaluable paired observations.')
    descriptive_tables(raw, frame, out)
    formula = 'hba1c_followup ~ ' + EXPOSURE + ' + ' + lifestyle.covariate_formula()
    fit = smf.wls(formula, data=frame, weights=frame.ipcw).fit(cov_type='HC3')
    joint_rows = []
    for level in sorted(frame.joint_lifestyle.unique()):
        count = int(frame.joint_lifestyle.eq(level).sum())
        if level == REFERENCE:
            joint_rows.append(dict(category=level, n=count, estimate=0., se=np.nan, ci_low=np.nan,
                                   ci_high=np.nan, p_value=np.nan))
        else:
            joint_rows.append(contrast(fit, {EXPOSURE+'[T.'+level+']':1}, category=level, n=count))
    save(out, 'joint_contrasts', joint_rows)
    _, secondary, _ = lifestyle.lifestyle_main(raw)
    secondary['p_holm_improvement'] = np.nan
    for _, index in secondary.groupby('model').groups.items():
        selected = secondary.loc[index].query("level == 'improved'").index
        secondary.loc[selected,'p_holm_improvement'] = multipletests(secondary.loc[selected,'p_value'],method='holm')[1]
    save(out, 'secondary_behaviours', secondary)
    counts, items = lifestyle.diet_item_model(raw)
    save(out, 'diet_item_counts', counts)
    save(out, 'diet_items', items)

    covariates = [
        'hba1c_baseline + I(hba1c_baseline ** 2)',
        'hba1c_baseline + I(hba1c_baseline ** 2) + age + sex_male + C(survey_source)',
        'hba1c_baseline + I(hba1c_baseline ** 2) + age + sex_male + bmi + whr + egfr + fbg + sbp + diabetes_duration + comorbidity_count + C(survey_source)',
        lifestyle.covariate_formula(),
    ]
    sequential = []
    for label, terms in zip(['Baseline HbA1c','Add demographics','Add clinical factors','Full primary model'],covariates):
        result = smf.wls('hba1c_followup ~ '+EXPOSURE+' + '+terms, data=frame, weights=frame.ipcw).fit(cov_type='HC3')
        sequential.append(contrast(result,{TARGET_TERM:1},label=label,n=int(result.nobs)))
    save(out, 'sequential_adjustment', sequential)

    support_columns = [c for c in raw if '组别' in c]
    date_columns = [c for c in raw if c.startswith('1st-') and '采集日期' in c]
    if not support_columns or not date_columns:
        raise ValueError('Management support and follow up date columns are required for sensitivity analyses.')
    aliases = {'传统':'Usual care','传统组':'Usual care','糖教授':'Education','糖教授组':'Education',
               'CGM组':'CGM','糖教授+CGM':'Education + CGM','糖教授+CGM+手环':'Education + CGM + wearable',
               '糖教授+手环':'Education + wearable'}
    support = raw.loc[frame.index,support_columns[0]].fillna('Not recorded').astype(str).str.strip()
    frame['management_arm'] = support.map(aliases).fillna(support)
    frame['interval_days'] = (pd.to_datetime(raw.loc[frame.index,date_columns[0]],errors='coerce',format='mixed')
                              -pd.to_datetime(raw.loc[frame.index,'Date'],errors='coerce',format='mixed')).dt.days
    save(out, 'management_support', [dict(category=name,n=len(group),
         joint_improved_n=int(group.joint_lifestyle.eq(TARGET).sum()),
         unchanged_n=int(group.joint_lifestyle.eq(REFERENCE).sum())) for name,group in frame.groupby('management_arm')])
    intervals = frame.interval_days.dropna()
    save(out, 'observation_interval', [dict(n=len(intervals),median=intervals.median(),q25=intervals.quantile(.25),
         q75=intervals.quantile(.75),minimum=intervals.min(),maximum=intervals.max())])
    specs = [
        ('Primary',frame,True,''), ('Unweighted',frame,False,''),
        ('Target and reference only',frame.loc[frame.joint_lifestyle.isin([TARGET,REFERENCE])],True,''),
        ('Medication change adjusted',frame,True,' + med_changed'),
        ('Exclude mixed or conflicting diet',frame.loc[~frame.diet_mixed_direction.fillna(False).astype(bool)
             &~frame.diet_nochange_conflict.fillna(False).astype(bool)],True,''),
        ('Absolute HbA1c change at most 3 points',frame.loc[frame.delta_hba1c.abs().le(3)],True,''),
        ('Management arm adjusted',frame,True,' + C(management_arm)'),
        ('Actual visit interval adjusted',frame,True,' + interval_days'),
    ]
    sensitivities = []
    for label,subset,weighted,extra in specs:
        model = (smf.wls(formula+extra,data=subset,weights=subset.ipcw) if weighted else smf.ols(formula+extra,data=subset))
        result = model.fit(cov_type='HC3')
        sensitivities.append(contrast(result,{TARGET_TERM:1},label=label,n=int(result.nobs)))
    rank_rows = []
    for label,subset,formula_here in [
        ('Primary',frame,formula), ('No medication change',frame.loc[frame.med_changed.eq(0)],formula),
        ('Management arm adjusted',frame,formula+' + C(management_arm)')]:
        result, diagnostic = stable_refit(formula_here,subset)
        row = contrast(result,{TARGET_TERM:1},label=label+' with rank-stable HC3',**diagnostic)
        rank_rows.append(row)
        if label == 'No medication change':
            sensitivities.append({**row,'label':label+' with rank-stable HC3'})
    save(out, 'rank_stable_sensitivity', rank_rows)
    save(out, 'sensitivity', sensitivities)
    direct = []
    for label,levels in [('Joint versus diet-only',['improved / unchanged']),
                         ('Joint versus activity-only',['unchanged / improved']),
                         ('Departure from additive joint-category effects',['improved / unchanged','unchanged / improved'])]:
        terms = {TARGET_TERM:1, **{EXPOSURE+'[T.'+level+']':-1 for level in levels}}
        direct.append(contrast(fit,terms,label=label))
    adjusted = multipletests([r['p_value'] for r in direct],method='holm')[1]
    for row,p in zip(direct,adjusted):
        row['p_holm_three_tests'] = p
    save(out, 'direct_joint_comparisons', direct)
    grouped = clinical.add_clinical_groups(frame)
    strata, interactions = clinical.lifestyle_strata(grouped)
    save(out, 'clinical_strata', strata)
    save(out, 'interactions', interactions)
    save(out, 'clinical_threshold_adjustment', clinical.lifestyle_adjustment_sensitivity(grouped))
    report = dict(input_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
                  enrolled_n=len(raw),analysis_n=len(frame),primary=contrast(fit,{TARGET_TERM:1}),
                  environment={'python':platform.python_version(), **{name:version(name) for name in
                       ['numpy','pandas','scipy','statsmodels','scikit-learn','patsy','openpyxl']}})
    (out/'xiangya_run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Xiangya analyses completed. Aggregate outputs were written outside the repository.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xiangya-workbook',required=True)
    parser.add_argument('--out-dir',required=True)
    args = parser.parse_args()
    workbook = require_input(args.xiangya_workbook)
    out = prepare_output(args.out_dir,[workbook])
    analyse(workbook,out)


if __name__ == '__main__':
    main()
