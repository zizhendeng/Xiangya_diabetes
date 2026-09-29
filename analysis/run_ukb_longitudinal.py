"""Run longitudinal behaviour comparisons using the primary joint-model cohort."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from models import ukb_longitudinal as legacy
from safe_io import require_input, prepare_output

FACTOR = .09148

def save(out, name, frame):
    pd.DataFrame(frame).to_csv(out / (name + '.csv'), index=False, encoding='utf-8-sig')

def linear_contrast(model, terms):
    vector = np.array([terms.get(x, 0.) for x in model.params.index])
    result = model.t_test(vector)
    ci = np.asarray(result.conf_int()).reshape(-1)
    return dict(estimate=float(np.asarray(result.effect).item())*FACTOR,
                ci_low=float(ci[0])*FACTOR, ci_high=float(ci[1])*FACTOR,
                se=float(np.asarray(result.sd).item())*FACTOR,
                p_value=float(np.asarray(result.pvalue).item()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ukb-dir', required=True)
    ap.add_argument('--repeat-file', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    input_dir = require_input(args.ukb_dir, directory=True)
    repeat_path = require_input(args.repeat_file)
    out = prepare_output(args.out_dir, [input_dir, repeat_path])
    legacy.DATA = input_dir
    frame = legacy.prepare_data()
    assert (frame.p130708 <= frame.p53_i0).all(), 'Inputs must satisfy recorded E11 eligibility'
    frame['ipw'], weight_fit = legacy.estimate_weights(frame)
    model, dat, coefficients = legacy.fit_model(frame, 'joint', True, 'primary_ipw_joint')
    model_n = len(dat)
    joint_term = next(t for t in model.params.index if 'T.improved__improved]' in t)
    primary = linear_contrast(model, {joint_term: 1})
    repeat = pd.read_csv(args.repeat_file)
    assert repeat.eid.is_unique and frame.eid.is_unique
    base_ids = frame[['eid']].copy()
    repeat = base_ids.merge(repeat[['eid','p1558_i1','p20116_i1']], on='eid', how='left', validate='one_to_one')
    assert list(repeat.eid) == list(frame.eid)
    frame['smoking_1'] = pd.to_numeric(repeat.p20116_i1, errors='coerce').where(lambda v:v.isin([0,1,2]))
    frame['alcohol_1'] = pd.to_numeric(repeat.p1558_i1, errors='coerce').where(lambda v:v.between(1,6))
    core = frame.loc[dat.index].copy()
    for col in dat:
        core[col] = dat[col]
    smoking = core.loc[core.smoking.eq(2) & core.smoking_1.isin([1,2])].copy()
    smoking['smoking_change'] = np.where(smoking.smoking_1.eq(1),'improved','unchanged')
    alcohol = core.loc[core.alcohol.between(1,5) & core.alcohol_1.notna()].copy()
    alcohol['alcohol_change'] = np.select([alcohol.alcohol_1.gt(alcohol.alcohol),alcohol.alcohol_1.lt(alcohol.alcohol)],['improved','worsened'],default='unchanged')
    # Every behavioural comparison uses the same observation weight, not a new response model.
    formula_base = ('hba1c0_c + I(hba1c0_c ** 2) + age_c + bmi_c + egfr_c + glucose_c + '
                    'interval_c + diet0_c + activity0_c + C(sex) + C(smoking) + C(alcohol)')
    rows=[]; all_coeff=[]; counts=[]; sensitive=[]
    domains=[('Diet improvement',core,'diet_change'),('Activity increase',core,'activity_change'),
             ('Smoking cessation',smoking,'smoking_change'),('Lower drinking frequency',alcohol,'alcohol_change')]
    for label, sub, exposure in domains:
        change_n=int(sub[exposure].eq('improved').sum()); reference_n=int(sub[exposure].eq('unchanged').sum())
        row=dict(behaviour=label,n=len(sub),changed_n=change_n,reference_n=reference_n,
                 other_n=int(sub[exposure].eq('worsened').sum()),source_population_n=model_n,
                 comparison='Improved versus unchanged for this behaviour',weight='Primary observation weight')
        for level,g in sub.groupby(exposure,observed=True):
            counts.append(dict(behaviour=label,category=level,n=len(g),baseline_mean=float(g.hba1c_i0.mean())*FACTOR+2.152,
                               followup_mean=float(g.hba1c_i1.mean())*FACTOR+2.152,
                               raw_change=float((g.hba1c_i1-g.hba1c_i0).mean())*FACTOR))
        if min(change_n,reference_n)<10:
            row.update(status='Not reliably estimable because of sparse transition groups',estimate=np.nan,ci_low=np.nan,ci_high=np.nan,se=np.nan,p_value=np.nan)
        else:
            formula=f"hba1c_i1 ~ C({exposure}, Treatment(reference='unchanged')) + "+formula_base
            if label=='Smoking cessation':formula=formula.replace(' + C(smoking)','')
            fitted=smf.wls(formula,sub,weights=sub.ipw).fit(cov_type='HC3')
            term=next(t for t in fitted.params.index if f"C({exposure}," in t and 'T.improved]' in t)
            row.update(status='Estimated',**linear_contrast(fitted,{term:1}))
            for t in fitted.params.index:
                if f"C({exposure}," in t:
                    all_coeff.append(dict(behaviour=label,term=t,n=len(sub),**linear_contrast(fitted,{t:1})))
            for name,fit in [('Without observation weights',smf.ols(formula,sub).fit(cov_type='HC3')),
                             ('Linear baseline HbA1c',smf.wls(formula.replace(' + I(hba1c0_c ** 2)',''),sub,weights=sub.ipw).fit(cov_type='HC3'))]:
                sensitive.append(dict(behaviour=label,analysis=name,n=len(sub),**linear_contrast(fit,{term:1})))
        rows.append(row)
    holm=multipletests([r['p_value'] if np.isfinite(r['p_value']) else 1 for r in rows],method='holm')[1]
    for row,p in zip(rows,holm):row['p_holm_4']=float(p) if np.isfinite(row['p_value']) else np.nan
    save(out,'ukb_four_longitudinal',rows);save(out,'ukb_four_longitudinal_categories',all_coeff)
    save(out,'ukb_four_longitudinal_groups',counts);save(out,'ukb_four_longitudinal_sensitivity',sensitive)
    joint_rows=[]
    for level in sorted(core.joint.unique()):
        terms=[t for t in model.params.index if f'T.{level}]' in t]
        result=linear_contrast(model,{terms[0]:1}) if terms else dict(estimate=0.,ci_low=np.nan,ci_high=np.nan,se=np.nan,p_value=np.nan)
        subset=core.loc[core.joint.eq(level)]
        joint_rows.append(dict(category=level,n=len(subset),model_n=model_n,baseline_mean=float(subset.hba1c_i0.mean())*FACTOR+2.152,
                               followup_mean=float(subset.hba1c_i1.mean())*FACTOR+2.152,
                               raw_change=float((subset.hba1c_i1-subset.hba1c_i0).mean())*FACTOR,**result))
    save(out,'ukb_joint',joint_rows)
    diet_term=next(t for t in model.params.index if 'T.improved__unchanged]' in t)
    act_term=next(t for t in model.params.index if 'T.unchanged__improved]' in t)
    comparisons=[]
    for label,terms in [('Joint versus diet only',{joint_term:1,diet_term:-1}),('Joint versus activity only',{joint_term:1,act_term:-1}),
                        ('Departure from additivity',{joint_term:1,diet_term:-1,act_term:-1})]:
        comparisons.append(dict(comparison=label,n=model_n,**linear_contrast(model,terms)))
    adj=multipletests([r['p_value'] for r in comparisons],method='holm')[1]
    for r,p in zip(comparisons,adj):r['p_holm_3']=p
    save(out,'ukb_joint_comparisons',comparisons)
    eligible=frame.hba1c_i0.notna() & frame.diet_level_i0.notna() & frame.activity_level_i0.notna()
    balance=[]
    baseline=frame.loc[eligible].copy()
    for column in ['age','hba1c_i0','bmi','egfr_i0','glucose_i0','diet_level_i0','activity_level_i0']:
        ref=baseline[column].fillna(baseline[column].median())
        obs=core[column].fillna(baseline[column].median())
        scale=float(ref.std(ddof=1))
        balance.append(dict(variable=column,source_n=len(baseline),observed_n=model_n,
                            source_mean=float(ref.mean()),unweighted_mean=float(obs.mean()),weighted_mean=float(np.average(obs,weights=core.ipw)),
                            smd_before=float((obs.mean()-ref.mean())/scale),smd_after=float((np.average(obs,weights=core.ipw)-ref.mean())/scale)))
    for column in ['sex','smoking','alcohol']:
        ref=baseline[column].fillna(baseline[column].mode().iloc[0])
        for level in sorted(ref.unique(),key=str):
            a=ref.eq(level).astype(float);b=core[column].eq(level).astype(float)
            sd=float(a.std(ddof=1))
            if sd>0:balance.append(dict(variable=f'{column} {level}',source_n=len(baseline),observed_n=model_n,
                                       source_mean=float(a.mean()),unweighted_mean=float(b.mean()),weighted_mean=float(np.average(b,weights=core.ipw)),
                                       smd_before=float((b.mean()-a.mean())/sd),smd_after=float((np.average(b,weights=core.ipw)-a.mean())/sd)))
    save(out,'ukb_observation_balance',balance)
    weight_fit['observation_available_in_full_export_n']=weight_fit.pop('observed_n')
    weight_fit['observed_n_in_weight_model']=int(round(weight_fit['observation_prevalence']*weight_fit['baseline_eligible_n']))
    diagnostics=dict(**weight_fit,final_model_n=model_n,final_weight_ess=float(core.ipw.sum()**2/(core.ipw**2).sum()),
                     final_weight_min=float(core.ipw.min()),final_weight_max=float(core.ipw.max()),
                     max_smd_before=max(abs(x['smd_before']) for x in balance),max_smd_after=max(abs(x['smd_after']) for x in balance))
    save(out,'ukb_observation_weights',[diagnostics])
    save(out, 'ukb_sensitivity', joint_sensitivities(legacy, frame, model, dat))
    manifest={'primary':primary,'primary_n':len(dat),'extension_results':rows,'weight_diagnostics':diagnostics,
              'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [repeat_path,*input_dir.glob('*.csv')]}}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf8')
    print(json.dumps({k:manifest[k] for k in ['primary','primary_n','extension_results','weight_diagnostics']},ensure_ascii=False,indent=2))

def joint_sensitivities(module, frame, fitted, model_frame):
    """Refit the manuscript's joint exposure sensitivity models."""
    cooks = smf.ols(fitted.model.formula, data=model_frame).fit().get_influence().cooks_distance[0]
    restricted = frame.drop(index=model_frame.index[cooks > 4 / len(model_frame)])
    specifications = [
        ('joint', False, 'sensitivity_unweighted_joint', frame),
        ('joint_dose', True, 'sensitivity_ipw_dose_joint', frame),
        ('joint', True, 'sensitivity_ipw_no_influential_joint', restricted),
        ('joint', True, 'sensitivity_ipw_hba1c_range_joint',
         frame.loc[frame.hba1c_i0.between(15, 200) & frame.hba1c_i1.between(15, 200)]),
    ]
    rows = []
    for exposure, weighted, label, subset in specifications:
        _, selected, coefficients = module.fit_model(subset, exposure, weighted, label)
        coefficients.attrs['n'] = len(selected)
        rows.append(module.contrast_row(coefficients, label, exposure, 'improved__improved'))
    return rows

if __name__ == '__main__':
    main()
