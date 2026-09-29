"""Reproduce the four baseline behavioural comparisons from licensed UK Biobank exports.

Reads participant data without modifying it; writes aggregate estimates only.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from safe_io import require_input, prepare_output

NUM=['age','year','recorded_duration','bmi','egfr','diet_index','activity_index']
CAT=['sex','ethnicity2','smoking','alcohol']
REF={'sex':'Female','ethnicity2':'White','smoking':2.,'alcohol':3.}
FACTOR=.09148

def clean(path):
    tables=[pd.read_csv(path/n) for n in ['base.csv','lifestyle.csv','bmi.csv','cystatin.csv','covariates.csv']]
    d=tables[0]
    for t in tables:
        assert t.eid.is_unique and t.eid.notna().all()
    for t in tables[1:]:d=d.merge(t,on='eid',how='left',validate='one_to_one')
    for k in ['p53_i0','p53_i1','p130706','p130708']:d[k]=pd.to_datetime(d[k],errors='coerce')
    assert (d.p130708<=d.p53_i0).all(), 'Inputs must satisfy recorded E11 eligibility'
    number=lambda x:pd.to_numeric(x,errors='coerce')
    valid=lambda x,vals:number(x).where(lambda v:v.isin(vals))
    quantity=lambda x:number(x).replace(-10,.5).where(lambda v:v>=0)
    freq=lambda x:number(x).map({0:0,1:.5,2:1,3:3,4:5.5,5:7})
    d['age']=number(d.p21022).fillna(d.p53_i0.dt.year-number(d.p34))
    d['sex']=d.p31.map({'Female':'Female','Male':'Male',0:'Female',1:'Male','0':'Female','1':'Male'})
    assert d.sex.notna().sum()==d.p31.notna().sum()
    e=number(d.p21000_i0)
    white=e.isin([1,1001,1002,1003])|d.p21000_i0.isin(['White','British','Irish','Any other white background'])
    nonwhite=e.isin([2,2001,2002,2003,2004,3,3001,3002,3003,3004,4,4001,4002,4003,5,6])|d.p21000_i0.isin(['Mixed','White and Black Caribbean','White and Black African','White and Asian','Any other mixed background','Asian or Asian British','Indian','Pakistani','Bangladeshi','Any other Asian background','Black or Black British','Caribbean','African','Any other Black background','Chinese','Other ethnic group'])
    d['ethnicity2']=pd.Series(np.nan,index=d.index,dtype=object)
    d.loc[white,'ethnicity2']='White';d.loc[nonwhite,'ethnicity2']='Non-White'
    d['year']=d.p53_i0.dt.year.astype(float)
    d['recorded_duration']=(d.p53_i0-d.p130708).dt.days/365.25
    d['hba0']=number(d.p30750_i0).where(lambda v:v>0)*FACTOR+2.152
    d['bmi']=number(d.p21001_i0).where(lambda v:v.between(10,80))
    cr=number(d.p30700_i0)/88.4;female=d.sex.eq('Female');ratio=cr/np.where(female,.7,.9)
    d['egfr']=(142*np.minimum(ratio,1)**np.where(female,-.241,-.302)*np.maximum(ratio,1)**-1.2*.9938**d.age*np.where(female,1.012,1)).where((cr>0)&d.sex.notna())
    d['smoking']=valid(d.p20116_i0,[0,1,2]);d['alcohol']=valid(d.p1558_i0,range(1,7))
    for label,field in [('cooked_veg',1289),('raw_veg',1299),('fresh_fruit',1309)]:d[label]=quantity(d[f'p{field}_i0'])
    for label,field in [('oily_fish',1329),('non_oily_fish',1339),('poultry',1359),('processed_meat',1349),('beef',1369),('lamb',1379),('pork',1389)]:d[label]=freq(d[f'p{field}_i0'])
    d['red_meat']=d[['beef','lamb','pork']].sum(axis=1,min_count=3)
    components=['cooked_veg','raw_veg','fresh_fruit','oily_fish','non_oily_fish','poultry','processed_meat','red_meat']
    z=(d[components]-d[components].mean())/d[components].std(ddof=1)
    z[['processed_meat','red_meat']]*=-1;d['diet_index']=z.sum(axis=1,min_count=8)
    d['activity_index']=valid(d.p864_i0,range(8))+2*valid(d.p884_i0,range(8))+3*valid(d.p904_i0,range(8))
    d['baseline_e10']=d.p130706.notna()&(d.p130706<=d.p53_i0)
    return d

def fit(d,iqrs,label='primary',no_clinical=False):
    x=pd.DataFrame({'Intercept':1.},index=d.index);scales={}
    for name in NUM:
        if no_clinical and name in ['bmi','egfr']:continue
        scales[name]=float(d[name].std(ddof=0))
        if scales[name]>1e-12:x[name]=(d[name]-d[name].mean())/scales[name]
    for name in CAT:
        levels=sorted(d[name].unique(),key=str);ref=REF[name] if REF[name] in levels else levels[0]
        for v in levels:
            if v!=ref:x[f'{name}={v}']=d[name].eq(v).astype(float)
    keep=[]
    for name in x:
        if np.linalg.matrix_rank(x[keep+[name]].to_numpy())>len(keep):keep.append(name)
    x=x[keep];model=sm.WLS(d.hba0,x,weights=np.ones(len(d))).fit(cov_type='HC3')
    b=model.params;cov=model.cov_params();z=stats.norm.ppf(.975)
    def contrast(name,k=1):
        beta=float(b[name])*k;se=np.sqrt(float(cov.loc[name,name]))*abs(k)
        return dict(estimate=beta,se=se,ci_low=beta-z*se,ci_high=beta+z*se,p_value=float(2*stats.norm.sf(abs(beta/se))),status='ok')
    rows=[];details=[]
    for behavior,zh in [('diet','饮食'),('activity','身体活动'),('smoking','吸烟'),('alcohol','饮酒频率')]:
        row=dict(analysis=label,behavior=behavior,label_zh=zh,n=len(d),n_parameters=len(keep),unit='absolute HbA1c difference on NGSP % scale')
        if behavior in ['diet','activity']:
            name=behavior+'_index';row.update(contrast(name,iqrs[behavior]/scales[name]),contrast='baseline Q75-Q25',contrast_size=iqrs[behavior]);details.append(row.copy())
        else:
            names=[n for n in x if n.startswith(behavior+'=')];bv=b[names].to_numpy();cv=cov.loc[names,names].to_numpy()
            row.update(p_value=float(stats.chi2.sf(bv @ np.linalg.pinv(cv) @ bv,np.linalg.matrix_rank(cv))),status='ok',contrast='omnibus categorical test',degrees_freedom=len(names))
            for name in names:
                code=float(name.split('=')[1]);details.append({**row,**contrast(name),'contrast':f'{code:g} vs {REF[behavior]:g}','code':code,'reference_code':REF[behavior],'n_exposed':int(d[behavior].eq(code).sum()),'n_reference':int(d[behavior].eq(REF[behavior]).sum())})
        rows.append(row)
    r=pd.DataFrame(rows)
    r['p_holm_4']=multipletests(r.p_value,method='holm')[1];r['q_bh_4']=multipletests(r.p_value,method='fdr_bh')[1];r['multiplicity_family_size']=4
    return r,pd.DataFrame(details)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ukb-dir',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
    src=require_input(args.ukb_dir, directory=True);out=prepare_output(args.out_dir, [src])
    d=clean(src);c=d.dropna(subset=['hba0']+NUM+CAT).copy()
    iqrs={b:float(c[b+'_index'].quantile(.75)-c[b+'_index'].quantile(.25)) for b in ['diet','activity']}
    r,detail=fit(c,iqrs)
    r.to_csv(out/'ukb_cross_sectional_primary.csv',index=False);detail.to_csv(out/'ukb_cross_sectional_category_contrasts.csv',index=False)
    sensitivities=[fit(c,iqrs,'without_BMI_eGFR',True)[0],fit(c.loc[~c.baseline_e10],iqrs,'no_baseline_E10')[0]]
    pd.concat(sensitivities,ignore_index=True).to_csv(out/'ukb_cross_sectional_sensitivities.csv',index=False)
    manifest={'source_n':len(d),'baseline_hba1c_n':int(d.hba0.notna().sum()),'baseline_model_n':len(c),'iqrs':iqrs,'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.csv')}}
    (out/'cross_sectional_run.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
