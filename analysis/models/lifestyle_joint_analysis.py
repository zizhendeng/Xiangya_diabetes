"""Xiangya lifestyle definitions, outcome models and dietary components."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from . import rebuild_full_cohort as base


DRUGS = [c for c in base.CORE_CLASSES]


DRUG_LABELS = base.DRUG_LABELS


DIET_ITEMS = {
    "whole_grain_more": ("全谷物/杂豆倾向增加", "better", "1st-20(主食类型更倾向全谷物 / 杂豆类（如糙米、燕麦、红豆等）)"),
    "refined_grain_more": ("精制谷物倾向增加", "worse", "1st-20(主食类型更倾向精制谷物（如白米饭、白面条等）)"),
    "staple_more": ("主食量明显增加", "worse", "1st-20(主食摄入量明显增加（较之前增加约 1/3 及以上）)"),
    "staple_less": ("主食量明显减少", "better", "1st-20(主食摄入量明显减少（较之前减少约 1/3 及以上）)"),
    "vegetable_more": ("蔬菜摄入明显增加", "better", "1st-20(蔬菜摄入量明显增加（较之前增加约 1/3 及以上）)"),
    "vegetable_less": ("蔬菜摄入明显减少", "worse", "1st-20(蔬菜摄入量明显减少（较之前减少约 1/3 及以上）)"),
    "lean_protein_more": ("鱼禽蛋豆来源增加", "better", "1st-20(蛋白质来源更倾向鱼类 / 禽类 / 蛋类 / 豆制品)"),
    "red_processed_more": ("红肉/加工肉倾向增加", "worse", "1st-20(蛋白质来源更倾向红肉（猪肉、牛肉等）/ 加工肉制品（香肠、火腿等）)"),
}


def _flag(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").eq(1).astype(int)


def add_baseline_lifestyle(d: pd.DataFrame) -> pd.DataFrame:
    """Harmonize compact baseline lifestyle covariates across WJX/TJS."""
    d = d.copy()
    wjx_ex = pd.to_numeric(d.get("WJX-您是否有定期运动的习惯?"), errors="coerce").map({1.0: 1.0, 2.0: 0.0})
    tjs_ex = d.get("TJS-您是否有定期运动的习惯", pd.Series(index=d.index, dtype=object)).map({"是": 1.0, "否": 0.0})
    d["baseline_exercise_regular"] = wjx_ex.combine_first(tjs_ex)

    # A compact channel-harmonized diet quality covariate.  It is used only for
    # confounding control, not as a validated dietary index or an outcome.
    wjx_obs = d.get("WJX-近一个月您每日蔬菜摄入量:（普通饭碗平铺量≈150克）").notna()
    wjx_score = (
        _flag(d.get("WJX-近一个月您每日主食的主要类型是?（可多选）(全谷物（如糙米、燕麦、全麦制品、苦荞/藜麦、小米/玉米）)"))
        + _flag(d.get("WJX-26(杂豆类（如红豆、绿豆、芸豆、鹰嘴豆、扁豆）)"))
        - _flag(d.get("WJX-26(精制谷物（如白米饭、白面条、年糕、汤圆）)"))
        + pd.to_numeric(d.get("WJX-近一个月您每日蔬菜摄入量:（普通饭碗平铺量≈150克）"), errors="coerce").map({1.0: -1.0, 2.0: 0.0, 3.0: 1.0}).fillna(0)
        + ((_flag(d.get("WJX-近一个月您的主要蛋白质来源是?（可多选）(鱼类（如三文鱼、鲈鱼、鲫鱼）)"))
            + _flag(d.get("WJX-29(禽类（如鸡、鸭、鹅）)"))
            + _flag(d.get("WJX-29(蛋类（如鸡蛋、鸭蛋、鹌鹑蛋、鹅蛋）)"))
            + _flag(d.get("WJX-29(豆类及豆制品（如黄豆、黑豆、豆腐、豆浆、鹰嘴豆、扁豆）)"))) > 0).astype(int)
        - _flag(d.get("WJX-29(加工肉制品（如香肠、火腿）)"))
    ).where(wjx_obs)

    staple = d.get("TJS-您每日主食的主要类型是？（可多选）", pd.Series(index=d.index, dtype=object)).fillna("").astype(str)
    veg = d.get("TJS-您每日蔬菜摄入量：（普通饭碗平铺量≈150克）", pd.Series(index=d.index, dtype=object)).fillna("").astype(str)
    protein = d.get("TJS-您的主要蛋白质来源是？（可多选）", pd.Series(index=d.index, dtype=object)).fillna("").astype(str)
    tjs_obs = d.get("TJS-您每日蔬菜摄入量：（普通饭碗平铺量≈150克）", pd.Series(index=d.index)).notna()
    tjs_score = (
        staple.str.contains("全谷物").astype(int)
        + staple.str.contains("杂豆").astype(int)
        - staple.str.contains("精制谷物").astype(int)
        + veg.map(lambda s: 1 if ">500" in s else (-1 if "<300" in s else 0))
        + protein.str.contains("鱼类|禽类|蛋类|豆类", regex=True).astype(int)
        - protein.str.contains("加工肉").astype(int)
    ).where(tjs_obs)
    d["baseline_diet_score"] = wjx_score.combine_first(tjs_score)
    return d


def prepare() -> pd.DataFrame:
    d = base.load_and_clean()
    d = add_baseline_lifestyle(d)
    d["ipcw"] = base.estimate_followup_weights(d)
    for short, (_, _, raw_col) in DIET_ITEMS.items():
        d[f"dietitem_{short}"] = _flag(d[raw_col])
    d["joint_lifestyle"] = d["diet_change3"] + " / " + d["exercise_change3"]
    d.loc[(d.diet_change3 == "missing") | (d.exercise_change3 == "missing"), "joint_lifestyle"] = "missing"
    d["hba1c_ge9"] = d.hba1c_baseline.ge(9).astype(float).where(d.hba1c_baseline.notna())
    d["bmi_ge28"] = d.bmi.ge(28).astype(float).where(d.bmi.notna())
    d["age_ge65"] = d.age.ge(65).astype(float).where(d.age.notna())
    d["central_obesity"] = np.where(
        d.sex_male.eq(1), d.whr.ge(.90), np.where(d.sex_male.eq(0), d.whr.ge(.85), np.nan)
    )
    d["hba1c_group"] = pd.cut(d.hba1c_baseline, [-np.inf, 7, 9, np.inf], right=False, labels=["<7%", "7-<9%", "≥9%"])
    d["bmi_group"] = pd.cut(d.bmi, [-np.inf, 24, 28, np.inf], right=False, labels=["<24", "24-<28", "≥28"])
    d["age_group"] = np.where(d.age.ge(65), "≥65", "<65")
    d.loc[d.age.isna(), "age_group"] = "missing"
    d["sex_group"] = d.sex_male.map({1.0: "male", 0.0: "female"}).fillna("missing")
    d["central_obesity_group"] = pd.Series(d.central_obesity, index=d.index).map({1.0: "yes", 0.0: "no"}).fillna("missing")
    d["med_changed"] = d.med_change_status.eq("changed").astype(float).where(d.med_change_status.ne("missing"))

    enroll_year = pd.to_numeric(d["Date"].astype(str).str.extract(r"(20\d{2})", expand=False), errors="coerce").fillna(2025)
    wjx_dx = pd.to_numeric(d.get("WJX-您被诊断为糖尿病的时间是？"), errors="coerce")
    tjs_dx = pd.to_datetime(d.get("TJS-您被诊断为糖尿病的年份"), errors="coerce").dt.year
    dx_year = wjx_dx.combine_first(tjs_dx)
    d["diabetes_duration"] = (enroll_year - dx_year).where(lambda s: s.between(0, 60))

    wjx_htn = pd.to_numeric(d.get("WJX-您是否曾被诊断为高血压?"), errors="coerce").map({1.0: 1.0, 2.0: 0.0})
    tjs_htn = d.get("TJS-您是否曾被诊断为高血压？", pd.Series(index=d.index, dtype=object)).map({"是": 1.0, "否": 0.0})
    wjx_lip = pd.to_numeric(d.get("WJX-您是否曾被诊断为高脂血症或血脂紊乱?"), errors="coerce").map({1.0: 1.0, 2.0: 0.0})
    tjs_lip = d.get("TJS-您是否曾被诊断为高脂血症或血脂紊乱？", pd.Series(index=d.index, dtype=object)).map({"是": 1.0, "否": 0.0})
    d["hypertension"] = wjx_htn.combine_first(tjs_htn)
    d["dyslipidemia"] = wjx_lip.combine_first(tjs_lip)
    d["comorbidity_count"] = d[["hypertension", "dyslipidemia"]].sum(axis=1, min_count=1)

    wjx_smoke = pd.to_numeric(d.get("WJX-您的吸烟情况?"), errors="coerce").map({1.0: 0.0, 2.0: 1.0, 3.0: 0.0})
    tjs_smoke = d.get("TJS-您的吸烟情况", pd.Series(index=d.index, dtype=object)).map({"从不吸烟": 0.0, "吸烟": 1.0, "已戒烟": 0.0})
    d["current_smoker"] = wjx_smoke.combine_first(tjs_smoke)
    wjx_alcohol = pd.to_numeric(d.get("WJX-您的饮酒情况?"), errors="coerce").map({1.0: 0.0, 2.0: 1.0, 3.0: 0.0})
    tjs_alcohol = d.get("TJS-您的饮酒情况", pd.Series(index=d.index, dtype=object)).map({"不饮酒": 0.0, "饮酒": 1.0, "已戒酒": 0.0})
    d["current_alcohol"] = wjx_alcohol.combine_first(tjs_alcohol)
    return d


NUMERIC_COVS = [
    "hba1c_baseline", "age", "sex_male", "bmi", "whr", "egfr", "fbg", "sbp",
    "baseline_exercise_regular", "baseline_diet_score", "diabetes_duration", "comorbidity_count",
    "current_smoker", "current_alcohol",
]


def model_frame(d: pd.DataFrame, require_lifestyle=True, classified=False) -> pd.DataFrame:
    mask = d.hba1c_followup.notna() & d.hba1c_baseline.notna()
    if require_lifestyle:
        mask &= d.diet_change3.ne("missing") & d.exercise_change3.ne("missing")
    if classified:
        mask &= d.baseline_med_status.eq("classified")
    x = d.loc[mask].copy()
    for c in NUMERIC_COVS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[c] = x[c].fillna(x[c].median())
    for c in [f"bl_{z}" for z in DRUGS]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    x["ipcw"] = pd.to_numeric(x.ipcw, errors="coerce").fillna(1).clip(.2, 5)
    return x


def covariate_formula(other_drugs: list[str] | None = None) -> str:
    drugs = other_drugs if other_drugs is not None else [f"bl_{z}" for z in DRUGS]
    return (
        "hba1c_baseline + I(hba1c_baseline ** 2) + age + sex_male + bmi + whr + egfr + fbg + sbp "
        "+ baseline_exercise_regular + baseline_diet_score + diabetes_duration + comorbidity_count "
        "+ current_smoker + current_alcohol + C(survey_source) + C(baseline_med_status)"
        + (" + " + " + ".join(drugs) if drugs else "")
    )


def _fit(formula: str, x: pd.DataFrame):
    return smf.wls(formula, data=x, weights=x.ipcw).fit(cov_type="HC3")


def _term_row(fit, term: str, **extra) -> dict | None:
    if term not in fit.params:
        return None
    est = float(fit.params[term]); se = float(fit.bse[term]); p = float(fit.pvalues[term])
    return {**extra, "estimate": est, "se": se, "ci_low": est-1.96*se, "ci_high": est+1.96*se, "p_value": p}


def lifestyle_main(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = model_frame(d, require_lifestyle=True)
    desc = x.groupby(["diet_change3", "exercise_change3"], observed=True).agg(
        n=("analysis_id", "size"), baseline_hba1c=("hba1c_baseline", "mean"),
        followup_hba1c=("hba1c_followup", "mean"), mean_delta_hba1c=("delta_hba1c", "mean"),
        sd_delta_hba1c=("delta_hba1c", "std"),
    ).reset_index()
    exposure = (
        'C(diet_change3, Treatment(reference="unchanged")) + '
        'C(exercise_change3, Treatment(reference="unchanged"))'
    )
    fits = {
        "baseline_adjusted": _fit("hba1c_followup ~ " + exposure + " + " + covariate_formula(), x),
        "plus_medication_change": _fit("hba1c_followup ~ " + exposure + " + med_changed + " + covariate_formula(), x),
    }
    rows = []
    for model_name, fit in fits.items():
        for exposure_name in ["diet_change3", "exercise_change3"]:
            for level in ["improved", "worsened"]:
                term = f'C({exposure_name}, Treatment(reference="unchanged"))[T.{level}]'
                row = _term_row(fit, term, model=model_name, exposure=exposure_name, level=level, reference="unchanged", n=len(x))
                if row: rows.append(row)
    effects = pd.DataFrame(rows)
    effects["q_value_fdr"] = np.nan
    for model_name, ix in effects.groupby("model").groups.items():
        effects.loc[ix, "q_value_fdr"] = multipletests(effects.loc[ix, "p_value"], method="fdr_bh")[1]

    reference = "unchanged / unchanged"
    joint_fit = _fit(f'hba1c_followup ~ C(joint_lifestyle, Treatment(reference="{reference}")) + ' + covariate_formula(), x)
    joint_rows = [{"joint_lifestyle": reference, "n": int((x.joint_lifestyle == reference).sum()), "estimate_vs_reference": 0.0,
                   "ci_low": 0.0, "ci_high": 0.0, "p_value": np.nan}]
    for level, n in x.joint_lifestyle.value_counts().items():
        if level == reference or n < 20:
            continue
        term = f'C(joint_lifestyle, Treatment(reference="{reference}"))[T.{level}]'
        r = _term_row(joint_fit, term, joint_lifestyle=level, n=int(n))
        if r:
            r["estimate_vs_reference"] = r.pop("estimate")
            joint_rows.append(r)
    joint = pd.DataFrame(joint_rows)
    ix = joint.p_value.notna()
    if ix.any(): joint.loc[ix, "q_value_fdr"] = multipletests(joint.loc[ix, "p_value"], method="fdr_bh")[1]
    return desc, effects, joint


def diet_item_model(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = model_frame(d, require_lifestyle=True)
    item_cols = [f"dietitem_{k}" for k in DIET_ITEMS]
    formula = (
        "hba1c_followup ~ " + " + ".join(item_cols)
        + ' + C(exercise_change3, Treatment(reference="unchanged")) + ' + covariate_formula()
    )
    fit = _fit(formula, x)
    rows, counts = [], []
    for short, (label, direction, _) in DIET_ITEMS.items():
        col = f"dietitem_{short}"
        counts.append({"item": short, "label_zh": label, "scoring_direction": direction, "selected_n": int(x[col].sum()), "analysis_n": len(x)})
        r = _term_row(fit, col, item=short, label_zh=label, scoring_direction=direction, selected_n=int(x[col].sum()))
        if r: rows.append(r)
    out = pd.DataFrame(rows)
    if len(out): out["q_value_fdr"] = multipletests(out.p_value, method="fdr_bh")[1]
    return pd.DataFrame(counts), out
