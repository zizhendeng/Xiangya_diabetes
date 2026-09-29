"""Clinical strata, global interaction tests and categorical adjustment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import lifestyle_joint_analysis as base


BMI_LEVELS = ["underweight", "normal", "overweight", "obesity"]


EGFR_LEVELS = ["G1_ge90", "G2_60_89", "G3_30_59", "G4_15_29", "G5_lt15"]


GROUP_SPECS = {
    "bmi_group4": {
        "reference": "normal",
        "label_zh": "BMI（中国成人标准）",
        "display": {
            "underweight": "<18.5 偏瘦",
            "normal": "18.5–<24 正常",
            "overweight": "24–<28 超重",
            "obesity": "≥28 肥胖",
        },
    },
    "egfr_group5": {
        "reference": "G1_ge90",
        "label_zh": "eGFR（KDOQI分期）",
        "display": {
            "G1_ge90": "G1 ≥90",
            "G2_60_89": "G2 60–89",
            "G3_30_59": "G3 30–59",
            "G4_15_29": "G4 15–29",
            "G5_lt15": "G5 <15",
        },
    },
    "central_obesity_group": {
        "reference": "no",
        "label_zh": "中心性肥胖（男WHR≥0.90、女≥0.85）",
        "display": {"no": "否", "yes": "是"},
    },
    "hba1c_group": {
        "reference": "<7%",
        "label_zh": "基线HbA1c",
        "display": {"<7%": "<7%", "7-<9%": "7–<9%", "≥9%": "≥9%"},
    },
    "age_group": {
        "reference": "<65",
        "label_zh": "年龄",
        "display": {"<65": "<65岁", "≥65": "≥65岁"},
    },
    "sex_group": {
        "reference": "female",
        "label_zh": "性别",
        "display": {"female": "女性", "male": "男性"},
    },
}


def add_clinical_groups(x: pd.DataFrame) -> pd.DataFrame:
    """Create clinical categories after numeric imputation in model_frame."""
    x = x.copy()
    x["bmi_group4"] = np.select(
        [x.bmi.lt(18.5), x.bmi.lt(24), x.bmi.lt(28)],
        BMI_LEVELS[:3],
        default="obesity",
    )
    x["egfr_group5"] = np.select(
        [x.egfr.ge(90), x.egfr.ge(60), x.egfr.ge(30), x.egfr.ge(15)],
        EGFR_LEVELS[:4],
        default="G5_lt15",
    )
    # Sparse categories are retained exactly for the requested clinical
    # description, but collapsed only when used as nuisance adjusters inside
    # smaller subgroup models.  This avoids HC3 leverage of one.
    x["bmi_adjust3"] = np.where(
        x["bmi_group4"].isin(["underweight", "normal"]),
        "underweight_or_normal",
        x["bmi_group4"],
    )
    x["egfr_adjust3"] = np.where(
        x["egfr_group5"].eq("G1_ge90"), "G1_ge90",
        np.where(x["egfr_group5"].eq("G2_60_89"), "G2_60_89", "G3_5_lt60"),
    )
    x["central_obesity_group"] = np.where(
        x.sex_male.eq(1), np.where(x.whr.ge(0.90), "yes", "no"),
        np.where(x.whr.ge(0.85), "yes", "no"),
    )
    x["hba1c_group"] = np.select(
        [x.hba1c_baseline.lt(7), x.hba1c_baseline.lt(9)],
        ["<7%", "7-<9%"],
        default="≥9%",
    )
    x["age_group"] = np.where(x.age.ge(65), "≥65", "<65")
    x["sex_group"] = np.where(x.sex_male.eq(1), "male", "female")
    return x


def categorical_term(variable: str, reference: str) -> str:
    return f'C({variable}, Treatment(reference="{reference}"))'


def discrete_covariate_formula(
    other_drugs: list[str] | None = None,
    omit_group: str | None = None,
    stable_nuisance: bool = False,
) -> str:
    drugs = other_drugs if other_drugs is not None else [f"bl_{z}" for z in base.DRUGS]
    if stable_nuisance:
        # GLP-1RA and TZD are sparse after stratification and can create
        # unit-leverage cells.  They remain described but are not used as
        # nuisance dummies in subgroup interaction models.
        drugs = [term for term in drugs if term not in {"bl_glp1ra", "bl_tzd"}]
    bmi_term = (
        'C(bmi_adjust3, Treatment(reference="underweight_or_normal"))'
        if stable_nuisance else 'C(bmi_group4, Treatment(reference="normal"))'
    )
    egfr_term = (
        'C(egfr_adjust3, Treatment(reference="G1_ge90"))'
        if stable_nuisance else 'C(egfr_group5, Treatment(reference="G1_ge90"))'
    )
    terms = [
        "hba1c_baseline", "I(hba1c_baseline ** 2)", "age", "sex_male",
        bmi_term,
        'C(central_obesity_group, Treatment(reference="no"))',
        egfr_term,
        "fbg", "sbp", "baseline_exercise_regular", "baseline_diet_score",
        "diabetes_duration", "comorbidity_count", "current_smoker", "current_alcohol",
        "C(survey_source)", "C(baseline_med_status)",
    ]
    omit_map = {
        "bmi_group4": bmi_term,
        "central_obesity_group": 'C(central_obesity_group, Treatment(reference="no"))',
        "egfr_group5": egfr_term,
        "age_group": "age",
        "sex_group": "sex_male",
    }
    if omit_group in omit_map and omit_map[omit_group] in terms:
        terms.remove(omit_map[omit_group])
    return " + ".join(terms + drugs)


def term_result(fit, term: str) -> dict | None:
    if term not in fit.params:
        return None
    estimate = float(fit.params[term])
    se = float(fit.bse[term])
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "p_value": float(fit.pvalues[term]),
    }


def bh_adjust(df: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    df = df.copy()
    df[q_col] = np.nan
    ok = df[p_col].notna()
    if ok.any():
        df.loc[ok, q_col] = base.multipletests(df.loc[ok, p_col], method="fdr_bh")[1]
    return df


def grouped_bh(df: pd.DataFrame, group_col: str, p_col: str, q_col: str) -> pd.DataFrame:
    df = df.copy()
    df[q_col] = np.nan
    for _, index in df.groupby(group_col).groups.items():
        ok = df.loc[index, p_col].notna()
        use = pd.Index(index)[ok.to_numpy()]
        if len(use):
            df.loc[use, q_col] = base.multipletests(df.loc[use, p_col], method="fdr_bh")[1]
    return df


def global_interaction_p(fit, exposure: str, modifier: str) -> tuple[float, int]:
    names = list(fit.params.index)
    interaction_terms = [
        name for name in names
        if ":" in name and exposure in name and modifier in name
    ]
    if not interaction_terms:
        return np.nan, 0
    restriction = np.zeros((len(interaction_terms), len(names)))
    for row, name in enumerate(interaction_terms):
        restriction[row, names.index(name)] = 1.0
    test = fit.wald_test(restriction, scalar=True)
    return float(np.asarray(test.pvalue).ravel()[0]), len(interaction_terms)


def usable_lifestyle_levels(x: pd.DataFrame, modifier: str, exposure: str) -> list[str]:
    levels = []
    for level, z in x.groupby(modifier):
        counts = z[exposure].value_counts()
        if (
            len(z) >= 50
            and int(counts.get("unchanged", 0)) >= 30
            and int(counts.get("improved", 0)) >= 10
            and int(counts.get("worsened", 0)) >= 10
        ):
            levels.append(str(level))
    return levels


def lifestyle_adjustment_sensitivity(life: pd.DataFrame) -> pd.DataFrame:
    exposure_formula = (
        'C(diet_change3, Treatment(reference="unchanged")) + '
        'C(exercise_change3, Treatment(reference="unchanged"))'
    )
    formulas = {
        "continuous_primary": "hba1c_followup ~ " + exposure_formula + " + " + base.covariate_formula(),
        "clinical_discrete": "hba1c_followup ~ " + exposure_formula + " + " + discrete_covariate_formula(),
        "clinical_discrete_plus_med_change": "hba1c_followup ~ " + exposure_formula + " + med_changed + " + discrete_covariate_formula(),
    }
    rows = []
    for model, formula in formulas.items():
        fit = base._fit(formula, life)
        for exposure in ["diet_change3", "exercise_change3"]:
            for level in ["improved", "worsened"]:
                term = f'C({exposure}, Treatment(reference="unchanged"))[T.{level}]'
                result = term_result(fit, term)
                if result:
                    rows.append({"model": model, "contrast": f"{exposure}:{level}_vs_unchanged", "n": len(life), **result})

    reference = "unchanged / unchanged"
    target = "improved / improved"
    joint_formulas = {
        "continuous_primary": base.covariate_formula(),
        "clinical_discrete": discrete_covariate_formula(),
        "clinical_discrete_plus_med_change": "med_changed + " + discrete_covariate_formula(),
    }
    for model, adjustment in joint_formulas.items():
        fit = base._fit(
            f'hba1c_followup ~ C(joint_lifestyle, Treatment(reference="{reference}")) + {adjustment}',
            life,
        )
        term = f'C(joint_lifestyle, Treatment(reference="{reference}"))[T.{target}]'
        result = term_result(fit, term)
        if result:
            rows.append({"model": model, "contrast": "diet_and_exercise_improved_vs_both_unchanged", "n": len(life), **result})
    return pd.DataFrame(rows)


def lifestyle_strata(life: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows = []
    global_rows = []
    for modifier, spec in GROUP_SPECS.items():
        reference = spec["reference"]
        for exposure, other_exposure in [
            ("diet_change3", "exercise_change3"),
            ("exercise_change3", "diet_change3"),
        ]:
            levels = usable_lifestyle_levels(life, modifier, exposure)
            if reference not in levels or len(levels) < 2:
                continue
            zall = life[life[modifier].isin(levels)].copy()
            modterm = categorical_term(modifier, reference)
            expterm = categorical_term(exposure, "unchanged")
            otherterm = categorical_term(other_exposure, "unchanged")
            formula = (
                f"hba1c_followup ~ {expterm} * {modterm} + {otherterm} + "
                + discrete_covariate_formula(omit_group=modifier, stable_nuisance=True)
            )
            try:
                fit = base._fit(formula, zall)
                p_global, df_global = global_interaction_p(fit, exposure, modifier)
            except Exception:
                p_global, df_global = np.nan, 0
            global_rows.append({
                "exposure": exposure,
                "modifier": modifier,
                "modifier_label_zh": spec["label_zh"],
                "n": len(zall),
                "levels_tested": ";".join(levels),
                "interaction_df": df_global,
                "global_interaction_p": p_global,
            })

            for level in levels:
                z = zall[zall[modifier] == level].copy()
                for change in ["improved", "worsened"]:
                    n_change = int((z[exposure] == change).sum())
                    n_reference = int((z[exposure] == "unchanged").sum())
                    if len(z) < 50 or n_change < 15 or n_reference < 30:
                        continue
                    try:
                        fit_s = base._fit(
                            f"hba1c_followup ~ {expterm} + {otherterm} + "
                            + discrete_covariate_formula(omit_group=modifier, stable_nuisance=True),
                            z,
                        )
                    except Exception:
                        continue
                    term = f'C({exposure}, Treatment(reference="unchanged"))[T.{change}]'
                    result = term_result(fit_s, term)
                    if result:
                        effect_rows.append({
                            "exposure": exposure,
                            "change": change,
                            "modifier": modifier,
                            "modifier_label_zh": spec["label_zh"],
                            "level": level,
                            "level_label_zh": spec["display"].get(level, level),
                            "subgroup_n": len(z),
                            "change_n": n_change,
                            "reference_n": n_reference,
                            **result,
                        })
    globals_df = bh_adjust(pd.DataFrame(global_rows), "global_interaction_p", "interaction_q_fdr")
    globals_df = grouped_bh(globals_df, "exposure", "global_interaction_p", "interaction_q_within_exposure")
    globals_df = grouped_bh(globals_df, "modifier", "global_interaction_p", "interaction_q_within_modifier")
    effects_df = pd.DataFrame(effect_rows)
    if len(effects_df):
        effects_df = effects_df.merge(
            globals_df[["exposure", "modifier", "global_interaction_p", "interaction_q_fdr", "interaction_q_within_exposure", "interaction_q_within_modifier"]],
            on=["exposure", "modifier"], how="left",
        )
    return effects_df, globals_df
