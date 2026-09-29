"""UK Biobank repeated-measure definitions, observation weights and ANCOVA."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set explicitly by the runner; importing this module does not access files.
DATA: Path | None = None


FREQ_WEEKLY = {0: 0.0, 1: 0.5, 2: 1.0, 3: 3.0, 4: 5.5, 5: 7.0}


POSITIVE_ITEMS = ["p1289", "p1299", "p1309", "p1329", "p1339", "p1359"]


RED_MEAT = ["p1369", "p1379", "p1389"]


def read_csv(name: str, dates: list[str] | None = None) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=dates or [], low_memory=False)


def assert_unique(df: pd.DataFrame, label: str) -> None:
    if "eid" not in df or df["eid"].isna().any() or df["eid"].duplicated().any():
        raise ValueError(f"{label}: eid missing or duplicated")


def quantity(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    x = x.mask(x.isin([-1, -3]))
    x = x.replace(-10, 0.5)
    return x.mask(x < 0)


def frequency(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.map(FREQ_WEEKLY)


def activity_days(series: pd.Series, unable_as_zero: bool = False) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if unable_as_zero:
        x = x.replace(-2, 0)
    return x.where(x.between(0, 7))


def activity_duration(series: pd.Series, days: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").where(lambda z: z.between(0, 1440))
    x = x.where(days != 0, 0.0)
    return x.where(days.notna())


def coded_category(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").mask(lambda z: z < 0).round().astype("Int64")
    return x.astype(float)


def classify_banded(delta: pd.Series, band: float) -> pd.Series:
    return pd.Series(np.select([delta > band, delta >= -band, delta < -band],
                               ["improved", "unchanged", "worsened"],
                               default=None), index=delta.index, dtype="object").where(delta.notna())


def egfr_2021(creatinine_umol: pd.Series, age: pd.Series, sex: pd.Series) -> pd.Series:
    scr = pd.to_numeric(creatinine_umol, errors="coerce") / 88.4
    age = pd.to_numeric(age, errors="coerce")
    female = sex.astype("string").str.lower().eq("female")
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    value = 142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.2
    value *= 0.9938 ** age
    value *= np.where(female, 1.012, 1.0)
    return pd.Series(value, index=creatinine_umol.index).where((scr > 0) & age.notna())


def prepare_data() -> pd.DataFrame:
    base = read_csv("base.csv", ["p53_i0", "p53_i1", "p130706", "p130708"])
    lifestyle = read_csv("lifestyle.csv")
    bmi = read_csv("bmi.csv")
    cystatin = read_csv("cystatin.csv")
    covariates = read_csv("covariates.csv")
    for label, frame in [("base", base), ("lifestyle", lifestyle), ("bmi", bmi),
                         ("cystatin", cystatin), ("covariates", covariates)]:
        assert_unique(frame, label)
    df = base.merge(lifestyle, on="eid", how="left", validate="one_to_one")
    df = df.merge(bmi, on="eid", how="left", validate="one_to_one")
    df = df.merge(cystatin, on="eid", how="left", validate="one_to_one")
    df = df.merge(covariates, on="eid", how="left", validate="one_to_one")

    df["age"] = pd.to_numeric(df["p21022"], errors="coerce")
    fallback_age = df["p53_i0"].dt.year - pd.to_numeric(df["p34"], errors="coerce")
    df["age"] = df["age"].fillna(fallback_age)
    df["sex"] = df["p31"].astype("string")
    df["bmi"] = pd.to_numeric(df["p21001_i0"], errors="coerce").where(lambda x: x.between(10, 80))
    df["hba1c_i0"] = pd.to_numeric(df["p30750_i0"], errors="coerce")
    df["hba1c_i1"] = pd.to_numeric(df["p30750_i1"], errors="coerce")
    df["glucose_i0"] = pd.to_numeric(df["p30740_i0"], errors="coerce").where(lambda x: x > 0)
    df["egfr_i0"] = egfr_2021(df["p30700_i0"], df["age"], df["sex"])
    df["interval_years"] = (df["p53_i1"] - df["p53_i0"]).dt.days / 365.25
    df["smoking"] = coded_category(df["p20116_i0"])
    df["alcohol"] = coded_category(df["p1558_i0"])
    df["sbp_i0"] = pd.concat([
        pd.to_numeric(df["p4080_i0_a0"], errors="coerce"),
        pd.to_numeric(df["p4080_i0_a1"], errors="coerce")], axis=1).mean(axis=1)

    item_levels: dict[str, tuple[pd.Series, pd.Series, int]] = {}
    for field in ["p1289", "p1299", "p1309"]:
        item_levels[field] = (quantity(df[f"{field}_i0"]), quantity(df[f"{field}_i1"]), 1)
    for field in ["p1329", "p1339", "p1359"]:
        item_levels[field] = (frequency(df[f"{field}_i0"]), frequency(df[f"{field}_i1"]), 1)
    item_levels["p1349"] = (frequency(df["p1349_i0"]), frequency(df["p1349_i1"]), -1)
    red0 = sum(frequency(df[f"{field}_i0"]) for field in RED_MEAT)
    red1 = sum(frequency(df[f"{field}_i1"]) for field in RED_MEAT)
    item_levels["red_meat"] = (red0, red1, -1)

    change_parts = []
    baseline_z = []
    for field, (v0, v1, direction) in item_levels.items():
        valid = v0.notna() & v1.notna()
        part = pd.Series(np.nan, index=df.index)
        part.loc[valid] = np.sign(direction * (v1.loc[valid] - v0.loc[valid]))
        df[f"diet_change_{field}"] = part
        change_parts.append(part)
        sd = v0.std(skipna=True)
        baseline_z.append(direction * (v0 - v0.mean(skipna=True)) / (sd if sd and sd > 0 else 1))
    change_matrix = pd.concat(change_parts, axis=1)
    df["diet_change_score"] = change_matrix.sum(axis=1, min_count=8)
    df["diet_change"] = classify_banded(df["diet_change_score"], 1)
    df["diet_level_i0"] = pd.concat(baseline_z, axis=1).sum(axis=1, min_count=8)

    for instance in (0, 1):
        walk = activity_days(df[f"p864_i{instance}"])
        moderate = activity_days(df[f"p884_i{instance}"])
        vigorous = activity_days(df[f"p904_i{instance}"])
        df[f"activity_level_i{instance}"] = walk + 2 * moderate + 3 * vigorous
        walk0 = activity_days(df[f"p864_i{instance}"], unable_as_zero=True)
        df[f"activity_level_unable0_i{instance}"] = walk0 + 2 * moderate + 3 * vigorous
        wdur = activity_duration(df[f"p874_i{instance}"], walk)
        mdur = activity_duration(df[f"p894_i{instance}"], moderate)
        vdur = activity_duration(df[f"p914_i{instance}"], vigorous)
        df[f"activity_met_i{instance}"] = 3.3 * walk * wdur + 4 * moderate * mdur + 8 * vigorous * vdur
    df["activity_change"] = classify_banded(df["activity_level_i1"] - df["activity_level_i0"], 2)
    df["activity_change_unable0"] = classify_banded(
        df["activity_level_unable0_i1"] - df["activity_level_unable0_i0"], 2)
    df["activity_change_dose"] = classify_banded(df["activity_met_i1"] - df["activity_met_i0"], 600)
    df["joint"] = np.where(df["diet_change"].notna() & df["activity_change"].notna(),
                           df["diet_change"] + "__" + df["activity_change"], None)
    df["joint_dose"] = np.where(df["diet_change"].notna() & df["activity_change_dose"].notna(),
                                df["diet_change"] + "__" + df["activity_change_dose"], None)
    return df


def estimate_weights(df: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    observed = df["hba1c_i1"].notna() & df["joint"].notna()
    numeric = ["age", "hba1c_i0", "bmi", "egfr_i0", "glucose_i0", "diet_level_i0", "activity_level_i0"]
    categorical = ["sex", "smoking", "alcohol"]
    eligible = df["hba1c_i0"].notna() & df["diet_level_i0"].notna() & df["activity_level_i0"].notna()
    train = df.loc[eligible, numeric + categorical].copy()
    y = observed.loc[eligible].astype(int)
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("pre", pre), ("logit", LogisticRegression(max_iter=2000, random_state=20260801))])
    model.fit(train, y)
    ps = np.clip(model.predict_proba(train)[:, 1], 0.005, 0.995)
    prevalence = float(y.mean())
    weights = pd.Series(np.nan, index=df.index, dtype=float)
    observed_idx = train.index[y.to_numpy() == 1]
    raw = prevalence / ps[y.to_numpy() == 1]
    lo, hi = np.quantile(raw, [0.01, 0.99])
    truncated = np.clip(raw, lo, hi)
    truncated /= truncated.mean()
    weights.loc[observed_idx] = truncated
    ess = float(truncated.sum() ** 2 / np.square(truncated).sum())
    diagnostics = {"baseline_eligible_n": int(eligible.sum()), "observed_n": int(observed.sum()),
                   "observation_prevalence": prevalence, "weight_p01": float(lo), "weight_p99": float(hi),
                   "weight_min": float(truncated.min()), "weight_max": float(truncated.max()), "weight_ess": ess}
    return weights, diagnostics


BASE_COVARS = ["hba1c_i0", "age", "bmi", "egfr_i0", "glucose_i0", "interval_years",
               "diet_level_i0", "activity_level_i0", "sex", "smoking", "alcohol"]


def fit_model(df: pd.DataFrame, exposure: str, weighted: bool, label: str):
    cols = ["hba1c_i1", exposure, *BASE_COVARS] + (["ipw"] if weighted else [])
    dat = df[cols].dropna().copy()
    if len(dat) < 50:
        raise RuntimeError(f"{label}: fewer than 50 complete cases")
    for col, new in [("hba1c_i0", "hba1c0_c"), ("age", "age_c"), ("bmi", "bmi_c"),
                     ("egfr_i0", "egfr_c"), ("glucose_i0", "glucose_c"),
                     ("interval_years", "interval_c"), ("diet_level_i0", "diet0_c"),
                     ("activity_level_i0", "activity0_c")]:
        scale = dat[col].std()
        dat[new] = (dat[col] - dat[col].mean()) / (scale if scale and scale > 0 else 1)
    reference = "unchanged__unchanged" if exposure.startswith("joint") else "unchanged"
    formula = (f"hba1c_i1 ~ C({exposure}, Treatment(reference='{reference}')) + hba1c0_c + "
               "I(hba1c0_c ** 2) + age_c + bmi_c + egfr_c + glucose_c + interval_c + "
               "diet0_c + activity0_c + C(sex) + C(smoking) + C(alcohol)")
    if weighted:
        result = smf.wls(formula, data=dat, weights=dat["ipw"]).fit(cov_type="HC3")
    else:
        result = smf.ols(formula, data=dat).fit(cov_type="HC3")
    coef = pd.DataFrame({"term": result.params.index, "estimate_mmolmol": result.params.values,
                         "se": result.bse, "p_value": result.pvalues})
    ci = result.conf_int()
    coef["ci_low"] = ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].to_numpy()
    coef["ci_high"] = ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].to_numpy()
    coef.insert(0, "model", label)
    return result, dat, coef


def contrast_row(coef: pd.DataFrame, model: str, exposure: str, level: str) -> dict:
    mask = coef["term"].str.contains(f"T.{level}]", regex=False)
    if not mask.any():
        return {"model": model, "contrast": f"{level} vs unchanged", "status": "not_estimable"}
    row = coef.loc[mask].iloc[0]
    return {"model": model, "contrast": f"{level} vs unchanged", "status": "ok",
            "n": int(coef.attrs.get("n", 0)), "estimate_mmolmol": row.estimate_mmolmol,
            "ci_low_mmolmol": row.ci_low, "ci_high_mmolmol": row.ci_high,
            "estimate_hba1c_pct": row.estimate_mmolmol * 0.09148,
            "ci_low_hba1c_pct": row.ci_low * 0.09148, "ci_high_hba1c_pct": row.ci_high * 0.09148,
            "p_value": row.p_value}
