"""Xiangya measurement cleaning, medication covariates and observation weights."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set explicitly by the runner; importing this module does not access files.
DATA_FILE: Path | None = None


DRUG_CLASSES = [
    "insulin", "metformin", "alpha_glucosidase", "sglt2", "dpp4",
    "glp1ra", "sulfonylurea", "tzd", "combination_product",
]


DRUG_LABELS = {
    "insulin": "Insulin",
    "metformin": "Metformin",
    "alpha_glucosidase": "α-glucosidase inhibitor",
    "sglt2": "SGLT2 inhibitor",
    "dpp4": "DPP-4 inhibitor",
    "glp1ra": "GLP-1 receptor agonist",
    "sulfonylurea": "Sulfonylurea",
    "tzd": "Thiazolidinedione",
    "combination_product": "Fixed-dose combination",
}


CORE_CLASSES = [c for c in DRUG_CLASSES if c != "combination_product"]


def _num(s: pd.Series, low=None, high=None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if low is not None:
        x = x.where(x >= low)
    if high is not None:
        x = x.where(x <= high)
    return x


def _positive(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    return s not in {"", "0", "0.0", "否", "无", "未使用", "没有", "nan", "None"}


def _text_positive(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    if not s or s in {"否", "未使用", "无", "0", "0.0"}:
        return False
    parts = [p.strip() for p in re.split(r"[;；,，]", s) if p.strip()]
    return any(p not in {"未使用", "否", "无"} for p in parts)


def _any_nonempty(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(False, index=df.index)
    return df[cols].apply(lambda c: c.notna() & c.astype(str).str.strip().ne(""), axis=0).any(axis=1)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _exact_header_flag(df: pd.DataFrame, headers: list[str], text_mode=False) -> pd.Series:
    cols = [c for c in headers if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    fn = _text_positive if text_mode else _positive
    return pd.DataFrame({c: df[c].map(fn) for c in cols}).any(axis=1)


def load_and_clean() -> pd.DataFrame:
    raw = pd.read_excel(DATA_FILE)
    d = raw.copy()
    d["analysis_id"] = np.arange(1, len(d) + 1)

    date_year = _col(d, "Date").astype(str).str.extract(r"(20\d{2})", expand=False)
    date_year = pd.to_numeric(date_year, errors="coerce").fillna(2025)
    d["age"] = (date_year - _num(_col(d, "Birth_year"), 1900, 2010)).where(lambda x: x.between(18, 100))
    d["sex_male"] = _col(d, "Sex").map({"男": 1.0, "女": 0.0})
    d["height_cm"] = _num(_col(d, "Height"), 120, 210)
    d["weight_kg"] = _num(_col(d, "Weight"), 30, 200)
    d["bmi"] = d["weight_kg"] / (d["height_cm"] / 100) ** 2
    d["bmi"] = d["bmi"].where(d["bmi"].between(12, 60))
    d["wc_cm"] = _num(_col(d, "WC"), 40, 180)
    d["hc_cm"] = _num(_col(d, "HC"), 50, 180)
    d["whr"] = (d["wc_cm"] / d["hc_cm"]).where(lambda x: x.between(0.5, 1.5))
    d["hba1c_baseline"] = _num(_col(d, "HBA1c"), 4, 16)
    d["hba1c_followup"] = _num(_col(d, "1st-糖化血红蛋白（%）"), 4, 16)
    d["delta_hba1c"] = d["hba1c_followup"] - d["hba1c_baseline"]
    d["fbg"] = _num(_col(d, "FBG"), 2, 40)
    d["sbp"] = _num(_col(d, "SBP"), 70, 250)
    d["dbp"] = _num(_col(d, "DBP"), 35, 150)
    d["creatinine"] = _num(_col(d, "Cr"), 20, 1000)
    d["egfr"] = _num(_col(d, "eGFR"), 5, 150)
    d["alt"] = _num(_col(d, "ALT"), 3, 500)

    wjx_cols = [c for c in d.columns if c.startswith("WJX-")]
    tjs_cols = [c for c in d.columns if c.startswith("TJS-")]
    d["has_wjx"] = _any_nonempty(d, wjx_cols)
    d["has_tjs"] = _any_nonempty(d, tjs_cols)
    d["survey_source"] = np.select(
        [d.has_wjx & d.has_tjs, d.has_wjx, d.has_tjs],
        ["WJX+TJS", "WJX", "TJS"], default="none",
    )
    return encode_medications(d)


def encode_medications(d: pd.DataFrame) -> pd.DataFrame:
    # WJX baseline uses the exported class checkboxes. TJS uses class-specific
    # text fields because its single summary field contains a known mislabeled option.
    wjx_headers = {
        "insulin": ["WJX-您近三个月正在使用的降糖药物是什么?(胰岛素类（如甘精胰岛素、地特胰岛素、预混胰岛素，如诺和灵30R、门冬胰岛素等）)"],
        "metformin": ["WJX-66(二甲双胍类（如二甲双胍等）)"],
        "alpha_glucosidase": ["WJX-66(α-糖苷酶抑制剂（如阿卡波糖、伏格列波糖等）)"],
        "sglt2": ["WJX-66(SGLT2抑制剂（如达格列净、恩格列净等）)"],
        "dpp4": ["WJX-66(DPP-4抑制剂（如西格列汀、沙格列汀）)"],
        "glp1ra": ["WJX-66(GLP-1受体激动剂（如利拉鲁肽、度拉糖肽、司美格鲁肽等）)"],
        "sulfonylurea": ["WJX-66(磺酰脲类（如格列美脲、格列齐特等）)"],
        "tzd": ["WJX-66(噻唑烷二酮类（如吡格列酮等）)"],
        "combination_product": ["WJX-66(复合制剂（如捷诺达、诺和益）)"],
    }
    tjs_headers = {
        "insulin": ["TJS-您是否使用以下胰岛素？是从哪一年开始使用的？"],
        "metformin": ["TJS-您是否使用以下二甲双胍类药物？是从哪一年开始使用的？"],
        "alpha_glucosidase": ["TJS-您是否使用以下α-糖苷酶抑制剂药物？是从哪一年开始使用的？"],
        "sglt2": ["TJS-您是否使用以下SGLT2抑制剂？是从哪一年开始使用的？"],
        "dpp4": ["TJS-您是否使用以下DPP-4抑制剂？是从哪一年开始使用的？"],
        "glp1ra": ["TJS-您是否使用以下GLP-1受体激动剂？是从哪一年开始使用的？"],
        "sulfonylurea": ["TJS-您是否使用以下磺酰脲类？是从哪一年开始使用的？"],
        "tzd": ["TJS-您是否使用以下噻唑烷二酮类（TZDs）？是从哪一年开始使用的？"],
        "combination_product": ["TJS-您是否使用以下复合制剂？是从哪一年开始使用的？"],
    }
    for cls in DRUG_CLASSES:
        w = _exact_header_flag(d, wjx_headers[cls])
        t = _exact_header_flag(d, tjs_headers[cls], text_mode=True)
        d[f"bl_{cls}"] = (w | t).astype(int)

    # Decompose explicitly named TJS fixed-dose combinations.
    combo_text = _col(d, "TJS-您是否使用以下复合制剂？是从哪一年开始使用的？").fillna("").astype(str)
    combo_map = {
        "metformin": ["二甲双胍", "格华止"], "sglt2": ["达格列净", "恩格列净", "卡格列净"],
        "dpp4": ["西格列汀", "沙格列汀", "维格列汀"], "glp1ra": ["利拉鲁肽"],
        "insulin": ["德谷胰岛素"], "sulfonylurea": ["格列本脲", "格列美脲"],
        "alpha_glucosidase": ["阿卡波糖"], "tzd": ["吡格列酮"],
    }
    for cls, kws in combo_map.items():
        hit = combo_text.map(lambda x: any(k in x for k in kws))
        d.loc[hit, f"bl_{cls}"] = 1

    # Regular-use question distinguishes confirmed no medication from missing survey.
    wreg = pd.to_numeric(_col(d, "WJX-您近三个月是否规律使用降糖药物?"), errors="coerce")
    treg = _col(d, "TJS-您近三个月是否规律使用降糖药物？").astype(str).str.strip()
    regular_yes = (wreg == 1) | (treg == "是")
    regular_no = (wreg == 2) | (treg == "否")
    d["baseline_regular_med"] = np.select([regular_yes, regular_no], [1.0, 0.0], default=np.nan)
    d["baseline_class_count"] = d[[f"bl_{c}" for c in CORE_CLASSES]].sum(axis=1)
    d["baseline_med_status"] = np.select(
        [d.baseline_class_count.gt(0), d.baseline_regular_med.eq(0), d.survey_source.eq("none")],
        ["classified", "confirmed_none", "no_baseline_survey"], default="unclassified",
    )

    fu_prefix = {
        "insulin": ["1st-19. 您目前使用的降糖药物有哪些？（请勾选所有正在使用的药物类别）(胰岛素类", "1st-近三个月，您是否使用以下胰岛素？"],
        "metformin": ["1st-19(二甲双胍类", "1st-近三个月，您是否使用以下二甲双胍类药物？"],
        "alpha_glucosidase": ["1st-19(α-糖苷酶抑制剂", "1st-近三个月，您是否使用以下α-糖苷酶抑制剂药物？"],
        "sglt2": ["1st-19(SGLT2抑制剂", "1st-近三个月，您是否使用以下 SGLT2 抑制剂？"],
        "dpp4": ["1st-19(DPP-4抑制剂", "1st-近三个月，您是否使用以下 DPP-4 抑制剂？"],
        "glp1ra": ["1st-19(GLP-1受体激动剂", "1st-近三个月，您是否使用以下 GLP-1 受体激动剂？"],
        "sulfonylurea": ["1st-19(磺酰脲类", "1st-近三个月，您是否使用以下磺酰脲类？"],
        "tzd": ["1st-19(噻唑烷二酮类", "1st-近三个月，您是否使用以下噻唑烷二酮类"],
        "combination_product": ["1st-19(复合制剂", "1st-近三个月，您是否使用以下复合制剂？"],
    }
    for cls, prefixes in fu_prefix.items():
        cols = [c for c in d.columns if any(c.startswith(p) for p in prefixes)]
        d[f"fu_{cls}"] = _exact_header_flag(d, cols).astype(int)

    # Decompose follow-up fixed-dose combination checkboxes by names in headers.
    fu_combo_cols = [c for c in d.columns if c.startswith("1st-29(") or c.startswith("1st-近三个月，您是否使用以下复合制剂？(")]
    for cls, kws in combo_map.items():
        cols = [c for c in fu_combo_cols if any(k in c for k in kws)]
        if cols:
            d.loc[_exact_header_flag(d, cols), f"fu_{cls}"] = 1

    change = pd.to_numeric(_col(d, "1st-18. 和三个月前相比，您使用的降糖药物有变化吗？"), errors="coerce")
    d["med_change_status"] = change.map({1.0: "changed", 2.0: "unchanged"}).fillna("missing")
    fu_any = d[[f"fu_{c}" for c in CORE_CLASSES]].sum(axis=1).gt(0)
    for cls in CORE_CLASSES:
        d[f"t1_{cls}"] = np.nan
        d.loc[d.med_change_status.eq("unchanged"), f"t1_{cls}"] = d.loc[d.med_change_status.eq("unchanged"), f"bl_{cls}"]
        ok_changed = d.med_change_status.eq("changed") & fu_any
        d.loc[ok_changed, f"t1_{cls}"] = d.loc[ok_changed, f"fu_{cls}"]

    d["transition"] = "change_status_missing"
    d.loc[d.med_change_status.eq("unchanged"), "transition"] = "stable"
    changed_known = d.med_change_status.eq("changed") & fu_any
    b = d[[f"bl_{c}" for c in CORE_CLASSES]].to_numpy(int)
    f = d[[f"fu_{c}" for c in CORE_CLASSES]].to_numpy(int)
    added = (f > b).sum(axis=1)
    removed = (f < b).sum(axis=1)
    d.loc[d.med_change_status.eq("changed") & ~fu_any, "transition"] = "changed_regimen_unknown"
    d.loc[changed_known & (added > 0) & (removed == 0), "transition"] = "add_on"
    d.loc[changed_known & (added == 0) & (removed > 0), "transition"] = "deintensification"
    d.loc[changed_known & (added > 0) & (removed > 0), "transition"] = "switch_or_complex"
    d.loc[changed_known & (added == 0) & (removed == 0), "transition"] = "reported_change_no_class_change"

    # Follow-up lifestyle trajectories.  Diet is directionally scored from the
    # individual response items (+1 healthier, -1 less healthy); exercise export
    # codes are 1=increased, 2=decreased, 3=unchanged.  Missing responses remain
    # missing and are not classified as "unchanged".
    diet_no_change_col = (
        "1st-20. 和三个月前相比，您的饮食习惯（包括主食类型、主食和蔬菜摄入量、"
        "蛋白质来源等）有变化吗？(无变化)"
    )
    diet_better_cols = [
        "1st-20(主食类型更倾向全谷物 / 杂豆类（如糙米、燕麦、红豆等）)",
        "1st-20(主食摄入量明显减少（较之前减少约 1/3 及以上）)",
        "1st-20(蔬菜摄入量明显增加（较之前增加约 1/3 及以上）)",
        "1st-20(蛋白质来源更倾向鱼类 / 禽类 / 蛋类 / 豆制品)",
    ]
    diet_worse_cols = [
        "1st-20(主食类型更倾向精制谷物（如白米饭、白面条等）)",
        "1st-20(主食摄入量明显增加（较之前增加约 1/3 及以上）)",
        "1st-20(蔬菜摄入量明显减少（较之前减少约 1/3 及以上）)",
        "1st-20(蛋白质来源更倾向红肉（猪肉、牛肉等）/ 加工肉制品（香肠、火腿等）)",
    ]
    better = pd.DataFrame({c: _col(d, c).map(_positive) for c in diet_better_cols})
    worse = pd.DataFrame({c: _col(d, c).map(_positive) for c in diet_worse_cols})
    diet_observed = _col(d, diet_no_change_col).notna()
    d["diet_score"] = (better.sum(axis=1) - worse.sum(axis=1)).where(diet_observed)
    d["diet_change3"] = np.select(
        [d.diet_score.gt(0), d.diet_score.lt(0), d.diet_score.eq(0) & diet_observed],
        ["improved", "worsened", "unchanged"], default="missing",
    )
    d["diet_mixed_direction"] = ((better.sum(axis=1) > 0) & (worse.sum(axis=1) > 0)).where(diet_observed)
    d["diet_nochange_conflict"] = (
        _col(d, diet_no_change_col).map(_positive) & ((better.sum(axis=1) + worse.sum(axis=1)) > 0)
    ).where(diet_observed)
    d["diet_change"] = d.diet_change3.isin(["improved", "worsened"]).astype("Int64").where(diet_observed)

    exercise = pd.to_numeric(_col(d, "1st-21. 和三个月前相比，您的运动量有变化吗？"), errors="coerce")
    d["exercise_change3"] = exercise.map({1.0: "improved", 2.0: "worsened", 3.0: "unchanged"}).fillna("missing")
    d["exercise_change"] = d.exercise_change3.isin(["improved", "worsened"]).astype("Int64").where(exercise.notna())
    both_lifestyle_observed = diet_observed & exercise.notna()
    d["behaviour_change"] = (
        d.diet_change3.isin(["improved", "worsened"]) | d.exercise_change3.isin(["improved", "worsened"])
    ).astype("Int64").where(both_lifestyle_observed)
    d["clinical_regimen"] = d.apply(classify_regimen, axis=1)
    return d


def classify_regimen(row: pd.Series) -> str:
    status = row["baseline_med_status"]
    if status == "confirmed_none":
        return "Confirmed no regular medication"
    if status != "classified":
        return "Medication unknown/unclassified"
    active = [c for c in CORE_CLASSES if row[f"bl_{c}"] == 1]
    n = len(active)
    if n == 1:
        return f"{DRUG_LABELS[active[0]]} monotherapy"
    if n >= 3:
        return "Complex regimen (≥3 classes)"
    if "metformin" in active and n == 2:
        other = next(c for c in active if c != "metformin")
        return f"Metformin + {DRUG_LABELS[other]}"
    if "insulin" in active:
        return "Insulin-based combination"
    return "Non-metformin oral combination"


def estimate_followup_weights(d: pd.DataFrame) -> pd.Series:
    model_df = d[d.age.notna() & d.hba1c_baseline.notna()].copy()
    features = ["age", "sex_male", "hba1c_baseline", "bmi", "egfr", "fbg", "sbp", "baseline_class_count", "survey_source"]
    continuous = [c for c in features if c != "survey_source"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), continuous),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["survey_source"]),
    ])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=1.0))])
    y = model_df.hba1c_followup.notna().astype(int)
    pipe.fit(model_df[features], y)
    p = np.clip(pipe.predict_proba(model_df[features])[:, 1], 0.05, 0.98)
    w = y.mean() / p
    observed_w = w[y.to_numpy() == 1]
    lo, hi = np.quantile(observed_w, [0.01, 0.99])
    out = pd.Series(np.nan, index=d.index, name="ipcw")
    out.loc[model_df.index] = np.clip(w, lo, hi)
    return out
