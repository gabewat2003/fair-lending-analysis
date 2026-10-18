"""Raw HMDA CSVs -> filtered, typed analysis table (data/processed/analysis.parquet).

Every filter step is logged to reports/tables/cleaning_log.csv so the README can
report exactly how the sample was built.
"""
import numpy as np
import pandas as pd

from src import config

MISSING_TOKENS = {"", "NA", "Exempt", "exempt", "N/A"}

ACTION_LABELS = {
    "1": "Originated", "2": "Approved, not accepted", "3": "Denied",
    "4": "Withdrawn", "5": "File closed incomplete", "6": "Purchased loan",
    "7": "Preapproval denied", "8": "Preapproval approved, not accepted",
}
LOAN_TYPE_LABELS = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA/RHS"}
DENIAL_LABELS = {
    "1": "Debt-to-income ratio", "2": "Employment history", "3": "Credit history",
    "4": "Collateral", "5": "Insufficient cash", "6": "Unverifiable information",
    "7": "Credit application incomplete", "8": "Mortgage insurance denied",
    "9": "Other", "10": "Not applicable", "1111": "Exempt",
}

# DTI arrives banded below 36% and above 49%, exact (as a string) for 36-49.
DTI_BAND_ORDER = ["<20", "20-30", "30-36", "36-40", "41-45", "46-49", "50-60", ">60", "Missing"]
_DTI_RANGES = {"<20%": ("<20", 15.0), "20%-<30%": ("20-30", 25.0), "30%-<36%": ("30-36", 33.0),
               "50%-60%": ("50-60", 55.0), ">60%": (">60", 65.0)}


def parse_dti(raw: str) -> tuple[str, float]:
    """Return (band, numeric value). Range strings map to band midpoints."""
    if raw in _DTI_RANGES:
        return _DTI_RANGES[raw]
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return "Missing", np.nan
    if 36 <= v <= 40:
        return "36-40", v
    if 41 <= v <= 45:
        return "41-45", v
    if 46 <= v <= 49:
        return "46-49", v
    return "Missing", np.nan


def to_num(s: pd.Series) -> pd.Series:
    """Strings -> float; 'NA'/'Exempt'/blank -> NaN."""
    return pd.to_numeric(s.where(~s.isin(MISSING_TOKENS)), errors="coerce")


def parse_ltv(s: pd.Series) -> pd.Series:
    """Combined LTV (%). Values <=0 or >150 treated as data errors -> NaN."""
    v = to_num(s)
    return v.where((v > 0) & (v <= 150))


def map_outcome(action: pd.Series) -> pd.Series:
    """1/2 -> 1 (approved), 3 -> 0 (denied), anything else -> NaN."""
    out = pd.Series(np.nan, index=action.index)
    out[action.isin(config.APPROVED_CODES)] = 1.0
    out[action.isin(config.DENIED_CODES)] = 0.0
    return out


class Log:
    def __init__(self):
        self.rows = []

    def add(self, step: str, df: pd.DataFrame):
        prev = self.rows[-1]["rows"] if self.rows else len(df)
        self.rows.append({"step": step, "rows": len(df), "dropped": prev - len(df)})
        print(f"{step:<45} {len(df):>10,}  (-{prev - len(df):,})")

    def save(self):
        pd.DataFrame(self.rows).to_csv(config.TABLE_DIR / "cleaning_log.csv", index=False)


def load_raw() -> pd.DataFrame:
    frames = []
    for st in config.STATES:
        path = config.RAW_DIR / f"hmda_{config.YEAR}_{st}.csv"
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def build(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    log.add("raw download (home purchase, 4 states)", df)

    for col, keep in config.LOCAL_FILTERS.items():
        df = df[df[col].isin(keep)]
        log.add(f"filter {col} in {sorted(keep)}", df)

    # Outcome. Record the full action_taken mix before dropping non-decisions.
    counts = (df["action_taken"].map(ACTION_LABELS).value_counts()
              .rename_axis("action_taken").reset_index(name="applications"))
    counts.to_csv(config.TABLE_DIR / "action_taken_counts.csv", index=False)
    df = df[df["action_taken"].isin(config.APPROVED_CODES | config.DENIED_CODES)].copy()
    log.add("keep action_taken 1,2 (approved) or 3 (denied)", df)

    out = pd.DataFrame(index=df.index)
    out["action_taken"] = df["action_taken"]
    out["approved"] = map_outcome(df["action_taken"]).astype(int)
    # Sensitivity outcome: originated vs denied only (approved-not-accepted -> NaN).
    out["approved_strict"] = np.where(df["action_taken"] == "2", np.nan, out["approved"])

    out["derived_race"] = df["derived_race"]
    out["derived_ethnicity"] = df["derived_ethnicity"]
    out["derived_sex"] = df["derived_sex"]
    out["applicant_age"] = df["applicant_age"].replace({"8888": "Not available", "9999": "Not available"})

    out["income"] = to_num(df["income"])                    # $ thousands
    out["loan_amount"] = to_num(df["loan_amount"])          # $
    out["property_value"] = to_num(df["property_value"])    # $
    out["ltv"] = parse_ltv(df["loan_to_value_ratio"])       # Data Browser name for CLTV
    dti = df["debt_to_income_ratio"].map(parse_dti)
    out["dti_band"] = pd.Categorical(dti.str[0], categories=DTI_BAND_ORDER, ordered=True)
    out["dti_num"] = dti.str[1].astype(float)

    pos = lambda s: np.log(s.where(s > 0))
    out["log_income"] = pos(out["income"])
    out["log_loan_amount"] = pos(out["loan_amount"])
    out["log_property_value"] = pos(out["property_value"])

    out["loan_type_code"] = df["loan_type"].astype(int)
    out["loan_type"] = df["loan_type"].map(LOAN_TYPE_LABELS)
    out["state"] = df["state_code"]
    out["state_num"] = out["state"].map({s: i for i, s in enumerate(config.STATES)})
    out["county_code"] = df["county_code"]
    out["census_tract"] = df["census_tract"]
    out["lei"] = df["lei"]
    for c in config.GEO_FEATURES:
        if c != "state_num":
            out[c] = to_num(df[c])

    out["denial_reason"] = df["denial_reason-1"].map(DENIAL_LABELS).fillna("Unknown")
    return out.reset_index(drop=True)


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    log = Log()
    out = build(load_raw(), log)
    log.save()

    # Missingness of key fields within the final sample (reported, not dropped here).
    miss = pd.DataFrame({
        "missing": out[["income", "loan_amount", "property_value", "ltv", "dti_num",
                        "tract_minority_population_percent"]].isna().sum(),
    })
    miss["dti_band_missing"] = np.nan
    miss.loc["dti_num", "dti_band_missing"] = (out["dti_band"] == "Missing").sum()
    miss["share"] = miss["missing"] / len(out)
    miss.rename_axis("field").to_csv(config.TABLE_DIR / "missingness.csv")

    for col in ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]:
        vc = out[col].value_counts()
        assert vc.sum() == len(out), f"{col} counts do not sum to total"
        vc.rename_axis(col).reset_index(name="applications").to_csv(
            config.TABLE_DIR / f"group_counts_{col}.csv", index=False)

    out.to_parquet(config.ANALYSIS_PATH, index=False)
    print(f"wrote {config.ANALYSIS_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
