"""Step 2: does the race gap persist after controlling for HMDA underwriting factors?

Models (all on the same regression sample):
  m0      logit, race (+ ethnicity) only                     -> raw gap
  m1      logit, + income, loan size, LTV, DTI, loan type, state
  m1_lpm  linear probability model with m1's covariates       -> apples-to-apples for m2
  m2      LPM with m1's covariates + lender (LEI) fixed effects
  m1_strict  m1 re-fit on originated-vs-denied only (sensitivity)

Race/ethnicity enter as explicit 0/1 dummy columns (reference = White,
Not Hispanic or Latino) so every model and the AME code share one coding.
Standard errors are clustered by lender throughout.
"""
import re

import altair as alt
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
from scipy.stats import norm

from src import config
from src.disparities import save_chart

LTV_BINS = [0, 60, 80, 90, 95, 100, 150]
LTV_LABELS = ["<=60", "60-80", "80-90", "90-95", "95-100", ">100"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def prepare(df: pd.DataFrame):
    """Build the regression sample and dummy columns. Returns (sample, race_map, eth_map, sample_log)."""
    n0 = len(df)
    d = df[df["log_income"].notna() & df["log_loan_amount"].notna()
           & df["ltv"].notna() & (df["dti_band"] != "Missing")].copy()
    d["dti_band"] = d["dti_band"].cat.remove_unused_categories()
    d["ltv_band"] = pd.cut(d["ltv"], LTV_BINS, labels=LTV_LABELS, right=True).astype(str)

    race_map = {f"race_{slug(g)}": g for g in sorted(d["derived_race"].unique()) if g != config.REF_RACE}
    eth_map = {f"eth_{slug(g)}": g for g in sorted(d["derived_ethnicity"].unique())
               if g != config.REF_ETHNICITY}
    for col, g in race_map.items():
        d[col] = (d["derived_race"] == g).astype(float)
    for col, g in eth_map.items():
        d[col] = (d["derived_ethnicity"] == g).astype(float)

    # Selection check: does dropping incomplete rows change the group mix?
    mix = pd.DataFrame({
        "share_full_sample": df["derived_race"].value_counts(normalize=True),
        "share_regression_sample": d["derived_race"].value_counts(normalize=True),
        "approval_full_sample": df.groupby("derived_race")["approved"].mean(),
        "approval_regression_sample": d.groupby("derived_race")["approved"].mean(),
    })
    mix.rename_axis("derived_race").to_csv(config.TABLE_DIR / "step2_sample_selection.csv")
    print(f"regression sample: {len(d):,} of {n0:,} rows ({n0 - len(d):,} dropped for missing "
          f"income/LTV/DTI)")
    return d.reset_index(drop=True), race_map, eth_map


def formulas(race_map, eth_map):
    race = " + ".join(race_map)
    eth = " + ".join(eth_map)
    controls = ("log_income + log_loan_amount + C(ltv_band) + C(dti_band) "
                "+ C(loan_type) + C(state)")
    return {
        "m0": f"approved ~ {race} + {eth}",
        "m1": f"approved ~ {race} + {eth} + {controls}",
        "m1_strict": f"approved_strict ~ {race} + {eth} + {controls}",
        "m2_rhs": f"{race} + {eth} + {controls}",
    }


def logit_ame(res, dummy: str, all_dummies: list[str]) -> tuple[float, float]:
    """Counterfactual average marginal effect of `dummy` vs the reference group.

    Every applicant is scored twice: once assigned to the group (dummy=1, other
    race dummies=0) and once to the reference group (all race dummies=0). The AME
    is the mean difference in predicted approval probability. SE by delta method.
    Race dummies are raw columns of the design matrix, so they are edited in place.
    """
    names = list(res.model.exog_names)
    idx = [names.index(c) for c in all_dummies]
    j = names.index(dummy)

    def X_for(on: bool):
        X = res.model.exog.copy()
        X[:, idx] = 0.0
        if on:
            X[:, j] = 1.0
        return X

    beta = res.params.values
    X1, X0 = X_for(True), X_for(False)
    xb1, xb0 = X1 @ beta, X0 @ beta
    p1, p0 = 1 / (1 + np.exp(-xb1)), 1 / (1 + np.exp(-xb0))
    ame = float(np.mean(p1 - p0))
    grad = (X1 * (p1 * (1 - p1))[:, None]).mean(0) - (X0 * (p0 * (1 - p0))[:, None]).mean(0)
    se = float(np.sqrt(grad @ res.cov_params().values @ grad))
    return ame, se


def fit_logit(formula: str, data: pd.DataFrame):
    d = data.dropna(subset=[formula.split("~")[0].strip()])
    groups = pd.factorize(d["lei"])[0]
    return smf.logit(formula, data=d).fit(disp=0, maxiter=200, cov_type="cluster",
                                          cov_kwds={"groups": groups})


def ame_rows(model: str, res, race_map) -> list[dict]:
    rows = []
    for col, g in race_map.items():
        ame, se = logit_ame(res, col, list(race_map))
        rows.append({"model": model, "group": g, "ame_pp": ame * 100, "se_pp": se * 100})
    return rows


def lpm_rows(model: str, fit, race_map) -> list[dict]:
    coef, se = fit.coef(), fit.se()
    return [{"model": model, "group": g, "ame_pp": coef[c] * 100, "se_pp": se[c] * 100}
            for c, g in race_map.items()]


def ame_chart(ame: pd.DataFrame, keep: list[str]) -> alt.Chart:
    d = ame[ame["group"].isin(keep)].copy()
    labels = {"m0": "m0 race + ethnicity only", "m1": "m1 + underwriting controls",
              "m1_lpm": "m1 as LPM", "m2": "m2 LPM + lender FE"}
    d["model_label"] = d["model"].map(labels)
    order = list(labels.values())
    base = alt.Chart(d).encode(
        y=alt.Y("group:N", title=None, sort=keep),
        yOffset=alt.YOffset("model_label:N", sort=order),
        color=alt.Color("model_label:N", sort=order, title="Model",
                        scale=alt.Scale(range=["#9D9D9D", "#4C78A8", "#72B7B2", "#F58518"])))
    pts = base.mark_point(filled=True, size=60).encode(
        x=alt.X("ame_pp:Q", title="Difference in approval probability vs White (pp)"),
        tooltip=["group", "model_label", alt.Tooltip("ame_pp", format=".2f"),
                 alt.Tooltip("ci_low", format=".2f"), alt.Tooltip("ci_high", format=".2f")])
    bars = base.mark_rule().encode(x="ci_low:Q", x2="ci_high:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
    return (zero + bars + pts).properties(
        title=alt.Title("Race gap in approval probability, raw vs controlled (95% CI)",
                        subtitle="Average marginal effects; SEs clustered by lender"),
        width=520, height=300)


def main() -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.ANALYSIS_PATH)
    d, race_map, eth_map = prepare(df)
    f = formulas(race_map, eth_map)

    rows, summaries = [], {}
    for name in ["m0", "m1", "m1_strict"]:
        res = fit_logit(f[name], d)
        summaries[name] = res
        rows += ame_rows(name, res, race_map)
        print(f"{name}: n={int(res.nobs):,}, pseudo R2={res.prsquared:.3f}")

    lpm = pf.feols(f"approved ~ {f['m2_rhs']}", data=d, vcov={"CRV1": "lei"})
    fe = pf.feols(f"approved ~ {f['m2_rhs']} | lei", data=d, vcov={"CRV1": "lei"})
    rows += lpm_rows("m1_lpm", lpm, race_map) + lpm_rows("m2", fe, race_map)

    ame = pd.DataFrame(rows)
    ame["ci_low"] = ame["ame_pp"] - 1.96 * ame["se_pp"]
    ame["ci_high"] = ame["ame_pp"] + 1.96 * ame["se_pp"]
    ame["p"] = 2 * norm.sf(np.abs(ame["ame_pp"] / ame["se_pp"]))
    n_by_group = d["derived_race"].value_counts()
    ame["apps_in_sample"] = ame["group"].map(n_by_group)
    ame.to_csv(config.TABLE_DIR / "step2_race_ame.csv", index=False)
    wide = ame.pivot(index="group", columns="model", values="ame_pp")
    print("\nAME (pp) by model\n", wide.round(2).to_string())

    # Odds ratios for m1 (all terms) with 95% CI.
    m1 = summaries["m1"]
    ci = m1.conf_int()
    ors = pd.DataFrame({"coef_logodds": m1.params, "odds_ratio": np.exp(m1.params),
                        "or_ci_low": np.exp(ci[0]), "or_ci_high": np.exp(ci[1]),
                        "p": m1.pvalues})
    ors.index = [race_map.get(i, eth_map.get(i, i)) for i in ors.index]
    ors.rename_axis("term").to_csv(config.TABLE_DIR / "step2_m1_odds_ratios.csv")

    with open(config.TABLE_DIR / "step2_model_summaries.txt", "w") as fh:
        for name, res in summaries.items():
            fh.write(f"===== {name} =====\n{res.summary()}\n\n")
        fh.write(f"===== m1_lpm =====\n{lpm.tidy().to_string()}\n\n")
        fh.write(f"===== m2 (lender FE, {d['lei'].nunique():,} lenders) =====\n"
                 f"{fe.tidy().to_string()}\n")

    # Sanity checks on control signs: income up -> approval up; very high DTI -> down.
    p = m1.params
    checks = {
        "log_income > 0": p["log_income"] > 0,
        "DTI >60 < 0 (vs <20)": p["C(dti_band)[T.>60]"] < 0,
        "DTI 50-60 < 0 (vs <20)": p["C(dti_band)[T.50-60]"] < 0,
    }
    pd.Series(checks, name="passed").rename_axis("check").to_csv(
        config.TABLE_DIR / "step2_sign_checks.csv")
    for k, v in checks.items():
        print(f"[{'ok' if v else 'FAIL'}] {k}")

    keep = [g for g in n_by_group.index if g != config.REF_RACE and n_by_group[g] >= config.MIN_GROUP_N]
    save_chart(ame_chart(ame[ame.model != "m1_strict"], keep), "step2_race_ame")


if __name__ == "__main__":
    main()
