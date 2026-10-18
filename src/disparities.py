"""Step 1 (raw approval-rate disparities, AIR, significance) and Step 4 (denial reasons)."""
import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportion_confint, proportions_ztest)

from src import config


def disparity_table(df: pd.DataFrame, group_col: str, ref: str,
                    outcome: str = "approved") -> pd.DataFrame:
    """Approval rate, AIR and two-sample tests for every group vs the reference group."""
    d = df.dropna(subset=[outcome])
    tbl = (d.groupby(group_col, observed=True)
             .agg(apps=(outcome, "size"), approvals=(outcome, "sum")))
    tbl["approvals"] = tbl["approvals"].astype(int)
    tbl["approval_rate"] = tbl["approvals"] / tbl["apps"]
    lo, hi = proportion_confint(tbl["approvals"], tbl["apps"], method="wilson")
    tbl["rate_ci_low"], tbl["rate_ci_high"] = lo, hi

    a_ref, n_ref = tbl.loc[ref, "approvals"], tbl.loc[ref, "apps"]
    ref_rate = a_ref / n_ref
    tbl["AIR"] = tbl["approval_rate"] / ref_rate
    tbl["flag_4_5ths"] = tbl["AIR"] < 0.80
    tbl["diff_pp"] = (tbl["approval_rate"] - ref_rate) * 100

    rows = []
    for g, r in tbl.iterrows():
        if g == ref:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        _, p = proportions_ztest([r.approvals, a_ref], [r.apps, n_ref])
        lo, hi = confint_proportions_2indep(r.approvals, r.apps, a_ref, n_ref,
                                            method="newcomb", compare="diff")
        # Fisher's exact as a check where any cell of the 2x2 table is small.
        cells = [r.approvals, r.apps - r.approvals, a_ref, n_ref - a_ref]
        p_fisher = (fisher_exact([[cells[0], cells[1]], [cells[2], cells[3]]])[1]
                    if min(cells) < 10 else np.nan)
        rows.append((lo * 100, hi * 100, p, p_fisher))
    tbl[["diff_ci_low_pp", "diff_ci_high_pp", "p_ztest", "p_fisher"]] = rows

    mask = tbl.index != ref
    tbl["p_holm"] = np.nan
    tbl.loc[mask, "p_holm"] = multipletests(tbl.loc[mask, "p_ztest"], method="holm")[1]
    tbl["small_n"] = tbl["apps"] < config.MIN_GROUP_N
    tbl["is_reference"] = ~mask
    return tbl.sort_values("apps", ascending=False).rename_axis("group").reset_index()


def approval_chart(tbl: pd.DataFrame, title: str) -> alt.LayerChart:
    ref_rate = tbl.loc[tbl.is_reference, "approval_rate"].iloc[0]
    d = tbl.assign(threshold=0.8 * ref_rate,
                   label=lambda t: t.group + np.where(t.small_n, " (small n)", ""))
    order = d.sort_values("approval_rate")["label"].tolist()
    y = alt.Y("label:N", sort=order, title=None)
    x_scale = alt.Scale(domain=[0.5, 1.0])
    bars = alt.Chart(d).mark_bar(clip=True).encode(
        x=alt.X("approval_rate:Q", title="Approval rate", scale=x_scale,
                axis=alt.Axis(format="%")),
        y=y,
        color=alt.condition("datum.is_reference", alt.value("#9D9D9D"), alt.value("#4C78A8")),
        tooltip=["group", "apps", alt.Tooltip("approval_rate", format=".1%"),
                 alt.Tooltip("AIR", format=".3f")])
    err = alt.Chart(d).mark_rule(color="black").encode(
        x=alt.X("rate_ci_low:Q", scale=x_scale), x2="rate_ci_high:Q", y=y)
    rule = alt.Chart(d.head(1)).mark_rule(color="#E45756", strokeDash=[6, 4], size=2).encode(
        x=alt.X("threshold:Q", scale=x_scale))
    return (bars + err + rule).properties(
        title=alt.Title(title, subtitle="Dashed red line = 80% of the reference (White) approval "
                                        "rate; grey bar = reference group"),
        width=520, height=260)


def denial_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each primary denial reason within each race group (denied apps only)."""
    den = df[df["approved"] == 0]
    ct = pd.crosstab(den["derived_race"], den["denial_reason"])
    share = ct.div(ct.sum(axis=1), axis=0)
    share.insert(0, "denied_apps", ct.sum(axis=1))
    return share.sort_values("denied_apps", ascending=False)


def denial_chart(share: pd.DataFrame) -> alt.Chart:
    keep = share[share["denied_apps"] >= 200].drop(columns="denied_apps")
    long = keep.reset_index().melt(id_vars="derived_race", var_name="reason", value_name="share")
    order = long.groupby("reason")["share"].mean().sort_values(ascending=False).index.tolist()
    return alt.Chart(long).mark_rect().encode(
        x=alt.X("reason:N", title="Primary denial reason", sort=order),
        y=alt.Y("derived_race:N", title=None),
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(format="%")),
        tooltip=["derived_race", "reason", alt.Tooltip("share", format=".1%")],
    ).properties(title="Primary denial reason by race (denied applications, groups with 200+ denials)",
                 width=560, height=220) + alt.Chart(long).mark_text(fontSize=10).encode(
        x=alt.X("reason:N", sort=order), y="derived_race:N",
        text=alt.Text("share:Q", format=".0%"),
        color=alt.condition("datum.share > 0.3", alt.value("white"), alt.value("black")))


def save_chart(chart, name: str) -> None:
    chart.save(config.FIG_DIR / f"{name}.html")
    chart.save(config.FIG_DIR / f"{name}.png", scale_factor=2)


def main() -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.ANALYSIS_PATH)

    specs = [("derived_race", config.REF_RACE), ("derived_ethnicity", config.REF_ETHNICITY),
             ("derived_sex", config.REF_SEX), ("applicant_age", config.REF_AGE)]
    for col, ref in specs:
        tbl = disparity_table(df, col, ref)
        tbl.to_csv(config.TABLE_DIR / f"step1_{col}.csv", index=False)
        print(f"\n{col}\n", tbl[["group", "apps", "approval_rate", "AIR", "diff_pp", "p_holm"]]
              .to_string(index=False))

    race = disparity_table(df, "derived_race", config.REF_RACE)
    save_chart(approval_chart(race, "Approval rate by race (95% CI), 2023 home purchase, GA/IL/TX/MD"),
               "step1_approval_by_race")

    # Sensitivity: originated (1) vs denied (3) only.
    disparity_table(df, "derived_race", config.REF_RACE, outcome="approved_strict").to_csv(
        config.TABLE_DIR / "step1_derived_race_strict.csv", index=False)

    # Race x state: where are raw gaps largest?
    by_state = pd.concat([disparity_table(g, "derived_race", config.REF_RACE).assign(state=s)
                          for s, g in df.groupby("state")])
    by_state.to_csv(config.TABLE_DIR / "step1_race_by_state.csv", index=False)

    share = denial_reasons(df)
    share.to_csv(config.TABLE_DIR / "step4_denial_reasons_by_race.csv")
    save_chart(denial_chart(share), "step4_denial_reasons")


if __name__ == "__main__":
    main()
