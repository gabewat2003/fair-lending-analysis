"""Collect headline numbers from reports/tables into reports/results.json and
render the markdown tables used in README.md and reports/memo.md
(reports/tables/summary_tables.md)."""
import json

import pandas as pd

from src import config

T = config.TABLE_DIR


def fmt_pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{x * 100:.{d}f}%"


def fmt_pp(x, d=1):
    return "n/a" if pd.isna(x) else f"{x:+.{d}f}"


def main_results_table() -> pd.DataFrame:
    s1 = pd.read_csv(T / "step1_derived_race.csv").set_index("group")
    ame = pd.read_csv(T / "step2_race_ame.csv").pivot(index="group", columns="model", values="ame_pp")
    ci = pd.read_csv(T / "step2_race_ame.csv").query("model == 'm1'").set_index("group")
    out = pd.DataFrame({
        "Applications": s1["apps"].map("{:,}".format),
        "Approval rate": s1["approval_rate"].map(fmt_pct),
        "AIR vs White": s1["AIR"].map("{:.3f}".format),
        "Raw diff (pp)": s1["diff_pp"].map(fmt_pp),
        "m1 adj. AME (pp) [95% CI]": [
            "n/a (reference)" if g == config.REF_RACE else
            f"{ame.loc[g, 'm1']:+.1f} [{ci.loc[g, 'ci_low']:+.1f}, {ci.loc[g, 'ci_high']:+.1f}]"
            for g in s1.index],
        "m2 lender-FE (pp)": ["n/a" if g == config.REF_RACE else fmt_pp(ame.loc[g, "m2"])
                              for g in s1.index],
    })
    out.index = [f"{g} (small n)" if s1.loc[g, "small_n"] else g for g in s1.index]
    return out.rename_axis("Group")


def main() -> None:
    log = pd.read_csv(T / "cleaning_log.csv")
    s1 = pd.read_csv(T / "step1_derived_race.csv").set_index("group")
    ame = pd.read_csv(T / "step2_race_ame.csv")
    auc = pd.read_csv(T / "step3_proxy_auc.csv")
    abl = pd.read_csv(T / "step3_ablation.csv").set_index("model")
    black = "Black or African American"

    results = {
        "final_sample": int(log["rows"].iloc[-1]),
        "overall_approval_rate": None,
        "white_rate": float(s1.loc["White", "approval_rate"]),
        "black_rate": float(s1.loc[black, "approval_rate"]),
        "black_air": float(s1.loc[black, "AIR"]),
        "black_raw_diff_pp": float(s1.loc[black, "diff_pp"]),
        "black_z_pooled": float(s1.loc[black, "z_ztest"]),
        "black_z_lender_clustered": float(s1.loc[black, "z_clustered"]),
        "black_ame_pp": {m: float(r.ame_pp) for m, r in
                         ame[ame.group == black].set_index("model").iterrows()},
        "black_ame_ci_m1": [float(x) for x in
                            ame.query("group == @black and model == 'm1'")[["ci_low", "ci_high"]].iloc[0]],
        "proxy_auc": {f"{r.target} | {r.feature_set}": round(float(r.test_auc), 4)
                      for r in auc.itertuples()},
        "ablation": {m: {"auc": None if pd.isna(r.test_auc) else round(float(r.test_auc), 4),
                         "black_air": round(float(r[f"AIR {black}"]), 4),
                         "min_air": round(float(r.min_AIR), 4)} for m, r in abl.iterrows()},
    }
    df = pd.read_parquet(config.ANALYSIS_PATH, columns=["approved"])
    results["overall_approval_rate"] = float(df["approved"].mean())
    (config.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))

    md = ["## Main results (race)\n", main_results_table().to_markdown(), "\n",
          "## Cleaning log\n", log.to_markdown(index=False), "\n",
          "## Proxy AUC\n",
          auc.pivot(index="feature_set", columns="target", values="test_auc").round(3).to_markdown(),
          "\n", "## Ablation\n", abl.round(4).to_markdown(), "\n"]
    (T / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/results.json and reports/tables/summary_tables.md")


if __name__ == "__main__":
    main()
