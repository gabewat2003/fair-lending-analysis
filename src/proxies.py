"""Step 3: proxy analysis.

(a) Can "neutral" features predict protected-class membership? (AUC + permutation importance)
(b) Ablation: does dropping geography from an approval model narrow group disparities,
    and what does it cost in accuracy? (a less-discriminatory-alternative style check)

Race, ethnicity and sex are never model inputs.
"""
import altair as alt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src.disparities import save_chart

CATEGORICAL = {"loan_type_code", "state_num"}
FEATURE_SETS = {
    "applicant/loan only": config.APPLICANT_FEATURES,
    "geography only": config.GEO_FEATURES,
    "all features": config.APPLICANT_FEATURES + config.GEO_FEATURES,
}
PROXY_TARGETS = [
    ("Black vs White", "derived_race", "Black or African American", config.REF_RACE),
    ("Asian vs White", "derived_race", "Asian", config.REF_RACE),
    ("Hispanic vs Not Hispanic", "derived_ethnicity", "Hispanic or Latino", config.REF_ETHNICITY),
]


def make_model(features: list[str]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True, random_state=config.SEED,
        categorical_features=[f in CATEGORICAL for f in features])


def proxy_detection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, importances = [], []
    for name, col, grp, ref in PROXY_TARGETS:
        d = df[df[col].isin([grp, ref])]
        y = (d[col] == grp).astype(int)
        tr, te = train_test_split(d.index, test_size=0.3, stratify=y, random_state=config.SEED)
        assert not set(tr) & set(te), "train/test overlap"
        for fs_name, feats in FEATURE_SETS.items():
            clf = make_model(feats).fit(d.loc[tr, feats], y[tr])
            auc = roc_auc_score(y[te], clf.predict_proba(d.loc[te, feats])[:, 1])
            rows.append({"target": name, "feature_set": fs_name, "test_auc": auc,
                         "n_train": len(tr), "n_test": len(te), "positive_share": y.mean()})
            print(f"{name:<26} {fs_name:<22} AUC={auc:.3f}")
            if fs_name == "all features":
                sub = pd.Index(te).to_series().sample(min(40_000, len(te)), random_state=config.SEED)
                pi = permutation_importance(clf, d.loc[sub, feats], y[sub], scoring="roc_auc",
                                            n_repeats=5, random_state=config.SEED)
                importances += [{"target": name, "feature": f, "auc_drop": m, "auc_drop_sd": s}
                                for f, m, s in zip(feats, pi.importances_mean, pi.importances_std)]
    return pd.DataFrame(rows), pd.DataFrame(importances)


def importance_chart(imp: pd.DataFrame) -> alt.Chart:
    order = (imp.groupby("feature")["auc_drop"].max().sort_values(ascending=False).index.tolist())
    return alt.Chart(imp).mark_bar().encode(
        x=alt.X("auc_drop:Q", title="Drop in test AUC when feature is shuffled",
                axis=alt.Axis(format=".2f", tickCount=8)),
        y=alt.Y("feature:N", sort=order, title=None),
        color=alt.Color("target:N", title="Proxy target",
                        scale=alt.Scale(range=["#4C78A8", "#F58518", "#54A24B"])),
        yOffset="target:N",
        tooltip=["target", "feature", alt.Tooltip("auc_drop", format=".3f")],
    ).properties(title=alt.Title("Which 'neutral' features carry group information?",
                                 subtitle="Permutation importance, all-features proxy model"),
                 width=480, height=360)


def group_air(pred: pd.Series, groups: pd.Series) -> pd.Series:
    rates = pred.groupby(groups).mean()
    return rates / rates[config.REF_RACE]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def noise_matched_scores(s_tr: np.ndarray, s_te: np.ndarray, y_te: pd.Series,
                         target_auc: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Add race-blind Gaussian noise to log-odds scores until test AUC falls to `target_auc`.

    This is a pure accuracy-loss control: it degrades the model by the same amount
    as dropping geography, but in a way that carries no information about any group.
    The noise draws are fixed once, so AUC is monotone in the noise scale.
    """
    rng = np.random.default_rng(config.SEED)
    z_tr, z_te = rng.standard_normal(len(s_tr)), rng.standard_normal(len(s_te))
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if roc_auc_score(y_te, s_te + mid * z_te) > target_auc:
            lo = mid
        else:
            hi = mid
    sigma = (lo + hi) / 2
    return s_tr + sigma * z_tr, s_te + sigma * z_te, sigma


def ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Approval model with vs without geography; AIR on predicted approvals (test set).

    The model is trained on the lenders' observed decisions, which are the thing
    under test, so "accuracy" here means agreement with those decisions.

    Controls for "dropping geography only narrows the gap because the model got worse":
      D: drop one non-geographic feature at a time (similar or larger AUC cost)
      E: model A with race-blind random noise added to its scores, tuned to B's AUC
    """
    y = df["approved"]
    race = df["derived_race"]
    tr, te = train_test_split(df.index, test_size=0.3, stratify=y, random_state=config.SEED)
    assert not set(tr) & set(te), "train/test overlap"
    all_feats = config.APPLICANT_FEATURES + config.GEO_FEATURES
    variants = {
        "A: all features": ("main", all_feats),
        "B: drop all geography": ("main", config.APPLICANT_FEATURES),
        "C: drop tract minority %": ("main", [f for f in all_feats
                                             if f != "tract_minority_population_percent"]),
    }
    for f in config.APPLICANT_FEATURES:
        variants[f"D: drop {f}"] = ("control", [x for x in all_feats if x != f])
    big = race[te].value_counts()
    big = [g for g in big.index if big[g] >= config.MIN_GROUP_N
           and g not in (config.REF_RACE, "Race Not Available", "Joint")]

    def score_row(name, kind, s_tr, s_te, auc, **extra):
        # Approve the same share of applicants the lenders approved in training data.
        thr = np.quantile(s_tr, 1 - y[tr].mean())
        pred = pd.Series((s_te >= thr).astype(int), index=te)
        air = group_air(pred, race[te])
        print(f"{name:<40} AUC={auc:.4f}  Black AIR={air['Black or African American']:.3f}"
              f"  min AIR={air[big].min():.3f}")
        return {"model": name, "kind": kind, "test_auc": auc,
                **{f"AIR {g}": air[g] for g in big}, "min_AIR": air[big].min(),
                "approval_rate_pred": pred.mean(), **extra}

    rows = []
    actual_air = group_air(y[te], race[te])
    rows.append({"model": "Observed outcomes (test set)", "kind": "observed", "test_auc": np.nan,
                 **{f"AIR {g}": actual_air[g] for g in big},
                 "min_AIR": actual_air[big].min()})
    scores = {}
    for name, (kind, feats) in variants.items():
        clf = make_model(feats).fit(df.loc[tr, feats], y[tr])
        s_tr = _logit(clf.predict_proba(df.loc[tr, feats])[:, 1])
        s_te = _logit(clf.predict_proba(df.loc[te, feats])[:, 1])
        scores[name] = (s_tr, s_te)
        rows.append(score_row(name, kind, s_tr, s_te, roc_auc_score(y[te], s_te)))

    auc_b = next(r["test_auc"] for r in rows if r["model"] == "B: drop all geography")
    n_tr, n_te, sigma = noise_matched_scores(*scores["A: all features"], y[te], auc_b)
    rows.append(score_row("E: all features + random score noise", "control", n_tr, n_te,
                          roc_auc_score(y[te], n_te), noise_sd_logodds=sigma))
    return pd.DataFrame(rows)


def main() -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.ANALYSIS_PATH)

    auc, imp = proxy_detection(df)
    auc.to_csv(config.TABLE_DIR / "step3_proxy_auc.csv", index=False)
    imp.to_csv(config.TABLE_DIR / "step3_permutation_importance.csv", index=False)
    save_chart(importance_chart(imp), "step3_proxy_importance")

    abl = ablation(df)
    abl.to_csv(config.TABLE_DIR / "step3_ablation.csv", index=False)
    print(abl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
