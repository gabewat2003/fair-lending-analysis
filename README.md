# Fair Lending Analysis on HMDA Mortgage Data

> **One-line pitch (rewrite last):** Measures approval-rate disparities across demographic groups in public mortgage application data, tests whether they persist after controlling for legitimate underwriting factors, and checks which "neutral" features act as proxies.

**Stack:** Python · pandas · statsmodels · scikit-learn · Altair · (optional) PostgreSQL

---

## 1. Business and regulatory question

> Do applicants from different racial/ethnic/sex groups receive approvals at different rates, and are those gaps explained by legitimate credit factors, or do they remain after controlling for them?

**Why a lender cares:** Fair lending laws (ECOA and Regulation B, the Fair Housing Act) prohibit discrimination on protected bases. Regulators and internal compliance teams test for two things:

- **Disparate treatment:** explicit differential treatment of protected groups.
- **Disparate impact:** a facially neutral policy or model that disproportionately harms a protected group without a sufficient business justification, or where a less discriminatory alternative exists.

This project is a **statistical screening analysis**, the first thing a compliance or analytics team runs. It cannot prove discrimination (see Limitations).

**WRITE (1 paragraph):** In your own words, why a Risk & Analytics team partners with Legal and Compliance on this. Tie it to your law interest and to how ML models can inherit bias through proxies.

---

## 2. Data

| Item | Detail |
|---|---|
| Source | HMDA (Home Mortgage Disclosure Act) public loan-level data, via the CFPB/FFIEC Data Browser (ffiec.cfpb.gov/data-browser) |
| Scope suggestion | **One recent year, one or a few states, home-purchase loans, first-lien, single-family, owner-occupied** (keeps it a manageable size and comparable) |
| Grain | One row per application |
| Download | Use the Data Browser filters to export CSV, or the bulk files. Document the exact filters in `data/README.md` |

### Key fields to use (verify names against the data dictionary for your year)

| Field | Use |
|---|---|
| `action_taken` | Outcome. `1` = originated, `3` = denied (approved-not-accepted `2`, withdrawn `4`, incomplete `5`, purchased `6`, preapproval `7/8` need decisions) |
| `derived_race`, `derived_ethnicity`, `derived_sex` | Protected-class groupings (HMDA provides derived fields combining applicant/co-applicant) |
| `applicant_age` | Age is also a protected basis (with limits) |
| `income` | Applicant income ($ thousands) |
| `loan_amount`, `property_value` | Size and collateral |
| `combined_loan_to_value_ratio`, `debt_to_income_ratio` | Core underwriting factors |
| `loan_type`, `loan_purpose`, `lien_status` | Filters / controls (conventional vs FHA/VA/USDA) |
| `census_tract`, `county_code`, `state_code` | Geography (proxy risk; handle carefully) |
| `denial_reason-1` | Lender-reported denial reasons, useful for descriptive analysis |

**Cleaning gotchas (write these in the README, they show real data experience):**
- Many numeric fields arrive as strings with values like `"Exempt"` or ranges like `"20%-<30%"` (DTI is often banded). Parse deliberately and document how.
- Decide how to treat `action_taken` values other than 1 and 3. A common, defensible choice: **keep only 1 (originated) and 3 (denied)** for approval-rate analysis, and state that you did so.
- "Race not provided" / "Not applicable" / "Joint" categories need explicit handling. Don't silently drop them; report counts.

---

## 3. Method (how it works, step by step)

```
HMDA CSV ──► clean.py ──► analysis table
                              │
      ┌───────────────────────┼─────────────────────────────┐
      ▼                       ▼                             ▼
 1. Raw disparities    2. Controlled regression     3. Proxy analysis
 (approval rates,      (logit w/ underwriting       (which features predict
  AIR)                  controls; group effect)      group membership?)
      │                       │                             │
      └───────────────┬───────┴─────────────────────────────┘
                      ▼
              4. Summary + memo
```

### Step 1: Raw approval-rate disparities and the Adverse Impact Ratio

```python
tbl = (df.groupby("derived_race")
         .agg(apps=("approved","size"), approvals=("approved","sum")))
tbl["approval_rate"] = tbl.approvals / tbl.apps

ref_rate = tbl.loc["White", "approval_rate"]       # choose and justify the reference group
tbl["AIR"] = tbl.approval_rate / ref_rate           # adverse impact ratio
tbl["flag_4_5ths"] = tbl.AIR < 0.80
```

- **AIR** = group approval rate ÷ reference group approval rate.
- The **four-fifths (80%) rule** is a screening heuristic from employment law (EEOC). It's commonly borrowed as a rough flag in lending analyses. It is **not** a legal standard for credit; say so in the README.
- Add a **significance test** so small groups aren't over-read: two-proportion z-test or Fisher's exact test per group vs. reference. Report confidence intervals on the rate difference, not only p-values.

```python
from statsmodels.stats.proportion import proportions_ztest
stat, p = proportions_ztest([a_group, a_ref], [n_group, n_ref])
```

**Chart:** horizontal bar chart of approval rate by group with a vertical line at 80% of the reference rate.

### Step 2: Controlled regression

**Goal:** see whether the gap remains after accounting for legitimate, applicant-level credit factors available in HMDA.

```python
import statsmodels.formula.api as smf

m0 = smf.logit("approved ~ C(derived_race, Treatment('White'))", data=df).fit()

m1 = smf.logit("""
    approved ~ C(derived_race, Treatment('White'))
             + np.log(income) + np.log(loan_amount)
             + ltv + dti_band + C(loan_type)
             + C(state_code)
""", data=df).fit()
```

Interpretation guide (write this in your own words in the README):

- **Coefficients are log-odds.** Exponentiate for **odds ratios**: `np.exp(m1.params)`.
- Better for communication: **average marginal effects**: "holding controls fixed, being in group X is associated with a Y percentage-point lower approval probability." Use `m1.get_margeff().summary()`.
- **Compare m0 → m1.** If the raw gap shrinks a lot after controls, much of the disparity is associated with income, LTV, DTI, etc. If a meaningful gap remains, that's the screening signal for deeper review.

**Important nuance to state:** controls like income and LTV may *themselves* reflect historical inequality (e.g., wealth gaps, appraisal differences). A "controlled" gap being small doesn't mean the system is fair, only that it's explained by the measured factors.

**Extension (optional but strong):** run separately by loan type (conventional vs FHA), by state, or by income band to find where the gaps concentrate. Mention the multiple-comparisons issue if you slice a lot.

### Step 3: Proxy analysis (the ML-bias angle)

**Question:** which seemingly neutral features carry information about protected-class membership? If a model uses them, it may reproduce disparities even without seeing race.

Two simple approaches:

**(a) Predict the protected attribute from the "neutral" features.** If features can predict group membership well, they act as proxies.

```python
# binary: group X vs reference, using ONLY features a model might use
X = df[["income","loan_amount","ltv","dti_num","census_tract_minority_pct", ...]]
y = (df.derived_race == "Black or African American").astype(int)

clf = GradientBoostingClassifier().fit(X_train, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
```

An AUC well above 0.5 means those features leak group information. Then check **permutation importance** to see which feature is doing it. Geography (tract/county, tract demographic percentages) is usually the strongest proxy; show it.

**(b) Ablation on the approval model.** Train an approval model with and without the suspicious proxy feature, then compare (i) predictive accuracy and (ii) the AIR for each group. If dropping the proxy narrows the disparity with little loss in accuracy, that's a **less-discriminatory-alternative** style finding, which is the concept regulators care about.

```
model A: all features         → AUC 0.xx, min AIR 0.xx
model B: drop geography       → AUC 0.xx, min AIR 0.xx
```

Report the tradeoff table honestly even if the result is "no meaningful change."

### Step 4: Denial reasons (descriptive add-on)

Tabulate `denial_reason-1` by group among denied applications (DTI, credit history, collateral, insufficient cash...). A skew can generate hypotheses (e.g., more collateral-related denials in certain tracts), though it doesn't prove cause.

---

## 4. Results (fill with real numbers)

| Group | Applications | Approval rate | AIR vs reference | Adj. marginal effect (pp) |
|---|---|---|---|---|
| Reference | TODO | TODO | 1.00 | n/a |
| Group A | TODO | TODO | TODO | TODO |
| Group B | TODO | TODO | TODO | TODO |

**Summary template:**
> Raw approval rates differ by X pp between A and the reference group (AIR = Y). After controlling for income, loan size, LTV, DTI, loan type, and state, the gap falls to Z pp (95% CI: ...). Geographic features predict group membership with AUC = W, and removing them from an approval model changes minimum AIR from P to Q at a cost of R AUC points.

---

## 5. Limitations (essential here; this is where credibility comes from)

- **HMDA lacks credit scores** (and other underwriting variables like reserves, full credit history). Omitted-variable bias is real: a remaining gap may reflect unmeasured credit quality, and a vanished gap doesn't prove fairness. This is the most important caveat.
- **No lender identifiers in the analysis scope:** pooling across lenders hides lender-level variation. An extension is lender fixed effects using `lei`.
- Observational data. Associations, not proof of intent or causation.
- Four-fifths rule is a screening heuristic, not a legal test for lending.
- Group definitions are derived categories with "not provided" cases; results depend on those choices.
- This analysis is for education and portfolio purposes, not a compliance determination.

## 6. AI-assisted development note

**WRITE honestly.** Suggested checks: confirm group counts sum to total after filters; reproduce one AIR and one z-test by hand; check regression coefficients' signs make sense (higher income → higher approval); confirm train/test separation in the proxy model.

## 7. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data      # see data/README.md for the Data Browser filters used
make clean     # writes data/processed/analysis.parquet
make analyze   # runs steps 1-4, writes reports/
```

## 8. Repo layout

```
.
├── README.md
├── requirements.txt
├── Makefile
├── data/             # gitignored raw; README with exact filters
├── src/              # clean.py, disparities.py, regression.py, proxies.py
├── notebooks/        # exploration
└── reports/          # figures + memo.md
```

## 9. Checklist before publishing

- [ ] Filters and `action_taken` handling documented
- [ ] Reference group choice justified
- [ ] Regression coefficients also shown as marginal effects
- [ ] Proxy analysis includes the ablation table
- [ ] Omitted-credit-score limitation stated prominently
- [ ] Language is careful: "associated with," "screening flag," not "discrimination proven"
