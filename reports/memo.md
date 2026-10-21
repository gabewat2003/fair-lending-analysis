# Memo: Fair lending screening results, 2023 HMDA home-purchase applications (GA, IL, TX, MD)

**To:** Risk & Analytics / Fair Lending Compliance
**Re:** Statistical screen for approval-rate disparities by race and ethnicity
**Status:** Screening analysis on public data, for education and portfolio use. Not a compliance determination.

## Purpose

We ran a first-pass screen on 635,352 lender decisions on first-lien, owner-occupied, site-built home-purchase applications. The screen asks whether approval rates differ by race and ethnicity, and whether any gaps remain after controlling for the underwriting factors HMDA reports. It also checks whether facially neutral features act as proxies for group membership.

## Key findings

1. **Raw gap.** The approval rate was 92.2% for White applicants and 83.4% for Black applicants. That is a gap of 8.8 percentage points (pp), with an adverse impact ratio (AIR) of 0.904. No group with 500 or more applications fell below the four-fifths (0.80) screening threshold. The gaps are still statistically significant after Holm correction, using standard errors clustered by lender (Black vs. White clustered z = −10.6). Black applicants have the largest gap among the larger groups. The same gap shows up in each of the four states, with raw differences from −7.9 pp (TX) to −12.2 pp (IL).
2. **After controls.** Controlling for income, loan amount, combined LTV, DTI band, loan type and state cuts the Black–White gap to **−5.9 pp (95% CI −6.8 to −5.0)**. Adding lender fixed effects (1,399 lenders) cuts it further, to **−4.9 pp**. About half the raw gap is associated with the measured factors and with which lender the applicant went to. A meaningful share remains. Asian applicants show a larger gap after controls (−3.3 pp) than before (−1.8 pp), because they have stronger measured profiles on average. Hispanic ethnicity has a controlled odds ratio of 0.77.
3. **Proxies.** Tract and MSA-level geography alone predicts Black vs. White applicant race with a test AUC of **0.83**. Tract minority population share is the strongest single proxy.
4. **Less-discriminatory-alternative check.** A gradient-boosted approval model without race inputs, with the approval threshold set to match the observed approval rate, gives a Black AIR of 0.902. Removing the geographic features raises it to **0.929**, at a cost of **1.1 AUC points** (0.811 → 0.800). Removing only tract minority share changes little (0.908), because the other geographic fields carry the same information. The improvement is not just a side effect of a less accurate model. Removing property value instead costs slightly more accuracy (AUC 0.798) and leaves Black AIR at 0.896. Adding random noise to the full model until its AUC matches the geography-free model gives 0.905. Note that the model is trained on the observed lender decisions, which are what is under test, so "accuracy" means agreement with those decisions.
5. **Denial reasons.** DTI is the most common primary denial reason for every group. It makes up 43% of denials of Black applicants and 38% for White applicants. Credit history is cited somewhat more often for Black applicants (15% vs 13%) and collateral less often (11% vs 14%).

## Caveats

- **HMDA has no credit scores.** It also lacks reserves, full credit history and other underwriting data. Some or all of the remaining gap may reflect unmeasured credit quality. Equally, a small controlled gap would not show that the process is fair.
- The controls themselves (income, LTV, property value) may reflect historical inequities in wealth and appraisal.
- The data is observational, so it shows associations, not causes or intent. The four-fifths rule is an employment-law heuristic, not a legal standard for credit.

## Recommended next steps

1. Re-run the controlled model with the lender's own credit data: credit score, reserves and AUS findings.
2. Rank lender-level gaps from the fixed-effects model, and do a matched-pair file review for outlier lenders.
3. For any internal approval model, run the geography ablation as a standing less-discriminatory-alternative test, and document the accuracy/fairness tradeoff.
4. Look at the "Race Not Available" group (15% of applications, −2.5 pp after controls). Gaps in demographic data collection can hide disparities.
