## Main results (race)

| Group                                     |   Applications | Approval rate   |   AIR vs White |   Raw diff (pp) | m1 adj. AME (pp) [95% CI]   | m2 lender-FE (pp)   |
|:------------------------------------------|---------------:|:----------------|---------------:|----------------:|:----------------------------|:--------------------|
| White                                     |        362,980 | 92.2%           |          1     |             0   | n/a (reference)             | n/a                 |
| Race Not Available                        |         97,328 | 87.6%           |          0.95  |            -4.7 | -2.5 [-3.2, -1.7]           | -1.9                |
| Black or African American                 |         88,848 | 83.4%           |          0.904 |            -8.8 | -5.9 [-6.8, -5.0]           | -4.9                |
| Asian                                     |         61,973 | 90.4%           |          0.98  |            -1.8 | -3.3 [-4.0, -2.6]           | -2.1                |
| Joint                                     |         17,130 | 92.5%           |          1.002 |             0.2 | -0.5 [-1.1, +0.1]           | -0.4                |
| American Indian or Alaska Native          |          4,007 | 85.7%           |          0.93  |            -6.5 | -2.8 [-3.8, -1.9]           | -2.8                |
| 2 or more minority races                  |          1,951 | 81.9%           |          0.888 |           -10.3 | -6.2 [-8.5, -3.9]           | -5.1                |
| Native Hawaiian or Other Pacific Islander |            982 | 84.8%           |          0.92  |            -7.4 | -3.6 [-5.2, -2.0]           | -3.5                |
| Free Form Text Only (small n)             |            153 | 79.7%           |          0.865 |           -12.5 | -2.8 [-8.9, +3.3]           | -0.2                |


## Cleaning log

| step                                           |    rows |   dropped |
|:-----------------------------------------------|--------:|----------:|
| raw download (home purchase, 4 states)         | 1325582 |         0 |
| filter loan_purpose in ['1']                   | 1325582 |         0 |
| filter occupancy_type in ['1']                 | 1214896 |    110686 |
| filter lien_status in ['1']                    | 1174317 |     40579 |
| filter construction_method in ['1']            | 1051991 |    122326 |
| filter total_units in ['1', '2', '3', '4']     | 1051980 |        11 |
| filter business_or_commercial_purpose in ['2'] | 1035246 |     16734 |
| filter reverse_mortgage in ['2']               | 1034841 |       405 |
| filter open-end_line_of_credit in ['2']        | 1031850 |      2991 |
| keep action_taken 1,2 (approved) or 3 (denied) |  635352 |    396498 |


## Proxy AUC

| feature_set         |   Asian vs White |   Black vs White |   Hispanic vs Not Hispanic |
|:--------------------|-----------------:|-----------------:|---------------------------:|
| all features        |            0.85  |            0.856 |                      0.813 |
| applicant/loan only |            0.762 |            0.72  |                      0.715 |
| geography only      |            0.83  |            0.831 |                      0.788 |


## Ablation

| model                                | kind     |   test_auc |   AIR Black or African American |   AIR Asian |   AIR American Indian or Alaska Native |   AIR 2 or more minority races |   min_AIR |   approval_rate_pred |   noise_sd_logodds |
|:-------------------------------------|:---------|-----------:|--------------------------------:|------------:|---------------------------------------:|-------------------------------:|----------:|---------------------:|-------------------:|
| Observed outcomes (test set)         | observed |   nan      |                          0.9091 |      0.9816 |                                 0.9204 |                         0.8888 |    0.8888 |             nan      |           nan      |
| A: all features                      | main     |     0.8105 |                          0.9023 |      0.9937 |                                 0.9397 |                         0.9095 |    0.9023 |               0.9011 |           nan      |
| B: drop all geography                | main     |     0.7997 |                          0.9288 |      0.9998 |                                 0.9485 |                         0.923  |    0.923  |               0.9005 |           nan      |
| C: drop tract minority %             | main     |     0.8097 |                          0.9083 |      0.9948 |                                 0.9423 |                         0.9169 |    0.9083 |               0.9009 |           nan      |
| D: drop log_income                   | control  |     0.8071 |                          0.9027 |      0.9962 |                                 0.9401 |                         0.9146 |    0.9027 |               0.9004 |           nan      |
| D: drop log_loan_amount              | control  |     0.8073 |                          0.9004 |      0.9939 |                                 0.9406 |                         0.9122 |    0.9004 |               0.9001 |           nan      |
| D: drop log_property_value           | control  |     0.7978 |                          0.8962 |      0.997  |                                 0.9354 |                         0.9194 |    0.8962 |               0.9009 |           nan      |
| D: drop ltv                          | control  |     0.8036 |                          0.906  |      0.9947 |                                 0.9367 |                         0.9102 |    0.906  |               0.9012 |           nan      |
| D: drop dti_num                      | control  |     0.7463 |                          0.8919 |      0.9886 |                                 0.9342 |                         0.9221 |    0.8919 |               0.9025 |           nan      |
| D: drop loan_type_code               | control  |     0.8018 |                          0.9072 |      0.9943 |                                 0.9456 |                         0.8954 |    0.8954 |               0.9    |           nan      |
| E: all features + random score noise | control  |     0.7997 |                          0.9051 |      0.9964 |                                 0.948  |                         0.9084 |    0.9051 |               0.901  |             0.3326 |

