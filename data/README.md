# Data

Raw and processed files are gitignored. `make data` recreates `data/raw/`, and `make clean` builds `data/processed/analysis.parquet` from it.

## Source

HMDA Snapshot National Loan-Level Dataset, **2023**, from the CFPB/FFIEC Data Browser API. It was downloaded on 2026-10-18.

- Endpoint: `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv`
- Request, one per state: `?states={GA|IL|TX|MD}&years=2023&loan_purposes=1`
- Data dictionary: https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
- API docs: https://ffiec.cfpb.gov/documentation/api/data-browser/

| File | Rows | Size |
|---|---:|---:|
| `raw/hmda_2023_GA.csv` | 246,651 | 94.6 MB |
| `raw/hmda_2023_IL.csv` | 223,024 | 85.6 MB |
| `raw/hmda_2023_TX.csv` | 740,889 | 281.5 MB |
| `raw/hmda_2023_MD.csv` | 115,018 | 44.4 MB |

### Why only one filter is applied in the API request

The Data Browser accepts **at most two** HMDA filter criteria per request; `states` and `years` don't count toward that limit. When asked for more, it returns `provide-two-or-less-filter-criteria`. A request that also passed `occupancy_types=1` returned the *same file* as the purpose-only query, which means that parameter was silently ignored. So the API request filters only on loan purpose, and every other scope filter is applied locally in `src/clean.py` and logged.

## Filters applied in `src/clean.py`

Filters are listed in order. Row counts come from `reports/tables/cleaning_log.csv`.

| Step | Field = kept values | Rows remaining | Dropped |
|---|---|---:|---:|
| Raw download (home purchase, 4 states) | — | 1,325,582 | — |
| Home purchase | `loan_purpose = 1` | 1,325,582 | 0 |
| Principal residence | `occupancy_type = 1` | 1,214,896 | 110,686 |
| First lien | `lien_status = 1` | 1,174,317 | 40,579 |
| Site-built (excludes manufactured) | `construction_method = 1` | 1,051,991 | 122,326 |
| 1–4 units | `total_units ∈ {1,2,3,4}` | 1,051,980 | 11 |
| Not business purpose | `business_or_commercial_purpose = 2` | 1,035,246 | 16,734 |
| Not reverse mortgage | `reverse_mortgage = 2` | 1,034,841 | 405 |
| Closed-end | `open-end_line_of_credit = 2` | 1,031,850 | 2,991 |
| Lender decision only | `action_taken ∈ {1,2,3}` | **635,352** | 396,498 |

The `action_taken` mix before the last step is in `reports/tables/action_taken_counts.csv`:

| action_taken | Applications |
|---|---:|
| 1 Originated | 553,589 |
| 2 Approved, not accepted | 18,335 |
| 3 Denied | 63,428 |
| 4 Withdrawn | 153,940 |
| 5 File closed incomplete | 18,659 |
| 6 Purchased loan | 217,007 |
| 7 Preapproval denied | 2,287 |
| 8 Preapproval approved, not accepted | 4,605 |

Codes 4 and 5 have no credit decision. Code 6 is a loan bought from another institution, so the purchaser made no underwriting decision. Codes 7 and 8 are preapproval requests, not applications for a specific property.

## Field notes

- `loan_to_value_ratio` in Data Browser files is the **combined** LTV (`combined_loan_to_value_ratio` in the LAR spec).
- Numeric fields are strings and can hold `NA` or `Exempt`. Both are parsed to missing.
- `debt_to_income_ratio` is banded below 36% and above 49%, and exact for 36–49. See the main README for the parsing rules.
- `applicant_age = 8888` means not applicable or not provided, and is relabeled "Not available".
