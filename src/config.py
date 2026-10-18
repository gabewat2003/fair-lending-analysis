"""Project-wide constants: scope, filters, paths, groups, feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYSIS_PATH = PROCESSED_DIR / "analysis.parquet"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

YEAR = 2023
STATES = ["GA", "IL", "TX", "MD"]

# The Data Browser API accepts at most two HMDA filter criteria per request and
# silently ignored occupancy_types in testing, so only loan purpose is applied
# server-side; every other scope filter is applied (and logged) in clean.py.
API_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
API_FILTERS = {"loan_purposes": "1"}  # home purchase
LOCAL_FILTERS = {
    "loan_purpose": {"1"},                     # home purchase (re-checked)
    "occupancy_type": {"1"},                   # principal residence
    "lien_status": {"1"},                      # first lien
    "construction_method": {"1"},              # site-built
    "total_units": {"1", "2", "3", "4"},       # 1-4 unit single family
    "business_or_commercial_purpose": {"2"},   # not business purpose
    "reverse_mortgage": {"2"},                 # not reverse mortgage
    "open-end_line_of_credit": {"2"},          # closed-end
}

# Outcome: lender approved (1 originated, 2 approved-not-accepted) vs denied (3).
APPROVED_CODES = {"1", "2"}
DENIED_CODES = {"3"}

REF_RACE = "White"
REF_ETHNICITY = "Not Hispanic or Latino"
REF_SEX = "Male"
REF_AGE = "35-44"

MIN_GROUP_N = 500   # below this a group is flagged "small_n" (estimates unstable)
SEED = 42

APPLICANT_FEATURES = ["log_income", "log_loan_amount", "log_property_value",
                      "ltv", "dti_num", "loan_type_code"]
GEO_FEATURES = ["tract_minority_population_percent", "tract_to_msa_income_percentage",
                "ffiec_msa_md_median_family_income", "tract_owner_occupied_units",
                "tract_median_age_of_housing_units", "state_num"]
