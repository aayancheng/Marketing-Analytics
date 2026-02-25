"""
fetch_data.py
-------------
Generates a realistic synthetic version of the IBM Telco Customer Churn dataset.

Real dataset reference: Kaggle blastchar/telco-customer-churn
Target: 7043 rows, exact column schema, realistic churn correlations.
Intercept calibrated (INTERCEPT=-1.30) to yield ~26.5% churn rate.
random_state = 42
"""

import os
import random
import string
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 7043
INTERCEPT = -1.30  # calibrated: yields 26.66% churn (~26.5% target)
OUTPUT_PATH = "/Users/aayan/MarketingAnalytics/m03_churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

rng = np.random.default_rng(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. customerID  -- format DDDD-AAAAA
# ---------------------------------------------------------------------------
def _make_customer_ids(n):
    ids = set()
    while len(ids) < n:
        d = "".join(random.choices(string.digits, k=4))
        a = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        ids.add(f"{d}-{a}")
    return list(ids)


# ---------------------------------------------------------------------------
# 2. Demographics
# ---------------------------------------------------------------------------
customer_ids   = _make_customer_ids(N_ROWS)
gender         = rng.choice(["Male", "Female"], size=N_ROWS, p=[0.505, 0.495])
senior_citizen = rng.choice([0, 1], size=N_ROWS, p=[0.84, 0.16])
partner        = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.48, 0.52])
dependents     = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.30, 0.70])
tenure         = np.clip(np.round(rng.beta(2.0, 3.0, N_ROWS) * 72).astype(int), 0, 72)

# ---------------------------------------------------------------------------
# 3. Services
# ---------------------------------------------------------------------------
phone_service    = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.90, 0.10])
multiple_lines   = np.where(
    phone_service == "No", "No phone service",
    rng.choice(["Yes", "No"], size=N_ROWS, p=[0.50, 0.50]),
)
internet_service = rng.choice(
    ["DSL", "Fiber optic", "No"], size=N_ROWS, p=[0.44, 0.44, 0.12]
)


def _internet_dep(svc_arr, p=0.50):
    return np.array([
        "No internet service" if s == "No"
        else rng.choice(["Yes", "No"], p=[p, 1 - p])
        for s in svc_arr
    ])


online_security   = _internet_dep(internet_service, 0.47)
online_backup     = _internet_dep(internet_service, 0.48)
device_protection = _internet_dep(internet_service, 0.48)
tech_support      = _internet_dep(internet_service, 0.47)
streaming_tv      = _internet_dep(internet_service, 0.50)
streaming_movies  = _internet_dep(internet_service, 0.50)

# ---------------------------------------------------------------------------
# 4. Contract / billing
# ---------------------------------------------------------------------------
contract = rng.choice(
    ["Month-to-month", "One year", "Two year"],
    size=N_ROWS, p=[0.55, 0.21, 0.24],
)
paperless_billing = rng.choice(["Yes", "No"], size=N_ROWS, p=[0.59, 0.41])
payment_method    = rng.choice(
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"],
    size=N_ROWS, p=[0.336, 0.228, 0.218, 0.218],
)

# ---------------------------------------------------------------------------
# 5. Charges
# ---------------------------------------------------------------------------
base_charge     = np.where(internet_service == "No", 20.0,
                           np.where(internet_service == "DSL", 50.0, 75.0))
monthly_charges = np.clip(
    np.round(base_charge + rng.normal(0, 15, N_ROWS), 2), 18.25, 118.75
)
total_charges = np.array([
    " " if tenure[i] == 0
    else str(round(max(monthly_charges[i] * tenure[i]
                       + rng.normal(0, tenure[i] * 0.5), 18.25), 2))
    for i in range(N_ROWS)
])

# ---------------------------------------------------------------------------
# 6. Churn -- logistic model with calibrated intercept
# ---------------------------------------------------------------------------
log_odds = (
    np.where(contract == "Month-to-month",  1.20, 0.0)
  + np.where(contract == "Two year",        -1.80, 0.0)
  + np.where(contract == "One year",        -0.40, 0.0)
  + np.where(internet_service == "Fiber optic",  0.70, 0.0)
  + np.where(internet_service == "No",           -0.50, 0.0)
  - tenure / 36.0
  + (monthly_charges - 64.76) / 60.0
  + np.where(payment_method == "Electronic check", 0.55, 0.0)
  + np.where(tech_support == "No",                 0.30, 0.0)
  + np.where(tech_support == "No internet service", -0.20, 0.0)
  + INTERCEPT
  + rng.normal(0, 0.25, N_ROWS)
)
churn = np.where(rng.random(N_ROWS) < 1.0 / (1.0 + np.exp(-log_odds)), "Yes", "No")

# ---------------------------------------------------------------------------
# 7. Assemble DataFrame
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "customerID":       customer_ids,
    "gender":           gender,
    "SeniorCitizen":    senior_citizen,
    "Partner":          partner,
    "Dependents":       dependents,
    "tenure":           tenure,
    "PhoneService":     phone_service,
    "MultipleLines":    multiple_lines,
    "InternetService":  internet_service,
    "OnlineSecurity":   online_security,
    "OnlineBackup":     online_backup,
    "DeviceProtection": device_protection,
    "TechSupport":      tech_support,
    "StreamingTV":      streaming_tv,
    "StreamingMovies":  streaming_movies,
    "Contract":         contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod":    payment_method,
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     total_charges,
    "Churn":            churn,
})

# ---------------------------------------------------------------------------
# 8. Write CSV
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------
sz  = os.path.getsize(OUTPUT_PATH) / 1024
cr  = (df["Churn"] == "Yes").mean() * 100
blk = (df["TotalCharges"].str.strip() == "").sum()
z_t = (df["tenure"] == 0).sum()

print("=" * 60)
print("Dataset generation complete")
print("=" * 60)
print(f"File path   : {OUTPUT_PATH}")
print(f"File size   : {sz:.1f} KB")
print(f"Rows        : {len(df):,}")
print(f"Columns     : {len(df.columns)}")
print(f"Column list : {list(df.columns)}")
print(f"Churn rate  : {cr:.2f}%")
print(f"Blank TotalCharges (tenure=0): {blk}")
print(f"Tenure=0 rows               : {z_t}")
print()
print("MonthlyCharges stats:")
print(df["MonthlyCharges"].describe().to_string())
print()
print("Churn by contract type:")
print(df.groupby("Contract")["Churn"].apply(lambda x: (x=="Yes").mean()).round(3).to_string())
print()
print("Churn by InternetService:")
print(df.groupby("InternetService")["Churn"].apply(lambda x: (x=="Yes").mean()).round(3).to_string())
print()
print("Churn by PaymentMethod:")
print(df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x=="Yes").mean()).round(3).to_string())
