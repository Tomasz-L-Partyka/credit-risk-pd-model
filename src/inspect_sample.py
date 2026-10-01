from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sf-loan-performance-data-sample.csv"
)

data = pd.read_csv(DATA_PATH, sep="|", header=None, low_memory=False)

print(f"Number of rows: {data.shape[0]:,}")
print(f"Number of columns: {data.shape[1]:,}")

data = data.rename(
    columns={
        1: "loan_id",
        2: "reporting_period",
        15: "loan_age",
        39: "delinquency_status",
    }
)


data["reporting_period"] = pd.to_datetime(
    data["reporting_period"].astype("string").str.zfill(6),
    format="%m%Y",
)

print(f"Number of unique loans: {data['loan_id'].nunique():,}")
print(f"First reporting month: {data['reporting_period'].min():%Y-%m}")
print(f"Last reporting month: {data['reporting_period'].max():%Y-%m}")

delinquency_status = (
    data["delinquency_status"]
    .astype("string")
    .str.zfill(2)
)

print("\nDelinquency status counts:")
print(delinquency_status.value_counts(dropna=False).sort_index())

delinquency_numeric = pd.to_numeric(
    delinquency_status,
    errors="coerce",
)
serious_delinquency = (
    delinquency_numeric
    .ge(3)
    .fillna(False)
)
print("\nEver-serious-delinquency counts by loan:")
ever_serious_delinquency_by_loan = (
    serious_delinquency
    .groupby(data["loan_id"])
    .any()
    .astype("int8")
    .rename("ever_serious_delinquency")
)
print(ever_serious_delinquency_by_loan.value_counts().sort_index())

print(
    data[
        [
            "loan_id",
            "reporting_period",
            "loan_age",
            "delinquency_status",
        ]
    ].head(12)
)