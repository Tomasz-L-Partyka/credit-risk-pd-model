from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "2015Q1.zip"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets_2015Q1.csv"
)

CHUNK_SIZE = 500_000
TARGET_WINDOWS = (12, 24, 36)

SELECTED_COLUMNS = {
    1: "loan_id",
    2: "reporting_period",
    39: "delinquency_status",
}

COLUMN_TYPES = {
    1: "string",
    2: "string",
    39: "string",
}

def create_chunk_reader():
    return pd.read_csv(
        ZIP_PATH,
        compression="zip",
        sep="|",
        header=None,
        usecols=SELECTED_COLUMNS.keys(),
        dtype=COLUMN_TYPES,
        chunksize=CHUNK_SIZE,
    )

first_month_by_loan = {}

for chunk_number, chunk in enumerate(create_chunk_reader(),start=1,):
    chunk = chunk.rename(columns=SELECTED_COLUMNS)

    chunk["reporting_period"] = pd.to_datetime(
        chunk["reporting_period"].str.zfill(6),
        format="%m%Y",
    )

    chunk_first_month = (
        chunk.groupby("loan_id")["reporting_period"].min()
    )

    for loan_id, first_month in chunk_first_month.items():
        previous_first_month = first_month_by_loan.get(loan_id)

        if (
            previous_first_month is None
            or first_month < previous_first_month
        ):
            first_month_by_loan[loan_id] = first_month

    if chunk_number % 10 == 0:
        print(
            f"First pass: processed chunk {chunk_number}, "
            f"loans found: {len(first_month_by_loan):,}"
        )

print(f"Loans with first month: {len(first_month_by_loan):,}")

serious_delinquency_loans = {
    12: set(),
    24: set(),
    36: set(),
}

for chunk_number, chunk in enumerate(
    create_chunk_reader(),
    start=1,
):
    chunk = chunk.rename(columns=SELECTED_COLUMNS)

    chunk["reporting_period"] = pd.to_datetime(
        chunk["reporting_period"].str.zfill(6),
        format="%m%Y",
    )

    chunk["first_month"] = (
        chunk["loan_id"].map(first_month_by_loan)
    )
    chunk["months_from_start"] = (
            (
                    chunk["reporting_period"].dt.year
                    - chunk["first_month"].dt.year
            )
            * 12
            + (
                    chunk["reporting_period"].dt.month
                    - chunk["first_month"].dt.month
            )
    )
    chunk["delinquency_numeric"] = pd.to_numeric(
        chunk["delinquency_status"],
        errors="coerce",
    )

    chunk["serious_delinquency"] = (
        chunk["delinquency_numeric"]
        .ge(3)
        .fillna(False)
    )

    for window in TARGET_WINDOWS:
        in_window = (
                chunk["months_from_start"].ge(0)
                & chunk["months_from_start"].lt(window)
        )

        selected_loans = chunk.loc[
            in_window & chunk["serious_delinquency"],
            "loan_id",
        ]

        serious_delinquency_loans[window].update(
            selected_loans.dropna().unique()
        )
    if chunk_number % 10 == 0:
        print(f"Second pass: processed chunk {chunk_number}")

total_loans = len(first_month_by_loan)

for window in TARGET_WINDOWS:
    defaults = len(serious_delinquency_loans[window])
    default_rate = defaults / total_loans

    print(
        f"{window} months: "
        f"{defaults:,} defaults, "
        f"default rate = {default_rate:.2%}"
    )

target_data = pd.DataFrame(
    {"loan_id": list(first_month_by_loan)}
)

target_data["target_24m"] = (
    target_data["loan_id"]
    .isin(serious_delinquency_loans[24])
    .astype("int8")
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

target_data.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"Saved target table: {OUTPUT_PATH}")