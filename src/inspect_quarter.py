from pathlib import Path

import pandas as pd
CHUNK_SIZE = 100_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "2015Q1.zip"

COLUMN_TYPES = {
    1: "string",
    2: "string",
    15: "Int16",
    39: "string",
}
SELECTED_COLUMNS = {
    1: "loan_id",
    2: "reporting_period",
    15: "loan_age",
    39: "delinquency_status",
}

chunks = pd.read_csv(
    ZIP_PATH,
    compression="zip",
    sep="|",
    header=None,
    usecols=SELECTED_COLUMNS.keys(),
    chunksize=CHUNK_SIZE,
    dtype=COLUMN_TYPES,
)

total_rows = 0
loan_ids = set()

for chunk_number, chunk in enumerate(chunks, start=1):
    chunk = chunk.rename(columns=SELECTED_COLUMNS)
    loan_ids.update(chunk["loan_id"].dropna().unique())
    total_rows += len(chunk)

    if chunk_number % 10 == 0:
        print(f"Processed {total_rows:,} rows")


print(f"Total rows: {total_rows:,}")
print(f"Unique loans: {len(loan_ids):,}")