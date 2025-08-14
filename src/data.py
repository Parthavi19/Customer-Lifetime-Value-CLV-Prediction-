import pandas as pd
from .config import CSV_PATH, DATE_COL, CUST_COL, QTY_COL, TOTAL_COL

def load_and_clean():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, CUST_COL])
    df[CUST_COL] = df[CUST_COL].astype(str)

    # Basic sanity filters
    if QTY_COL in df.columns:
        df = df[df[QTY_COL] > 0]
    if TOTAL_COL in df.columns:
        df = df[df[TOTAL_COL] >= 0]

    return df
