from __future__ import annotations
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def basic_profile(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_stats = {}
    if numeric_cols:

        desc = df[numeric_cols].describe().round(2).to_dict()
        numeric_stats = desc

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "columns": df.columns.tolist(),
        "numeric_stats": numeric_stats,
    }

def sample_rows_as_dicts(df: pd.DataFrame, n: int = 8) -> list[dict]:
    # JSON-friendly sample
    return df.head(n).fillna("").to_dict(orient="records")
