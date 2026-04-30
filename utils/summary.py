import pandas as pd


def get_summary(df: pd.DataFrame) -> dict:
    """Return a rich summary dictionary for the given dataframe."""
    missing = df.isnull().sum()
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes,
        "missing": missing,
        "missing_pct": (missing / len(df) * 100).round(2),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
    }
