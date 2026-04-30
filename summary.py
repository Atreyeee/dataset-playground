import pandas as pd
import numpy as np

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

def generate_insights(df: pd.DataFrame) -> list:
    """Generates automated insights based on missing values, skewness, and correlation."""
    insights = []
    
    # 1. High missing values (>20%)
    for col in df.columns:
        pct = df[col].isnull().mean()
        if pct > 0.20:
            insights.append(f"⚠️ **High Missing Values**: `{col}` is missing {pct*100:.1f}% of its data.")
            
    num_cols = df.select_dtypes(include=np.number).columns
    
    # 2. Skewed data
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) > 1.0:
            insights.append(f"📊 **Skewed Data**: `{col}` is highly skewed (score: {skew:.2f}).")
            
    # 3. Highly correlated features (>0.8)
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8:
                    insights.append(f"🔗 **High Correlation**: `{num_cols[i]}` & `{num_cols[j]}` are strongly correlated (r={val:.2f}).")
                    
    return insights
