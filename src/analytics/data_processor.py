import pandas as pd
from typing import Dict, Any

class DataProcessor:
    def summarize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)
        complete_summary = {
            "rows": total,
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        # Decide which columns to summarise — no copy of df
        columns_to_summarise = []
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count > max(10, 0.5 * total):
                continue
            if df[col].dtype == object:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if pd.notna(avg_len) and avg_len > 100:
                    continue
            columns_to_summarise.append(col)

        column_summary = {}
        for col in columns_to_summarise:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                column_summary[col] = {
                    "mean":   float(series.mean()),
                    "median": float(series.median()),
                    "min":    float(series.min()),
                    "max":    float(series.max()),
                    "std":    float(series.std()),
                }
            else:
                column_summary[col] = {
                    "top_values": series.value_counts().head(50).to_dict(),
                    "unique": int(series.nunique()),
                }

        return {
            "dataset_info": complete_summary,
            "column_details": column_summary,
        }