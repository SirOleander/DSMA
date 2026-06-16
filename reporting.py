"""
Shared reporting helpers for the Yelp restaurant satisfaction project.

This module centralises the terminal-table formatting and the variable /
distribution / skewness overviews that were previously duplicated across
process_data.py and eda.py. It contains no project-specific data logic, only
generic, reusable diagnostics.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Printing primitives
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a clear top-level section header."""
    line = "=" * 100
    print("\n" + line)
    print(title.upper())
    print(line)


def print_subsection(title: str) -> None:
    """Print a clear subsection header."""
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def print_table(
    df: pd.DataFrame,
    title: str | None = None,
    max_rows: int = 50,
) -> None:
    """Print a DataFrame as a clean, left-aligned terminal table."""
    if title:
        print_subsection(title)

    if df is None or df.empty:
        print("No data available.")
        return

    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:,.3f}".format,
    ):
        print(df.head(max_rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Overviews
# ---------------------------------------------------------------------------

def build_variable_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Variable-level overview: dtype, missing values, unique values."""
    rows = [
        {
            "variable": col,
            "dtype": str(df[col].dtype),
            "missing_values": int(df[col].isna().sum()),
            "missing_percent": round(df[col].isna().mean() * 100, 2),
            "unique_values": int(df[col].nunique(dropna=True)),
        }
        for col in df.columns
    ]

    return pd.DataFrame(rows)


def build_numeric_distribution_overview(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Numeric distribution / skewness overview for original numeric variables.

    Intended to run before log features are created, to decide which variables
    benefit from a log transform. log_* columns and any excluded columns are
    skipped.
    """
    exclude_columns = set(exclude_columns or [])

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_columns
        and not col.startswith("log_")
    ]

    if not numeric_cols:
        return pd.DataFrame()

    rows = []

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if values.empty:
            continue

        skewness = values.skew()

        if abs(skewness) < 0.5:
            skew_label = "approximately symmetric"
        elif abs(skewness) < 1:
            skew_label = "moderately skewed"
        else:
            skew_label = "highly skewed"

        non_negative = values.min() >= 0

        if skewness > 1 and non_negative:
            suggested_action = "consider log1p"
        elif skewness > 1 and not non_negative:
            suggested_action = "high skew, but log1p not directly suitable"
        elif skewness < -1:
            suggested_action = "left-skewed; log1p usually not useful"
        else:
            suggested_action = "no log transform likely needed"

        rows.append({
            "variable": col,
            "count": values.shape[0],
            "missing_values": int(df[col].isna().sum()),
            "missing_percent": round(df[col].isna().mean() * 100, 2),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p01": values.quantile(0.01),
            "p05": values.quantile(0.05),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
            "max": values.max(),
            "zero_count": int((values == 0).sum()),
            "zero_percent": round((values == 0).mean() * 100, 2),
            "skewness": skewness,
            "abs_skewness": abs(skewness),
            "skew_label": skew_label,
            "suggested_action": suggested_action,
        })

    overview = pd.DataFrame(rows)

    if overview.empty:
        return overview

    return overview.sort_values("abs_skewness", ascending=False)


def build_negative_value_check(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Check for negative values in variables that should be non-negative."""
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        negative_count = int((values < 0).sum())

        rows.append({
            "variable": col,
            "negative_count": negative_count,
            "negative_percent": round(negative_count / len(df) * 100, 4),
            "min_value": values.min(),
        })

    check = pd.DataFrame(rows)

    if check.empty:
        return check

    return check.sort_values("negative_count", ascending=False)


def build_log_skewness_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare skewness before and after log1p transformation.

    Pairs each log_<x> column with its original <x> column. Intended to run
    after the log features have been created.
    """
    rows = []

    log_cols = [col for col in df.columns if col.startswith("log_")]

    for log_col in log_cols:
        original_col = log_col.replace("log_", "", 1)

        if original_col not in df.columns:
            continue

        original_values = pd.to_numeric(df[original_col], errors="coerce").dropna()
        log_values = pd.to_numeric(df[log_col], errors="coerce").dropna()

        if original_values.empty or log_values.empty:
            continue

        original_skewness = original_values.skew()
        log_skewness = log_values.skew()

        rows.append({
            "original_variable": original_col,
            "log_variable": log_col,
            "original_skewness": original_skewness,
            "log_skewness": log_skewness,
            "absolute_original_skewness": abs(original_skewness),
            "absolute_log_skewness": abs(log_skewness),
            "skewness_reduction": abs(original_skewness) - abs(log_skewness),
            "skewness_reduction_percent": (
                (abs(original_skewness) - abs(log_skewness))
                / abs(original_skewness)
                * 100
                if abs(original_skewness) > 0
                else np.nan
            ),
            "original_min": original_values.min(),
            "original_median": original_values.median(),
            "original_p99": original_values.quantile(0.99),
            "original_max": original_values.max(),
            "log_min": log_values.min(),
            "log_median": log_values.median(),
            "log_p99": log_values.quantile(0.99),
            "log_max": log_values.max(),
        })

    comparison = pd.DataFrame(rows)

    if comparison.empty:
        return comparison

    comparison["skewness_reduction_percent"] = (
        comparison["skewness_reduction_percent"].round(2)
    )

    return comparison.sort_values("skewness_reduction", ascending=False)
