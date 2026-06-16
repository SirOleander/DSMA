"""
Compact Exploratory Data Analysis for the Yelp restaurant satisfaction project.

This script analyzes the final review-level Yelp restaurant dataset enriched
with NOAA weather data.

Design choices:
- Tables are printed to the terminal only.
- No CSV tables are saved.
- Plots are saved as PNG files.
- EDA focuses on report-relevant variables instead of all variables.
- Outliers are diagnosed, not automatically removed.
- No modeling, no train/test split, no learned preprocessing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import WEATHER_ENRICHED_PKL, PROCESSED_DATA_DIR, ID_COLS, LEAKAGE_COLS
from reporting import print_section, print_subsection, print_table


# =============================================================================
# Configuration
# =============================================================================

EDA_OUTPUT_DIR = PROCESSED_DATA_DIR / "eda_outputs"
PLOT_OUTPUT_DIR = EDA_OUTPUT_DIR / "plots"
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "satisfied"
# review_stars is the canonical leakage column for the stars distribution plot;
# LEAKAGE_COLS (from config) additionally covers business_stars and
# user_average_stars, which leak the target via all-time averages.
LEAKAGE_COL = "review_stars"

TOP_N = 15
MIN_CATEGORY_COUNT = 1_000
PLOT_SAMPLE_N = 500_000

# Focused variables for report-ready EDA
KEY_NUMERIC_COLS = [
    "review_text_length",
    "log_review_text_length",
    "business_stars",
    "business_review_count",
    "log_business_review_count",
    "user_average_stars",
    "user_review_count",
    "log_user_review_count",
    "checkin_count",
    "log_checkin_count",
    "tip_count",
    "log_tip_count",
    "photo_count",
    "log_photo_count",
    "weather_prcp",
    "weather_tmax",
    "weather_tmin",
    "weather_temp_range",
]

KEY_CATEGORICAL_COLS = [
    "state",
    "city",
    "is_open",
    "attributes.RestaurantsPriceRange2",
    "attributes.RestaurantsTakeOut",
    "attributes.RestaurantsDelivery",
    "attributes.OutdoorSeating",
    "attributes.RestaurantsReservations",
    "attributes.Alcohol",
    "attributes.NoiseLevel",
    "weather_available",
    "is_rainy",
    "is_snowy",
    "is_hot",
    "is_cold",
]

OUTLIER_COLS = [
    "review_text_length",
    "review_useful",
    "review_funny",
    "review_cool",
    "business_review_count",
    "user_review_count",
    "user_fans",
    "user_useful",
    "user_funny",
    "user_cool",
    "checkin_count",
    "tip_count",
    "tip_compliment_count",
    "photo_count",
    "weather_prcp",
    "weather_snow",
    "weather_snow_depth",
    "weather_tmax",
    "weather_tmin",
    "weather_temp_range",
]

IMPOSSIBLE_VALUE_RULES = {
    "satisfied": (0, 1),
    "review_stars": (1, 5),
    "business_stars": (1, 5),
    "user_average_stars": (1, 5),
    "review_text_length": (0, None),
    "review_useful": (0, None),
    "review_funny": (0, None),
    "review_cool": (0, None),
    "business_review_count": (0, None),
    "user_review_count": (0, None),
    "user_fans": (0, None),
    "user_useful": (0, None),
    "user_funny": (0, None),
    "user_cool": (0, None),
    "checkin_count": (0, None),
    "tip_count": (0, None),
    "tip_compliment_count": (0, None),
    "photo_count": (0, None),
    "weather_prcp": (0, None),
    "weather_snow": (0, None),
    "weather_snow_depth": (0, None),
    "weather_temp_range": (0, None),
}


# =============================================================================
# General helpers
# =============================================================================

def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def save_current_plot(filename: str) -> None:
    output_path = PLOT_OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    rotate_labels: bool = True,
) -> None:
    if table.empty:
        return

    plt.figure(figsize=(11, 6))
    plt.bar(table[x_col].astype(str), table[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    save_current_plot(filename)


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    filename: str,
    bins: int = 50,
    sample_n: int | None = None,
) -> None:
    if column not in df.columns:
        return

    data = df[column].dropna()

    if data.empty:
        return

    if sample_n is not None and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    save_current_plot(filename)


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    sample_n: int | None = None,
) -> None:
    if column not in df.columns:
        return

    data = df[column].dropna()

    if data.empty:
        return

    if sample_n is not None and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42)

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, vert=False, showfliers=True)
    plt.title(title)
    plt.xlabel(column)

    save_current_plot(filename)


def satisfaction_by_category(
    df: pd.DataFrame,
    category_col: str,
    top_n: int | None = None,
    min_count: int = 0,
) -> pd.DataFrame:
    if category_col not in df.columns:
        return pd.DataFrame()

    table = (
        df.groupby(category_col, dropna=False, observed=True)
        .agg(
            review_count=(TARGET_COL, "size"),
            satisfaction_rate=(TARGET_COL, "mean"),
        )
        .reset_index()
    )

    table["satisfaction_rate_percent"] = (
        table["satisfaction_rate"] * 100
    ).round(2)

    table = table.sort_values("review_count", ascending=False)

    if min_count > 0:
        table = table[table["review_count"] >= min_count].copy()

    if top_n is not None:
        table = table.head(top_n).copy()

    return table


def satisfaction_by_quantile_bins(
    df: pd.DataFrame,
    column: str,
    q: int = 5,
) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()

    temp = df[[column, TARGET_COL]].dropna().copy()

    if temp.empty:
        return pd.DataFrame()

    try:
        temp[f"{column}_bin"] = pd.qcut(
            temp[column],
            q=q,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        return pd.DataFrame()

    table = (
        temp.groupby(f"{column}_bin", dropna=False)
        .agg(
            review_count=(TARGET_COL, "size"),
            satisfaction_rate=(TARGET_COL, "mean"),
            min_value=(column, "min"),
            max_value=(column, "max"),
        )
        .reset_index()
    )

    table["satisfaction_rate_percent"] = (
        table["satisfaction_rate"] * 100
    ).round(2)

    return table


# =============================================================================
# Load data
# =============================================================================

def load_data() -> pd.DataFrame:
    print_section("Loading final EDA dataset")
    print(f"Loading: {WEATHER_ENRICHED_PKL}")

    df = pd.read_pickle(WEATHER_ENRICHED_PKL)

    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")

    if "weather_tobs" not in df.columns:
        print("Note: weather_tobs is not present. This is expected because NOAA did not return TOBS.")

    return df


# =============================================================================
# Dataset overview
# =============================================================================

def run_dataset_overview(df: pd.DataFrame) -> None:
    print_section("Dataset overview")

    overview = pd.DataFrame([{
        "rows": df.shape[0],
        "columns": df.shape[1],
        "unique_reviews": df["review_id"].nunique() if "review_id" in df.columns else np.nan,
        "unique_businesses": df["business_id"].nunique() if "business_id" in df.columns else np.nan,
        "unique_users": df["user_id"].nunique() if "user_id" in df.columns else np.nan,
        "unique_cities": df["city"].nunique() if "city" in df.columns else np.nan,
        "unique_states": df["state"].nunique() if "state" in df.columns else np.nan,
        "overall_satisfaction_rate_percent": round(df[TARGET_COL].mean() * 100, 2),
    }])

    print_table(overview, "Dataset shape and key counts")

    if "review_id" in df.columns:
        duplicate_review_count = df["review_id"].duplicated().sum()
        duplicate_table = pd.DataFrame([{
            "duplicate_review_id_count": duplicate_review_count,
            "duplicate_review_id_percent": round(duplicate_review_count / len(df) * 100, 4),
        }])
        print_table(duplicate_table, "Duplicate review_id check")

    missing = pd.DataFrame({
        "variable": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean().values * 100).round(2),
        "dtype": [str(dtype) for dtype in df.dtypes],
        "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
    }).sort_values(["missing_percent", "missing_count"], ascending=False)

    print_table(missing, "Variables with most missing values", max_rows=25)


# =============================================================================
# Target analysis
# =============================================================================

def run_target_analysis(df: pd.DataFrame) -> None:
    print_section("Target analysis")

    target = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COL)
        .reset_index(name="review_count")
        .sort_values(TARGET_COL)
    )

    target["percent"] = (
        target["review_count"] / target["review_count"].sum() * 100
    ).round(2)

    print_table(target, "Target distribution")

    plot_bar(
        target,
        x_col=TARGET_COL,
        y_col="review_count",
        title="Target distribution: satisfied",
        xlabel="Satisfied",
        ylabel="Number of reviews",
        filename="target_distribution.png",
        rotate_labels=False,
    )

    if LEAKAGE_COL in df.columns:
        stars = (
            df[LEAKAGE_COL]
            .value_counts(dropna=False)
            .rename_axis(LEAKAGE_COL)
            .reset_index(name="review_count")
            .sort_values(LEAKAGE_COL)
        )

        stars["percent"] = (
            stars["review_count"] / stars["review_count"].sum() * 100
        ).round(2)

        print_table(stars, "Review stars distribution")

        plot_bar(
            stars,
            x_col=LEAKAGE_COL,
            y_col="review_count",
            title="Distribution of review stars",
            xlabel="Review stars",
            ylabel="Number of reviews",
            filename="review_stars_distribution.png",
            rotate_labels=False,
        )


# =============================================================================
# Geographic EDA
# =============================================================================

def run_geographic_analysis(df: pd.DataFrame) -> None:
    print_section("Geographic analysis")

    if "state" in df.columns:
        top_states = (
            df["state"]
            .value_counts(dropna=False)
            .head(TOP_N)
            .rename_axis("state")
            .reset_index(name="review_count")
        )

        print_table(top_states, f"Top {TOP_N} states by review count")

        plot_bar(
            top_states,
            x_col="state",
            y_col="review_count",
            title=f"Top {TOP_N} states by review count",
            xlabel="State",
            ylabel="Number of reviews",
            filename="top_states_review_count.png",
        )

        sat_by_state = satisfaction_by_category(
            df,
            "state",
            top_n=TOP_N,
            min_count=MIN_CATEGORY_COUNT,
        )

        print_table(sat_by_state, f"Satisfaction by top {TOP_N} states")

    if "city" in df.columns:
        top_cities = (
            df["city"]
            .value_counts(dropna=False)
            .head(TOP_N)
            .rename_axis("city")
            .reset_index(name="review_count")
        )

        print_table(top_cities, f"Top {TOP_N} cities by review count")

        plot_bar(
            top_cities,
            x_col="city",
            y_col="review_count",
            title=f"Top {TOP_N} cities by review count",
            xlabel="City",
            ylabel="Number of reviews",
            filename="top_cities_review_count.png",
        )

        sat_by_city = satisfaction_by_category(
            df,
            "city",
            top_n=TOP_N,
            min_count=MIN_CATEGORY_COUNT,
        )

        print_table(sat_by_city, f"Satisfaction by top {TOP_N} cities")

        plot_bar(
            sat_by_city,
            x_col="city",
            y_col="satisfaction_rate_percent",
            title=f"Satisfaction rate by top {TOP_N} cities",
            xlabel="City",
            ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_top_city.png",
        )


# =============================================================================
# Focused feature EDA
# =============================================================================

def run_focused_numeric_eda(df: pd.DataFrame) -> None:
    print_section("Focused numeric EDA")

    cols = available_columns(df, KEY_NUMERIC_COLS)

    if not cols:
        print("No selected numeric columns available.")
        return

    summary = (
        df[cols]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    print_table(summary, "Numeric summary for selected variables", max_rows=50)

    plot_cols = [
        "log_review_text_length",
        "business_stars",
        "log_business_review_count",
        "user_average_stars",
        "log_user_review_count",
        "log_checkin_count",
        "log_tip_count",
        "log_photo_count",
        "weather_tmax",
        "weather_prcp",
    ]

    for col in available_columns(df, plot_cols):
        plot_histogram(
            df=df,
            column=col,
            title=f"Distribution of {col}",
            xlabel=col,
            filename=f"distribution_{col}.png",
            bins=50,
            sample_n=PLOT_SAMPLE_N,
        )

    bin_cols = [
        "review_text_length",
        "business_stars",
        "log_business_review_count",
        "user_average_stars",
        "log_user_review_count",
        "log_checkin_count",
        "log_tip_count",
        "log_photo_count",
    ]

    for col in available_columns(df, bin_cols):
        table = satisfaction_by_quantile_bins(df, col, q=5)
        print_table(table, f"Satisfaction by {col} bins", max_rows=10)

        if not table.empty:
            plot_bar(
                table,
                x_col=f"{col}_bin",
                y_col="satisfaction_rate_percent",
                title=f"Satisfaction rate by {col} bins",
                xlabel=f"{col} bin",
                ylabel="Satisfaction rate (%)",
                filename=f"satisfaction_by_{col}_bins.png",
            )


def run_focused_categorical_eda(df: pd.DataFrame) -> None:
    print_section("Focused categorical EDA")

    selected_cols = [
        "attributes.RestaurantsPriceRange2",
        "is_open",
        "attributes.RestaurantsTakeOut",
        "attributes.RestaurantsDelivery",
        "attributes.OutdoorSeating",
        "attributes.RestaurantsReservations",
        "attributes.Alcohol",
        "attributes.NoiseLevel",
    ]

    for col in available_columns(df, selected_cols):
        table = satisfaction_by_category(
            df,
            col,
            min_count=MIN_CATEGORY_COUNT,
        )

        print_table(table, f"Satisfaction by {col}", max_rows=20)

        if not table.empty and table.shape[0] <= 15:
            safe_col = col.replace(".", "_")
            plot_bar(
                table,
                x_col=col,
                y_col="satisfaction_rate_percent",
                title=f"Satisfaction rate by {col}",
                xlabel=col,
                ylabel="Satisfaction rate (%)",
                filename=f"satisfaction_by_{safe_col}.png",
            )


# =============================================================================
# Weather EDA
# =============================================================================

def run_weather_analysis(df: pd.DataFrame) -> None:
    print_section("Weather EDA")

    if "weather_available" not in df.columns:
        print("weather_available is missing. Skipping weather EDA.")
        return

    coverage = pd.DataFrame([{
        "reviews": len(df),
        "weather_available_reviews": int(df["weather_available"].sum()),
        "weather_coverage_percent": round(df["weather_available"].mean() * 100, 2),
    }])

    print_table(coverage, "Overall weather coverage")

    if "city" in df.columns and "state" in df.columns:
        coverage_by_city = (
            df.groupby(["city", "state"], dropna=False, observed=True)
            .agg(
                review_count=(TARGET_COL, "size"),
                weather_coverage_percent=("weather_available", lambda x: x.mean() * 100),
            )
            .reset_index()
            .sort_values("review_count", ascending=False)
            .head(TOP_N)
        )

        coverage_by_city["weather_coverage_percent"] = (
            coverage_by_city["weather_coverage_percent"].round(2)
        )

        print_table(coverage_by_city, f"Weather coverage by top {TOP_N} cities")

        plot_bar(
            coverage_by_city,
            x_col="city",
            y_col="weather_coverage_percent",
            title=f"Weather coverage by top {TOP_N} cities",
            xlabel="City",
            ylabel="Weather coverage (%)",
            filename="weather_coverage_by_city.png",
        )

    df_weather = df[df["weather_available"] == 1].copy()

    if df_weather.empty:
        print("No weather-covered reviews found.")
        return

    weather_value_cols = available_columns(
        df_weather,
        [
            "weather_prcp",
            "weather_snow",
            "weather_snow_depth",
            "weather_tmax",
            "weather_tmin",
            "weather_temp_range",
        ],
    )

    if weather_value_cols:
        weather_summary = (
            df_weather[weather_value_cols]
            .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            .T
            .reset_index()
            .rename(columns={"index": "variable"})
        )

        print_table(weather_summary, "Weather summary among weather-covered reviews")

    for col in available_columns(df_weather, ["is_rainy", "is_snowy", "is_hot", "is_cold"]):
        table = satisfaction_by_category(df_weather, col, min_count=0)
        print_table(table, f"Satisfaction by {col}")

        plot_bar(
            table,
            x_col=col,
            y_col="satisfaction_rate_percent",
            title=f"Satisfaction rate by {col}",
            xlabel=col,
            ylabel="Satisfaction rate (%)",
            filename=f"satisfaction_by_{col}.png",
            rotate_labels=False,
        )

    for col in available_columns(df_weather, ["weather_tmax", "weather_prcp"]):
        table = satisfaction_by_quantile_bins(df_weather, col, q=5)
        print_table(table, f"Satisfaction by {col} bins", max_rows=10)

        if not table.empty:
            plot_bar(
                table,
                x_col=f"{col}_bin",
                y_col="satisfaction_rate_percent",
                title=f"Satisfaction rate by {col} bins",
                xlabel=f"{col} bin",
                ylabel="Satisfaction rate (%)",
                filename=f"satisfaction_by_{col}_bins.png",
            )


# =============================================================================
# Correlation EDA
# =============================================================================

def run_correlation_analysis(df: pd.DataFrame) -> None:
    print_section("Correlation analysis")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    drop_cols = [
        col for col in ID_COLS + ["weather_available"]
        if col in numeric_df.columns
    ]

    numeric_df = numeric_df.drop(columns=drop_cols, errors="ignore")

    if TARGET_COL not in numeric_df.columns:
        print("Target is missing from numeric columns. Skipping correlation analysis.")
        return

    corr = numeric_df.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)

    corr_table = (
        corr
        .reset_index()
        .rename(columns={
            "index": "variable",
            TARGET_COL: "correlation_with_satisfied",
        })
    )

    corr_table["absolute_correlation"] = corr_table["correlation_with_satisfied"].abs()

    corr_table["note"] = np.where(
        corr_table["variable"].isin(LEAKAGE_COLS),
        "LEAKAGE: leaks the target; exclude from modelling",
        "",
    )

    corr_table = corr_table.sort_values("absolute_correlation", ascending=False)

    print_table(corr_table, "Top numeric correlations with satisfied", max_rows=30)

    plot_corr = (
        corr_table[~corr_table["variable"].isin(LEAKAGE_COLS)]
        .head(20)
        .sort_values("correlation_with_satisfied")
    )

    if not plot_corr.empty:
        plt.figure(figsize=(10, 8))
        plt.barh(
            plot_corr["variable"],
            plot_corr["correlation_with_satisfied"],
        )
        plt.title("Top numeric correlations with satisfied, excluding leakage columns")
        plt.xlabel("Correlation with satisfied")
        plt.ylabel("Variable")
        save_current_plot("numeric_correlation_with_satisfied.png")


# =============================================================================
# Outlier analysis
# =============================================================================

def impossible_value_check(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col, (min_allowed, max_allowed) in IMPOSSIBLE_VALUE_RULES.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        invalid_mask = pd.Series(False, index=df.index)

        if min_allowed is not None:
            invalid_mask = invalid_mask | (values < min_allowed)

        if max_allowed is not None:
            invalid_mask = invalid_mask | (values > max_allowed)

        invalid_count = int(invalid_mask.sum())

        rows.append({
            "variable": col,
            "min_allowed": min_allowed,
            "max_allowed": max_allowed,
            "observed_min": values.min(),
            "observed_max": values.max(),
            "invalid_count": invalid_count,
            "invalid_percent": round(invalid_count / len(df) * 100, 4),
        })

    return pd.DataFrame(rows).sort_values("invalid_count", ascending=False)


def iqr_outlier_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []

    for col in available_columns(df, columns):
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        lower_outliers = int((values < lower_bound).sum())
        upper_outliers = int((values > upper_bound).sum())
        total_outliers = lower_outliers + upper_outliers

        rows.append({
            "variable": col,
            "count": values.shape[0],
            "min": values.min(),
            "p01": values.quantile(0.01),
            "p05": values.quantile(0.05),
            "q1": q1,
            "median": values.median(),
            "q3": q3,
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
            "max": values.max(),
            "iqr_lower_bound": lower_bound,
            "iqr_upper_bound": upper_bound,
            "lower_outliers": lower_outliers,
            "upper_outliers": upper_outliers,
            "total_outliers": total_outliers,
            "outlier_percent": round(total_outliers / values.shape[0] * 100, 2),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("outlier_percent", ascending=False)
    )


def top_extreme_values(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    id_cols = available_columns(df, ["review_id", "business_id", "user_id", "city", "state", TARGET_COL, LEAKAGE_COL])
    cols = id_cols + [column]

    return (
        df[cols]
        .dropna(subset=[column])
        .sort_values(column, ascending=False)
        .head(n)
    )


def run_outlier_analysis(df: pd.DataFrame) -> None:
    print_section("Outlier analysis")

    print(
        "Outlier policy: Do not drop rows only because they are statistically extreme. "
        "For this Yelp dataset, many count variables are naturally right-skewed. "
        "Rows should only be removed if values are impossible or clearly caused by data corruption."
    )

    invalid_table = impossible_value_check(df)
    print_table(invalid_table, "Impossible value checks", max_rows=50)

    outlier_table = iqr_outlier_table(df, OUTLIER_COLS)
    print_table(outlier_table, "IQR-based outlier diagnostics", max_rows=50)

    boxplot_cols = [
        "review_text_length",
        "business_review_count",
        "user_review_count",
        "checkin_count",
        "tip_count",
        "photo_count",
        "weather_prcp",
        "weather_tmax",
        "weather_tmin",
    ]

    for col in available_columns(df, boxplot_cols):
        plot_boxplot(
            df=df,
            column=col,
            title=f"Boxplot for {col}",
            filename=f"boxplot_{col}.png",
            sample_n=PLOT_SAMPLE_N,
        )

    extreme_cols = [
        "review_text_length",
        "business_review_count",
        "user_review_count",
        "checkin_count",
        "tip_count",
        "photo_count",
        "weather_prcp",
    ]

    for col in available_columns(df, extreme_cols):
        extreme = top_extreme_values(df, col, n=10)
        print_table(extreme, f"Top 10 largest values for {col}", max_rows=10)


# =============================================================================
# EDA notes
# =============================================================================

def run_eda_notes(df: pd.DataFrame) -> None:
    print_section("EDA notes for report")

    notes = [
        {
            "topic": "EDA scope",
            "note": "The EDA is intentionally focused on key variables and variable groups rather than deeply analyzing all available columns.",
        },
        {
            "topic": "Leakage",
            "note": "review_stars, business_stars and user_average_stars are shown in EDA but must be excluded from modelling: satisfied is derived from review_stars, and the two averages already include the current review's rating (target + temporal leakage).",
        },
        {
            "topic": "Outliers",
            "note": "Large count values are expected in Yelp data and should usually be handled with log transformations rather than deleted.",
        },
        {
            "topic": "Dropping rows",
            "note": "Rows should only be removed if they contain impossible values or clear data errors, not merely because they are extreme under the IQR rule.",
        },
    ]

    if "weather_available" in df.columns:
        notes.append({
            "topic": "Weather coverage",
            "note": (
                f"Weather data are available for {df['weather_available'].mean() * 100:.2f}% of reviews. "
                "Missing weather is structural because only selected cities were matched to NOAA stations."
            ),
        })

    if "weather_tobs" not in df.columns:
        notes.append({
            "topic": "Weather TOBS",
            "note": "weather_tobs is absent because NOAA did not return TOBS for the selected station requests.",
        })

    print_table(pd.DataFrame(notes), "Summary notes", max_rows=20)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    df = load_data()

    run_dataset_overview(df)
    run_target_analysis(df)
    run_geographic_analysis(df)
    run_focused_numeric_eda(df)
    run_focused_categorical_eda(df)
    run_weather_analysis(df)
    run_correlation_analysis(df)
    run_outlier_analysis(df)
    run_eda_notes(df)

    print_section("EDA completed")
    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")

if __name__ == "__main__":
    main()