"""
Exploratory Data Analysis for the Yelp restaurant satisfaction project.

This script analyzes the final review-level Yelp restaurant dataset enriched
with NOAA weather data.

Important:
- One row = one Yelp review.
- Target variable: satisfied = 1 if review_stars >= 4, else 0.
- review_stars is useful for EDA but is leakage for later modeling.
- This script does NOT create a train/test split.
- This script does NOT train models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import WEATHER_ENRICHED_PKL, PROCESSED_DATA_DIR


# =============================================================================
# Configuration
# =============================================================================

EDA_OUTPUT_DIR = PROCESSED_DATA_DIR / "eda_outputs"
TABLE_OUTPUT_DIR = EDA_OUTPUT_DIR / "tables"
PLOT_OUTPUT_DIR = EDA_OUTPUT_DIR / "plots"

TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "satisfied"
LEAKAGE_COL = "review_stars"

ID_COLS = [
    "review_id",
    "business_id",
    "user_id",
]

DATE_COLS = [
    "review_date",
    "review_year",
    "review_month",
    "review_weekday",
]

WEATHER_COLS = [
    "weather_prcp",
    "weather_snow",
    "weather_snow_depth",
    "weather_tmax",
    "weather_tmin",
    "weather_tobs",  # optional; NOAA may not return it
    "weather_temp_range",
    "is_rainy",
    "is_snowy",
    "is_hot",
    "is_cold",
    "weather_available",
]

WEATHER_VALUE_COLS = [
    "weather_prcp",
    "weather_snow",
    "weather_snow_depth",
    "weather_tmax",
    "weather_tmin",
    "weather_tobs",  # optional
    "weather_temp_range",
]

WEATHER_FLAG_COLS = [
    "is_rainy",
    "is_snowy",
    "is_hot",
    "is_cold",
]

REVIEW_BEHAVIOR_COLS = [
    "review_text_length",
    "review_useful",
    "review_funny",
    "review_cool",
    "log_review_text_length",
    "log_review_useful",
    "log_review_funny",
    "log_review_cool",
]

BUSINESS_NUMERIC_COLS = [
    "business_stars",
    "business_review_count",
    "log_business_review_count",
]

BUSINESS_CATEGORICAL_COLS = [
    "is_open",
    "attributes.RestaurantsPriceRange2",
    "attributes.BusinessAcceptsCreditCards",
    "attributes.RestaurantsTakeOut",
    "attributes.RestaurantsDelivery",
    "attributes.OutdoorSeating",
    "attributes.BikeParking",
    "attributes.GoodForKids",
    "attributes.RestaurantsReservations",
    "attributes.Alcohol",
    "attributes.WiFi",
    "attributes.HasTV",
    "attributes.NoiseLevel",
    "attributes.RestaurantsGoodForGroups",
    "attributes.RestaurantsTableService",
]

USER_NUMERIC_COLS = [
    "user_review_count",
    "user_average_stars",
    "user_fans",
    "user_useful",
    "user_funny",
    "user_cool",
    "log_user_review_count",
    "log_user_fans",
    "log_user_useful",
    "log_user_funny",
    "log_user_cool",
]

ACTIVITY_COLS = [
    "checkin_count",
    "tip_count",
    "tip_compliment_count",
    "photo_count",
    "photo_food",
    "photo_drink",
    "photo_menu",
    "photo_inside",
    "photo_outside",
    "log_checkin_count",
    "log_tip_count",
    "log_tip_compliment_count",
    "log_photo_count",
]

TOP_N = 20
MIN_CATEGORY_COUNT = 1_000


# =============================================================================
# Helper functions
# =============================================================================

def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return only columns that exist in the DataFrame."""
    return [col for col in columns if col in df.columns]


def save_table(df: pd.DataFrame, filename: str) -> None:
    """Save a table to the EDA tables folder."""
    output_path = TABLE_OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)


def save_current_plot(filename: str) -> None:
    """Save the active matplotlib figure and close it."""
    output_path = PLOT_OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_section(title: str) -> None:
    """Print a clear console section header."""
    line = "=" * 100
    print("\n" + line)
    print(title.upper())
    print(line)


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value summary table."""
    table = pd.DataFrame({
        "variable": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean().values * 100).round(2),
        "dtype": [str(dtype) for dtype in df.dtypes],
        "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
    })

    return table.sort_values(
        ["missing_percent", "missing_count"],
        ascending=False
    )


def numeric_summary_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create a numeric summary table for selected columns."""
    cols = available_columns(df, columns)

    if not cols:
        return pd.DataFrame()

    summary = (
        df[cols]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    return summary


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the target variable."""
    summary = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COL)
        .reset_index(name="review_count")
    )

    summary["percent"] = (
        summary["review_count"] / summary["review_count"].sum() * 100
    ).round(2)

    return summary.sort_values(TARGET_COL)


def satisfaction_by_category(
    df: pd.DataFrame,
    category_col: str,
    top_n: int | None = None,
    min_count: int = 0,
) -> pd.DataFrame:
    """
    Calculate review count and satisfaction rate by category.

    Parameters
    ----------
    df:
        Input DataFrame.
    category_col:
        Categorical variable.
    top_n:
        If provided, keep only the top N categories by review count.
    min_count:
        Minimum number of reviews required to keep a category.
    """
    if category_col not in df.columns:
        return pd.DataFrame()

    table = (
        df.groupby(category_col, dropna=False)
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


def plot_bar_count(
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    rotate_labels: bool = True,
) -> None:
    """Create and save a bar chart."""
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
    """Create and save a histogram for a numeric variable."""
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


def plot_satisfaction_rate(
    table: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    filename: str,
    rotate_labels: bool = True,
) -> None:
    """Create and save a satisfaction-rate bar chart."""
    if table.empty or "satisfaction_rate_percent" not in table.columns:
        return

    plt.figure(figsize=(11, 6))
    plt.bar(table[x_col].astype(str), table["satisfaction_rate_percent"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Satisfaction rate (%)")
    plt.ylim(0, 100)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    save_current_plot(filename)


def create_quantile_bins(
    df: pd.DataFrame,
    column: str,
    q: int = 5,
    label_prefix: str = "Q",
) -> pd.Series:
    """
    Create quantile bins for a numeric variable.

    Duplicate bin edges are dropped to avoid errors for variables with many ties.
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype="object")

    values = df[column]

    try:
        bins = pd.qcut(
            values,
            q=q,
            duplicates="drop"
        )

        return bins.astype(str)

    except ValueError:
        return pd.Series(index=df.index, dtype="object")


def satisfaction_by_numeric_bins(
    df: pd.DataFrame,
    column: str,
    q: int = 5,
) -> pd.DataFrame:
    """Calculate satisfaction by quantile bins for a numeric variable."""
    if column not in df.columns:
        return pd.DataFrame()

    temp = df[[column, TARGET_COL]].copy()
    temp = temp.dropna(subset=[column, TARGET_COL])

    if temp.empty:
        return pd.DataFrame()

    temp[f"{column}_bin"] = create_quantile_bins(temp, column, q=q)

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
# Load and validate data
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load the final weather-enriched Yelp dataset."""
    print_section("Loading final EDA dataset")
    print(f"Loading data from: {WEATHER_ENRICHED_PKL}")

    df = pd.read_pickle(WEATHER_ENRICHED_PKL)

    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")

    if LEAKAGE_COL not in df.columns:
        print(f"Warning: {LEAKAGE_COL} is missing. Star distribution EDA will be skipped.")

    if "weather_tobs" not in df.columns:
        print(
            "Note: weather_tobs is not present. "
            "This is expected if NOAA did not return TOBS for the selected stations."
        )

    return df


# =============================================================================
# EDA sections
# =============================================================================

def run_dataset_overview(df: pd.DataFrame) -> None:
    """Create dataset overview tables."""
    print_section("Dataset overview")

    overview = pd.DataFrame([{
        "rows": df.shape[0],
        "columns": df.shape[1],
        "unique_reviews": df["review_id"].nunique() if "review_id" in df.columns else np.nan,
        "unique_businesses": df["business_id"].nunique() if "business_id" in df.columns else np.nan,
        "unique_users": df["user_id"].nunique() if "user_id" in df.columns else np.nan,
        "unique_cities": df["city"].nunique() if "city" in df.columns else np.nan,
        "unique_states": df["state"].nunique() if "state" in df.columns else np.nan,
        "overall_satisfaction_rate": df[TARGET_COL].mean(),
    }])

    save_table(overview, "dataset_overview.csv")
    print(overview.to_string(index=False))

    missing = missing_value_table(df)
    save_table(missing, "missing_values.csv")

    dtype_table = pd.DataFrame({
        "variable": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
    })
    save_table(dtype_table, "data_types.csv")

    if "review_id" in df.columns:
        duplicate_review_count = df["review_id"].duplicated().sum()
        duplicate_table = pd.DataFrame([{
            "duplicate_review_id_count": duplicate_review_count,
            "duplicate_review_id_percent": duplicate_review_count / len(df) * 100,
        }])
        save_table(duplicate_table, "duplicate_review_id_check.csv")
        print(duplicate_table.to_string(index=False))


def run_target_analysis(df: pd.DataFrame) -> None:
    """Analyze target and review-star distribution."""
    print_section("Target analysis")

    target = target_summary(df)
    save_table(target, "target_distribution.csv")
    print(target.to_string(index=False))

    plot_bar_count(
        table=target,
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

        save_table(stars, "review_stars_distribution.csv")

        plot_bar_count(
            table=stars,
            x_col=LEAKAGE_COL,
            y_col="review_count",
            title="Distribution of review stars",
            xlabel="Review stars",
            ylabel="Number of reviews",
            filename="review_stars_distribution.png",
            rotate_labels=False,
        )


def run_geographic_analysis(df: pd.DataFrame) -> None:
    """Analyze states, cities, and geographic satisfaction differences."""
    print_section("Geographic analysis")

    if "state" in df.columns:
        top_states = (
            df["state"]
            .value_counts(dropna=False)
            .head(TOP_N)
            .rename_axis("state")
            .reset_index(name="review_count")
        )

        save_table(top_states, "top_states_review_count.csv")

        plot_bar_count(
            table=top_states,
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

        save_table(sat_by_state, "satisfaction_by_top_state.csv")

        plot_satisfaction_rate(
            table=sat_by_state,
            x_col="state",
            title=f"Satisfaction rate by top {TOP_N} states",
            xlabel="State",
            filename="satisfaction_by_top_state.png",
        )

    if "city" in df.columns:
        top_cities = (
            df["city"]
            .value_counts(dropna=False)
            .head(TOP_N)
            .rename_axis("city")
            .reset_index(name="review_count")
        )

        save_table(top_cities, "top_cities_review_count.csv")

        plot_bar_count(
            table=top_cities,
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

        save_table(sat_by_city, "satisfaction_by_top_city.csv")

        plot_satisfaction_rate(
            table=sat_by_city,
            x_col="city",
            title=f"Satisfaction rate by top {TOP_N} cities",
            xlabel="City",
            filename="satisfaction_by_top_city.png",
        )

    if "weather_available" in df.columns and "city" in df.columns and "state" in df.columns:
        coverage_by_city = (
            df.groupby(["city", "state"], dropna=False)
            .agg(
                review_count=(TARGET_COL, "size"),
                weather_coverage_rate=("weather_available", "mean"),
            )
            .reset_index()
            .sort_values("review_count", ascending=False)
            .head(TOP_N)
        )

        coverage_by_city["weather_coverage_percent"] = (
            coverage_by_city["weather_coverage_rate"] * 100
        ).round(2)

        save_table(coverage_by_city, "weather_coverage_by_top_city.csv")

        plot_bar_count(
            table=coverage_by_city,
            x_col="city",
            y_col="weather_coverage_percent",
            title=f"Weather coverage by top {TOP_N} cities",
            xlabel="City",
            ylabel="Weather coverage (%)",
            filename="weather_coverage_by_city.png",
        )


def run_review_behavior_analysis(df: pd.DataFrame) -> None:
    """Analyze review text length and review vote behavior."""
    print_section("Review behavior analysis")

    cols = available_columns(df, REVIEW_BEHAVIOR_COLS)
    summary = numeric_summary_table(df, cols)
    save_table(summary, "review_behavior_numeric_summary.csv")

    for col in cols:
        plot_histogram(
            df=df,
            column=col,
            title=f"Distribution of {col}",
            xlabel=col,
            filename=f"distribution_{col}.png",
            bins=50,
            sample_n=500_000,
        )

    if "review_text_length" in df.columns:
        text_bins = satisfaction_by_numeric_bins(df, "review_text_length", q=5)
        save_table(text_bins, "satisfaction_by_review_text_length_bins.csv")

        plot_satisfaction_rate(
            table=text_bins,
            x_col="review_text_length_bin",
            title="Satisfaction rate by review text length bins",
            xlabel="Review text length bin",
            filename="satisfaction_by_review_text_length_bins.png",
        )


def run_business_analysis(df: pd.DataFrame) -> None:
    """Analyze business characteristics and restaurant attributes."""
    print_section("Business characteristics analysis")

    numeric_cols = available_columns(df, BUSINESS_NUMERIC_COLS)
    summary = numeric_summary_table(df, numeric_cols)
    save_table(summary, "business_numeric_summary.csv")

    for col in numeric_cols:
        plot_histogram(
            df=df,
            column=col,
            title=f"Distribution of {col}",
            xlabel=col,
            filename=f"distribution_{col}.png",
            bins=50,
            sample_n=500_000,
        )

    if "business_stars" in df.columns:
        sat_by_business_stars = satisfaction_by_category(
            df,
            "business_stars",
            min_count=MIN_CATEGORY_COUNT,
        ).sort_values("business_stars")

        save_table(sat_by_business_stars, "satisfaction_by_business_stars.csv")

        plot_satisfaction_rate(
            table=sat_by_business_stars,
            x_col="business_stars",
            title="Satisfaction rate by business star rating",
            xlabel="Business stars",
            filename="satisfaction_by_business_stars.png",
            rotate_labels=False,
        )

    if "log_business_review_count" in df.columns:
        business_review_bins = satisfaction_by_numeric_bins(
            df,
            "log_business_review_count",
            q=5,
        )

        save_table(
            business_review_bins,
            "satisfaction_by_log_business_review_count_bins.csv",
        )

        plot_satisfaction_rate(
            table=business_review_bins,
            x_col="log_business_review_count_bin",
            title="Satisfaction rate by business review count bins",
            xlabel="Log business review count bin",
            filename="satisfaction_by_log_business_review_count_bins.png",
        )

    categorical_cols = available_columns(df, BUSINESS_CATEGORICAL_COLS)

    for col in categorical_cols:
        table = satisfaction_by_category(
            df,
            col,
            top_n=None,
            min_count=MIN_CATEGORY_COUNT,
        )

        safe_name = col.replace(".", "_")
        save_table(table, f"satisfaction_by_{safe_name}.csv")

        if not table.empty and table.shape[0] <= 15:
            plot_satisfaction_rate(
                table=table,
                x_col=col,
                title=f"Satisfaction rate by {col}",
                xlabel=col,
                filename=f"satisfaction_by_{safe_name}.png",
            )


def run_user_analysis(df: pd.DataFrame) -> None:
    """Analyze user characteristics."""
    print_section("User characteristics analysis")

    cols = available_columns(df, USER_NUMERIC_COLS)
    summary = numeric_summary_table(df, cols)
    save_table(summary, "user_numeric_summary.csv")

    for col in cols:
        plot_histogram(
            df=df,
            column=col,
            title=f"Distribution of {col}",
            xlabel=col,
            filename=f"distribution_{col}.png",
            bins=50,
            sample_n=500_000,
        )

    for col in [
        "user_average_stars",
        "log_user_review_count",
        "log_user_fans",
        "log_user_useful",
        "log_user_funny",
        "log_user_cool",
    ]:
        if col in df.columns:
            table = satisfaction_by_numeric_bins(df, col, q=5)
            save_table(table, f"satisfaction_by_{col}_bins.csv")

            plot_satisfaction_rate(
                table=table,
                x_col=f"{col}_bin",
                title=f"Satisfaction rate by {col} bins",
                xlabel=f"{col} bin",
                filename=f"satisfaction_by_{col}_bins.png",
            )


def run_activity_analysis(df: pd.DataFrame) -> None:
    """Analyze restaurant activity variables such as checkins, tips, and photos."""
    print_section("Restaurant activity analysis")

    cols = available_columns(df, ACTIVITY_COLS)
    summary = numeric_summary_table(df, cols)
    save_table(summary, "activity_numeric_summary.csv")

    for col in cols:
        plot_histogram(
            df=df,
            column=col,
            title=f"Distribution of {col}",
            xlabel=col,
            filename=f"distribution_{col}.png",
            bins=50,
            sample_n=500_000,
        )

    for col in [
        "log_checkin_count",
        "log_tip_count",
        "log_tip_compliment_count",
        "log_photo_count",
    ]:
        if col in df.columns:
            table = satisfaction_by_numeric_bins(df, col, q=5)
            save_table(table, f"satisfaction_by_{col}_bins.csv")

            plot_satisfaction_rate(
                table=table,
                x_col=f"{col}_bin",
                title=f"Satisfaction rate by {col} bins",
                xlabel=f"{col} bin",
                filename=f"satisfaction_by_{col}_bins.png",
            )


def run_weather_analysis(df: pd.DataFrame) -> None:
    """Analyze weather coverage and descriptive weather relationships."""
    print_section("Weather data analysis")

    if "weather_available" not in df.columns:
        print("weather_available column is missing. Skipping weather analysis.")
        return

    coverage_overall = pd.DataFrame([{
        "reviews": len(df),
        "weather_available_reviews": int(df["weather_available"].sum()),
        "weather_coverage_percent": round(df["weather_available"].mean() * 100, 2),
    }])

    save_table(coverage_overall, "weather_coverage_overall.csv")
    print(coverage_overall.to_string(index=False))

    coverage_target = satisfaction_by_category(
        df,
        "weather_available",
        min_count=0,
    )

    save_table(coverage_target, "satisfaction_by_weather_available.csv")

    plot_satisfaction_rate(
        table=coverage_target,
        x_col="weather_available",
        title="Satisfaction rate by weather availability",
        xlabel="Weather available",
        filename="satisfaction_by_weather_available.png",
        rotate_labels=False,
    )

    df_weather = df[df["weather_available"] == 1].copy()

    if df_weather.empty:
        print("No matched weather rows available. Skipping weather-value analysis.")
        return

    existing_weather_value_cols = available_columns(df_weather, WEATHER_VALUE_COLS)
    existing_weather_flag_cols = available_columns(df_weather, WEATHER_FLAG_COLS)

    weather_missing = missing_value_table(
        df_weather[existing_weather_value_cols + existing_weather_flag_cols]
    )

    save_table(weather_missing, "weather_missing_values_among_available_rows.csv")

    weather_summary = numeric_summary_table(
        df_weather,
        existing_weather_value_cols + existing_weather_flag_cols,
    )

    save_table(weather_summary, "weather_numeric_summary.csv")

    for col in existing_weather_value_cols:
        plot_histogram(
            df=df_weather,
            column=col,
            title=f"Distribution of {col} among weather-covered reviews",
            xlabel=col,
            filename=f"distribution_{col}.png",
            bins=50,
            sample_n=500_000,
        )

    for col in existing_weather_flag_cols:
        table = satisfaction_by_category(
            df_weather,
            col,
            min_count=0,
        )

        save_table(table, f"satisfaction_by_{col}.csv")

        plot_satisfaction_rate(
            table=table,
            x_col=col,
            title=f"Satisfaction rate by {col}",
            xlabel=col,
            filename=f"satisfaction_by_{col}.png",
            rotate_labels=False,
        )

    if "weather_tmax" in df_weather.columns:
        temp_bins = satisfaction_by_numeric_bins(df_weather, "weather_tmax", q=5)
        save_table(temp_bins, "satisfaction_by_weather_tmax_bins.csv")

        plot_satisfaction_rate(
            table=temp_bins,
            x_col="weather_tmax_bin",
            title="Satisfaction rate by maximum temperature bins",
            xlabel="Maximum temperature bin",
            filename="satisfaction_by_weather_tmax_bins.png",
        )

    if "weather_tmin" in df_weather.columns:
        tmin_bins = satisfaction_by_numeric_bins(df_weather, "weather_tmin", q=5)
        save_table(tmin_bins, "satisfaction_by_weather_tmin_bins.csv")

        plot_satisfaction_rate(
            table=tmin_bins,
            x_col="weather_tmin_bin",
            title="Satisfaction rate by minimum temperature bins",
            xlabel="Minimum temperature bin",
            filename="satisfaction_by_weather_tmin_bins.png",
        )

    if "weather_prcp" in df_weather.columns:
        prcp = df_weather[[TARGET_COL, "weather_prcp"]].copy()
        prcp = prcp.dropna(subset=["weather_prcp"])

        prcp["precipitation_bin"] = pd.cut(
            prcp["weather_prcp"],
            bins=[-0.001, 0, 1, 5, 10, np.inf],
            labels=[
                "0 mm",
                "0-1 mm",
                "1-5 mm",
                "5-10 mm",
                ">10 mm",
            ],
        )

        prcp_table = (
            prcp.groupby("precipitation_bin", observed=False)
            .agg(
                review_count=(TARGET_COL, "size"),
                satisfaction_rate=(TARGET_COL, "mean"),
            )
            .reset_index()
        )

        prcp_table["satisfaction_rate_percent"] = (
            prcp_table["satisfaction_rate"] * 100
        ).round(2)

        save_table(prcp_table, "satisfaction_by_precipitation_bins.csv")

        plot_satisfaction_rate(
            table=prcp_table,
            x_col="precipitation_bin",
            title="Satisfaction rate by precipitation bins",
            xlabel="Precipitation bin",
            filename="satisfaction_by_precipitation_bins.png",
        )


def run_correlation_analysis(df: pd.DataFrame) -> None:
    """Analyze numeric correlations with the satisfaction target."""
    print_section("Correlation analysis")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    cols_to_drop = [
        col for col in ID_COLS + ["weather_available"]
        if col in numeric_df.columns
    ]

    numeric_df = numeric_df.drop(columns=cols_to_drop, errors="ignore")

    if TARGET_COL not in numeric_df.columns:
        print("Target column is not numeric or is missing. Skipping correlations.")
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

    corr_table["absolute_correlation"] = (
        corr_table["correlation_with_satisfied"].abs()
    )

    corr_table["modeling_note"] = np.where(
        corr_table["variable"] == LEAKAGE_COL,
        "leakage: target is derived from review_stars",
        "",
    )

    corr_table = corr_table.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    save_table(corr_table, "numeric_correlation_with_satisfied.csv")

    # Plot top correlations excluding review_stars leakage.
    plot_corr = corr_table[
        corr_table["variable"] != LEAKAGE_COL
    ].head(25).copy()

    if not plot_corr.empty:
        plot_corr = plot_corr.sort_values("correlation_with_satisfied")

        plt.figure(figsize=(10, 8))
        plt.barh(
            plot_corr["variable"],
            plot_corr["correlation_with_satisfied"],
        )
        plt.title("Top numeric correlations with satisfied, excluding review_stars")
        plt.xlabel("Correlation with satisfied")
        plt.ylabel("Variable")

        save_current_plot("numeric_correlation_with_satisfied.png")

    if LEAKAGE_COL in corr_table["variable"].values:
        leakage_corr = corr_table[corr_table["variable"] == LEAKAGE_COL]
        save_table(leakage_corr, "review_stars_leakage_correlation.csv")


def run_eda_conclusion(df: pd.DataFrame) -> None:
    """
    Save compact EDA notes for the report.

    This is not final paper text. It is a structured summary to support writing.
    """
    print_section("EDA conclusion notes")

    notes = []

    notes.append({
        "topic": "Dataset unit",
        "note": "The dataset is review-level: one row represents one Yelp restaurant review.",
    })

    notes.append({
        "topic": "Target",
        "note": "The target is satisfied, equal to 1 for reviews with review_stars >= 4 and 0 otherwise.",
    })

    notes.append({
        "topic": "Leakage",
        "note": "review_stars is useful for EDA but must be excluded from later modeling because the target is derived from it.",
    })

    if "weather_available" in df.columns:
        coverage = df["weather_available"].mean() * 100
        notes.append({
            "topic": "Weather coverage",
            "note": (
                f"Weather data are available for {coverage:.2f}% of reviews. "
                "Missing weather is not necessarily random because only selected top cities were matched to NOAA stations."
            ),
        })

    if "weather_tobs" not in df.columns:
        notes.append({
            "topic": "Weather TOBS",
            "note": (
                "weather_tobs is not included because the NOAA response did not return TOBS "
                "for the selected station requests. EDA should rely on weather_tmax, weather_tmin, "
                "and weather_temp_range instead."
            ),
        })

    notes.append({
        "topic": "Modeling preparation",
        "note": (
            "This EDA script does not create a train/test split or fit models. "
            "Learned preprocessing should happen later inside the modeling workflow."
        ),
    })

    notes_df = pd.DataFrame(notes)
    save_table(notes_df, "eda_conclusion_notes.csv")
    print(notes_df.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run the full EDA workflow."""
    df = load_data()

    run_dataset_overview(df)
    run_target_analysis(df)
    run_geographic_analysis(df)
    run_review_behavior_analysis(df)
    run_business_analysis(df)
    run_user_analysis(df)
    run_activity_analysis(df)
    run_weather_analysis(df)
    run_correlation_analysis(df)
    run_eda_conclusion(df)

    print_section("EDA completed")
    print(f"Tables saved to: {TABLE_OUTPUT_DIR}")
    print(f"Plots saved to:  {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()