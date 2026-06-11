import pandas as pd

from config import (
    YELP_SAMPLE_PKL,
    WEATHER_CLEAN_CSV,
    WEATHER_ENRICHED_PKL,
    WEATHER_ENRICHED_CSV,
)


YELP_DATA_FILE = YELP_SAMPLE_PKL
WEATHER_FILE = WEATHER_CLEAN_CSV
OUTPUT_PKL = WEATHER_ENRICHED_PKL
OUTPUT_CSV = WEATHER_ENRICHED_CSV


# Weather column groups (single source of truth for the merge).
WEATHER_KEY_COLS = ["city", "state", "review_date"]

WEATHER_NUMERIC_COLS = [
    "weather_prcp",
    "weather_snow",
    "weather_snow_depth",
    "weather_tmax",
    "weather_tmin",
    "weather_temp_range",
]

WEATHER_BINARY_COLS = [
    "is_rainy",
    "is_snowy",
    "is_hot",
    "is_cold",
]

WEATHER_FEATURE_COLS = WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS

# Columns that determine whether a weather record was matched.
# Deliberately excludes weather_tobs so that weather_available keeps the exact
# same meaning (and coverage percentage) it had before TOBS was added.
WEATHER_AVAILABILITY_COLS = [
    "weather_prcp",
    "weather_snow",
    "weather_snow_depth",
    "weather_tmax",
    "weather_tmin",
    "weather_temp_range",
    "is_rainy",
    "is_snowy",
    "is_hot",
    "is_cold",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed Yelp review data and clean NOAA weather data."""
    yelp = pd.read_pickle(YELP_DATA_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    return yelp, weather


def prepare_merge_keys(
    yelp: pd.DataFrame,
    weather: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare city, state, and date keys for merging."""
    yelp = yelp.copy()
    weather = weather.copy()

    # Clean city/state keys on both sides
    yelp["city"] = (
        yelp["city"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    weather["city"] = (
        weather["city"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    yelp["state"] = (
        yelp["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    weather["state"] = (
        weather["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Important: both sides must use exact calendar date
    yelp["review_date"] = pd.to_datetime(
        yelp["review_date"],
        errors="coerce"
    ).dt.date

    weather["review_date"] = pd.to_datetime(
        weather["review_date"],
        errors="coerce"
    ).dt.date

    return yelp, weather


def validate_weather_file(weather: pd.DataFrame) -> None:
    """Check whether weather data has duplicate city-state-date rows."""
    duplicate_count = weather.duplicated(subset=WEATHER_KEY_COLS).sum()

    print("\nWeather file validation")
    print("-" * 80)
    print(f"Weather rows: {weather.shape[0]}")
    print(f"Duplicate city-state-date rows: {duplicate_count}")

    if duplicate_count > 0:
        print("\nWARNING: Duplicate weather keys found.")
        print("This would increase rows during the merge.")
        print("The script will aggregate duplicate rows before merging.")


def aggregate_duplicate_weather_rows(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one weather row per city-state-date.

    This protects the Yelp dataset from row multiplication during the merge.
    """
    numeric_weather_cols = [
        col for col in WEATHER_NUMERIC_COLS if col in weather.columns
    ]

    binary_weather_cols = [
        col for col in WEATHER_BINARY_COLS if col in weather.columns
    ]

    agg_dict = {col: "mean" for col in numeric_weather_cols}
    agg_dict.update({col: "max" for col in binary_weather_cols})

    if "station" in weather.columns:
        agg_dict["station"] = "first"

    weather_aggregated = (
        weather
        .groupby(WEATHER_KEY_COLS, as_index=False)
        .agg(agg_dict)
    )

    return weather_aggregated


def merge_weather(
    yelp: pd.DataFrame,
    weather: pd.DataFrame
) -> pd.DataFrame:
    """Merge NOAA weather variables into the Yelp review-level dataset."""
    weather_columns = WEATHER_KEY_COLS + ["station"] + WEATHER_FEATURE_COLS

    weather_columns = [
        col for col in weather_columns if col in weather.columns
    ]

    weather = weather[weather_columns].copy()

    enriched = yelp.merge(
        weather,
        on=WEATHER_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    available_weather_columns = [
        col for col in WEATHER_AVAILABILITY_COLS if col in enriched.columns
    ]

    enriched["weather_available"] = (
        enriched[available_weather_columns]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    return enriched


def print_weather_merge_summary(
    yelp: pd.DataFrame,
    weather: pd.DataFrame,
    enriched: pd.DataFrame
) -> None:
    """Print summary of the weather merge."""
    print("\nWeather merge summary")
    print("-" * 80)

    print(f"Yelp rows before merge: {yelp.shape[0]}")
    print(f"Weather rows:           {weather.shape[0]}")
    print(f"Rows after merge:       {enriched.shape[0]}")
    print(f"Columns after merge:    {enriched.shape[1]}")

    if yelp.shape[0] == enriched.shape[0]:
        print("Row check: OK - merge did not duplicate Yelp reviews.")
    else:
        print("Row check: WARNING - row count changed during merge.")

    coverage = enriched["weather_available"].mean() * 100
    print(f"\nWeather coverage: {coverage:.2f}% of reviews")

    print("\nWeather coverage by top cities:")
    coverage_by_city = (
        enriched
        .groupby(["city", "state"], dropna=False)
        .agg(
            reviews=("review_id", "count"),
            weather_available_percent=("weather_available", lambda x: x.mean() * 100),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
        .head(30)
    )

    print(coverage_by_city.to_string(index=False))

    print("\nMissing values in weather columns:")
    weather_cols = [
        col for col in enriched.columns
        if col.startswith("weather_")
        or col in ["is_rainy", "is_snowy", "is_hot", "is_cold"]
    ]

    missing = (
        enriched[weather_cols]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print(missing.to_string())


def save_outputs(enriched: pd.DataFrame) -> None:
    """Save weather-enriched Yelp dataset."""
    enriched.to_pickle(OUTPUT_PKL)
    enriched.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved weather-enriched files")
    print("-" * 80)
    print(f"PKL: {OUTPUT_PKL}")
    print(f"CSV: {OUTPUT_CSV}")


def main() -> None:
    print("Loading Yelp and NOAA weather data...")
    yelp, weather = load_data()

    print("Preparing merge keys...")
    yelp, weather = prepare_merge_keys(yelp, weather)

    validate_weather_file(weather)

    print("Aggregating duplicate weather rows if necessary...")
    weather = aggregate_duplicate_weather_rows(weather)

    print("Merging NOAA weather into Yelp data...")
    enriched = merge_weather(yelp, weather)

    print_weather_merge_summary(yelp, weather, enriched)

    save_outputs(enriched)


if __name__ == "__main__":
    main()