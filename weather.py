"""
NOAA weather stage for the Yelp restaurant satisfaction project.

This single module replaces the former noaa.py + merge.py pair. It:
  1. determines the NOAA date range from the processed Yelp review years,
  2. downloads (or reuses a cached) NOAA daily weather file for the selected
     representative city stations,
  3. cleans the weather data and derives interpretable weather indicators,
  4. merges weather onto the review-level Yelp dataset by city/state/date,
  5. creates weather_available and saves the weather-enriched dataset.

Run standalone:           python weather.py            (uses cached weather)
Force a fresh download:   python weather.py --refresh
Or import build_weather_enriched(yelp_df) from a pipeline script.

Design notes preserved from the original scripts:
  - One representative station per city/metro (usually a main airport): long,
    reliable daily records that are easy to reproduce and explain.
  - TOBS is not requested; weather_available therefore keeps its original
    meaning and coverage (see config.WEATHER_AVAILABILITY_COLS).
  - The merge is many-to-one (many reviews per city-date, one weather row),
    validated as such, with duplicate weather keys aggregated as a safeguard.
  - Missing weather stays NaN: it is structural (only selected cities were
    downloaded) and must not be treated as zero rain / normal temperature.
"""

import sys
from time import sleep

import pandas as pd
import requests

from config import (
    YELP_SAMPLE_PKL,
    WEATHER_RAW_CSV,
    WEATHER_CLEAN_CSV,
    WEATHER_ENRICHED_PKL,
    WEATHER_ENRICHED_CSV,
    WRITE_FULL_CSV_OUTPUTS,
    WEATHER_KEY_COLS,
    WEATHER_NUMERIC_COLS,
    WEATHER_BINARY_COLS,
    WEATHER_FEATURE_COLS,
    WEATHER_AVAILABILITY_COLS,
)


# ---------------------------------------------------------------------------
# NOAA download configuration
# ---------------------------------------------------------------------------

NOAA_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1"

# Requested NOAA data types (metric units). TOBS is intentionally not requested.
WEATHER_VARIABLES = [
    "PRCP",  # precipitation
    "SNOW",  # snowfall
    "SNWD",  # snow depth
    "TMAX",  # maximum temperature
    "TMIN",  # minimum temperature
]

# NOAA raw column -> project column.
RENAME_MAP = {
    "DATE": "review_date",
    "PRCP": "weather_prcp",
    "SNOW": "weather_snow",
    "SNWD": "weather_snow_depth",
    "TMAX": "weather_tmax",
    "TMIN": "weather_tmin",
}

# One representative station per city/metro area.
TOP_CITIES = [
    {"city": "Philadelphia", "state": "PA", "station": "USW00013739"},
    {"city": "New Orleans", "state": "LA", "station": "USW00012916"},
    {"city": "Nashville", "state": "TN", "station": "USW00013897"},
    {"city": "Tampa", "state": "FL", "station": "USW00012842"},
    {"city": "Indianapolis", "state": "IN", "station": "USW00093819"},
    {"city": "Tucson", "state": "AZ", "station": "USW00023160"},
    {"city": "Saint Louis", "state": "MO", "station": "USW00013994"},
    {"city": "Reno", "state": "NV", "station": "USW00023185"},
    {"city": "Santa Barbara", "state": "CA", "station": "USW00023190"},
    {"city": "Saint Petersburg", "state": "FL", "station": "USW00092806"},
    {"city": "Boise", "state": "ID", "station": "USW00024131"},
    {"city": "Edmonton", "state": "AB", "station": "CA003012209"},
    {"city": "Clearwater", "state": "FL", "station": "USW00012842"},
    {"city": "Metairie", "state": "LA", "station": "USW00012916"},
    {"city": "Sparks", "state": "NV", "station": "USW00023185"},
    {"city": "Franklin", "state": "TN", "station": "USW00013897"},
    {"city": "Wilmington", "state": "DE", "station": "USW00013781"},
    {"city": "Brandon", "state": "FL", "station": "USW00012842"},
    {"city": "Carmel", "state": "IN", "station": "USW00093819"},
    {"city": "Saint Pete Beach", "state": "FL", "station": "USW00012842"},
    {"city": "Goleta", "state": "CA", "station": "USW00023190"},
    {"city": "Cherry Hill", "state": "NJ", "station": "USW00013739"},
    {"city": "Meridian", "state": "ID", "station": "USW00024131"},
    {"city": "King of Prussia", "state": "PA", "station": "USW00013739"},
    {"city": "Dunedin", "state": "FL", "station": "USW00012842"},
]


# ---------------------------------------------------------------------------
# Download + clean
# ---------------------------------------------------------------------------

def load_yelp_data() -> pd.DataFrame:
    """Load the processed review-level Yelp dataset."""
    return pd.read_pickle(YELP_SAMPLE_PKL)


def get_review_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """
    Whole-year range (Jan 1 to Dec 31) spanning the observed review years, so
    every exact review date can find a matching weather day.
    """
    min_year = int(df["review_year"].min())
    max_year = int(df["review_year"].max())
    return f"{min_year}-01-01", f"{max_year}-12-31"


def request_noaa_weather(station: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download NOAA daily summaries for one station."""
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(WEATHER_VARIABLES),
        "format": "json",
        "units": "metric",
    }

    response = requests.get(NOAA_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def download_weather_for_top_cities(start_date: str, end_date: str) -> pd.DataFrame:
    """Download NOAA weather for every selected city."""
    weather_frames = []

    for city_info in TOP_CITIES:
        city = city_info["city"]
        state = city_info["state"]
        station = city_info["station"]

        print(f"Downloading NOAA weather: {city}, {state} ({station})")

        try:
            city_weather = request_noaa_weather(station, start_date, end_date)

            if city_weather.empty:
                print(f"No data returned for {city}, {state}")
                continue

            city_weather["city"] = city
            city_weather["state"] = state
            city_weather["station"] = station
            weather_frames.append(city_weather)

        except requests.HTTPError as error:
            print(f"HTTP error for {city}, {state}: {error}")
        except requests.RequestException as error:
            print(f"Request error for {city}, {state}: {error}")

        sleep(1)

    if not weather_frames:
        raise ValueError("No NOAA weather data were downloaded.")

    return pd.concat(weather_frames, ignore_index=True)


def clean_weather_data(weather: pd.DataFrame) -> pd.DataFrame:
    """Clean NOAA weather data and create interpretable weather variables."""
    weather = weather.copy()

    weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce")

    for col in WEATHER_VARIABLES:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather.rename(columns=RENAME_MAP)

    if "weather_tmax" in weather.columns and "weather_tmin" in weather.columns:
        weather["weather_temp_range"] = weather["weather_tmax"] - weather["weather_tmin"]

    if "weather_prcp" in weather.columns:
        weather["is_rainy"] = (weather["weather_prcp"].fillna(0) > 0).astype(int)

    if "weather_snow" in weather.columns or "weather_snow_depth" in weather.columns:
        snow = weather.get("weather_snow", 0)
        snow_depth = weather.get("weather_snow_depth", 0)
        weather["is_snowy"] = (
            pd.Series(snow).fillna(0) + pd.Series(snow_depth).fillna(0) > 0
        ).astype(int)

    if "weather_tmax" in weather.columns:
        weather["is_hot"] = (weather["weather_tmax"] >= 30).astype(int)

    if "weather_tmin" in weather.columns:
        weather["is_cold"] = (weather["weather_tmin"] <= 0).astype(int)

    columns_to_keep = (
        ["city", "state", "station", "review_date"] + WEATHER_FEATURE_COLS
    )
    columns_to_keep = [col for col in columns_to_keep if col in weather.columns]

    weather = weather[columns_to_keep].copy()
    weather = weather.sort_values(["city", "state", "review_date"])

    return weather


def get_clean_weather(yelp: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """
    Return the clean weather table, reusing cached files when possible.

    - If the clean CSV exists and refresh is False, load and return it.
    - Otherwise reuse the cached raw download if present (and not refreshing),
      else download fresh; then clean and cache both raw and clean files.
    """
    if WEATHER_CLEAN_CSV.exists() and not refresh:
        print(f"Using cached clean weather: {WEATHER_CLEAN_CSV}")
        return pd.read_csv(WEATHER_CLEAN_CSV)

    if WEATHER_RAW_CSV.exists() and not refresh:
        print(f"Using cached raw weather: {WEATHER_RAW_CSV}")
        weather_raw = pd.read_csv(WEATHER_RAW_CSV)
    else:
        start_date, end_date = get_review_date_range(yelp)
        print(f"NOAA date range: {start_date} to {end_date}")
        weather_raw = download_weather_for_top_cities(start_date, end_date)
        weather_raw.to_csv(WEATHER_RAW_CSV, index=False)
        print(f"Raw weather saved to: {WEATHER_RAW_CSV}")

    weather_clean = clean_weather_data(weather_raw)
    weather_clean.to_csv(WEATHER_CLEAN_CSV, index=False)
    print(f"Clean weather saved to: {WEATHER_CLEAN_CSV}")

    return weather_clean


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def prepare_merge_keys(
    yelp: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise city, state, and date keys on both sides."""
    yelp = yelp.copy()
    weather = weather.copy()

    for frame in (yelp, weather):
        frame["city"] = (
            frame["city"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        )
        frame["state"] = frame["state"].astype("string").str.strip().str.upper()
        # Normalise to midnight datetime64 so the merge key dtype stays
        # consistent with the Yelp dataset (avoids an object-dtype date column).
        frame["review_date"] = pd.to_datetime(
            frame["review_date"], errors="coerce"
        ).dt.normalize()

    return yelp, weather


def validate_weather_file(weather: pd.DataFrame) -> None:
    """Report duplicate city-state-date weather rows."""
    duplicate_count = weather.duplicated(subset=WEATHER_KEY_COLS).sum()

    print("\nWeather file validation")
    print("-" * 80)
    print(f"Weather rows: {weather.shape[0]}")
    print(f"Duplicate city-state-date rows: {duplicate_count}")

    if duplicate_count > 0:
        print("\nWARNING: Duplicate weather keys found; rows will be aggregated.")


def aggregate_duplicate_weather_rows(weather: pd.DataFrame) -> pd.DataFrame:
    """Ensure one weather row per city-state-date to protect the merge."""
    numeric_cols = [c for c in WEATHER_NUMERIC_COLS if c in weather.columns]
    binary_cols = [c for c in WEATHER_BINARY_COLS if c in weather.columns]

    agg_dict = {c: "mean" for c in numeric_cols}
    agg_dict.update({c: "max" for c in binary_cols})

    if "station" in weather.columns:
        agg_dict["station"] = "first"

    return weather.groupby(WEATHER_KEY_COLS, as_index=False).agg(agg_dict)


def merge_weather(yelp: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge NOAA weather into the review-level Yelp dataset (many-to-one)."""
    weather_columns = WEATHER_KEY_COLS + ["station"] + WEATHER_FEATURE_COLS
    weather_columns = [c for c in weather_columns if c in weather.columns]
    weather = weather[weather_columns].copy()

    enriched = yelp.merge(
        weather,
        on=WEATHER_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    available_cols = [c for c in WEATHER_AVAILABILITY_COLS if c in enriched.columns]
    enriched["weather_available"] = (
        enriched[available_cols].notna().any(axis=1).astype(int)
    )

    return enriched


def print_weather_merge_summary(
    yelp: pd.DataFrame,
    weather: pd.DataFrame,
    enriched: pd.DataFrame,
) -> None:
    """Print a summary of the weather merge."""
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
        .groupby(["city", "state"], dropna=False, observed=True)
        .agg(
            reviews=("review_id", "count"),
            weather_available_percent=("weather_available", lambda x: x.mean() * 100),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
        .head(30)
    )
    print(coverage_by_city.to_string(index=False))


def save_outputs(enriched: pd.DataFrame) -> None:
    """Save the weather-enriched dataset (pickle always; CSV only if enabled)."""
    enriched.to_pickle(WEATHER_ENRICHED_PKL)
    print("\nSaved weather-enriched files")
    print("-" * 80)
    print(f"PKL: {WEATHER_ENRICHED_PKL}")

    if WRITE_FULL_CSV_OUTPUTS:
        enriched.to_csv(WEATHER_ENRICHED_CSV, index=False)
        print(f"CSV: {WEATHER_ENRICHED_CSV}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_weather_enriched(
    yelp: pd.DataFrame | None = None,
    refresh: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the weather-enriched dataset end to end.

    If yelp is None it is loaded from YELP_SAMPLE_PKL, so this works both as a
    standalone stage and as a step called from a pipeline with the dataset
    already in memory.
    """
    if yelp is None:
        print("Loading processed Yelp dataset...")
        yelp = load_yelp_data()

    weather = get_clean_weather(yelp, refresh=refresh)

    print("Preparing merge keys...")
    yelp, weather = prepare_merge_keys(yelp, weather)

    validate_weather_file(weather)
    weather = aggregate_duplicate_weather_rows(weather)

    print("Merging NOAA weather into Yelp data...")
    enriched = merge_weather(yelp, weather)

    print_weather_merge_summary(yelp, weather, enriched)

    if save:
        save_outputs(enriched)

    return enriched


def main() -> None:
    refresh = "--refresh" in sys.argv
    build_weather_enriched(refresh=refresh)


if __name__ == "__main__":
    main()
