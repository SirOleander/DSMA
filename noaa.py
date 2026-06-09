from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests

from config import PROCESSED_DATA_DIR, DATA_DIR


SAMPLE_SIZE = 3_000_000

YELP_DATA_FILE = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"

EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_WEATHER_OUTPUT = EXTERNAL_DATA_DIR / "noaa_weather_raw_top_cities.csv"
CLEAN_WEATHER_OUTPUT = EXTERNAL_DATA_DIR / "noaa_weather_daily_top_cities.csv"


NOAA_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1"

WEATHER_VARIABLES = [
    "PRCP",  # precipitation
    "SNOW",  # snowfall
    "SNWD",  # snow depth
    "TMAX",  # maximum temperature
    "TMIN",  # minimum temperature
    "TOBS",  # observed temperature
]


TOP_CITIES = [
    {
        "city": "Philadelphia",
        "state": "PA",
        "station": "GHCND:USW00013739",
    },
    {
        "city": "New Orleans",
        "state": "LA",
        "station": "GHCND:USW00012916",
    },
    {
        "city": "Nashville",
        "state": "TN",
        "station": "GHCND:USW00013897",
    },
    {
        "city": "Tampa",
        "state": "FL",
        "station": "GHCND:USW00012842",
    },
    {
        "city": "Indianapolis",
        "state": "IN",
        "station": "GHCND:USW00093819",
    },
    {
        "city": "Tucson",
        "state": "AZ",
        "station": "GHCND:USW00023160",
    },
    {
        "city": "Reno",
        "state": "NV",
        "station": "GHCND:USW00023185",
    },
    {
        "city": "Saint Louis",
        "state": "MO",
        "station": "GHCND:USW00013994",
    },
    {
        "city": "Santa Barbara",
        "state": "CA",
        "station": "GHCND:USW00023190",
    },
    {
        "city": "Boise",
        "state": "ID",
        "station": "GHCND:USW00024131",
    },
]


def load_yelp_data() -> pd.DataFrame:
    """Load processed Yelp dataset."""
    df = pd.read_pickle(YELP_DATA_FILE)
    df["review_date"] = pd.to_datetime(df["review_year"].astype(str) + "-" +
                                       df["review_month"].astype(str) + "-01",
                                       errors="coerce")

    return df


def get_review_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """
    Get approximate date range from processed Yelp data.

    Since the final processed dataset currently keeps year/month/weekday but not
    the original review date, we use the available year range here.
    """
    min_year = int(df["review_year"].min())
    max_year = int(df["review_year"].max())

    start_date = f"{min_year}-01-01"
    end_date = f"{max_year}-12-31"

    return start_date, end_date


def request_noaa_weather(
    station: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download NOAA daily weather data for one station.
    """
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": ",".join(WEATHER_VARIABLES),
        "format": "json",
        "units": "metric",
    }

    response = requests.get(
        NOAA_ENDPOINT,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def download_weather_for_top_cities(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download NOAA weather data for all selected cities.
    """
    weather_frames = []

    for city_info in TOP_CITIES:
        city = city_info["city"]
        state = city_info["state"]
        station = city_info["station"]

        print(f"Downloading NOAA weather: {city}, {state} ({station})")

        try:
            city_weather = request_noaa_weather(
                station=station,
                start_date=start_date,
                end_date=end_date
            )

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
    """
    Clean NOAA weather data and create interpretable weather variables.
    """
    weather = weather.copy()

    weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce")

    for col in WEATHER_VARIABLES:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    rename_map = {
        "DATE": "review_date",
        "PRCP": "weather_prcp",
        "SNOW": "weather_snow",
        "SNWD": "weather_snow_depth",
        "TMAX": "weather_tmax",
        "TMIN": "weather_tmin",
        "TOBS": "weather_tobs",
    }

    weather = weather.rename(columns=rename_map)

    if "weather_tmax" in weather.columns and "weather_tmin" in weather.columns:
        weather["weather_temp_range"] = (
            weather["weather_tmax"] - weather["weather_tmin"]
        )

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

    columns_to_keep = [
        "city",
        "state",
        "station",
        "review_date",
        "weather_prcp",
        "weather_snow",
        "weather_snow_depth",
        "weather_tmax",
        "weather_tmin",
        "weather_tobs",
        "weather_temp_range",
        "is_rainy",
        "is_snowy",
        "is_hot",
        "is_cold",
    ]

    columns_to_keep = [col for col in columns_to_keep if col in weather.columns]

    weather = weather[columns_to_keep].copy()

    weather = weather.sort_values(["city", "state", "review_date"])

    return weather


def main() -> None:
    print("Loading processed Yelp dataset...")
    yelp = load_yelp_data()

    start_date, end_date = get_review_date_range(yelp)

    print(f"NOAA date range: {start_date} to {end_date}")

    weather_raw = download_weather_for_top_cities(
        start_date=start_date,
        end_date=end_date
    )

    weather_raw.to_csv(RAW_WEATHER_OUTPUT, index=False)

    weather_clean = clean_weather_data(weather_raw)

    weather_clean.to_csv(CLEAN_WEATHER_OUTPUT, index=False)

    print("\nNOAA weather download completed.")
    print(f"Raw weather saved to: {RAW_WEATHER_OUTPUT}")
    print(f"Clean weather saved to: {CLEAN_WEATHER_OUTPUT}")
    print(f"Clean weather shape: {weather_clean.shape}")

    print("\nWeather columns:")
    print(weather_clean.columns.tolist())

    print("\nPreview:")
    print(weather_clean.head())


if __name__ == "__main__":
    main()