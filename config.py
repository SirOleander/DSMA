from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(r"C:\Users\fynne\University\Data Science and Marketing Analytics\data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Review sample size used to build the modeling/EDA dataset.
SAMPLE_SIZE = 4_622_091

# Derived dataset paths (single source of truth shared across scripts).
YELP_SAMPLE_PKL = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"
YELP_SAMPLE_CSV = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.csv"

WEATHER_RAW_CSV = EXTERNAL_DATA_DIR / "noaa_weather_raw_top_cities.csv"
WEATHER_CLEAN_CSV = EXTERNAL_DATA_DIR / "noaa_weather_daily_top_cities.csv"

WEATHER_ENRICHED_PKL = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k_weather.pkl"
WEATHER_ENRICHED_CSV = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k_weather.csv"