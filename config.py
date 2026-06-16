from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(r"C:\Users\fynne\University\Data Science and Marketing Analytics\data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------
# Review sample size used to build the modeling/EDA dataset.
SAMPLE_SIZE = 4_622_091

# ---------------------------------------------------------------------------
# Output options
# ---------------------------------------------------------------------------
# The pipeline always saves pickles (fast, typed, lossless). The full-size CSV
# copies of the ~4.6M-row datasets are large and slow to write and nothing in
# the pipeline reads them back, so they are off by default. Flip to True if you
# need a CSV to open externally.
WRITE_FULL_CSV_OUTPUTS = False

# ---------------------------------------------------------------------------
# Derived dataset paths (single source of truth shared across scripts)
# ---------------------------------------------------------------------------
YELP_SAMPLE_PKL = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"
YELP_SAMPLE_CSV = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.csv"

WEATHER_RAW_CSV = EXTERNAL_DATA_DIR / "noaa_weather_raw_top_cities.csv"
WEATHER_CLEAN_CSV = EXTERNAL_DATA_DIR / "noaa_weather_daily_top_cities.csv"

WEATHER_ENRICHED_PKL = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k_weather.pkl"
WEATHER_ENRICHED_CSV = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k_weather.csv"

# ---------------------------------------------------------------------------
# Modelling metadata (single source of truth for downstream scripts)
# ---------------------------------------------------------------------------
ID_COLS = ["review_id", "business_id", "user_id"]

# Columns that must be excluded from modelling because they leak the target.
# - review_stars: satisfied is derived directly from it.
# - business_stars / user_average_stars: all-time averages that already include
#   the current review's rating (target + temporal leakage). Kept for EDA only.
LEAKAGE_COLS = ["review_stars", "business_stars", "user_average_stars"]

# ---------------------------------------------------------------------------
# Weather schema (single source of truth for the NOAA download + merge)
# ---------------------------------------------------------------------------
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

# Columns used to decide whether a weather record was matched. Deliberately
# excludes any TOBS-based column so weather_available keeps the exact same
# meaning (and coverage %) it had before TOBS was ever requested.
WEATHER_AVAILABILITY_COLS = WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS
