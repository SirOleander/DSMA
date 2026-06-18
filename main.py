"""
Yelp restaurant-satisfaction project - consolidated single-file version.

The project is delivered as one file (for the report appendix) and runs end to
end in a single pass, entirely in memory:

    1. Import data  -> 2. Process data (+ NOAA weather enrichment)
    -> 3. EDA  -> 4. Feature engineering  -> 5. Modelling  -> 6. Results

Just run the file:

    python main.py

Only two files are written: the final enriched dataset and a cached copy of the
NOAA weather download (so reruns don't re-download). Plots and the model-metrics
table are written to the output folders.
"""

import json
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ================================================================================
# CONFIGURATION
# ================================================================================

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
# Saved files (single source of truth)
# ---------------------------------------------------------------------------
# The whole pipeline runs in memory in one pass. Only two files are written:
#   - DATASET_PKL: the final weather-enriched dataset (the one full dataset).
#   - WEATHER_CACHE_PKL: a cache of the cleaned NOAA download so repeated runs
#     don't re-hit the NOAA API. Both are pickles (fast, lossless, and they
#     preserve the optimised dtypes); they are never opened outside this code.
DATASET_PKL = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k_weather.pkl"
WEATHER_CACHE_PKL = EXTERNAL_DATA_DIR / "noaa_weather_daily_top_cities.pkl"

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

# Global random seed (single source of truth).
RANDOM_STATE = 42


# ================================================================================
# SHARED REPORTING HELPERS
# ================================================================================

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

# ================================================================================
# 1. IMPORT DATA  (parse raw pgAdmin JSON exports)
# ================================================================================

RAW_FILES = {
    "business": "restaurant_business_raw.csv",
    "reviews": "restaurant_reviews_raw.csv",
    "users": "restaurant_users_raw.csv",
    "checkin": "restaurant_checkin_raw.csv",
    "tip": "restaurant_tip_raw.csv",
    "photo": "restaurant_photo_raw.csv",
}


def check_required_files() -> None:
    """Check whether all required raw CSV files exist."""
    missing_files = []

    for filename in RAW_FILES.values():
        path = RAW_DATA_DIR / filename

        if not path.exists():
            missing_files.append(path)

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "The following required raw files are missing:\n"
            f"{missing_text}"
        )


def load_json_csv(filename: str) -> pd.DataFrame:
    """
    Load a CSV file exported from pgAdmin.

    Expected structure:
    - one column
    - each row contains one JSON object as text
    """
    path = RAW_DATA_DIR / filename

    raw = pd.read_csv(path)

    if raw.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one column in {filename}, "
            f"but found {raw.shape[1]} columns."
        )

    json_col = raw.columns[0]

    try:
        df = pd.json_normalize(raw[json_col].apply(json.loads))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON conversion failed for {filename}. "
            "Please check whether the file was exported correctly."
        ) from error

    return df


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Parse the raw pgAdmin JSON exports into in-memory tables."""
    check_required_files()

    tables = {}
    for name, filename in RAW_FILES.items():
        df = load_json_csv(filename)
        tables[name] = df
        print(f"Loaded {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("Raw data loading completed.")
    return tables

# ================================================================================
# 2. PROCESS DATA  (clean, merge, sample, log features)
# ================================================================================

# ---------------------------------------------------------------------------
# Study scope
# ---------------------------------------------------------------------------

MIN_REVIEW_YEAR = 2010

SELECTED_CITY_STATE_SCOPE = [
    ("Philadelphia", "PA"),
    ("New Orleans", "LA"),
    ("Nashville", "TN"),
    ("Tampa", "FL"),
    ("Indianapolis", "IN"),
    ("Tucson", "AZ"),
    ("Saint Louis", "MO"),
    ("Reno", "NV"),
    ("Santa Barbara", "CA"),
    ("Saint Petersburg", "FL"),
]


# String / categorical columns. Single source of truth used both for the
# "Unknown" cleaning step and for the category-dtype memory optimisation.
CATEGORICAL_COLUMNS = [
    "city",
    "state",
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

# Count columns whose missing values genuinely mean "no record" -> fill with 0.
COUNT_COLUMNS = [
    "checkin_count",
    "tip_count",
    "tip_compliment_count",
    "photo_count",
    "photo_drink",
    "photo_food",
    "photo_inside",
    "photo_menu",
    "photo_outside",
]


def prepare_core_tables(
    business: pd.DataFrame,
    reviews: pd.DataFrame,
    users: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rename columns and create target variable."""

    business = business.rename(columns={
        "name": "business_name",
        "stars": "business_stars",
        "review_count": "business_review_count",
    })

    reviews = reviews.rename(columns={
        "stars": "review_stars",
        "date": "review_date",
        "text": "review_text",
        "useful": "review_useful",
        "funny": "review_funny",
        "cool": "review_cool",
    })

    users = users.rename(columns={
        "name": "user_name",
        "review_count": "user_review_count",
        "average_stars": "user_average_stars",
        "useful": "user_useful",
        "funny": "user_funny",
        "cool": "user_cool",
        "fans": "user_fans",
    })

    reviews["satisfied"] = (reviews["review_stars"] >= 4).astype(int)

    return business, reviews, users


def select_columns_before_merge(
    business: pd.DataFrame,
    reviews: pd.DataFrame,
    users: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep only relevant columns before merging.

    This reduces memory usage and runtime because unused columns are not repeated
    across millions of review rows.
    """

    reviews = reviews.copy()

    if "review_text" in reviews.columns:
        reviews["review_text_length"] = reviews["review_text"].fillna("").str.len()
        reviews = reviews.drop(columns=["review_text"])

    business_columns = [
        "business_id",
        "business_stars",
        "business_review_count",
        "is_open",
        "city",
        "state",
        "latitude",
        "longitude",
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

    review_columns = [
        "review_id",
        "business_id",
        "user_id",
        "satisfied",
        "review_stars",
        "review_date",
        "review_text_length",
        "review_useful",
        "review_funny",
        "review_cool",
    ]

    user_columns = [
        "user_id",
        "user_review_count",
        "user_average_stars",
        "user_fans",
        "user_useful",
        "user_funny",
        "user_cool",
    ]

    business = business[
        [col for col in business_columns if col in business.columns]
    ].copy()

    reviews = reviews[
        [col for col in review_columns if col in reviews.columns]
    ].copy()

    users = users[
        [col for col in user_columns if col in users.columns]
    ].copy()

    return business, reviews, users


def sample_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Sample reviews before merging to keep processing manageable.

    When SAMPLE_SIZE covers the whole table (the current setting), we return
    the data as-is instead of shuffling several million rows for no reason.
    """
    if SAMPLE_SIZE >= len(reviews):
        return reviews.copy()

    return reviews.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()


def create_checkin_features(checkin: pd.DataFrame) -> pd.DataFrame:
    """Create business-level check-in features."""
    checkin = checkin.copy()

    checkin["checkin_count"] = checkin["date"].fillna("").apply(
        lambda x: 0 if x == "" else len(str(x).split(","))
    )

    return checkin[["business_id", "checkin_count"]]


def create_tip_features(tip: pd.DataFrame) -> pd.DataFrame:
    """Create business-level tip features."""
    tip_features = tip.groupby("business_id").agg(
        tip_count=("text", "count"),
        tip_compliment_count=("compliment_count", "sum"),
    ).reset_index()

    return tip_features


def create_photo_features(photo: pd.DataFrame) -> pd.DataFrame:
    """Create business-level photo features."""
    photo_features = photo.groupby("business_id").agg(
        photo_count=("photo_id", "count")
    ).reset_index()

    photo_label_features = (
        photo.pivot_table(
            index="business_id",
            columns="label",
            values="photo_id",
            aggfunc="count",
            fill_value=0,
        )
        .add_prefix("photo_")
        .reset_index()
    )

    photo_features = photo_features.merge(
        photo_label_features,
        on="business_id",
        how="left"
    )

    return photo_features


def merge_tables(
    reviews: pd.DataFrame,
    business: pd.DataFrame,
    users: pd.DataFrame,
    checkin_features: pd.DataFrame,
    tip_features: pd.DataFrame,
    photo_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge all information into one review-level dataset."""

    dataset = reviews.merge(
        business,
        on="business_id",
        how="left"
    )

    dataset = dataset.merge(
        users,
        on="user_id",
        how="left"
    )

    dataset = dataset.merge(
        checkin_features,
        on="business_id",
        how="left"
    )

    dataset = dataset.merge(
        tip_features,
        on="business_id",
        how="left"
    )

    dataset = dataset.merge(
        photo_features,
        on="business_id",
        how="left"
    )

    return dataset

def clean_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean obvious missing values and create simple derived variables."""
    dataset = dataset.copy()

    for col in COUNT_COLUMNS:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0)

    for col in CATEGORICAL_COLUMNS:
        if col in dataset.columns:
            dataset[col] = (
                dataset[col]
                .astype("string")
                .replace({"None": "Unknown", "nan": "Unknown"})
                .fillna("Unknown")
            )

    if "review_date" in dataset.columns:
        dataset["review_date"] = pd.to_datetime(dataset["review_date"], errors="coerce")
        dataset["review_year"] = dataset["review_date"].dt.year
        dataset["review_month"] = dataset["review_date"].dt.month
        dataset["review_weekday"] = dataset["review_date"].dt.dayofweek
    
    review_vote_columns = [
        "review_useful",
        "review_funny",
        "review_cool",
    ]

    for col in review_vote_columns:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
            dataset[col] = dataset[col].clip(lower=0)

    return dataset

def standardize_city_names(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize obvious duplicate city names and spelling variants.

    This avoids treating the same city as multiple cities due to casing,
    punctuation, abbreviations, or spelling differences.
    """
    dataset = dataset.copy()

    if "city" not in dataset.columns:
        return dataset

    dataset["city"] = (
        dataset["city"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Only keep city-standardisation rules that can map into the final
    # top-10 study scope. Cities outside this scope are intentionally not
    # standardised here because filter_study_scope() removes them later.
    city_replacements = {
        # Philadelphia
        'Bala Cynwyd': 'Philadelphia',
        'Center City': 'Philadelphia',
        'Chestnut Hill': 'Philadelphia',
        'East Passyunk': 'Philadelphia',
        'Fishtown': 'Philadelphia',
        'Manayunk': 'Philadelphia',
        'Mount Airy': 'Philadelphia',
        'Mt Airy': 'Philadelphia',
        'Northern Liberties': 'Philadelphia',
        'Old City': 'Philadelphia',
        'Passyunk Square': 'Philadelphia',
        'Phialdelphia': 'Philadelphia',
        'Phila': 'Philadelphia',
        'PHILADELPHIA': 'Philadelphia',
        'philadelphia': 'Philadelphia',
        'Philadelphia ': 'Philadelphia',
        'Philly': 'Philadelphia',
        'Queen Village': 'Philadelphia',
        'Rittenhouse': 'Philadelphia',
        'Rittenhouse Square': 'Philadelphia',
        'Roxborough': 'Philadelphia',
        'South Philadelphia': 'Philadelphia',
        'University City': 'Philadelphia',

        # New Orleans
        'Algiers': 'New Orleans',
        'Bucktown': 'New Orleans',
        'Bywater': 'New Orleans',
        'Downtown': 'New Orleans',
        'new orleans': 'New Orleans',
        'NEW ORLEANS': 'New Orleans',

        # Nashville
        'Antioch': 'Nashville',
        'Belle Meade': 'Nashville',
        'Berry Hill': 'Nashville',
        'Cane Ridge': 'Nashville',
        'DONELSON': 'Nashville',
        'Donelson': 'Nashville',
        'East Nashville': 'Nashville',
        'Hermitage': 'Nashville',
        'Inglewood': 'Nashville',
        'MADISON': 'Nashville',
        'Madison': 'Nashville',
        'NASHVILLE': 'Nashville',
        'nashville': 'Nashville',
        'Nashville ': 'Nashville',
        'Nashville-Davidson metropolitan gover': 'Nashville',
        'Old Hickory': 'Nashville',

        # Tampa
        'Carrollwood': 'Tampa',
        'Greater Northdale': 'Tampa',
        'South Tampa': 'Tampa',
        'Southwest Tampa': 'Tampa',
        'TAMPA': 'Tampa',
        'tampa': 'Tampa',
        'Tampa Bay': 'Tampa',
        'Tampa,Fl': 'Tampa',
        'Temple Terr': 'Tampa',
        'TEMPLE TERR': 'Tampa',
        'Temple Terrace': 'Tampa',
        "Town 'n' Country": 'Tampa',
        "Town 'N' Country": 'Tampa',
        'Town n Country': 'Tampa',
        'Westchase': 'Tampa',

        # Indianapolis
        'Indianaplois': 'Indianapolis',
        'INDIANAPOLIS': 'Indianapolis',
        'indianapolis': 'Indianapolis',
        'indianopolis': 'Indianapolis',
        'INpolis': 'Indianapolis',

        # Tucson
        'tucson': 'Tucson',
        'TUCSON': 'Tucson',

        # Saint Louis
        'SAINT LOUIS': 'Saint Louis',
        'saint louis': 'Saint Louis',
        'Saint Louis,': 'Saint Louis',
        'St Louis': 'Saint Louis',
        'st louis': 'Saint Louis',
        'ST LOUIS': 'Saint Louis',
        'St Louis County': 'Saint Louis',
        'St Louis Downtown': 'Saint Louis',
        'St. Louis': 'Saint Louis',
        'ST. Louis': 'Saint Louis',
        'ST. LOUIS': 'Saint Louis',
        'st. louis': 'Saint Louis',
        'St. Louis County': 'Saint Louis',
        'St.Louis': 'Saint Louis',

        # Reno
        'RENO': 'Reno',
        'reno': 'Reno',

        # Santa Barbara
        'SANTA BARBARA': 'Santa Barbara',
        'santa barbara': 'Santa Barbara',
        'Santa barbra': 'Santa Barbara',
        'Santa Barbra': 'Santa Barbara',
        'santa barbra': 'Santa Barbara',

        # Saint Petersburg
        'saint petersburg': 'Saint Petersburg',
        'SAINT PETERSBURG': 'Saint Petersburg',
        'Saint Petersburg': 'Saint Petersburg',
        'Saintt Petersburg': 'Saint Petersburg',
        'St Pete': 'Saint Petersburg',
        'St Petersburg': 'Saint Petersburg',
        'st petersburg': 'Saint Petersburg',
        'St. Petersburg': 'Saint Petersburg',
        'ST. PETERSBURG': 'Saint Petersburg',
        'st. petersburg': 'Saint Petersburg',
        'St.Petersburg': 'Saint Petersburg',

    }

    dataset["city"] = dataset["city"].replace(city_replacements)

    return dataset


def add_log_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of skewed non-negative count variables."""
    dataset = dataset.copy()

    log_candidates = [
        # Business activity / popularity
        "business_review_count",

        # User activity / history
        "user_review_count",
        "user_useful",
        "user_funny",
        "user_cool",
        "user_fans",

        # Business engagement / activity
        "checkin_count",
        "tip_count",
        "tip_compliment_count",
        "photo_count",
        "photo_food",
        "photo_drink",
        "photo_menu",
        "photo_inside",
        "photo_outside",

        # Review-level behavior
        "review_useful",
        "review_funny",
        "review_cool",
        "review_text_length",
    ]

    for col in log_candidates:
        if col in dataset.columns:
            values = pd.to_numeric(dataset[col], errors="coerce")
            # log1p of the non-negative value. Missing values stay NaN on
            # purpose: imputation is deferred to the modelling pipeline so it
            # can be fit on training data only. Genuine count columns have
            # already been 0-filled in clean_values(), so their logs are 0.
            dataset[f"log_{col}"] = np.log1p(values.clip(lower=0))

    return dataset

def filter_study_scope(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataset to the selected study scope.

    Scope:
    - selected top 10 city-state restaurant markets
    - reviews from 2010 onward, including 2010

    This filter is applied after city standardization and date cleaning, so the
    selected city names match the cleaned city names and the NOAA merge keys.
    """
    dataset = dataset.copy()

    required_columns = {"city", "state", "review_year"}
    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            "Cannot filter study scope because these columns are missing: "
            f"{sorted(missing_columns)}"
        )

    rows_before = len(dataset)

    selected_pairs = pd.DataFrame(
        SELECTED_CITY_STATE_SCOPE,
        columns=["city", "state"],
    )

    dataset = dataset[dataset["review_year"] >= MIN_REVIEW_YEAR].copy()

    dataset = dataset.merge(
        selected_pairs,
        on=["city", "state"],
        how="inner",
    )

    rows_after = len(dataset)

    print("\nApplied study-scope filter.")
    print(f"Rows before filter: {rows_before:,}")
    print(f"Rows after filter:  {rows_after:,}")
    print(f"Minimum review year: {MIN_REVIEW_YEAR}")

    print("\nReviews by selected city-state:")
    city_summary = (
        dataset.groupby(["city", "state"], observed=True)
        .size()
        .reset_index(name="review_count")
        .sort_values("review_count", ascending=False)
    )
    print(city_summary.to_string(index=False))

    print("\nReviews by year:")
    year_summary = (
        dataset.groupby("review_year", observed=True)
        .size()
        .reset_index(name="review_count")
        .sort_values("review_year")
    )
    print(year_summary.to_string(index=False))

    missing_scope_pairs = selected_pairs.merge(
        city_summary[["city", "state"]],
        on=["city", "state"],
        how="left",
        indicator=True,
    )

    missing_scope_pairs = missing_scope_pairs[
        missing_scope_pairs["_merge"] == "left_only"
    ][["city", "state"]]

    if not missing_scope_pairs.empty:
        print("\nWARNING: These selected city-state pairs have zero rows:")
        print(missing_scope_pairs.to_string(index=False))

    if dataset.empty:
        raise ValueError("Study-scope filter removed all rows.")

    return dataset

def select_model_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select final columns for EDA and modeling."""

    model_columns = [
        # IDs
        "review_id",
        "business_id",
        "user_id",

        # Target
        "satisfied",

        # LEAKAGE WARNING: keep for EDA only, exclude from modelling.
        # review_stars: the target is derived directly from it.
        # business_stars / user_average_stars (below): all-time averages that
        # already include this review's rating, so they leak the target and are
        # also temporal leakage. See config.LEAKAGE_COLS.
        "review_stars",

        # Review-level variables
        "review_text_length",
        "review_useful",
        "review_funny",
        "review_cool",
        "log_review_useful",
        "log_review_funny",
        "log_review_cool",
        "log_review_text_length",
        "review_year",
        "review_month",
        "review_weekday",
        "review_date",

        # Business-level variables
        "business_stars",
        "business_review_count",
        "log_business_review_count",
        "is_open",
        "city",
        "state",
        "latitude",
        "longitude",
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

        # User-level variables
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

        # Restaurant activity features
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
        "log_photo_food",
        "log_photo_drink",
        "log_photo_menu",
        "log_photo_inside",
        "log_photo_outside",
    ]

    available_columns = [col for col in model_columns if col in dataset.columns]

    missing_columns = [col for col in model_columns if col not in dataset.columns]
    if missing_columns:
        print("\nColumns not found and therefore skipped:")
        print(missing_columns)

    return dataset[available_columns].copy()

def optimize_dtypes(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory use without changing any values.

    - Known string/attribute columns become 'category'.
    - Integer-valued numeric columns with no missing values are downcast to the
      smallest safe integer type. Columns containing NaN (e.g. the 7 unmatched
      users) or genuine floats (star ratings, log features) are left untouched.
    """
    dataset = dataset.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype("category")

    for col in dataset.select_dtypes(include="number").columns:
        values = dataset[col]
        if values.isna().any():
            continue
        if (values == values.round()).all():
            dataset[col] = pd.to_numeric(values, downcast="integer")

    return dataset


def process_data(tables: dict) -> pd.DataFrame:
    """Build the review-level modelling dataset from the raw tables (in memory)."""
    business = tables["business"]
    reviews = tables["reviews"]
    users = tables["users"]
    checkin = tables["checkin"]
    tip = tables["tip"]
    photo = tables["photo"]

    business, reviews, users = prepare_core_tables(
        business=business,
        reviews=reviews,
        users=users,
    )

    business, reviews, users = select_columns_before_merge(
        business=business,
        reviews=reviews,
        users=users,
    )

    review_sample = sample_reviews(reviews)

    checkin_features = create_checkin_features(checkin)
    tip_features = create_tip_features(tip)
    photo_features = create_photo_features(photo)

    print("\nMerging tables...")
    dataset = merge_tables(
        reviews=review_sample,
        business=business,
        users=users,
        checkin_features=checkin_features,
        tip_features=tip_features,
        photo_features=photo_features,
    )

    print("\nCleaning values...")
    dataset = clean_values(dataset)
    dataset = standardize_city_names(dataset)

    print("\nFiltering study scope...")
    dataset = filter_study_scope(dataset)

    print("\nAdding log features...")
    dataset = add_log_features(dataset)

    print("\nSelecting final columns...")
    model_data = select_model_columns(dataset)

    print("\nOptimizing dtypes...")
    model_data = optimize_dtypes(model_data)

    print("\nProcessing completed.")
    return model_data


# ================================================================================
# 2b. WEATHER ENRICHMENT  (download + merge NOAA daily weather)
# ================================================================================

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
]


# ---------------------------------------------------------------------------
# Download + clean
# ---------------------------------------------------------------------------

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
    Return the clean weather table, reusing the cached download when possible.

    If the cache exists and refresh is False, load it. Otherwise download from
    NOAA, clean it, cache it as a single pickle, and return it.
    """
    if WEATHER_CACHE_PKL.exists() and not refresh:
        print(f"Using cached weather: {WEATHER_CACHE_PKL}")
        return pd.read_pickle(WEATHER_CACHE_PKL)

    start_date, end_date = get_review_date_range(yelp)
    print(f"NOAA date range: {start_date} to {end_date}")
    weather_raw = download_weather_for_top_cities(start_date, end_date)

    weather_clean = clean_weather_data(weather_raw)
    weather_clean.to_pickle(WEATHER_CACHE_PKL)
    print(f"Weather cache saved to: {WEATHER_CACHE_PKL}")

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


def save_weather_outputs(enriched: pd.DataFrame) -> None:
    """Save the final weather-enriched dataset (the one full dataset)."""
    enriched.to_pickle(DATASET_PKL)
    print(f"\nSaved final dataset: {DATASET_PKL}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_weather_enriched(yelp: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Download/merge NOAA weather into the review-level dataset and save it."""
    weather = get_clean_weather(yelp, refresh=refresh)

    print("Preparing merge keys...")
    yelp, weather = prepare_merge_keys(yelp, weather)

    validate_weather_file(weather)
    weather = aggregate_duplicate_weather_rows(weather)

    print("Merging NOAA weather into Yelp data...")
    enriched = merge_weather(yelp, weather)

    save_weather_outputs(enriched)
    return enriched

# ================================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ================================================================================

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

def eda_main(df: pd.DataFrame) -> None:
    print_section("Exploratory data analysis")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

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

# ================================================================================
# 4. FEATURE ENGINEERING  (leakage-safe feature set + preprocessing)
# ================================================================================

TARGET = "satisfied"
DATE_COL = "review_date"

# Votes a review accumulates from OTHER users after it is posted. They are not
# available at the moment a review is written, so under a "predict at posting
# time" framing they are look-ahead leakage. Excluded by default; flip
# include_review_votes=True only if you adopt a purely descriptive framing.
REVIEW_VOTE_COLS = [
    "review_useful", "review_funny", "review_cool",
    "log_review_useful", "log_review_funny", "log_review_cool",
]


def select_features(
    df: pd.DataFrame,
    include_review_votes: bool = False,
    include_city: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) with a leakage-safe feature set.

    Dropped: IDs, review_date, the leakage columns (review_stars,
    business_stars, user_average_stars), each raw count that has a log_ twin,
    and -- by default -- the review-vote columns and the very high-cardinality
    'city' column (state already captures the regional signal cleanly).
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    drop = set(ID_COLS) | {DATE_COL} | set(LEAKAGE_COLS)

    # 'station' is the NOAA station id added by the weather merge: an
    # identifier and a near-duplicate of the high-cardinality 'city' we drop.
    drop.add("station")

    if not include_review_votes:
        drop |= set(REVIEW_VOTE_COLS)

    if not include_city:
        drop.add("city")

    # For every raw count with a log_ version, keep the log and drop the raw.
    for col in list(df.columns):
        if f"log_{col}" in df.columns:
            drop.add(col)

    drop.discard(TARGET)

    feature_cols = [c for c in df.columns if c not in drop and c != TARGET]

    X = df[feature_cols].copy()

    # Weekday and month are cyclical / non-monotonic: treat them as categorical
    # so they are one-hot / natively encoded rather than read as ordinal numbers
    # (e.g. Sunday=6 is not "greater than" Monday=0). review_year stays numeric
    # because a monotonic time trend there is meaningful.
    for col in ["review_weekday", "review_month"]:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y = df[TARGET].astype(int)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split (preserves the satisfied class balance)."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical lists."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    return numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build an UNFITTED preprocessing ColumnTransformer.

    - Numeric: median imputation (covers the 7 unmatched users and the
      structural weather NaNs), then optional standardisation. Set
      scale_numeric=False for tree-based models, which don't need scaling.
    - Categorical: constant 'Unknown' imputation, then one-hot encoding with
      handle_unknown='ignore' so unseen categories at test time are safe.

    weather_available stays in the numeric block, so the model can still tell
    weather-covered rows apart from imputed ones.
    """
    numeric, categorical = split_feature_types(X)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    return preprocessor, numeric, categorical


# ================================================================================
# 5. MODELLING  (train candidate models)
# ================================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROCESSED_DATA_DIR / "model_outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Light cross-validation for stability. Off by default because k-fold on
# millions of rows is slow; when on, it runs on a stratified subsample.
DO_CV = False
CV_FOLDS = 3
CV_SUBSAMPLE = 300_000


# ---------------------------------------------------------------------------
# Tree-specific preprocessor (compact, no one-hot blow-up)
# ---------------------------------------------------------------------------

def build_tree_preprocessor(X: pd.DataFrame):
    """
    Preprocessor for tree models: median-impute numerics (no scaling) and
    ordinal-encode categoricals so HistGradientBoosting can treat them as true
    categoricals natively. Returns (preprocessor, categorical_mask) where the
    mask aligns with the transformed column order [numeric..., categorical...].
    """
    numeric, categorical = split_feature_types(X)

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,          # unseen categories -> treated as missing
            encoded_missing_value=np.nan,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    return preprocessor, categorical_mask


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models(X_train: pd.DataFrame) -> dict:
    """Build the candidate models, each as a complete leakage-safe Pipeline."""
    linear_pre, _, _ = build_preprocessor(X_train, scale_numeric=True)
    tree_pre, cat_mask = build_tree_preprocessor(X_train)

    return {
        "baseline_most_frequent": Pipeline([
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("pre", linear_pre),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("pre", tree_pre),
            ("model", HistGradientBoostingClassifier(
                categorical_features=cat_mask,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Classification metrics suited to an imbalanced binary target."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def get_scores(model, X) -> np.ndarray | None:
    """Positive-class probabilities if available, else None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(models, X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict]:
    """Fit each model on train, evaluate on the held-out test set."""
    rows = []
    fitted = {}

    for name, model in models.items():
        print(f"  fitting {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)

        row = {"model": name}
        row.update(compute_metrics(y_test, y_pred, y_score))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return results, fitted


def run_cross_validation(models, X_train, y_train) -> pd.DataFrame:
    """Optional light stratified CV on a subsample, for stability estimates."""
    if len(X_train) > CV_SUBSAMPLE:
        Xs = X_train.sample(CV_SUBSAMPLE, random_state=RANDOM_STATE)
        ys = y_train.loc[Xs.index]
    else:
        Xs, ys = X_train, y_train

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision"]

    rows = []
    for name, model in models.items():
        print(f"  cross-validating {name} ...")
        scores = cross_validate(model, Xs, ys, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name, "cv_rows": len(Xs)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def plot_confusion(model, X_test, y_test, name: str) -> None:
    """Save a confusion matrix for the given fitted model."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["not satisfied", "satisfied"])
    disp.plot(cmap="Blues", values_format=",")
    plt.title(f"Confusion matrix: {name}")
    plt.tight_layout()
    path = PLOT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def models_main(df: pd.DataFrame) -> None:
    print("Building feature set...")
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows "
          f"| positive rate {y.mean() * 100:.2f}%")

    models = build_models(X_train)

    print("\nEvaluating on held-out test set...")
    results, fitted = evaluate_holdout(models, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON (held-out test set)")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(results.to_string(index=False))

    results.to_csv(OUTPUT_DIR / "model_comparison_holdout.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison_holdout.csv'}")

    best_name = results.iloc[0]["model"]
    print(f"\nBest model by ROC-AUC (baseline-proof): {best_name}")
    plot_confusion(fitted[best_name], X_test, y_test, best_name)

    if DO_CV:
        print("\nRunning light cross-validation (subsample)...")
        cv_results = run_cross_validation(models, X_train, y_train)
        print("\n" + "=" * 100)
        print(f"CROSS-VALIDATION ({CV_FOLDS}-fold, subsample)")
        print("=" * 100)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(cv_results.to_string(index=False))
        cv_results.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    print("\nModel comparison completed.")

# ================================================================================
# 6. RESULTS & COMMAND-LINE ENTRY POINTS
# ================================================================================

def main() -> None:
    """Run the project end to end.

    The expensive stages (import raw -> process -> weather) run only once: they
    save the final dataset to DATASET_PKL. On every later run that file is loaded
    directly and we go straight to EDA + modelling, so the raw data is not
    re-parsed each time. To force a rebuild, delete DATASET_PKL or set
    REBUILD = True.
    """
    REBUILD = False

    if DATASET_PKL.exists() and not REBUILD:
        print(f"Loading existing dataset: {DATASET_PKL}")
        enriched = pd.read_pickle(DATASET_PKL)
    else:
        tables = load_raw_data()                      # 1. import raw data
        processed = process_data(tables)              # 2. clean / merge / sample
        enriched = build_weather_enriched(processed)  # 2b. + NOAA weather (saved)

    eda_main(enriched)                                # 3. EDA on the final dataset
    models_main(enriched)                             # 4-5. features + train + results

    print("\nDone.")
    print(f"Final dataset: {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")


if __name__ == "__main__":
    main()
