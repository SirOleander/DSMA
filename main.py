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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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
# review_stars is shown for context only: the target is derived from it, so it
# (and the other LEAKAGE_COLS) are excluded from the modelling correlation matrix.
LEAKAGE_COL = "review_stars"

# Numeric variables summarised in the descriptive-statistics table.
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

# |r| at or above this flags two predictors as redundant (multicollinearity).
CORR_THRESHOLD = 0.8


# =============================================================================
# Helpers
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


# =============================================================================
# Descriptive statistics
# =============================================================================

def run_descriptive_stats(df: pd.DataFrame) -> None:
    print_section("Descriptive statistics")

    overview = pd.DataFrame([{
        "rows": df.shape[0],
        "columns": df.shape[1],
        "overall_satisfaction_rate_percent": round(df[TARGET_COL].mean() * 100, 2),
    }])
    print_table(overview, "Dataset shape")

    cols = available_columns(df, KEY_NUMERIC_COLS)
    if cols:
        summary = (
            df[cols]
            .describe(percentiles=[0.25, 0.50, 0.75])
            .T
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        print_table(
            summary,
            "Numeric summary (count, mean, std, min, quartiles, max)",
            max_rows=50,
        )

    missing = pd.DataFrame({
        "variable": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean().values * 100).round(2),
    })
    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_percent", ascending=False
    )
    if missing.empty:
        print("\nNo missing values.")
    else:
        print_table(missing, "Variables with missing values", max_rows=25)


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
# Correlation matrix (relevance + multicollinearity)
# =============================================================================

def run_correlation_matrix(df: pd.DataFrame) -> None:
    print_section("Correlation matrix")

    numeric_df = df.select_dtypes(include="number")
    # Drop leakage columns so they don't dominate the matrix or the ranking.
    leak = [c for c in LEAKAGE_COLS if c in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=leak)

    if TARGET_COL not in numeric_df.columns or numeric_df.shape[1] < 2:
        print("Not enough numeric columns for a correlation matrix.")
        return

    corr = numeric_df.corr(numeric_only=True)

    # 1. Relevance: how each numeric feature correlates with the target.
    target_corr = (
        corr[TARGET_COL]
        .drop(TARGET_COL)
        .rename("correlation_with_satisfied")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    target_corr["abs"] = target_corr["correlation_with_satisfied"].abs()
    target_corr = (
        target_corr.sort_values("abs", ascending=False).drop(columns="abs")
    )
    print_table(target_corr, "Correlation with target (satisfied)", max_rows=40)

    # 2. Multicollinearity: predictor pairs with |r| >= CORR_THRESHOLD.
    predictors = corr.drop(index=TARGET_COL, columns=TARGET_COL)
    names = predictors.columns.tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = predictors.iloc[i, j]
            if pd.notna(r) and abs(r) >= CORR_THRESHOLD:
                pairs.append({
                    "feature_1": names[i],
                    "feature_2": names[j],
                    "correlation": round(float(r), 3),
                })

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        print(f"\nNo feature pairs with |correlation| >= {CORR_THRESHOLD}.")
    else:
        pairs_df = pairs_df.reindex(
            pairs_df["correlation"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
        print_table(
            pairs_df,
            f"Highly correlated feature pairs (|r| >= {CORR_THRESHOLD}) - candidates to drop",
            max_rows=40,
        )

    # 3. Full correlation matrix as a heatmap for the report.
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Numeric feature correlation matrix")
    save_current_plot("correlation_matrix.png")


# =============================================================================
# Main
# =============================================================================

def eda_main(df: pd.DataFrame) -> None:
    print_section("Exploratory data analysis")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    run_descriptive_stats(df)
    run_target_analysis(df)
    run_correlation_matrix(df)

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


def stratified_subsample(X: pd.DataFrame, y: pd.Series, n: int | None):
    """Draw a stratified subsample of n rows (keeps the class balance)."""
    if n is None or len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=RANDOM_STATE
    )
    return Xs, ys


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
        # sparse_output=False -> dense matrix, required by GaussianNB and the
        # MLP. Safe here because the categoricals are low-cardinality (10 cities,
        # binary attributes, etc.), so the one-hot width stays small.
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=50, sparse_output=False
        )),
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


# Several required models (SVM, KNN, the neural net) cannot train on millions
# of rows. For a fair, tractable comparison every model is fit on the same
# stratified subsample of this size (set to None to use the full dataset, only
# sensible if you drop SVM/KNN). 100k preserves the class balance and is ample
# for ranking algorithms.
MODEL_SAMPLE_N = 100_000


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


def build_ordinal_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Ordinal preprocessor for the classic tree ensembles (decision tree, bagging,
    random forest). Like build_tree_preprocessor but maps unseen/missing
    categories to a -1 integer sentinel instead of NaN, because those estimators
    reject NaN in older scikit-learn versions. No scaling (trees don't need it).
    """
    numeric, categorical = split_feature_types(X)

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    return preprocessor, numeric, categorical


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models(X_train: pd.DataFrame) -> dict:
    """Build the candidate models, each as a complete leakage-safe Pipeline.

    Preprocessor routing:
      - scaled_pre (standardised + dense one-hot): logit, SVM, KNN, NB, MLP
      - ordinal_pre (-1 sentinel, no scaling):     decision tree, bagging, RF
      - hgb_pre (NaN + native categoricals):       histogram gradient boosting
    class_weight="balanced" is set where the estimator supports it; KNN, NB and
    the MLP have no such parameter and rely on the balanced metrics instead.
    """
    scaled_pre, _, _ = build_preprocessor(X_train, scale_numeric=True)
    ordinal_pre, _, _ = build_ordinal_preprocessor(X_train)
    hgb_pre, cat_mask = build_tree_preprocessor(X_train)

    return {
        "baseline_most_frequent": Pipeline([
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("pre", scaled_pre),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "svm_linear": Pipeline([
            ("pre", scaled_pre),
            # Linear SVM (kernel SVM is infeasible at this scale). dual=False is
            # the recommended setting when n_samples > n_features.
            ("model", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,
                max_iter=5000,
            )),
        ]),
        "knn": Pipeline([
            ("pre", scaled_pre),
            ("model", KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        ]),
        "naive_bayes": Pipeline([
            ("pre", scaled_pre),
            ("model", GaussianNB()),
        ]),
        "neural_net": Pipeline([
            ("pre", scaled_pre),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64,),
                early_stopping=True,
                max_iter=100,
                random_state=RANDOM_STATE,
            )),
        ]),
        "decision_tree": Pipeline([
            ("pre", ordinal_pre),
            ("model", DecisionTreeClassifier(
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "bagging": Pipeline([
            ("pre", ordinal_pre),
            ("model", BaggingClassifier(
                n_estimators=50,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("pre", ordinal_pre),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("pre", hgb_pre),
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
    """Positive-class confidence scores: predict_proba if available, else the
    decision_function (for LinearSVC), else None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
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

    X, y = stratified_subsample(X, y, MODEL_SAMPLE_N)
    if MODEL_SAMPLE_N is not None:
        print(f"Using a stratified subsample of {len(X):,} rows for the model comparison.")

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
