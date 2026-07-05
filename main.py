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

import ast
import json
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from colorspace import qualitative_hcl
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.inspection import permutation_importance
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

# Binary classification target: 1 if the review awarded >= 4 stars. Single
# source of truth used by both the EDA and the modelling stages.
TARGET = "satisfied"

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

# Columns used to decide whether a weather record was matched. Identical to the
# full weather feature set; kept as a named alias to document that intent.
WEATHER_AVAILABILITY_COLS = WEATHER_FEATURE_COLS

# Global random seed (single source of truth).
RANDOM_STATE = 42

# Script-level run lever.
# - "full_pipeline" preserves the normal end-to-end workflow.
# - "inspect_yelp_features" builds/loads the fuller Yelp-only review-level
#   dataset and writes a feature inventory.
# - "eda_full_yelp_inspection" loads that cached full Yelp dataset and runs EDA
#   without reparsing/cleaning/merging the raw files each time.
RUN_MODE = "eda_full_yelp_inspection"

REBUILD_FULL_YELP_INSPECTION_DATASET = False
YELP_FEATURE_INVENTORY_CSV = PROCESSED_DATA_DIR / "yelp_full_feature_inventory.csv"
YELP_FULL_INSPECTION_PKL = PROCESSED_DATA_DIR / "yelp_full_merged_inspection.pkl"


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


def report_feature_change(step: str, before, after) -> None:
    """Print which columns a processing step added or dropped."""
    before_cols, after_cols = list(before), list(after)
    added = [c for c in after_cols if c not in set(before_cols)]
    dropped = [c for c in before_cols if c not in set(after_cols)]

    print(f"\n[{step}] columns: {len(before_cols)} -> {len(after_cols)}")
    if added:
        print(f"  + added ({len(added)}): {added}")
    if dropped:
        print(f"  - dropped ({len(dropped)}): {dropped}")
    if not added and not dropped:
        print("  (no column changes)")


def report_dropped_variable_table(
    step: str,
    before_df: pd.DataFrame,
    after_columns,
    max_rows: int = 100,
) -> None:
    """Print a table describing variables removed by a column-selection step."""
    after_set = set(after_columns)
    dropped = [col for col in before_df.columns if col not in after_set]

    if not dropped:
        return

    n_rows = len(before_df)
    rows = []
    for col in dropped:
        missing_count = int(before_df[col].isna().sum())
        rows.append({
            "variable": col,
            "dtype": str(before_df[col].dtype),
            "non_null_count": int(before_df[col].notna().sum()),
            "missing_count": missing_count,
            "missing_percent": round((missing_count / n_rows * 100), 2) if n_rows else np.nan,
        })

    print_table(
        pd.DataFrame(rows),
        f"Dropped-variable details [{step}]",
        max_rows=max_rows,
    )

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
    "attributes.Caters",
    "attributes.RestaurantsAttire",
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

# Business-level subfeature extraction. Category dummies are limited to the most
# common categories (excluding the universal "Restaurants" tag) so the
# review-level modelling table stays tractable.
CATEGORY_DUMMY_TOP_N = 30
EXCLUDED_CATEGORY_DUMMIES = {"Restaurants"}

ATTRIBUTE_SUBFEATURES = {
    "attributes.Ambience": "ambience",
    "attributes.BusinessParking": "parking",
    "attributes.GoodForMeal": "meal",
}

HOURS_DAYS = [
    ("Monday", "monday"),
    ("Tuesday", "tuesday"),
    ("Wednesday", "wednesday"),
    ("Thursday", "thursday"),
    ("Friday", "friday"),
    ("Saturday", "saturday"),
    ("Sunday", "sunday"),
]

ENGINEERED_BUSINESS_PREFIXES = (
    "category_",
    "ambience_",
    "parking_",
    "meal_",
    "hours_",
)


def is_engineered_business_feature(column: str) -> bool:
    """Return True for business subfeatures created from nested/list fields."""
    return column.startswith(ENGINEERED_BUSINESS_PREFIXES)


def sanitize_feature_name(value: str) -> str:
    """Convert a Yelp label/key into a readable snake_case feature suffix."""
    chars = []
    for char in str(value).lower():
        chars.append(char if char.isalnum() else "_")
    cleaned = "".join(chars).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "unknown"


def split_categories(value) -> list[str]:
    """Split Yelp's comma-separated categories field into clean labels."""
    if pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip() and item.strip() != "None"
    ]


def parse_yelp_dict(value) -> dict:
    """Parse Yelp dictionary-like attribute strings safely.

    Yelp JSON exports sometimes contain nested dicts as strings with Python
    booleans/None rather than JSON true/false/null, so ast.literal_eval is the
    appropriate parser here.
    """
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def add_category_dummies(business: pd.DataFrame) -> pd.DataFrame:
    """Add top category/cuisine dummy variables to the business table."""
    if "categories" not in business.columns:
        return business

    business = business.copy()
    category_lists = business["categories"].map(split_categories)
    counts = {}
    for categories in category_lists:
        for category in categories:
            if category not in EXCLUDED_CATEGORY_DUMMIES:
                counts[category] = counts.get(category, 0) + 1

    top_categories = [
        category
        for category, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:CATEGORY_DUMMY_TOP_N]
    ]

    used_columns = set(business.columns)
    for category in top_categories:
        base_name = f"category_{sanitize_feature_name(category)}"
        column = base_name
        suffix = 2
        while column in used_columns:
            column = f"{base_name}_{suffix}"
            suffix += 1
        used_columns.add(column)
        business[column] = category_lists.map(
            lambda values, c=category: int(c in values)
        ).astype("int8")

    return business


def add_attribute_dummies(business: pd.DataFrame) -> pd.DataFrame:
    """Expand selected nested Yelp attribute dictionaries into dummy columns."""
    business = business.copy()

    for source_col, prefix in ATTRIBUTE_SUBFEATURES.items():
        if source_col not in business.columns:
            continue

        parsed = business[source_col].map(parse_yelp_dict)
        keys = sorted({key for values in parsed for key in values})
        for key in keys:
            column = f"{prefix}_{sanitize_feature_name(key)}"
            business[column] = parsed.map(
                lambda values, k=key: int(values.get(k) is True)
            ).astype("int8")

    return business


def parse_hours_range(value) -> tuple[int, float]:
    """Parse a Yelp hours string into (is_open, duration_hours).

    Yelp stores hours as strings like "7:0-21:0". Closing times earlier than
    opening times mean the venue closes after midnight. "0:0-0:0" is treated as
    24 hours open, which is Yelp's convention for all-day opening.
    """
    if pd.isna(value):
        return 0, 0.0

    text = str(value).strip()
    if not text or "-" not in text:
        return 0, 0.0

    try:
        open_text, close_text = text.split("-", 1)
        open_hour, open_minute = [int(part) for part in open_text.split(":", 1)]
        close_hour, close_minute = [int(part) for part in close_text.split(":", 1)]
    except (TypeError, ValueError):
        return 0, 0.0

    open_minutes = open_hour * 60 + open_minute
    close_minutes = close_hour * 60 + close_minute

    if open_minutes == close_minutes:
        return 1, 24.0

    duration = close_minutes - open_minutes
    if duration < 0:
        duration += 24 * 60

    return 1, duration / 60


def add_hours_features(business: pd.DataFrame) -> pd.DataFrame:
    """Create opening-hours features from flattened hours.* columns."""
    hour_cols = [f"hours.{day}" for day, _ in HOURS_DAYS]
    if not any(col in business.columns for col in hour_cols):
        return business

    business = business.copy()
    duration_cols = []
    open_cols = []

    for day, day_key in HOURS_DAYS:
        source_col = f"hours.{day}"
        open_col = f"hours_open_{day_key}"
        duration_col = f"hours_duration_{day_key}"
        open_cols.append(open_col)
        duration_cols.append(duration_col)

        if source_col in business.columns:
            parsed = business[source_col].map(parse_hours_range)
            business[open_col] = parsed.map(lambda value: value[0]).astype("int8")
            business[duration_col] = parsed.map(lambda value: value[1]).astype("float32")
        else:
            business[open_col] = np.int8(0)
            business[duration_col] = np.float32(0.0)

    weekend_duration_cols = ["hours_duration_saturday", "hours_duration_sunday"]
    business["hours_open_days_count"] = business[open_cols].sum(axis=1).astype("int8")
    business["hours_total_weekly_open_hours"] = business[duration_cols].sum(axis=1).astype("float32")
    business["hours_weekend_open"] = (
        business[["hours_open_saturday", "hours_open_sunday"]].sum(axis=1) > 0
    ).astype("int8")
    business["hours_weekend_open_hours"] = (
        business[weekend_duration_cols].sum(axis=1).astype("float32")
    )

    return business


def add_business_subfeatures(business: pd.DataFrame) -> pd.DataFrame:
    """Create structured dummies from categories, nested attributes and hours."""
    business = add_category_dummies(business)
    business = add_attribute_dummies(business)
    business = add_hours_features(business)
    return business


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
        "attributes.Caters",
        "attributes.RestaurantsAttire",
    ]
    business_columns += [
        col for col in business.columns
        if is_engineered_business_feature(col)
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
    """Create business-level check-in features.

    checkin['date'] holds a comma-separated list of timestamps per business, so
    the number of check-ins is (number of commas + 1), with an empty/missing
    value meaning zero. Computed vectorised rather than row-by-row.
    """
    checkin = checkin.copy()

    dates = checkin["date"].fillna("").astype(str)
    counts = dates.str.count(",").add(1).where(dates.ne(""), 0)
    checkin["checkin_count"] = counts.astype("int64")

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

    for col in [c for c in dataset.columns if is_engineered_business_feature(c)]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(0)

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


def plot_yearly_review_count_and_missingness(
    dataset: pd.DataFrame,
    output_dir: Path,
    min_year_marker: int = MIN_REVIEW_YEAR,
) -> None:
    """
    Plot yearly review volume and missingness before the year cutoff is applied.

    This preprocessing diagnostic is intentionally run before filter_study_scope()
    so the pre-2010 rows remain visible when justifying the cutoff.
    """
    if "review_year" not in dataset.columns:
        print("Cannot plot yearly coverage because 'review_year' is missing.")
        return

    review_year = pd.to_numeric(dataset["review_year"], errors="coerce")
    valid_years = review_year.notna()
    if not valid_years.any():
        print("Cannot plot yearly coverage because all review years are missing.")
        return

    coverage = dataset.loc[valid_years].copy()
    coverage["review_year"] = review_year.loc[valid_years].astype(int)
    row_missing_percent = coverage.isna().mean(axis=1) * 100

    yearly = (
        pd.DataFrame({
            "review_year": coverage["review_year"],
            "row_missing_percent": row_missing_percent,
        })
        .groupby("review_year", observed=True)
        .agg(
            review_count=("review_year", "size"),
            missing_percent=("row_missing_percent", "mean"),
        )
        .reset_index()
        .sort_values("review_year")
    )

    if yearly.empty:
        print("No yearly review coverage data available to plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    palette = qualitative_hcl("Warm").colors(2)
    count_color = palette[0]
    missing_color = palette[1]

    fig, ax_count = plt.subplots(figsize=(11, 6))
    x_positions = np.arange(len(yearly))
    bar_width = 0.4

    ax_count.bar(
        x_positions - bar_width / 2,
        yearly["review_count"],
        width=bar_width,
        color=count_color,
        label="Review count",
    )
    ax_count.set_xlabel("Review year")
    ax_count.set_ylabel("Number of reviews")
    ax_count.set_xticks(x_positions)
    ax_count.set_xticklabels(yearly["review_year"].astype(str), rotation=45)

    ax_missing = ax_count.twinx()
    ax_missing.bar(
        x_positions + bar_width / 2,
        yearly["missing_percent"],
        width=bar_width,
        color=missing_color,
        label="Average missing values per row (%)",
    )
    ax_missing.set_ylabel("Average missing values per row (%)")
    ax_missing.set_ylim(0, 100)
    ax_missing.set_yticks(np.arange(0, 101, 10))

    handles_count, labels_count = ax_count.get_legend_handles_labels()
    handles_missing, labels_missing = ax_missing.get_legend_handles_labels()
    ax_count.legend(
        handles_count + handles_missing,
        labels_count + labels_missing,
        loc="upper left",
    )

    plt.title("Yearly review count and missingness before study-scope filtering")
    fig.tight_layout()

    output_path = output_dir / "yearly_review_count_and_missingness.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved yearly review-count and missingness plot to: {output_path}")


def plot_top_city_review_count_and_missingness(
    dataset: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
    min_year: int = MIN_REVIEW_YEAR,
) -> None:
    """
    Plot the city-state markets with the most review data before city filtering.

    City names should already be standardized when this helper is called, so the
    plot justifies the final city-scope decision using the same market names as
    the research dataset.
    """
    required_columns = {"city", "state"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        print(
            "Cannot plot top city review coverage because these columns are missing: "
            f"{sorted(missing_columns)}"
        )
        return

    coverage = dataset.copy()
    if "review_year" in coverage.columns:
        review_year = pd.to_numeric(coverage["review_year"], errors="coerce")
        coverage = coverage[review_year >= min_year].copy()

    if coverage.empty:
        print("No city review coverage data available to plot.")
        return

    city_label = (
        coverage["city"].astype("string").fillna("Unknown").str.strip()
        + ", "
        + coverage["state"].astype("string").fillna("Unknown").str.strip()
    )

    city_coverage = (
        pd.DataFrame({
            "city_state": city_label,
        })
        .groupby("city_state", observed=True)
        .size()
        .reset_index(name="review_count")
        .sort_values("review_count", ascending=False)
        .head(top_n)
        .sort_values("review_count")
    )

    if city_coverage.empty:
        print("No top city review coverage data available to plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    colors = qualitative_hcl("Pastel1").colors(len(city_coverage))

    fig, ax_count = plt.subplots(figsize=(11, 7))
    y_positions = np.arange(len(city_coverage))

    ax_count.barh(
        y_positions,
        city_coverage["review_count"],
        color=colors,
        label="Review count",
    )
    ax_count.set_xlabel("Number of reviews")
    ax_count.set_ylabel("City-state market")
    ax_count.set_yticks(y_positions)
    ax_count.set_yticklabels(city_coverage["city_state"].astype(str))

    research_scope_n = len(SELECTED_CITY_STATE_SCOPE)
    if len(city_coverage) > research_scope_n:
        boundary_y = len(city_coverage) - research_scope_n - 0.5
        ax_count.axhline(
            boundary_y,
            color="0.25",
            linestyle="--",
            linewidth=1.2,
        )
        ax_count.text(
            0.98,
            boundary_y + 0.15,
            f"Top {research_scope_n} research scope",
            transform=ax_count.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="0.25",
        )

    plt.title(
        f"Top {len(city_coverage)} city-state markets by review count "
        f"({min_year} onward, before city-scope filtering)"
    )
    fig.tight_layout()

    output_path = output_dir / "top_city_review_count_and_missingness.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved top city review-count plot to: {output_path}")


def plot_philadelphia_city_variants_before_standardization(
    dataset: pd.DataFrame,
    output_dir: Path,
    top_n: int = 12,
) -> None:
    """
    Plot non-standard raw city-name variants collapsed into Philadelphia.

    This is a preprocessing diagnostic and must run before
    standardize_city_names(), while the original raw city values are still
    available for the paper's justification of the standardisation step.
    """
    if "city" not in dataset.columns:
        print("Cannot plot Philadelphia city variants because 'city' is missing.")
        return

    philadelphia_variants = {
        "Bala Cynwyd",
        "Center City",
        "Chestnut Hill",
        "East Passyunk",
        "Fishtown",
        "Manayunk",
        "Mount Airy",
        "Mt Airy",
        "Northern Liberties",
        "Old City",
        "Passyunk Square",
        "Phialdelphia",
        "Phila",
        "PHILADELPHIA",
        "philadelphia",
        "Philadelphia ",
        "Philadelphia",
        "Philly",
        "Queen Village",
        "Rittenhouse",
        "Rittenhouse Square",
        "Roxborough",
        "South Philadelphia",
        "University City",
    }
    canonical_philadelphia_values = {
        "Philadelphia",
        "PHILADELPHIA",
        "philadelphia",
    }

    raw_city = (
        dataset["city"]
        .astype("string")
        .fillna("Unknown")
        .str.strip()
    )

    non_standard_variants = raw_city[
        raw_city.isin(philadelphia_variants)
        & ~raw_city.isin(canonical_philadelphia_values)
    ]
    all_counts = non_standard_variants.value_counts()
    counts = (
        all_counts
        .head(top_n)
        .sort_values()
    )

    if counts.empty:
        print("No non-standard Philadelphia city-name variants found before standardization.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    colors = qualitative_hcl("Pastel1").colors(len(counts))

    plt.figure(figsize=(10, 6))
    plt.barh(counts.index.astype(str), counts.values, color=colors)
    plt.xlabel("Number of records")
    plt.ylabel("Raw city value")
    plt.title("Most frequent non-standard raw city-name variants mapped to Philadelphia")
    plt.tight_layout()

    output_path = output_dir / "philadelphia_city_variants_before_standardization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Philadelphia city-variant plot to: {output_path}")

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("Skipping Philadelphia city-variant wordcloud because wordcloud is not installed.")
        return

    wordcloud = WordCloud(
        width=1400,
        height=800,
        background_color="white",
        colormap="viridis",
        random_state=RANDOM_STATE,
    ).generate_from_frequencies(all_counts.to_dict())

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    wordcloud_path = output_dir / "philadelphia_city_variants_wordcloud.png"
    plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Philadelphia city-variant wordcloud to: {wordcloud_path}")


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


# Skewed, non-negative count features that are log1p-transformed at the
# modelling stage (after EDA). Shared by the EDA skewness comparison and the
# log-and-replace step so the two never drift apart.
LOG_CANDIDATES = [
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
        "review_year",
        "review_month",
        "review_weekday",
        "review_date",

        # Business-level variables
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
        "attributes.Caters",
        "attributes.RestaurantsAttire",

        # Business subfeatures expanded from multi-label / nested fields.
        # These are structured descriptors, created before the review merge.
        *[
            col for col in dataset.columns
            if is_engineered_business_feature(col)
        ],

        # User-level variables
        "user_review_count",
        "user_average_stars",
        "user_fans",
        "user_useful",
        "user_funny",
        "user_cool",

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

    # prepare_core_tables renames columns (e.g. stars -> review_stars) and adds
    # the target. Renames are not feature add/drops, so we only flag the target.
    business, reviews, users = prepare_core_tables(
        business=business,
        reviews=reviews,
        users=users,
    )
    print("\n[prepare_core_tables] created target column 'satisfied' on reviews")

    business_before = list(business.columns)
    business = add_business_subfeatures(business)
    report_feature_change("add_business_subfeatures [business]",
                          business_before, business.columns)

    # First feature drop: keep only the columns we need from each table, before
    # the merge copies them across millions of rows.
    tables_before_selection = {
        "business": business,
        "reviews": reviews,
        "users": users,
    }
    business, reviews, users = select_columns_before_merge(
        business=business,
        reviews=reviews,
        users=users,
    )
    after_tables = {"business": business, "reviews": reviews, "users": users}
    for name, before_df in tables_before_selection.items():
        report_feature_change(f"select_columns_before_merge [{name}]",
                              before_df.columns, after_tables[name].columns)
        report_dropped_variable_table(
            f"select_columns_before_merge [{name}]",
            before_df=before_df,
            after_columns=after_tables[name].columns,
        )

    review_sample = sample_reviews(reviews)

    # The check-in / tip / photo tables are collapsed into per-business count
    # features here; they get attached to the dataset at the merge step below.
    checkin_features = create_checkin_features(checkin)
    tip_features = create_tip_features(tip)
    photo_features = create_photo_features(photo)

    print("\nMerging tables...")
    review_cols = list(review_sample.columns)
    dataset = merge_tables(
        reviews=review_sample,
        business=business,
        users=users,
        checkin_features=checkin_features,
        tip_features=tip_features,
        photo_features=photo_features,
    )
    # Merge ADDS the business, user, and count columns onto each review row.
    report_feature_change("merge_tables", review_cols, dataset.columns)

    print("\nCleaning values...")
    clean_before = list(dataset.columns)
    dataset = clean_values(dataset)
    plot_yearly_review_count_and_missingness(
        dataset=dataset,
        output_dir=PLOT_OUTPUT_DIR,
        min_year_marker=MIN_REVIEW_YEAR,
    )
    plot_philadelphia_city_variants_before_standardization(
        dataset=dataset,
        output_dir=PLOT_OUTPUT_DIR,
        top_n=12,
    )
    dataset = standardize_city_names(dataset)
    report_feature_change("clean_values (+ standardize_city_names)",
                          clean_before, dataset.columns)
    plot_top_city_review_count_and_missingness(
        dataset=dataset,
        output_dir=PLOT_OUTPUT_DIR,
        top_n=15,
        min_year=MIN_REVIEW_YEAR,
    )

    print("\nFiltering study scope...")
    rows_before = len(dataset)
    dataset = filter_study_scope(dataset)
    print(f"\n[filter_study_scope] dropped ROWS, not columns: "
          f"{rows_before:,} -> {len(dataset):,}")

    print("\nSelecting final columns...")
    select_before = dataset
    model_data = select_model_columns(dataset)
    report_feature_change("select_model_columns", select_before.columns, model_data.columns)
    report_dropped_variable_table(
        "select_model_columns",
        before_df=select_before,
        after_columns=model_data.columns,
    )

    print("\nOptimizing dtypes...")
    model_data = optimize_dtypes(model_data)

    print("\nProcessing completed.")
    return model_data


def _rename_remaining_overlaps(
    left: pd.DataFrame,
    right: pd.DataFrame,
    merge_keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Rename non-key columns that would otherwise become *_x / *_y.

    The known Yelp overlaps are handled by prepare_core_tables() using readable
    project names such as review_stars and business_stars. This helper catches
    any extra raw columns that overlap, preserving them for inspection without
    creating ambiguous merge suffixes.
    """
    protected = set(merge_keys)
    overlaps = (set(left.columns) & set(right.columns)) - protected
    rename_map = {
        col: f"{prefix}_{col}"
        for col in overlaps
        if not col.startswith(f"{prefix}_")
    }
    return right.rename(columns=rename_map)


def create_tip_inspection_features(tip: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-business tip features for the inspection dataset."""
    if tip.empty or "business_id" not in tip.columns:
        return pd.DataFrame(columns=["business_id"])

    tip = tip.copy()
    if "text" in tip.columns:
        tip["tip_text_length"] = tip["text"].fillna("").astype(str).str.len()

    aggregations = {}
    if "text" in tip.columns:
        aggregations["tip_count"] = ("text", "count")
    else:
        aggregations["tip_count"] = ("business_id", "size")
    if "compliment_count" in tip.columns:
        aggregations["tip_compliment_count"] = ("compliment_count", "sum")
    if "tip_text_length" in tip.columns:
        aggregations["tip_text_length_mean"] = ("tip_text_length", "mean")
        aggregations["tip_text_length_median"] = ("tip_text_length", "median")
        aggregations["tip_text_length_max"] = ("tip_text_length", "max")
    if "user_id" in tip.columns:
        aggregations["tip_user_count"] = ("user_id", "nunique")
    if "date" in tip.columns:
        aggregations["tip_first_date"] = ("date", "min")
        aggregations["tip_last_date"] = ("date", "max")

    return tip.groupby("business_id").agg(**aggregations).reset_index()


def create_photo_inspection_features(photo: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-business photo features, including label counts."""
    if photo.empty or "business_id" not in photo.columns:
        return pd.DataFrame(columns=["business_id"])

    photo = photo.copy()
    photo_id_col = "photo_id" if "photo_id" in photo.columns else "business_id"
    photo_features = photo.groupby("business_id").agg(
        photo_count=(photo_id_col, "count")
    ).reset_index()

    if "label" in photo.columns:
        photo_label_features = (
            photo.pivot_table(
                index="business_id",
                columns="label",
                values=photo_id_col,
                aggfunc="count",
                fill_value=0,
            )
            .add_prefix("photo_")
            .reset_index()
        )
        photo_features = photo_features.merge(
            photo_label_features,
            on="business_id",
            how="left",
        )

    if "caption" in photo.columns:
        photo["photo_caption_length"] = photo["caption"].fillna("").astype(str).str.len()
        caption_features = photo.groupby("business_id").agg(
            photo_caption_count=("caption", "count"),
            photo_caption_length_mean=("photo_caption_length", "mean"),
            photo_caption_length_max=("photo_caption_length", "max"),
        ).reset_index()
        photo_features = photo_features.merge(
            caption_features,
            on="business_id",
            how="left",
        )

    return photo_features


def build_full_yelp_inspection_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a full Yelp-only review-level dataset for feature discovery.

    Unlike process_data(), this path does not select a narrow modelling feature
    set before merging. It keeps the business/review/user tables broadly intact
    and only aggregates one-to-many tables first so each output row remains one
    review.
    """
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
    if "review_text" in reviews.columns and "review_text_length" not in reviews.columns:
        reviews = reviews.copy()
        reviews["review_text_length"] = reviews["review_text"].fillna("").str.len()

    business = add_business_subfeatures(business)

    business = _rename_remaining_overlaps(
        left=reviews,
        right=business,
        merge_keys=["business_id"],
        prefix="business",
    )
    users = _rename_remaining_overlaps(
        left=reviews,
        right=users,
        merge_keys=["user_id"],
        prefix="user",
    )

    checkin_features = create_checkin_features(checkin)
    tip_features = create_tip_inspection_features(tip)
    photo_features = create_photo_inspection_features(photo)

    print("\nBuilding full Yelp inspection dataset...")
    rows_before = len(reviews)
    dataset = reviews.merge(business, on="business_id", how="left")
    dataset = dataset.merge(users, on="user_id", how="left")
    dataset = dataset.merge(checkin_features, on="business_id", how="left")
    dataset = dataset.merge(tip_features, on="business_id", how="left")
    dataset = dataset.merge(photo_features, on="business_id", how="left")

    if len(dataset) != rows_before:
        raise ValueError(
            "Inspection merge changed the number of review rows. "
            f"Expected {rows_before:,}, got {len(dataset):,}."
        )

    dataset = clean_values(dataset)
    plot_yearly_review_count_and_missingness(
        dataset=dataset,
        output_dir=PLOT_OUTPUT_DIR,
        min_year_marker=MIN_REVIEW_YEAR,
    )
    plot_philadelphia_city_variants_before_standardization(
        dataset=dataset,
        output_dir=PLOT_OUTPUT_DIR,
        top_n=12,
    )
    plot_top_city_review_count_and_missingness(
        dataset=standardize_city_names(dataset),
        output_dir=PLOT_OUTPUT_DIR,
        top_n=15,
        min_year=MIN_REVIEW_YEAR,
    )
    print(f"Full Yelp inspection dataset: {dataset.shape[0]:,} rows x "
          f"{dataset.shape[1]:,} columns")
    return dataset


def is_nested_or_flattened_json_column(column: str) -> bool:
    """Flag columns that look like JSON-normalized or nested Yelp fields."""
    lower = column.lower()
    return (
        "." in column
        or lower.startswith("attributes.")
        or lower.startswith("hours.")
        or lower.startswith("ambience.")
        or "ambience" in lower
        or "categories" in lower
        or "businessparking" in lower
        or "goodfor" in lower
    )


def classify_feature_family(column: str) -> str:
    """Assign a human-readable feature family for the inspection inventory."""
    lower = column.lower()

    if column in ID_COLS or lower.endswith("_id"):
        return "ID columns"
    if lower.startswith("attributes.") or "ambience" in lower:
        return "business attribute columns"
    if lower.startswith("hours."):
        return "hours columns"
    if "categories" in lower:
        return "category columns"
    if lower.startswith("checkin"):
        return "check-in columns"
    if lower.startswith("tip"):
        return "tip columns"
    if lower.startswith("photo"):
        return "photo columns"
    if lower.startswith("review_") or column == TARGET:
        return "review columns"
    if lower.startswith("user_"):
        return "user columns"
    if (
        lower.startswith("business_")
        or column in {"is_open", "city", "state", "latitude", "longitude",
                      "postal_code", "address"}
    ):
        return "business columns"
    return "other columns"


def _safe_nunique(series: pd.Series):
    """Return nunique where practical; skip very large free-text columns."""
    lower = series.name.lower()
    if "text" in lower:
        return np.nan
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return np.nan


def create_feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level feature inventory for the inspection dataset."""
    rows = []
    n_rows = len(df)
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        rows.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "missing_count": missing_count,
            "missing_percent": (missing_count / n_rows * 100) if n_rows else np.nan,
            "n_unique": _safe_nunique(df[column]),
            "feature_family": classify_feature_family(column),
            "is_nested_or_flattened_json": is_nested_or_flattened_json_column(column),
        })

    return pd.DataFrame(rows).sort_values(
        ["feature_family", "column"],
        kind="stable",
    ).reset_index(drop=True)


def load_or_build_full_yelp_inspection_dataset(rebuild: bool = False) -> pd.DataFrame:
    """Load the cached full merged Yelp dataset, or build and cache it once."""
    if YELP_FULL_INSPECTION_PKL.exists() and not rebuild:
        print(f"Loading full Yelp inspection dataset: {YELP_FULL_INSPECTION_PKL}")
        return pd.read_pickle(YELP_FULL_INSPECTION_PKL)

    tables = load_raw_data()
    dataset = build_full_yelp_inspection_dataset(tables)
    dataset.to_pickle(YELP_FULL_INSPECTION_PKL)
    print(f"Saved full Yelp inspection dataset: {YELP_FULL_INSPECTION_PKL}")
    return dataset


def inspect_yelp_features_main() -> None:
    """Run only the Yelp ingestion and full feature-inspection workflow."""
    print_section("Yelp feature inspection mode")
    dataset = load_or_build_full_yelp_inspection_dataset(
        rebuild=REBUILD_FULL_YELP_INSPECTION_DATASET
    )
    inventory = create_feature_inventory(dataset)

    print(f"\nDataset shape: {dataset.shape[0]:,} rows x "
          f"{dataset.shape[1]:,} columns")
    print(f"Feature inventory created for {len(inventory):,} columns.")

    inventory.to_csv(YELP_FEATURE_INVENTORY_CSV, index=False)


def prepare_cached_full_yelp_for_eda(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the research EDA scope to the cached full Yelp inspection dataset.

    The cache keeps the broad 180-column merged data so raw import/clean/merge
    does not repeat. This step then makes the actual research EDA dataset from
    that cache and prints the dropped-variable audit table.
    """
    print_section("Preparing cached full Yelp dataset for EDA")

    clean_before = list(dataset.columns)
    dataset = standardize_city_names(dataset)
    report_feature_change(
        "standardize_city_names [cached full Yelp EDA]",
        clean_before,
        dataset.columns,
    )

    plot_top_city_review_count_and_missingness(
        dataset=dataset,
        output_dir=PLOT_OUTPUT_DIR,
        top_n=15,
        min_year=MIN_REVIEW_YEAR,
    )

    print("\nFiltering study scope...")
    rows_before = len(dataset)
    dataset = filter_study_scope(dataset)
    print(f"\n[filter_study_scope] dropped ROWS, not columns: "
          f"{rows_before:,} -> {len(dataset):,}")

    print("\nSelecting final EDA columns...")
    select_before = dataset
    eda_dataset = select_model_columns(dataset)
    report_feature_change(
        "select_model_columns [cached full Yelp EDA]",
        select_before.columns,
        eda_dataset.columns,
    )
    report_dropped_variable_table(
        "select_model_columns [cached full Yelp EDA]",
        before_df=select_before,
        after_columns=eda_dataset.columns,
    )

    print("\nOptimizing dtypes...")
    return optimize_dtypes(eda_dataset)


def eda_full_yelp_inspection_main() -> None:
    """Run EDA from the cached full merged Yelp inspection dataset."""
    print_section("EDA on full Yelp inspection dataset")
    dataset = load_or_build_full_yelp_inspection_dataset(
        rebuild=REBUILD_FULL_YELP_INSPECTION_DATASET
    )
    eda_dataset = prepare_cached_full_yelp_for_eda(dataset)
    eda_main(eda_dataset)

    print("\nDone.")
    print(f"EDA dataset: {eda_dataset.shape[0]:,} rows x {eda_dataset.shape[1]} columns")


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
    yelp_cols = list(yelp.columns)
    enriched = merge_weather(yelp, weather)
    report_feature_change("weather merge", yelp_cols, enriched.columns)

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

# review_stars is shown for context only: the target is derived from it, so it
# (and the other LEAKAGE_COLS) are excluded from the modelling correlation matrix.

# Fallback raw numeric variables for the descriptive-statistics table. In the
# normal EDA flow, the numeric summary is produced after the correlation/VIF
# workflow and uses the reduced numeric EDA feature set instead.
KEY_NUMERIC_COLS = [
    "review_text_length",
    "business_stars",
    "business_review_count",
    "user_average_stars",
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
    "weather_tmax",
    "weather_tmin",
    "weather_temp_range",
]

# |r| at or above this flags two predictors as redundant (multicollinearity).
CORR_THRESHOLD = 0.8

# VIF above this threshold is treated as severe multicollinearity and removed
# from the final EDA correlation plot. This is diagnostic only; the modelling
# feature-selection logic below remains the single source of truth for models.
VIF_THRESHOLD = 10.0

# Binary indicators are candidates for closer review when they are both rare
# and have almost no marginal relationship with satisfaction. This is a
# conservative EDA flag, not a tuning step on the test set.
EDA_DUMMY_LOW_PREVALENCE_PERCENT = 1.0
EDA_DUMMY_LOW_ABS_TARGET_CORR = 0.01

EDA_EXCLUDE_EXACT = (
    set(ID_COLS)
    | {
        "station",
        "review_date",
        "review_text",
        "tip_text",
        "caption",
        "review_text_length",
        "log_review_text_length",
        "review_useful",
        "review_funny",
        "review_cool",
        "log_review_useful",
        "log_review_funny",
        "log_review_cool",
    }
    | set(LEAKAGE_COLS)
)

EDA_INTERPRETABILITY_KEEP_PRIORITY = {
    "business_review_count": 0,
    "user_review_count": 0,
    "user_useful": 0,
    "photo_count": 0,
    "weather_tmax": 0,
    "weather_temp_range": 0,
}

EDA_INTERPRETABILITY_DROP_PRIORITY = {
    "weather_tmin": 90,
    "user_funny": 90,
    "user_cool": 90,
    "checkin_count": 85,
    "tip_count": 85,
    "tip_compliment_count": 75,
}


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

def run_dataset_overview(df: pd.DataFrame) -> None:
    """Print whole-dataset EDA context without pretending every column is a predictor."""
    print_section("Dataset overview")
    overview = pd.DataFrame([{
        "rows": df.shape[0],
        "columns": df.shape[1],
        "overall_satisfaction_rate_percent": round(df[TARGET].mean() * 100, 2),
    }])
    print_table(overview, "Dataset shape")

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


def run_numeric_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Numeric summary (reduced EDA feature set)",
) -> None:
    """Summarise only the numeric variables that are valid EDA/model candidates."""
    print_section("Descriptive statistics")

    if columns is None:
        columns = KEY_NUMERIC_COLS

    cols = available_columns(df, columns)
    if not cols:
        print("No numeric columns available for the descriptive-statistics table.")
        return

    summary = (
        df[cols]
        .describe(percentiles=[0.25, 0.50, 0.75])
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    print_table(summary, title, max_rows=max(50, len(cols)))


def run_binary_summary(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Binary indicator summary",
) -> None:
    """Summarise binary indicators by prevalence and marginal target correlation."""
    print_section("Binary indicator descriptive statistics")

    if not columns:
        print("No binary indicator columns available.")
        return

    target_corr = pd.Series(dtype=float)
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        target_corr = (
            pd.concat([df[[TARGET]], df[columns]], axis=1)
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
        )

    rows = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        prevalence = float(values.mean() * 100)
        corr_value = target_corr.get(col, np.nan)
        flags = []
        if prevalence < EDA_DUMMY_LOW_PREVALENCE_PERCENT:
            flags.append("rare")
        if pd.notna(corr_value) and abs(corr_value) < EDA_DUMMY_LOW_ABS_TARGET_CORR:
            flags.append("near-zero target correlation")
        rows.append({
            "variable": col,
            "count": int(values.notna().sum()),
            "prevalence_percent": round(prevalence, 2),
            "correlation_with_satisfied": round(float(corr_value), 4) if pd.notna(corr_value) else np.nan,
            "drop_flag": "; ".join(flags),
        })

    summary = pd.DataFrame(rows)
    summary["abs_target_corr"] = summary["correlation_with_satisfied"].abs()
    summary = summary.sort_values(
        ["abs_target_corr", "prevalence_percent"],
        ascending=[False, False],
    ).drop(columns="abs_target_corr")
    print_table(summary, title, max_rows=max(80, len(summary)))

    flagged = summary[summary["drop_flag"] != ""]
    print_table(
        flagged,
        "Binary indicators flagged as rare and/or near-zero target correlation",
        max_rows=max(50, len(flagged)),
    )


def run_categorical_summary(df: pd.DataFrame) -> None:
    """Summarise true categorical variables that are encoded later in pipelines."""
    print_section("Categorical feature summary")

    categorical_cols = [
        col for col in CATEGORICAL_COLUMNS
        if col in df.columns
    ]
    if not categorical_cols:
        print("No true categorical columns found in the EDA dataset.")
        return

    rows = []
    for col in categorical_cols:
        values = df[col]
        counts = values.value_counts(dropna=False)
        top_value = counts.index[0] if not counts.empty else np.nan
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rows.append({
            "variable": col,
            "levels": int(values.nunique(dropna=False)),
            "top_level": top_value,
            "top_level_percent": round(top_count / len(values) * 100, 2) if len(values) else np.nan,
            "missing_percent": round(float(values.isna().mean() * 100), 2),
        })

    print_table(
        pd.DataFrame(rows).sort_values("levels", ascending=False),
        "True categorical variables retained for model-pipeline encoding",
        max_rows=50,
    )


# =============================================================================
# Skewness: raw vs log-transformed
# =============================================================================

def run_skewness_comparison(df: pd.DataFrame) -> None:
    """
    Compare the skewness of each count feature before and after a log1p
    transform. This is the EDA justification for the transform applied at the
    modelling stage: a large drop in |skew| toward 0 means the log makes the
    distribution far more symmetric.
    """
    print_section("Skewness: raw vs log-transformed")

    rows = []
    for col in LOG_CANDIDATES:
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        logged = np.log1p(raw.clip(lower=0))
        skew_raw = float(raw.skew())
        skew_log = float(logged.skew())
        rows.append({
            "variable": col,
            "skew_raw": round(skew_raw, 3),
            "skew_log": round(skew_log, 3),
            "abs_improvement": round(abs(skew_raw) - abs(skew_log), 3),
        })

    table = pd.DataFrame(rows).sort_values("abs_improvement", ascending=False)
    print_table(
        table,
        "Skewness reduction from log1p (higher abs_improvement = more symmetric)",
        max_rows=40,
    )


# =============================================================================
# Target analysis
# =============================================================================

def run_target_analysis(df: pd.DataFrame) -> None:
    print_section("Target analysis")

    target = (
        df[TARGET]
        .value_counts(dropna=False)
        .rename_axis(TARGET)
        .reset_index(name="review_count")
        .sort_values(TARGET)
    )
    target["percent"] = (
        target["review_count"] / target["review_count"].sum() * 100
    ).round(2)
    print_table(target, "Target distribution")

    plot_bar(
        target,
        x_col=TARGET,
        y_col="review_count",
        title="Target distribution: satisfied",
        xlabel="Satisfied",
        ylabel="Number of reviews",
        filename="target_distribution.png",
        rotate_labels=False,
    )

    if LEAKAGE_COLS[0] in df.columns:
        stars = (
            df[LEAKAGE_COLS[0]]
            .value_counts(dropna=False)
            .rename_axis(LEAKAGE_COLS[0])
            .reset_index(name="review_count")
            .sort_values(LEAKAGE_COLS[0])
        )
        stars["percent"] = (
            stars["review_count"] / stars["review_count"].sum() * 100
        ).round(2)
        print_table(stars, "Review stars distribution")

        plot_bar(
            stars,
            x_col=LEAKAGE_COLS[0],
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

def empty_multicollinearity_diagnostics() -> pd.DataFrame:
    """Return the standard schema for EDA multicollinearity drop diagnostics."""
    return pd.DataFrame(
        columns=["variable", "reason", "correlated_with", "correlation", "vif", "step"]
    )


def _drop_record(
    variable: str,
    reason: str,
    step: str,
    correlated_with: str | None = None,
    correlation: float | None = None,
    vif: float | None = None,
) -> dict:
    return {
        "variable": variable,
        "reason": reason,
        "correlated_with": correlated_with,
        "correlation": correlation,
        "vif": vif,
        "step": step,
    }


def _eda_candidate_exclusion_reason(column: str) -> str | None:
    """Return why a numeric column is excluded from the EDA correlation matrix."""
    if column == TARGET:
        return "target shown separately; not treated as a predictor candidate"
    if column in EDA_EXCLUDE_EXACT:
        if column in ID_COLS or column == "station" or column.endswith("_id"):
            return "identifier"
        if column in LEAKAGE_COLS:
            return "target leakage / contemporaneous rating signal"
        if column in {"review_date"}:
            return "date string"
        if column in {"review_text", "tip_text", "caption"}:
            return "raw text field"
        return "post-review variable"
    return None


def is_binary_numeric_series(series: pd.Series) -> bool:
    """Return True when a numeric series contains only 0/1 values."""
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna().unique()
    if len(values) == 0:
        return False
    return set(values).issubset({0, 1})


def select_eda_candidate_sets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select continuous numeric and binary indicator candidates for EDA diagnostics.

    Invalid predictors (IDs, leakage, dates and post-review fields) are removed
    first. Remaining numeric features are then split by measurement type: binary
    indicators get prevalence/phi-correlation style diagnostics, while
    continuous/count variables get numeric summaries and the main heatmap.
    """
    numeric = df.select_dtypes(include="number").copy()
    drop_records = []
    keep_columns = []

    for col in numeric.columns:
        reason = _eda_candidate_exclusion_reason(col)
        if reason is None:
            keep_columns.append(col)
        elif col != TARGET:
            drop_records.append(
                _drop_record(
                    variable=col,
                    reason=f"Excluded before EDA correlation matrix: {reason}",
                    step="candidate_screening",
                )
            )

    candidates = numeric[keep_columns].copy()
    unusable = []
    for col in candidates.columns:
        values = candidates[col]
        if not values.notna().any():
            unusable.append((col, "all values missing"))
        elif values.nunique(dropna=True) <= 1:
            unusable.append((col, "constant numeric variable"))

    if unusable:
        candidates = candidates.drop(columns=[col for col, _reason in unusable])
        for col, reason in unusable:
            drop_records.append(
                _drop_record(
                    variable=col,
                    reason=f"Excluded before EDA correlation matrix: {reason}",
                    step="candidate_screening",
                )
            )

    diagnostics = pd.DataFrame(drop_records)
    if diagnostics.empty:
        diagnostics = empty_multicollinearity_diagnostics()

    binary_cols = [
        col for col in candidates.columns
        if is_binary_numeric_series(candidates[col])
    ]
    continuous_cols = [
        col for col in candidates.columns
        if col not in binary_cols
    ]
    return candidates[continuous_cols].copy(), candidates[binary_cols].copy(), diagnostics


def _interpretability_drop_score(column: str) -> int:
    """Higher score means the variable is less preferred when redundancy exists."""
    if column in EDA_INTERPRETABILITY_KEEP_PRIORITY:
        return EDA_INTERPRETABILITY_KEEP_PRIORITY[column]
    if column in EDA_INTERPRETABILITY_DROP_PRIORITY:
        return EDA_INTERPRETABILITY_DROP_PRIORITY[column]
    if column.startswith("log_"):
        return 70
    if column.startswith("photo_") and column != "photo_count":
        return 80
    if column.endswith("_range") or column.endswith("_ratio"):
        return 65
    if column.startswith(("is_", "has_")):
        return 55
    return 50


def _choose_redundant_variable_to_drop(feature_1: str, feature_2: str) -> tuple[str, str]:
    """Deterministically choose the less interpretable variable in a high-r pair."""
    scores = {
        feature_1: _interpretability_drop_score(feature_1),
        feature_2: _interpretability_drop_score(feature_2),
    }
    if scores[feature_1] != scores[feature_2]:
        drop = max(scores, key=lambda col: (scores[col], col))
    else:
        drop = max(feature_1, feature_2)
    keep = feature_2 if drop == feature_1 else feature_1
    reason = (
        f"Dropped redundant variable from a high-correlation pair; kept {keep} "
        "by the deterministic interpretability/tie-break rule"
    )
    return drop, reason


def remove_highly_correlated_variables(
    numeric: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove one variable from each high-correlation pair using a stable rule."""
    if numeric.shape[1] < 2:
        return numeric.copy(), empty_multicollinearity_diagnostics(), pd.DataFrame()

    corr = numeric.corr(numeric_only=True)
    names = corr.columns.tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({
                    "feature_1": names[i],
                    "feature_2": names[j],
                    "correlation": float(r),
                    "abs_correlation": abs(float(r)),
                })

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return numeric.copy(), empty_multicollinearity_diagnostics(), pairs_df

    pairs_df = pairs_df.sort_values(
        ["abs_correlation", "feature_1", "feature_2"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    dropped = set()
    records = []
    for row in pairs_df.itertuples(index=False):
        feature_1 = row.feature_1
        feature_2 = row.feature_2
        if feature_1 in dropped or feature_2 in dropped:
            continue

        drop, reason = _choose_redundant_variable_to_drop(feature_1, feature_2)
        keep = feature_2 if drop == feature_1 else feature_1
        dropped.add(drop)
        records.append(
            _drop_record(
                variable=drop,
                reason=reason,
                correlated_with=keep,
                correlation=round(float(row.correlation), 3),
                step="pairwise_correlation",
            )
        )

    reduced = numeric.drop(columns=sorted(dropped))
    diagnostics = pd.DataFrame(records)
    if diagnostics.empty:
        diagnostics = empty_multicollinearity_diagnostics()
    return reduced, diagnostics, pairs_df.drop(columns="abs_correlation")


def calculate_vif(numeric: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for numeric predictors using least-squares regressions.

    VIF measures multivariate collinearity: how well each predictor can be
    reconstructed from all of the others. Median imputation is used here only
    for the EDA diagnostic matrix; model imputation remains inside pipelines.
    """
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["variable", "VIF"])

    work = numeric.copy()
    work = work.loc[:, work.notna().any()]
    constant = work.nunique(dropna=True)[lambda s: s <= 1].index.tolist()
    if constant:
        work = work.drop(columns=constant)
    if work.shape[1] < 2:
        return pd.DataFrame(columns=["variable", "VIF"])

    work = work.fillna(work.median(numeric_only=True))
    if len(work) > 200_000:
        work = work.sample(200_000, random_state=RANDOM_STATE)

    cols = work.columns.tolist()
    matrix = work.to_numpy(dtype=float)
    n_rows = len(matrix)

    records = []
    for j, col in enumerate(cols):
        y = matrix[:, j]
        others = np.delete(matrix, j, axis=1)
        others = np.column_stack([np.ones(n_rows), others])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        residual = y - others @ beta
        ss_res = float(residual @ residual)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = min(max(r2, 0.0), 1.0)
        vif_value = np.inf if r2 >= 0.9999 else 1.0 / (1.0 - r2)
        records.append({"variable": col, "VIF": vif_value})

    return pd.DataFrame(records).sort_values("VIF", ascending=False).reset_index(drop=True)


def _format_vif_table(vif: pd.DataFrame) -> pd.DataFrame:
    if vif.empty:
        return vif

    def _flag(v: float) -> str:
        if v > 1000:
            return "near-perfect dependency"
        if v > 10:
            return "severe (>10)"
        if v > 5:
            return "moderate (>5)"
        return ""

    return pd.DataFrame({
        "variable": vif["variable"],
        "VIF": vif["VIF"].map(lambda v: ">1000" if v > 1000 else f"{v:,.2f}"),
        "flag": vif["VIF"].map(_flag),
    })


def iteratively_reduce_high_vif(
    numeric: pd.DataFrame,
    threshold: float = VIF_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop the highest-VIF variable until all remaining VIFs are acceptable."""
    reduced = numeric.copy()
    records = []

    while reduced.shape[1] >= 2:
        vif = calculate_vif(reduced)
        if vif.empty:
            break

        high_vif = vif[vif["VIF"] > threshold]
        if high_vif.empty:
            break

        high_vif = high_vif.assign(
            drop_score=high_vif["variable"].map(_interpretability_drop_score)
        )
        high_vif = high_vif.sort_values(
            ["VIF", "drop_score", "variable"],
            ascending=[False, False, True],
        )
        row = high_vif.iloc[0]
        variable = row["variable"]
        records.append(
            _drop_record(
                variable=variable,
                reason=f"VIF above {threshold:g}; removed highest-VIF variable",
                vif=round(float(row["VIF"]), 3) if np.isfinite(row["VIF"]) else np.inf,
                step="vif_reduction",
            )
        )
        reduced = reduced.drop(columns=[variable])

    final_vif = calculate_vif(reduced)
    diagnostics = pd.DataFrame(records)
    if diagnostics.empty:
        diagnostics = empty_multicollinearity_diagnostics()
    return reduced, diagnostics, final_vif


def plot_reduced_correlation_matrix(
    numeric: pd.DataFrame,
    df: pd.DataFrame,
    filename: str = "correlation_matrix.png",
    title: str = "Reduced EDA feature-set correlation matrix (lower triangle)",
) -> None:
    """Plot the final lower-triangle correlation matrix after reduction."""
    if numeric.shape[1] < 2:
        print("Not enough reduced numeric columns for a correlation matrix plot.")
        return

    plot_df = numeric.copy()
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        plot_df = pd.concat([df[[TARGET]], plot_df], axis=1)

    corr = plot_df.corr(numeric_only=True)
    corr_plot = corr.mask(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    fig_size = max(10, min(18, 0.45 * len(corr.columns) + 6))
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    plt.imshow(corr_plot, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    for row_idx in range(corr_plot.shape[0]):
        for col_idx in range(corr_plot.shape[1]):
            value = corr_plot.iloc[row_idx, col_idx]
            if pd.isna(value):
                continue
            text_color = "white" if abs(value) >= 0.5 else "black"
            plt.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )
    plt.title(title)
    save_current_plot(filename)


def run_reduced_correlation_vif_workflow(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_type: str,
    plot_filename: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run target correlation, pairwise correlation, VIF and final heatmap."""
    print_section(f"{feature_type} correlation and multicollinearity diagnostics")

    print(f"\n{feature_type} candidates after screening: {candidates.shape[1]:,}")
    if candidates.shape[1] < 2:
        print(f"Not enough {feature_type.lower()} candidates for correlation/VIF diagnostics.")
        return candidates, empty_multicollinearity_diagnostics()

    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        target_corr = (
            pd.concat([df[[TARGET]], candidates], axis=1)
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .rename("correlation_with_satisfied")
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        target_corr["abs"] = target_corr["correlation_with_satisfied"].abs()
        target_corr = target_corr.sort_values("abs", ascending=False).drop(columns="abs")
        print_table(
            target_corr,
            f"Correlation with target ({feature_type.lower()} candidates)",
            max_rows=max(50, len(target_corr)),
        )

    reduced_corr, corr_drops, pairs_df = remove_highly_correlated_variables(
        candidates,
        threshold=CORR_THRESHOLD,
    )
    if pairs_df.empty:
        print(f"\nNo {feature_type.lower()} pairs with |correlation| >= {CORR_THRESHOLD}.")
    else:
        display_pairs = pairs_df.copy()
        display_pairs["correlation"] = display_pairs["correlation"].round(3)
        print_table(
            display_pairs,
            f"Highly correlated {feature_type.lower()} pairs (|r| >= {CORR_THRESHOLD})",
            max_rows=max(50, len(display_pairs)),
        )
    if not corr_drops.empty:
        if feature_type == "Binary indicator":
            removed = corr_drops[["variable"]].drop_duplicates().reset_index(drop=True)
            print_table(
                removed,
                "Binary indicator variables removed by pairwise-correlation rule",
                max_rows=max(50, len(removed)),
            )
        elif feature_type != "Continuous numeric":
            print_table(
                corr_drops,
                f"{feature_type} variables removed by pairwise-correlation rule",
                max_rows=max(50, len(corr_drops)),
            )

    vif_before = calculate_vif(reduced_corr)
    print_table(
        _format_vif_table(vif_before),
        f"VIF after pairwise-correlation reduction ({feature_type.lower()})",
        max_rows=max(60, len(vif_before)),
    )

    final_features, vif_drops, _ = iteratively_reduce_high_vif(
        reduced_corr,
        threshold=VIF_THRESHOLD,
    )
    if not vif_drops.empty:
        if feature_type not in {"Continuous numeric", "Binary indicator"}:
            print_table(
                vif_drops,
                f"{feature_type} variables removed by iterative VIF rule",
                max_rows=max(50, len(vif_drops)),
            )

    print(
        f"\nFinal reduced {feature_type.lower()} EDA feature set: "
        f"{final_features.shape[1]:,} predictors"
    )

    diagnostics = pd.concat([corr_drops, vif_drops], ignore_index=True)
    diagnostics = diagnostics.reindex(
        columns=["variable", "reason", "correlated_with", "correlation", "vif", "step"]
    )
    if feature_type == "Continuous numeric":
        removed = diagnostics[["variable"]].drop_duplicates().reset_index(drop=True)
        print_table(
            removed,
            "Continuous numeric variables removed by multicollinearity rule",
            max_rows=max(80, len(removed)),
        )
    elif feature_type == "Binary indicator":
        removed = diagnostics[["variable"]].drop_duplicates().reset_index(drop=True)
        print_table(
            removed,
            "Binary indicator variables removed by multicollinearity rule",
            max_rows=max(80, len(removed)),
        )
    else:
        print_table(
            diagnostics,
            f"{feature_type} variables removed from the final {feature_type.lower()} matrix",
            max_rows=max(80, len(diagnostics)),
        )

    plot_reduced_correlation_matrix(
        final_features,
        df,
        filename=plot_filename,
        title=f"Reduced {feature_type.lower()} EDA feature-set correlation matrix (lower triangle)",
    )
    return final_features, diagnostics


def run_eda_correlation_workflows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("EDA feature screening")

    continuous_candidates, binary_candidates, screening_drops = select_eda_candidate_sets(df)
    total_numeric = df.select_dtypes(include="number").shape[1]
    print(
        f"\nNumeric columns: {total_numeric:,} | "
        f"continuous/count candidates: {continuous_candidates.shape[1]:,} | "
        f"binary indicator candidates: {binary_candidates.shape[1]:,}"
    )
    print_table(
        screening_drops,
        "Variables excluded before numeric/binary diagnostics",
        max_rows=max(80, len(screening_drops)),
    )

    final_continuous, _ = run_reduced_correlation_vif_workflow(
        df=df,
        candidates=continuous_candidates,
        feature_type="Continuous numeric",
        plot_filename="correlation_matrix.png",
    )
    final_binary, _ = run_reduced_correlation_vif_workflow(
        df=df,
        candidates=binary_candidates,
        feature_type="Binary indicator",
        plot_filename="binary_correlation_matrix.png",
    )

    return final_continuous, final_binary


# =============================================================================
# Main
# =============================================================================

def eda_main(df: pd.DataFrame) -> None:
    print_section("Exploratory data analysis")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    run_dataset_overview(df)
    run_target_analysis(df)
    run_categorical_summary(df)
    final_numeric_eda, final_binary_eda = run_eda_correlation_workflows(df)
    run_numeric_summary(
        df,
        columns=final_numeric_eda.columns.tolist(),
        title="Numeric summary (final reduced continuous/count EDA feature set)",
    )
    run_binary_summary(
        df,
        columns=final_binary_eda.columns.tolist(),
        title="Binary indicator summary (final reduced EDA feature set)",
    )

    print_section("EDA completed")
    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")


# ================================================================================
# 4. FEATURE ENGINEERING  (leakage-safe feature set + preprocessing)
# ================================================================================

DATE_COL = "review_date"

# Variables observed only after the review is written. They are useful for EDA
# or optional robustness checks, but excluded from the main structured model by
# default because they are not available before/at the satisfaction outcome.
POST_REVIEW_COLS = [
    "review_text_length",
    "log_review_text_length",
    "review_useful",
    "review_funny",
    "review_cool",
    "log_review_useful",
    "log_review_funny",
    "log_review_cool",
]

# Features removed after EDA because of multicollinearity / VIF. Names are given
# in their pre-log form; the drop step below removes whichever form (raw or
# log_) actually exists after the log-and-replace.
DROP_FOR_REDUNDANCY = [
    # Photo sub-types: photo_count is their exact sum (near-perfect dependency,
    # VIF -> infinity). Keep photo_count.
    "photo_food", "photo_drink", "photo_menu", "photo_inside", "photo_outside",
    # Temperature: weather_temp_range = tmax - tmin (near-perfect dependency).
    # Keep weather_tmax and weather_temp_range.
    "weather_tmin",
    # User votes: severe multicollinearity (VIF >> 10). Keep user_useful.
    "user_funny", "user_cool",
    # Business activity: moderate multicollinearity. Keep business_review_count.
    "checkin_count", "tip_count",
]

# Features we don't use at all, removed BEFORE the EDA so they don't appear in
# the EDA tables. weather_available is the weather missing-indicator: correct to
# include in principle, but it carries ~0 signal here, so we choose not to use it.
DROP_LOW_SIGNAL = ["weather_available"]


def drop_low_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features we won't use (e.g. the weather_available missing-indicator).
    Called before EDA so they stay out of the analysis as well."""
    df = df.copy()
    before = list(df.columns)
    cols = [c for c in DROP_LOW_SIGNAL if c in df.columns]
    df = df.drop(columns=cols)
    report_feature_change("drop_low_signal_features", before, df.columns)
    return df


def log_and_replace_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each skewed count feature with its log1p version (raw dropped).

    Missing values stay NaN on purpose: imputation is deferred to the modelling
    pipeline so it is fit on training data only.
    """
    df = df.copy()
    before = list(df.columns)
    for col in LOG_CANDIDATES:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            df[f"log_{col}"] = np.log1p(values.clip(lower=0))
            df = df.drop(columns=col)
    report_feature_change("log_and_replace_skewed", before, df.columns)
    return df


def drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the features flagged by the EDA correlation/VIF analysis."""
    df = df.copy()
    before = list(df.columns)
    to_drop = [
        candidate
        for name in DROP_FOR_REDUNDANCY
        for candidate in (name, f"log_{name}")
        if candidate in df.columns
    ]
    df = df.drop(columns=to_drop)
    report_feature_change("drop_redundant_features", before, df.columns)
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pure identifiers: the review/business/user IDs and the NOAA station
    id. They are merge keys needed up to this point but carry no EDA or
    modelling value."""
    df = df.copy()
    before = list(df.columns)
    cols = [c for c in ID_COLS + ["station"] if c in df.columns]
    df = df.drop(columns=cols)
    report_feature_change("drop_identifier_columns", before, df.columns)
    return df


def build_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA-informed transition from the EDA dataset to the modelling dataset:
      1. drop pure identifiers (IDs + station) now that all merges are done,
      2. drop the redundant features (justified by the correlation/VIF tables),
      3. show the skewness comparison for the features we keep, then
      4. log-and-replace those skewed counts.

    (Low-signal features such as weather_available are removed earlier, before
    EDA, so they don't appear in the EDA tables.)

    Dropping first means we never transform a feature we won't use, and the
    skewness comparison only covers the features that survive into the model.
    """
    print_section("Building modelling dataset (post-EDA feature decisions)")
    df = drop_identifier_columns(df)
    df = drop_redundant_features(df)
    run_skewness_comparison(df)
    df = log_and_replace_skewed(df)
    return df


def select_features(
    df: pd.DataFrame,
    include_post_review_features: bool = False,
    include_city: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) with a leakage-safe feature set.

    Dropped: IDs, review_date, the leakage columns (review_stars,
    business_stars, user_average_stars), each raw count that has a log_ twin,
    and -- by default -- post-review variables including review votes and
    review text length, plus the very high-cardinality 'city' column (state
    already captures the regional signal cleanly).
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    drop_groups = {
        "identifiers/date": set(ID_COLS) | {DATE_COL},
        "target leakage": set(LEAKAGE_COLS),
    }
    drop = set().union(*drop_groups.values())

    # 'station' is the NOAA station id added by the weather merge: an
    # identifier and a near-duplicate of the high-cardinality 'city' we drop.
    drop.add("station")
    drop_groups["weather station identifier"] = {"station"}

    if not include_post_review_features:
        drop_groups["post-review variables"] = set(POST_REVIEW_COLS)
        drop |= drop_groups["post-review variables"]

    if not include_city:
        drop.add("city")
        drop_groups["high-cardinality location"] = {"city"}

    # For every raw count with a log_ version, keep the log and drop the raw.
    raw_with_log_twin = set()
    for col in list(df.columns):
        if f"log_{col}" in df.columns:
            drop.add(col)
            raw_with_log_twin.add(col)
    if raw_with_log_twin:
        drop_groups["raw variables with log_ replacement"] = raw_with_log_twin

    drop.discard(TARGET)

    feature_cols = [c for c in df.columns if c not in drop and c != TARGET]
    dropped_from_x = [c for c in df.columns if c in drop and c != TARGET]

    print(f"\n[select_features] X columns: {df.shape[1] - 1} -> {len(feature_cols)}")
    if dropped_from_x:
        reported = set()
        for reason, columns in drop_groups.items():
            actual = [
                col for col in df.columns
                if col in columns and col in drop and col != TARGET and col not in reported
            ]
            if actual:
                print(f"  - dropped ({reason}, {len(actual)}): {actual}")
                reported.update(actual)
        remaining = [col for col in dropped_from_x if col not in reported]
        if remaining:
            print(f"  - dropped (other, {len(remaining)}): {remaining}")
    else:
        print("  (no feature columns dropped)")

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
    random_state: int = RANDOM_STATE,
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


def _assemble_preprocessor(
    X: pd.DataFrame,
    categorical_encoder,
    scale_numeric: bool,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build an UNFITTED numeric + categorical ColumnTransformer.

    Shared scaffolding behind the three public preprocessor builders. Only two
    things vary between models: whether numerics are standardised, and which
    encoder handles the categoricals.

    - Numeric: median imputation (covers the 7 unmatched users and the
      structural weather NaNs), then optional standardisation (trees skip it).
    - Categorical: constant 'Unknown' imputation, then the supplied encoder.

    Returns (preprocessor, numeric_cols, categorical_cols).
    """
    numeric, categorical = split_feature_types(X)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", categorical_encoder),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    return preprocessor, numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Preprocessor for the scaled models (logit, SVM, KNN, NB, MLP).

    Categoricals are one-hot encoded with handle_unknown='ignore' so unseen
    categories at test time are safe. sparse_output=False gives a dense matrix
    (required by GaussianNB and the MLP); safe here because the categoricals are
    low-cardinality (10 cities, binary attributes), so the width stays small.

    Weather NaNs are median-imputed in the numeric block. (The weather_available
    missing-indicator was dropped earlier for ~0 signal; if reinstated it would
    ride along here so the model could tell imputed rows apart.)
    """
    encoder = OneHotEncoder(
        handle_unknown="ignore", min_frequency=50, sparse_output=False
    )
    return _assemble_preprocessor(X, encoder, scale_numeric)


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


# Light cross-validation for stability. Runs on a stratified subsample.
DO_CV = True
CV_FOLDS = 3
CV_SUBSAMPLE = 300_000

# Focused hyperparameter search on the top models (HGB + random forest), and a
# permutation-importance plot across all models. Both are extra work, so they
# run on subsamples; set the DO_ flags to False to skip.
DO_GRID_SEARCH = True
GRID_SAMPLE = 30_000
DO_FEATURE_IMPORTANCE = True
PERM_SAMPLE = 3_000      # rows used for permutation importance (KNN is the bottleneck)
PERM_REPEATS = 5


# ---------------------------------------------------------------------------
# Tree-specific preprocessor (compact, no one-hot blow-up)
# ---------------------------------------------------------------------------

def build_tree_preprocessor(X: pd.DataFrame):
    """
    Preprocessor for HistGradientBoosting: median-impute numerics (no scaling)
    and ordinal-encode categoricals so the model can treat them as true
    categoricals natively (unseen/missing -> NaN). Returns
    (preprocessor, categorical_mask) where the mask aligns with the transformed
    column order [numeric..., categorical...].
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,          # unseen categories -> treated as missing
        encoded_missing_value=np.nan,
    )
    preprocessor, numeric, categorical = _assemble_preprocessor(
        X, encoder, scale_numeric=False
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
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    return _assemble_preprocessor(X, encoder, scale_numeric=False)


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
            # min_samples_leaf raised to curb the overfitting seen in the
            # train-vs-test gap (a single tree memorises small leaves otherwise).
            ("model", DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "bagging": Pipeline([
            ("pre", ordinal_pre),
            # Default bagging uses unpruned trees -> train ROC-AUC = 1.0. Bound
            # the base tree's depth and leaf size so it generalises instead of
            # memorising.
            ("model", BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=16, min_samples_leaf=50),
                n_estimators=50,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("pre", ordinal_pre),
            # max_depth + min_samples_leaf added for the same reason: the
            # unconstrained forest hit train ROC-AUC = 1.0 (severe overfitting).
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=50,
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

def top_decile_lift(y_true, y_score, fraction: float = 0.1, pos_label: int = 1) -> float:
    """Lift in the top-scored fraction (default top decile): the rate of
    `pos_label` among the highest-ranked `fraction` of cases divided by the
    overall `pos_label` base rate. Lift = 1 is random; higher means the model
    concentrates that class near the top of its ranking. A standard marketing/
    CRM model-assessment metric.

    For pos_label=1 (satisfied) cases are ranked by descending score; for
    pos_label=0 (dissatisfied) by ascending score (lowest satisfied-score =
    highest dissatisfied probability)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    target = (y_true == pos_label)
    base_rate = target.mean()
    if base_rate == 0:
        return np.nan
    k = max(1, int(round(len(y_true) * fraction)))
    order = np.argsort(y_score)
    if pos_label == 1:
        order = order[::-1]
    top = target[order[:k]]
    return float(top.mean() / base_rate)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Classification metrics suited to an imbalanced binary target.

    The plain precision/recall/f1 are for the positive (satisfied = 1) class;
    the *_dissat versions are the same metrics for the minority dissatisfied
    (= 0) class, which the majority-class metrics hide. gini (= 2*AUC - 1) and
    top_decile_lift are standard marketing-analytics ranking metrics;
    top_decile_lift is for satisfied, top_decile_lift_dissat for the (more
    actionable) dissatisfied minority.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_dissat": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_dissat": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_dissat": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_score is not None:
        roc = roc_auc_score(y_true, y_score)
        metrics["roc_auc"] = roc
        metrics["gini"] = 2 * roc - 1            # Gini = 2*AUC - 1 (rescaling of AUC)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
        metrics["top_decile_lift"] = top_decile_lift(y_true, y_score, pos_label=1)
        metrics["top_decile_lift_dissat"] = top_decile_lift(y_true, y_score, pos_label=0)
    else:
        metrics["roc_auc"] = np.nan
        metrics["gini"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["top_decile_lift"] = np.nan
        metrics["top_decile_lift_dissat"] = np.nan
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

def evaluate_holdout(
    models, X_train, X_test, y_train, y_test
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit each model on train; return full metric tables for both the held-out
    test set and the training set. Returns (test_results, fitted, train_results)."""
    test_rows = []
    train_rows = []
    fitted = {}

    for name, model in models.items():
        print(f"  fitting {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model

        test_rows.append({"model": name, **compute_metrics(
            y_test, model.predict(X_test), get_scores(model, X_test))})
        train_rows.append({"model": name, **compute_metrics(
            y_train, model.predict(X_train), get_scores(model, X_train))})

    test_results = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    train_results = pd.DataFrame(train_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return test_results, fitted, train_results


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


def run_grid_search(X_train, y_train) -> None:
    """Focused hyperparameter search on the two strongest models (HGB + random
    forest), on a subsample for speed. Prints best CV ROC-AUC and parameters."""
    print_section("Hyperparameter search (top models, ROC-AUC, CV)")

    if len(X_train) > GRID_SAMPLE:
        Xs = X_train.sample(GRID_SAMPLE, random_state=RANDOM_STATE)
        ys = y_train.loc[Xs.index]
    else:
        Xs, ys = X_train, y_train
    print(f"Searching on {len(Xs):,} rows, {CV_FOLDS}-fold CV.")

    pipes = build_models(Xs)
    grids = {
        "hist_gradient_boosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": [31, 63],
            "model__max_iter": [100, 200],
        },
        "random_forest": {
            "model__max_depth": [12, 16, 20],
            "model__min_samples_leaf": [20, 50],
        },
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, grid in grids.items():
        print(f"\nSearching {name} ...")
        search = GridSearchCV(pipes[name], grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(Xs, ys)
        print(f"  best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"  best params:     {search.best_params_}")


def plot_feature_importance(fitted, X_test, y_test, top_models) -> None:
    """Grouped horizontal bar chart of the top-10 features for the top-3 models.

    Features on the y-axis, permutation importance (drop in ROC-AUC when a
    feature is shuffled) on the x-axis. Permutation importance is the only
    importance measure comparable across model types; computed on a subsample
    for speed. Only the three best models (by test ROC-AUC) are shown.
    """
    print_section("Permutation feature importance (top 3 models)")

    model_names = [m for m in top_models
                   if m in fitted and m != "baseline_most_frequent"][:3]

    if len(X_test) > PERM_SAMPLE:
        Xp = X_test.sample(PERM_SAMPLE, random_state=RANDOM_STATE)
        yp = y_test.loc[Xp.index]
    else:
        Xp, yp = X_test, y_test
    print(f"Models: {model_names}")
    print(f"Computing on {len(Xp):,} rows, {PERM_REPEATS} repeats per feature...")

    importances = {}
    for name in model_names:
        print(f"  importance: {name} ...")
        result = permutation_importance(
            fitted[name], Xp, yp, scoring="roc_auc",
            n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
        )
        importances[name] = pd.Series(result.importances_mean, index=Xp.columns).clip(lower=0)

    imp_df = pd.DataFrame(importances)
    top = imp_df.mean(axis=1).sort_values(ascending=False).head(10).index.tolist()
    imp_df = imp_df.loc[top]

    n_models = len(model_names)
    y = np.arange(len(top))
    height = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    plt.figure(figsize=(12, 8))
    for i, mdl in enumerate(model_names):
        offset = (i - n_models / 2) * height + height / 2
        plt.barh(y + offset, imp_df[mdl].to_numpy(), height, label=mdl, color=colors[i])
    plt.yticks(y, top)
    plt.gca().invert_yaxis()                      # most important feature on top
    plt.xlabel("Permutation importance (drop in ROC-AUC)")
    plt.title("Top 10 features: permutation importance (top 3 models)")
    plt.legend(title="model")
    plt.tight_layout()
    path = PLOT_DIR / "feature_importance_by_model.png"
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

    # ---- Training-phase work first: nothing here touches the test set. ----
    if DO_CV:
        print("\nRunning light cross-validation (training subsample)...")
        cv_results = run_cross_validation(models, X_train, y_train)
        print("\n" + "=" * 100)
        print(f"CROSS-VALIDATION ({CV_FOLDS}-fold, training subsample)")
        print("=" * 100)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(cv_results.to_string(index=False))
        cv_results.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    if DO_GRID_SEARCH:
        run_grid_search(X_train, y_train)

    # ---- Fit on train; report training-set performance. ----
    print("\nFitting models and evaluating on training and held-out test sets...")
    results, fitted, train_results = evaluate_holdout(models, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON (training set)")
    print("=" * 100)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(train_results.to_string(index=False))
    train_results.to_csv(OUTPUT_DIR / "model_comparison_train.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'model_comparison_train.csv'}")

    # ---- Held-out test set: the final measurement, touched last. ----
    print("\n" + "=" * 100)
    print("MODEL COMPARISON (held-out test set)")
    print("=" * 100)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(results.to_string(index=False))
    print("\nCompare against the training table above: scores far higher on train")
    print("than test = overfitting; both low and similar = underfitting.")
    results.to_csv(OUTPUT_DIR / "model_comparison_holdout.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'model_comparison_holdout.csv'}")

    # ---- Interpretation of the final model(s) on the test set. ----
    best_name = results.iloc[0]["model"]
    print(f"\nBest model by ROC-AUC (baseline-proof): {best_name}")
    plot_confusion(fitted[best_name], X_test, y_test, best_name)

    if DO_FEATURE_IMPORTANCE:
        plot_feature_importance(fitted, X_test, y_test, results["model"].head(3).tolist())

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
    RUN_MODELS = True   # set True to also run the log transform + modelling

    valid_run_modes = {
        "full_pipeline",
        "inspect_yelp_features",
        "eda_full_yelp_inspection",
    }
    if RUN_MODE not in valid_run_modes:
        raise ValueError(
            f"Unknown RUN_MODE={RUN_MODE!r}. "
            f"Choose one of {sorted(valid_run_modes)}."
        )

    if RUN_MODE == "inspect_yelp_features":
        inspect_yelp_features_main()
        return

    if RUN_MODE == "eda_full_yelp_inspection":
        eda_full_yelp_inspection_main()
        return

    if DATASET_PKL.exists() and not REBUILD:
        print(f"Loading existing dataset: {DATASET_PKL}")
        enriched = pd.read_pickle(DATASET_PKL)
    else:
        tables = load_raw_data()                      # 1. import raw data
        processed = process_data(tables)              # 2. clean / merge / sample
        enriched = build_weather_enriched(processed)  # 2b. + NOAA weather (saved)

    # Remove features we won't use at all (e.g. the weather_available missing-
    # indicator) before EDA, so they don't appear in the EDA tables either.
    enriched = drop_low_signal_features(enriched)

    eda_main(enriched)                                # 3. EDA on the raw (untransformed) data

    # 4. EDA-informed transition (always runs, so the feature decisions are
    #    visible even in an EDA-only run): drop identifiers + the redundant
    #    features from the correlation/VIF analysis, then log-and-replace.
    model_df = build_model_dataset(enriched)

    if RUN_MODELS:
        models_main(model_df)                         # 5-6. features + train + results

    print("\nDone.")
    print(f"EDA dataset:       {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")
    print(f"Modelling dataset: {model_df.shape[0]:,} rows x {model_df.shape[1]} columns")


if __name__ == "__main__":
    main()
