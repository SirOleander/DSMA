"""
Yelp restaurant-satisfaction project - consolidated single-file version.

The project is delivered as one file (for the report appendix) and runs end to
end in a single pass, entirely in memory:

    1. Import data  -> 2. Process data (+ NOAA weather enrichment)
    -> 3. EDA  -> 4. Feature engineering  -> 5. Modelling  -> 6. Results

Just run the file:

    python main.py

Every stage runs on every invocation; there are no toggles and nothing is
resumed from a cached dataset.

Only two files are written: the final enriched dataset and a cached copy of the
NOAA weather download (so reruns don't re-download). Plots and the model-metrics
table are written to the output folders.
"""

import ast
import json
import os
import re
import warnings
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from matplotlib.colors import to_hex

try:
    from colorspace import qualitative_hcl
    _HAS_COLORSPACE = True
except ImportError:
    qualitative_hcl = None
    _HAS_COLORSPACE = False

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    KBinsDiscretizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)



def _resolve_data_dir() -> Path:
    """Locate the data directory without hard-coding one machine's layout.

    Order:
      1. the DSMA_DATA_DIR environment variable, if set;
      2. ./data beside this script      (the layout a grader gets: unzip and go);
      3. ../data beside the repository  (the author's layout);
      4. ./data, created on demand, if neither exists yet.

    Candidates 2 and 3 are only accepted when they already contain a raw/
    subdirectory, so an empty stub directory cannot shadow the real one.
    """
    override = os.environ.get("DSMA_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve().parent
    for candidate in (here / "data", here.parent / "data"):
        if (candidate / "raw").is_dir():
            return candidate
    return here / "data"


DATA_DIR = _resolve_data_dir()
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 4_622_091

DATASET_PKL = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k_weather.pkl"
WEATHER_CACHE_PKL = EXTERNAL_DATA_DIR / "noaa_weather_daily_top_cities.pkl"

ID_COLS = ["review_id", "business_id", "user_id"]

TARGET = "satisfied"

LEAKAGE_COLS = ["review_stars", "business_stars", "user_average_stars"]

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

WEATHER_AVAILABILITY_COLS = WEATHER_FEATURE_COLS

WEATHER_UNITS = {
    "weather_prcp": "mm",
    "weather_snow": "mm",
    "weather_snow_depth": "mm",
    "weather_tmax": "deg C",
    "weather_tmin": "deg C",
    "weather_temp_range": "deg C",
    "is_rainy": "0/1",
    "is_snowy": "0/1",
    "is_hot": "0/1",
    "is_cold": "0/1",
}

RANDOM_STATE = 42

warnings.filterwarnings(
    "ignore",
    message="Bins whose width are too small",
    category=UserWarning,
    module="sklearn.preprocessing._discretization",
)




_FALLBACK_PALETTES = {"Warm": "Set2", "Pastel1": "Pastel1"}


def apply_simple_plot_style(ax) -> None:
    """The house plot style: a plain box, no grid, ticks outward, no chart junk.

    Kept deliberately unadorned so every figure in the report looks the same.
    Titles are omitted from the figures themselves -- the caption names the plot.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.grid(False)
    ax.tick_params(direction="out", length=4, width=0.8, colors="black")


def qualitative_palette(name: str, n_colors: int) -> list[str]:
    """Return n distinct hex colours from a named qualitative palette.

    Uses colorspace when available and matplotlib otherwise, so a missing plotting
    dependency degrades the figures rather than aborting the whole pipeline.
    """
    if _HAS_COLORSPACE:
        return qualitative_hcl(name).colors(n_colors)
    cmap = plt.get_cmap(_FALLBACK_PALETTES.get(name, "tab10"))
    return [to_hex(cmap(i % cmap.N)) for i in range(n_colors)]


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

_QUOTED_LEVEL = re.compile(r"^[ub]?(['\"])(?P<value>.*)\1$")


def _strip_python_repr_quotes(value):
    """u'quiet' -> quiet, 'quiet' -> quiet, Unknown -> Unknown."""
    if not isinstance(value, str):
        return value
    match = _QUOTED_LEVEL.match(value.strip())
    return match.group("value") if match else value.strip()


def normalize_attribute_levels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Collapse the u'x' / 'x' duplicate levels in the attributes.* columns.

    Idempotent: running it on already-clean data changes nothing.
    """
    dataset = dataset.copy()
    columns = [c for c in dataset.columns if c.startswith("attributes.")]
    merged = {}

    for column in columns:
        series = dataset[column]
        if not (isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)):
            continue
        as_text = series.astype("string")
        before = as_text.nunique(dropna=True)
        cleaned = as_text.map(_strip_python_repr_quotes)
        after = cleaned.nunique(dropna=True)
        dataset[column] = cleaned.astype("category")
        if after < before:
            merged[column] = (before, after)

    if merged:
        print_subsection("Normalising Python-2 repr strings in business attributes")
        for column, (before, after) in merged.items():
            print(f"  {column:45s} {before:>2} -> {after:>2} levels")
        print(f"  collapsed {sum(b - a for b, a in merged.values())} duplicate levels "
              f"across {len(merged)} columns")
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

    palette = qualitative_palette("Warm", 2)
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

    colors = qualitative_palette("Pastel1", len(city_coverage))

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

    colors = qualitative_palette("Pastel1", len(counts))

    plt.figure(figsize=(10, 6))
    plt.barh(counts.index.astype(str), counts.values, color=colors)
    plt.xlabel("Number of records")
    plt.ylabel("Raw city value")
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

    city_replacements = {
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

        'Algiers': 'New Orleans',
        'Bucktown': 'New Orleans',
        'Bywater': 'New Orleans',
        'Downtown': 'New Orleans',
        'new orleans': 'New Orleans',
        'NEW ORLEANS': 'New Orleans',

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

        'Indianaplois': 'Indianapolis',
        'INDIANAPOLIS': 'Indianapolis',
        'indianapolis': 'Indianapolis',
        'indianopolis': 'Indianapolis',
        'INpolis': 'Indianapolis',

        'tucson': 'Tucson',
        'TUCSON': 'Tucson',

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

        'RENO': 'Reno',
        'reno': 'Reno',

        'SANTA BARBARA': 'Santa Barbara',
        'santa barbara': 'Santa Barbara',
        'Santa barbra': 'Santa Barbara',
        'Santa Barbra': 'Santa Barbara',
        'santa barbra': 'Santa Barbara',

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


LOG_CANDIDATES = [
    "business_review_count",

    "user_review_count",
    "user_useful",
    "user_funny",
    "user_cool",
    "user_fans",

    "checkin_count",
    "tip_count",
    "tip_compliment_count",
    "photo_count",
    "photo_food",
    "photo_drink",
    "photo_menu",
    "photo_inside",
    "photo_outside",

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
        "review_id",
        "business_id",
        "user_id",

        "satisfied",

        "review_stars",

        "review_text_length",
        "review_useful",
        "review_funny",
        "review_cool",
        "review_year",
        "review_month",
        "review_weekday",
        "review_date",

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

        *[
            col for col in dataset.columns
            if is_engineered_business_feature(col)
        ],

        "user_review_count",
        "user_average_stars",
        "user_fans",
        "user_useful",
        "user_funny",
        "user_cool",

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
    report_feature_change("merge_tables", review_cols, dataset.columns)

    print("\nCleaning values...")
    clean_before = list(dataset.columns)
    dataset = normalize_attribute_levels(dataset)
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




NOAA_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1"

WEATHER_VARIABLES = [
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX",
    "TMIN",
]

RENAME_MAP = {
    "DATE": "review_date",
    "PRCP": "weather_prcp",
    "SNOW": "weather_snow",
    "SNWD": "weather_snow_depth",
    "TMAX": "weather_tmax",
    "TMIN": "weather_tmin",
}

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
    """Download NOAA weather for every selected city.

    Raises if ANY city fails. A partial download used to pass silently: the
    affected reviews received NaN weather, which the modelling pipelines then
    median-imputed without complaint, so a transient NOAA outage produced a
    quietly different dataset and quietly different results. Failing loudly is
    the only way a rerun can be trusted to reproduce the cached numbers.
    """
    weather_frames = []
    failed: list[str] = []

    for city_info in TOP_CITIES:
        city = city_info["city"]
        state = city_info["state"]
        station = city_info["station"]

        print(f"Downloading NOAA weather: {city}, {state} ({station})")

        try:
            city_weather = request_noaa_weather(station, start_date, end_date)

            if city_weather.empty:
                print(f"No data returned for {city}, {state}")
                failed.append(f"{city}, {state} ({station}): no rows returned")
                continue

            city_weather["city"] = city
            city_weather["state"] = state
            city_weather["station"] = station
            weather_frames.append(city_weather)

        except requests.HTTPError as error:
            print(f"HTTP error for {city}, {state}: {error}")
            failed.append(f"{city}, {state} ({station}): HTTP error: {error}")
        except requests.RequestException as error:
            print(f"Request error for {city}, {state}: {error}")
            failed.append(f"{city}, {state} ({station}): request error: {error}")

        sleep(1)

    if failed:
        detail = "\n  ".join(failed)
        raise RuntimeError(
            f"NOAA weather download incomplete: {len(failed)} of {len(TOP_CITIES)} "
            f"cities returned no data.\n  {detail}\n\n"
            "The pipeline stops here on purpose. Continuing would leave those cities' "
            "reviews with missing weather, which the modelling pipelines would silently "
            "median-impute, producing results that differ from a complete run without "
            "any visible error. Re-run when NOAA is reachable, or delete the affected "
            "cities from TOP_CITIES and document the change."
        )

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




EDA_OUTPUT_DIR = PROCESSED_DATA_DIR / "eda_outputs"
PLOT_OUTPUT_DIR = EDA_OUTPUT_DIR / "plots"
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

CORR_THRESHOLD = 0.8

VIF_THRESHOLD = 10.0

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
    hline: float | None = None,
    colors: list[str] | None = None,
) -> None:
    if table.empty:
        return

    plt.figure(figsize=(11, 6))
    plt.bar(table[x_col].astype(str), table[y_col], color=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hline is not None:
        plt.axhline(hline, color="0.35", linestyle="--", linewidth=1.2)
    if rotate_labels:
        plt.xticks(rotation=45, ha="right")
    save_current_plot(filename)



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


def run_weather_summary(df: pd.DataFrame) -> None:
    """Inventory the NOAA weather variables present in the clean dataset.

    Weather missingness has two distinct sources and the table separates them:

      unmatched  -- the review's city-date found no NOAA row at all, so *every*
                    weather variable is missing on that row.
      station_gap -- the row matched, but the station did not report that one
                    variable that day. Snow depth is the usual culprit.

    The binary flags are built from the numeric columns with .fillna(0) *before*
    the merge (see clean_weather_data), so on a matched row they are never
    missing: a day whose station reported no precipitation is coded is_rainy = 0,
    indistinguishable from a genuinely dry day. Their station_gap is therefore 0
    by construction, not by luck, and their prevalence is a mild underestimate.
    """
    print_section("Weather variables in the clean dataset")

    numeric_cols = available_columns(df, WEATHER_NUMERIC_COLS)
    binary_cols = available_columns(df, WEATHER_BINARY_COLS)
    cols = numeric_cols + binary_cols
    if not cols:
        print("No weather variables found in the dataset.")
        return

    n_rows = len(df)
    unmatched = int(df[cols].isna().all(axis=1).sum())

    rows = []
    for col in cols:
        series = df[col]
        present = int(series.notna().sum())
        missing = n_rows - present
        rows.append({
            "variable": col,
            "role": "binary" if col in binary_cols else "continuous",
            "unit": WEATHER_UNITS.get(col, ""),
            "rows_present": present,
            "rows_missing": missing,
            "missing_pct": round(missing / n_rows * 100, 2),
            "station_gap": missing - unmatched,
            "mean": series.mean(),
            "min": series.min(),
            "max": series.max(),
            "in_model": "no" if col in DROP_FOR_REDUNDANCY else "yes",
        })

    print_table(
        pd.DataFrame(rows),
        f"NOAA weather variables ({len(cols)}) across {n_rows:,} reviews",
        max_rows=len(cols),
    )

    print(
        f"\nReviews with no NOAA match at all: {unmatched:,} "
        f"({unmatched / n_rows * 100:.2f}%) -- missing on every weather variable."
    )
    dropped = [c for c in cols if c in DROP_FOR_REDUNDANCY]
    if dropped:
        print(
            f"Dropped before modelling ({len(dropped)}): {', '.join(dropped)} -- "
            "weather_temp_range = weather_tmax - weather_tmin is an exact identity, "
            "so keeping all three would make the design matrix singular."
        )
    if "weather_available" in df.columns:
        print(
            "weather_available (the missing-indicator) is still present; it is "
            "removed by drop_low_signal_features before the EDA tables."
        )


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
            colors=qualitative_palette("Pastel1", len(stars)),
        )



WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

SATISFACTION_COMPARE_COLS = [
    "weather_tmax", "weather_tmin", "weather_temp_range",
    "weather_prcp", "weather_snow", "weather_snow_depth",
    "business_review_count", "checkin_count", "tip_count",
    "tip_compliment_count", "photo_count",
    "user_review_count", "user_fans", "user_useful",
    "review_text_length", "review_year",
]


def satisfaction_rate_by(
    df: pd.DataFrame,
    group,
    group_name: str,
    min_count: int = 200,
    sort_by: str = "rate",
) -> pd.DataFrame:
    """Satisfaction rate per level of a grouping variable.

    Returns the level, its review_count, satisfaction_rate_percent, and the
    deviation in percentage points from the overall rate. Groups with fewer than
    ``min_count`` reviews are dropped so small, noisy cells do not distort the
    comparison. ``sort_by`` is "rate" (descending satisfaction) or "group".
    """
    overall = df[TARGET].mean() * 100
    work = pd.DataFrame({group_name: np.asarray(group), TARGET: df[TARGET].to_numpy()})
    grouped = work.groupby(group_name, observed=True)[TARGET]
    tbl = grouped.agg(review_count="size", rate="mean").reset_index()
    tbl["satisfaction_rate_percent"] = (tbl["rate"] * 100).round(2)
    tbl["vs_overall_pp"] = (tbl["satisfaction_rate_percent"] - overall).round(2)
    tbl = tbl.drop(columns="rate")
    tbl = tbl[tbl["review_count"] >= min_count]
    if sort_by == "rate":
        tbl = tbl.sort_values("satisfaction_rate_percent", ascending=False)
    else:
        tbl = tbl.sort_values(group_name)
    return tbl.reset_index(drop=True)


def run_satisfaction_by_group(df: pd.DataFrame) -> None:
    """Satisfaction rate broken down by location, timing, user experience, and
    weather conditions -- the descriptive core that supports the managerial
    interpretation in the paper."""
    print_section("Satisfaction rate by group")
    overall = df[TARGET].mean() * 100
    print(f"\nOverall satisfaction rate: {overall:.2f}%")

    if "city" in df.columns:
        by_city = satisfaction_rate_by(df, df["city"], "city", min_count=500)
        print_table(by_city, "Satisfaction rate by city")
        plot_bar(
            by_city, "city", "satisfaction_rate_percent",
            title="", xlabel="City", ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_city.png", hline=overall,
            colors=qualitative_palette("Pastel1", len(by_city)),
        )

    if "review_month" in df.columns:
        season = pd.to_numeric(df["review_month"], errors="coerce").map(MONTH_TO_SEASON)
        by_season = satisfaction_rate_by(df, season, "season", sort_by="group")
        by_season["season"] = pd.Categorical(
            by_season["season"], categories=SEASON_ORDER, ordered=True
        )
        by_season = by_season.sort_values("season").reset_index(drop=True)
        print_table(by_season, "Satisfaction rate by season")
        plot_bar(
            by_season, "season", "satisfaction_rate_percent",
            title="", xlabel="Season", ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_season.png", rotate_labels=False, hline=overall,
            colors=qualitative_palette("Pastel1", len(by_season)),
        )

    if "review_weekday" in df.columns:
        weekday = pd.to_numeric(df["review_weekday"], errors="coerce").map(WEEKDAY_NAMES)
        by_weekday = satisfaction_rate_by(df, weekday, "weekday", sort_by="group")
        by_weekday["weekday"] = pd.Categorical(
            by_weekday["weekday"], categories=WEEKDAY_ORDER, ordered=True
        )
        by_weekday = by_weekday.sort_values("weekday").reset_index(drop=True)
        print_table(by_weekday, "Satisfaction rate by day of week")
        plot_bar(
            by_weekday, "weekday", "satisfaction_rate_percent",
            title="", xlabel="Day of week", ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_weekday.png", rotate_labels=False, hline=overall,
        )

    if "review_year" in df.columns:
        year_values = pd.to_numeric(df["review_year"], errors="coerce")
        by_year = satisfaction_rate_by(df, year_values, "review_year", sort_by="group")
        print_table(by_year, "Satisfaction rate by review year")
        plot_bar(
            by_year, "review_year", "satisfaction_rate_percent",
            title="", xlabel="Review year", ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_year.png", hline=overall,
        )

    if "user_review_count" in df.columns:
        uec = pd.to_numeric(df["user_review_count"], errors="coerce")
        bucket = pd.cut(
            uec,
            bins=[0, 1, 5, 20, 100, np.inf],
            labels=["1", "2-5", "6-20", "21-100", "100+"],
            include_lowest=True,
        )
        bucket = bucket.cat.add_categories("Unknown").fillna("Unknown")
        by_user = satisfaction_rate_by(df, bucket, "user_experience", sort_by="group")
        user_order = ["1", "2-5", "6-20", "21-100", "100+", "Unknown"]
        by_user["user_experience"] = pd.Categorical(
            by_user["user_experience"], categories=user_order, ordered=True
        )
        by_user = by_user.sort_values("user_experience").reset_index(drop=True)
        print_table(
            by_user,
            "Satisfaction rate by user experience (total user review count)",
        )
        plot_bar(
            by_user, "user_experience", "satisfaction_rate_percent",
            title="", xlabel="User review-count bucket", ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_user_experience.png", rotate_labels=False, hline=overall,
            colors=qualitative_palette("Pastel1", len(by_user)),
        )

    weather_flags = [c for c in ["is_rainy", "is_snowy", "is_hot", "is_cold"] if c in df.columns]
    if weather_flags:
        rows = []
        for col in weather_flags:
            flag = pd.to_numeric(df[col], errors="coerce")
            present, absent = flag == 1, flag == 0
            rate_present = df.loc[present, TARGET].mean() * 100 if present.any() else np.nan
            rate_absent = df.loc[absent, TARGET].mean() * 100 if absent.any() else np.nan
            rows.append({
                "condition": col.replace("is_", ""),
                "days_present_count": int(present.sum()),
                "sat_rate_present_percent": round(rate_present, 2),
                "sat_rate_absent_percent": round(rate_absent, 2),
                "difference_pp": round(rate_present - rate_absent, 2),
            })
        cond = pd.DataFrame(rows).sort_values("difference_pp").reset_index(drop=True)
        print_table(cond, "Satisfaction rate by weather condition (present vs absent)")
        plot_bar(
            cond, "condition", "sat_rate_present_percent",
            title="", xlabel="Weather condition present", ylabel="Satisfaction rate (%)",
            filename="satisfaction_by_weather_condition.png", rotate_labels=False, hline=overall,
        )


def run_satisfaction_by_category(df: pd.DataFrame, top_n: int = 15) -> None:
    """Satisfaction rate per restaurant category / cuisine, for the most common
    category tags. A review can carry several category dummies, so each rate is
    computed over the reviews tagged with that category."""
    print_section("Satisfaction rate by restaurant category / cuisine")

    category_cols = [c for c in df.columns if c.startswith("category_")]
    if not category_cols:
        print("No category_* dummy columns available.")
        return

    overall = df[TARGET].mean() * 100
    rows = []
    for col in category_cols:
        flag = pd.to_numeric(df[col], errors="coerce").fillna(0)
        mask = flag == 1
        n = int(mask.sum())
        if n == 0:
            continue
        rate = df.loc[mask, TARGET].mean() * 100
        rows.append({
            "category": col[len("category_"):],
            "review_count": n,
            "satisfaction_rate_percent": round(rate, 2),
            "vs_overall_pp": round(rate - overall, 2),
        })

    if not rows:
        print("No tagged category reviews found.")
        return

    tbl = pd.DataFrame(rows)
    top = tbl.sort_values("review_count", ascending=False).head(top_n)
    top = top.sort_values("satisfaction_rate_percent", ascending=False).reset_index(drop=True)
    print_table(
        top,
        f"Satisfaction rate for the {len(top)} most common categories "
        f"(overall {overall:.2f}%)",
        max_rows=max(top_n, len(top)),
    )

    plot_df = top.sort_values("satisfaction_rate_percent", ascending=False)
    plot_bar(
        plot_df, "category", "satisfaction_rate_percent",
        title="", xlabel="Restaurant category", ylabel="Satisfaction rate (%)",
        filename="satisfaction_by_category.png", hline=overall,
        colors=qualitative_palette("Pastel1", len(plot_df)),
    )


def run_satisfied_vs_dissatisfied_comparison(df: pd.DataFrame) -> None:
    """Compare satisfied and dissatisfied reviews on continuous variables and on
    weather-condition prevalence -- answers 'how do the two groups differ?'."""
    print_section("Satisfied vs dissatisfied comparison")

    if TARGET not in df.columns or df[TARGET].nunique() < 2:
        print("Both satisfaction classes are required for the comparison.")
        return

    cols = [c for c in SATISFACTION_COMPARE_COLS if c in df.columns]
    if cols:
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        grouped = numeric.groupby(df[TARGET].to_numpy()).mean()
        out = pd.DataFrame({"variable": cols})
        out["mean_dissatisfied"] = [grouped.loc[0, c] if 0 in grouped.index else np.nan for c in cols]
        out["mean_satisfied"] = [grouped.loc[1, c] if 1 in grouped.index else np.nan for c in cols]
        out["difference"] = out["mean_satisfied"] - out["mean_dissatisfied"]
        print_table(
            out.round(3),
            "Mean of continuous variables by satisfaction class (satisfied minus dissatisfied)",
            max_rows=max(40, len(out)),
        )

    weather_flags = [c for c in ["is_rainy", "is_snowy", "is_hot", "is_cold"] if c in df.columns]
    if weather_flags:
        shares = df.groupby(df[TARGET].to_numpy())[weather_flags].mean().mul(100)
        rows = []
        for col in weather_flags:
            share_dissat = shares.loc[0, col] if 0 in shares.index else np.nan
            share_sat = shares.loc[1, col] if 1 in shares.index else np.nan
            rows.append({
                "weather_condition": col.replace("is_", ""),
                "share_dissatisfied_percent": round(share_dissat, 2),
                "share_satisfied_percent": round(share_sat, 2),
                "difference_pp": round(share_sat - share_dissat, 2),
            })
        print_table(
            pd.DataFrame(rows),
            "Weather-condition prevalence (%) by satisfaction class",
        )



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


def drop_screened_eda_exclusions(
    df: pd.DataFrame,
    screening_drops: pd.DataFrame,
) -> pd.DataFrame:
    """Return the EDA view after removing variables reported as excluded."""
    if screening_drops.empty:
        return df

    to_drop = available_columns(df, screening_drops["variable"].dropna().unique().tolist())
    return df.drop(columns=to_drop)


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


def add_vif_evidence_to_drops(
    drops: pd.DataFrame,
    vif_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Attach candidate-set VIF scores to variables removed before VIF reduction."""
    if drops.empty or vif_reference.empty:
        return drops

    drops = drops.copy()
    vif_lookup = vif_reference.set_index("variable")["VIF"]
    missing_vif = drops["vif"].isna()
    drops.loc[missing_vif, "vif"] = drops.loc[missing_vif, "variable"].map(vif_lookup)
    return drops


def format_multicollinearity_drop_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Show the evidence used to exclude redundant variables from the EDA matrix."""
    if diagnostics.empty:
        return diagnostics

    display = diagnostics[["variable", "vif", "correlated_with", "correlation"]].copy()
    display = display.rename(columns={"vif": "VIF"})
    display["VIF"] = display["VIF"].map(
        lambda v: "" if pd.isna(v) else (">1000" if v > 1000 else f"{v:,.2f}")
    )
    return display


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
    title: str = "Reduced EDA feature-set correlation matrix",
    annotate_cells: bool = True,
) -> None:
    """Plot the final reduced correlation matrix after reduction."""
    if numeric.shape[1] < 2:
        print("Not enough reduced numeric columns for a correlation matrix plot.")
        return

    plot_df = numeric.copy()
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        plot_df = pd.concat([df[[TARGET]], plot_df], axis=1)

    corr = plot_df.corr(numeric_only=True)
    lower_triangle = np.tril(np.ones(corr.shape, dtype=bool), k=-1)
    corr_plot = corr.mask(lower_triangle)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="white")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig_size = max(8, min(14, 0.38 * len(corr.columns) + 4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(corr_plot, cmap=cmap, norm=norm, aspect="equal")
    fig.colorbar(image, ax=ax, label="Pearson correlation", fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, ha="center", va="bottom", fontsize=7)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False, labeltop=True, bottom=False, labelbottom=False, pad=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for idx, label in enumerate(corr.index):
        ax.text(
            idx - 0.62,
            idx,
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=7,
            color="black",
            clip_on=False,
        )
    if annotate_cells:
        for row_idx in range(corr_plot.shape[0]):
            for col_idx in range(corr_plot.shape[1]):
                value = corr_plot.iloc[row_idx, col_idx]
                if pd.isna(value):
                    continue
                text_color = "white" if abs(value) >= 0.5 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6,
                )
    save_current_plot(filename)


def select_compact_target_correlated_features(
    final_features: pd.DataFrame,
    df: pd.DataFrame,
    threshold: float = 0.05,
    min_features: int = 8,
    max_features: int = 10,
    feature_label: str = "numerical",
) -> pd.DataFrame:
    """Select reduced features most associated with the target for the paper plot."""
    if TARGET not in df.columns or not pd.api.types.is_numeric_dtype(df[TARGET]):
        return final_features.iloc[:, :0].copy()

    correlations = (
        pd.concat([df[[TARGET]], final_features], axis=1)
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .dropna()
        .rename("correlation_with_satisfied")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    if correlations.empty:
        return final_features.iloc[:, :0].copy()

    correlations["abs_correlation"] = correlations["correlation_with_satisfied"].abs()
    correlations = correlations.sort_values(
        ["abs_correlation", "variable"],
        ascending=[False, True],
    )

    selected = correlations[correlations["abs_correlation"] >= threshold]
    if len(selected) < min_features:
        selected = correlations.head(min(min_features, len(correlations)))
    elif len(selected) > max_features:
        selected = selected.head(max_features)

    selected = selected.copy()
    selected["correlation_with_satisfied"] = selected["correlation_with_satisfied"].round(4)
    selected["abs_correlation"] = selected["abs_correlation"].round(4)
    print_table(
        selected,
        f"Variables selected for compact main-text {feature_label} correlation matrix",
        max_rows=max(max_features, len(selected)),
    )
    return final_features[selected["variable"].tolist()].copy()


def plot_compact_main_text_correlation_matrix(
    final_features: pd.DataFrame,
    df: pd.DataFrame,
    feature_label: str = "numerical",
    filename: str = "correlation_matrix_main_text.png",
    max_features: int = 10,
) -> None:
    """Plot a compact target-focused correlation matrix for the main paper."""
    compact_features = select_compact_target_correlated_features(
        final_features,
        df,
        max_features=max_features,
        feature_label=feature_label,
    )
    if compact_features.shape[1] < 2:
        print(f"Not enough compact {feature_label} features for the main-text correlation matrix.")
        return

    plot_reduced_correlation_matrix(
        compact_features,
        df,
        filename=filename,
        title=f"Compact {feature_label} correlation matrix for variables most associated with satisfaction",
        annotate_cells=True,
    )


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

    vif_candidates = calculate_vif(candidates)
    reduced_corr, corr_drops, pairs_df = remove_highly_correlated_variables(
        candidates,
        threshold=CORR_THRESHOLD,
    )
    corr_drops = add_vif_evidence_to_drops(corr_drops, vif_candidates)
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
        removed = format_multicollinearity_drop_table(diagnostics)
        print_table(
            removed,
            "Continuous numeric variables removed by multicollinearity rule",
            max_rows=max(80, len(removed)),
        )
    elif feature_type == "Binary indicator":
        removed = format_multicollinearity_drop_table(diagnostics)
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
        title=f"Reduced {feature_type.lower()} EDA feature-set correlation matrix",
        annotate_cells=feature_type != "Binary indicator",
    )
    if feature_type == "Continuous numeric":
        plot_compact_main_text_correlation_matrix(final_features, df)
    elif feature_type == "Binary indicator":
        plot_compact_main_text_correlation_matrix(
            final_features,
            df,
            feature_label="binary",
            filename="binary_correlation_matrix_main_text.png",
            max_features=15,
        )
    return final_features, diagnostics


def run_eda_correlation_workflows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("EDA feature screening")

    continuous_candidates, binary_candidates, screening_drops = select_eda_candidate_sets(df)
    total_numeric = df.select_dtypes(include="number").shape[1]
    print(
        f"\nNumeric columns: {total_numeric:,} | "
        f"continuous/count candidates: {continuous_candidates.shape[1]:,} | "
        f"binary indicator candidates: {binary_candidates.shape[1]:,}"
    )
    print_table(
        screening_drops[["variable"]] if not screening_drops.empty else screening_drops,
        "Variables excluded because of leakage rule",
        max_rows=max(80, len(screening_drops)),
    )
    eda_df = drop_screened_eda_exclusions(df, screening_drops)

    final_continuous, continuous_drops = run_reduced_correlation_vif_workflow(
        df=eda_df,
        candidates=continuous_candidates,
        feature_type="Continuous numeric",
        plot_filename="correlation_matrix.png",
    )
    final_binary, binary_drops = run_reduced_correlation_vif_workflow(
        df=eda_df,
        candidates=binary_candidates,
        feature_type="Binary indicator",
        plot_filename="binary_correlation_matrix.png",
    )

    kept_numeric = set(final_continuous.columns) | set(final_binary.columns)
    candidate_numeric = set(continuous_candidates.columns) | set(binary_candidates.columns)
    multicollinearity_dropped = sorted(candidate_numeric - kept_numeric)
    eda_df = eda_df.drop(
        columns=[c for c in multicollinearity_dropped if c in eda_df.columns]
    )

    consolidated = pd.concat(
        [screening_drops, continuous_drops, binary_drops],
        ignore_index=True,
    ).reindex(columns=["variable", "step", "reason", "correlated_with", "correlation", "vif"])
    consolidated = (
        consolidated.dropna(subset=["variable"])
        .drop_duplicates(subset=["variable"])
        .reset_index(drop=True)
    )
    print_table(
        consolidated,
        "Complete EDA exclusion audit (all variables removed from the feature set)",
        max_rows=max(120, len(consolidated)),
    )
    report_dropped_variable_table("EDA numeric screening", df, eda_df.columns)
    print(
        f"\nFinal EDA feature set after clean removal: {eda_df.shape[1]:,} columns "
        f"({len(final_continuous.columns):,} continuous, "
        f"{len(final_binary.columns):,} binary, plus categoricals and the target)."
    )

    return eda_df, final_continuous, final_binary



def eda_main(df: pd.DataFrame) -> None:
    print_section("Exploratory data analysis")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    run_dataset_overview(df)
    run_weather_summary(df)
    run_target_analysis(df)
    run_satisfaction_by_group(df)
    run_satisfaction_by_category(df)
    run_satisfied_vs_dissatisfied_comparison(df)
    eda_feature_df, final_numeric_eda, final_binary_eda = run_eda_correlation_workflows(df)
    run_categorical_summary(eda_feature_df)
    run_numeric_summary(
        eda_feature_df,
        columns=final_numeric_eda.columns.tolist(),
        title="Numeric summary (final reduced continuous/count EDA feature set)",
    )
    run_skewness_comparison(eda_feature_df)
    run_binary_summary(
        eda_feature_df,
        columns=final_binary_eda.columns.tolist(),
        title="Binary indicator summary (final reduced EDA feature set)",
    )

    print_section("EDA completed")
    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")



DATE_COL = "review_date"

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

DROP_FOR_REDUNDANCY = [
    "photo_food", "photo_drink", "photo_menu", "photo_inside", "photo_outside",
    "weather_tmin",

    "user_funny", "user_cool",
    "checkin_count", "tip_count",
    "hours_duration_wednesday", "hours_duration_thursday", "hours_duration_saturday",
    "hours_total_weekly_open_hours", "hours_weekend_open_hours",
    "hours_open_thursday", "hours_open_saturday", "hours_weekend_open",
    "category_nightlife",
]

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
      3. log-and-replace the skewed count variables.

    The raw-vs-log skewness comparison that justifies step 3 is produced earlier,
    in the EDA (run_skewness_comparison, on the surviving features), so it is not
    repeated here.

    (Low-signal features such as weather_available are removed earlier, before
    EDA, so they don't appear in the EDA tables.)

    Dropping first means we never transform a feature we won't use.
    """
    print_section("Building modelling dataset (post-EDA feature decisions)")
    df = drop_identifier_columns(df)
    df = drop_redundant_features(df)
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

    drop.add("station")
    drop_groups["weather station identifier"] = {"station"}

    if not include_post_review_features:
        drop_groups["post-review variables"] = set(POST_REVIEW_COLS)
        drop |= drop_groups["post-review variables"]

    if not include_city:
        drop.add("city")
        drop_groups["high-cardinality location"] = {"city"}

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




OUTPUT_DIR = PROCESSED_DATA_DIR / "model_outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_SAMPLE_N = 100_000


DO_CV = True
CV_FOLDS = 3
CV_SUBSAMPLE = 300_000

DO_GRID_SEARCH = True
GRID_SAMPLE = 30_000
DO_FEATURE_IMPORTANCE = True
PERM_SAMPLE = 3_000
PERM_REPEATS = 5

DO_UNTUNED_COMPARISON = True

DO_GROUP_ABLATION = True

DO_LEARNING_CURVE = True
LEARNING_CURVE_MODELS = ("hist_gradient_boosting", "logistic_regression")
LEARNING_CURVE_SIZES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
LEARNING_CURVE_TEST_SIZE = 250_000

DO_PR_CURVE = True
PR_CURVE_MODELS = ("hist_gradient_boosting", "logistic_regression")

DO_TOPK_CURVE = True
TOPK_VALUES = (5, 10, 25, 50, None)
TOPK_PERM_SAMPLE = 5_000
TOPK_PERM_REPEATS = 3
TOPK_IMPORTANCE_PLOT_N = 15

IMPORTANCE_PLOT_PRIMARY = "hist_gradient_boosting"

IMPORTANCE_PLOT_COLOURS = ("#F2CDB2", "#D4D8A7")

IMPORTANCE_COMPARISON_TOP_N = 12

IMPORTANCE_PLOT_WIDTH_PX = 2800
IMPORTANCE_PLOT_HEIGHT_PX = 2000
IMPORTANCE_PLOT_DPI = 300

PAIRED_BOOTSTRAP_RESAMPLES = 2_000

DO_MODEL_BOOTSTRAP = True
DO_IMPORTANCE_AGREEMENT = True


PARAM_GRIDS: dict[str, dict] = {
    "logistic_regression": {
        "model__C": [0.0001, 0.001, 0.01, 0.1, 1.0],
    },
    "svm_linear": {
        "model__C": [0.0001, 0.001, 0.01, 0.1, 1.0],
    },
    "knn": {
        "model__n_neighbors": [100, 200, 400, 800, 1600],
    },
    "naive_bayes": {
        "pre__num_cont__bin__n_bins": [5, 10, 20],
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "neural_net": {
        "model__hidden_layer_sizes": [(64,), (128,), (256,), (64, 32)],
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    },
    "decision_tree": {
        "model__max_depth": [4, 6, 8, 12],
        "model__min_samples_leaf": [50, 100, 200],
        "model__criterion": ["gini", "entropy"],
    },
    "bagging": {
        "model__estimator__max_depth": [12, 16, 20],
        "model__estimator__min_samples_leaf": [20, 50],
        "model__max_features": [0.5, 1.0],
    },
    "random_forest": {
        "model__max_depth": [8, 12, 16],
        "model__min_samples_leaf": [10, 20, 50],
        "model__max_features": ["sqrt", 0.5, 0.8],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 1.0, 10.0],
        "model__min_samples_leaf": [20, 50, 100],
    },
}



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
        unknown_value=np.nan,
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



def build_nb_preprocessor(X: pd.DataFrame, n_bins: int = 10) -> ColumnTransformer:
    """Distribution-free preprocessor for Naive Bayes: everything becomes binary.

    GaussianNB was the wrong likelihood for this feature set. It fits a mean and a
    variance to every column, including the one-hot city indicators and the binary
    attribute flags, which take only the values 0 and 1 and are in no sense
    Gaussian; and to zero-inflated counts, which are not Gaussian either.

    Instead:
      - continuous numerics are cut into QUANTILE bins, then one-hot encoded,
      - binary numerics (0/1 flags) pass through untouched, already indicators,
      - categoricals are one-hot encoded,
    and BernoulliNB estimates one probability per indicator. Nothing assumes a
    parametric density, which is the same idea as R's `usekernel = TRUE`.

    A useful consequence: quantile binning is invariant to any monotone transform,
    so this model gives identical results whether or not the skewed counts were
    log-transformed upstream. (The log transform stays, because the linear and
    distance-based models still need it.)
    """
    numeric, categorical = split_feature_types(X)
    binary = [c for c in numeric if is_binary_numeric_series(X[c])]
    continuous = [c for c in numeric if c not in binary]

    continuous_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("bin", KBinsDiscretizer(
            n_bins=n_bins, encode="onehot-dense", strategy="quantile",
        )),
    ])
    binary_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(
            handle_unknown="ignore", min_frequency=50, sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num_cont", continuous_pipe, continuous),
            ("num_bin", binary_pipe, binary),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_models(X_train: pd.DataFrame) -> dict:
    """Build the candidate models, each as a complete leakage-safe Pipeline.

    Preprocessor routing:
      - scaled_pre (standardised + dense one-hot): logit, SVM, KNN, NB, MLP
      - ordinal_pre (-1 sentinel, no scaling):     decision tree, bagging, RF
      - hgb_pre (NaN + native categoricals):       histogram gradient boosting
    class_weight="balanced" is set wherever it is reachable: directly on the
    estimator, or (for bagging) on its base tree. KNN, NB and the MLP expose no
    such parameter anywhere and rely on the balanced metrics instead. Because of
    that, their accuracy is NOT comparable with the balanced models': they score
    above the no-information rate by mostly predicting the majority class.

    The hyperparameter values below are the defaults used when DO_GRID_SEARCH is
    off. When it is on, run_grid_search overwrites the tunable ones (PARAM_GRIDS)
    before the models are fit. Values kept deliberately fixed rather than tuned:
      - class_weight="balanced" is the documented imbalance strategy, not a knob.
      - n_estimators (RF, bagging): more trees is monotonically non-worse in
        expectation, so searching it only buys compute.
      - The tree depth / min_samples_leaf caps below were added because the
        unconstrained ensembles reached train ROC-AUC = 1.0. The grids search
        *within* those caps; none of them offers max_depth=None.
    """
    scaled_pre, _, _ = build_preprocessor(X_train, scale_numeric=True)
    ordinal_pre, _, _ = build_ordinal_preprocessor(X_train)
    hgb_pre, cat_mask = build_tree_preprocessor(X_train)

    return {
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
            ("model", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,
                max_iter=5000,
            )),
        ]),
        "knn": Pipeline([
            ("pre", scaled_pre),
            ("model", KNeighborsClassifier(
                n_neighbors=25, weights="uniform", n_jobs=-1,
            )),
        ]),
        "naive_bayes": Pipeline([
            ("pre", build_nb_preprocessor(X_train)),
            ("model", BernoulliNB(alpha=1.0)),
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
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "bagging": Pipeline([
            ("pre", ordinal_pre),
            ("model", BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=16,
                    min_samples_leaf=50,
                    class_weight="balanced",
                ),
                n_estimators=50,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("pre", ordinal_pre),
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
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=500,
                random_state=RANDOM_STATE,
            )),
        ]),
    }



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
        metrics["gini"] = 2 * roc - 1
        y_dissat = (np.asarray(y_true) == 0).astype(int)
        metrics["pr_auc_dissat"] = average_precision_score(y_dissat, -np.asarray(y_score))
        metrics["top_decile_lift"] = top_decile_lift(y_true, y_score, pos_label=1)
        metrics["top_decile_lift_dissat"] = top_decile_lift(y_true, y_score, pos_label=0)
    else:
        metrics["gini"] = np.nan
        metrics["pr_auc_dissat"] = np.nan
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



def fit_and_time(models, X_train, y_train) -> tuple[dict, dict]:
    """Fit each model on the training set, recording wall-clock fit seconds.

    Fit time is a reported column in the evaluation tables, so it is measured
    here rather than estimated. It covers the whole pipeline (preprocessing +
    estimator), which is what a practitioner actually pays.
    """
    fitted = {}
    seconds = {}
    for name, model in models.items():
        print(f"  fitting {name} ...")
        start = perf_counter()
        model.fit(X_train, y_train)
        seconds[name] = perf_counter() - start
        fitted[name] = model
    return fitted, seconds


def evaluate_fitted(fitted, X, y, seconds: dict | None = None) -> pd.DataFrame:
    """Score already-fitted models on (X, y). Never fits, so it is safe to call
    on the test set without re-touching training data."""
    rows = []
    for name, model in fitted.items():
        row = {"model": name, **compute_metrics(y, model.predict(X), get_scores(model, X))}
        if seconds is not None:
            row["fit_seconds"] = seconds[name]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)


SELECTION_METRIC = "gini"
SELECTION_METRIC_CV = f"{SELECTION_METRIC}_mean"

GRID_SCORING = "roc_auc"

REPORT_COLUMNS = {
    "model": "Model",
    "balanced_accuracy": "Balanced Acc.",
    "pr_auc_dissat": "PR-AUC",
    "gini": "GINI",
    "top_decile_lift": "TDL",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "recall_dissat": "Specificity",
    "fit_seconds": "Time (s)",
}

REPORT_LOWER_IS_BETTER = {"Time (s)"}

MODEL_DISPLAY_ORDER = [
    "logistic_regression",
    "naive_bayes",
    "knn",
    "svm_linear",
    "decision_tree",
    "random_forest",
    "bagging",
    "hist_gradient_boosting",
    "neural_net",
]

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "knn": "KNN",
    "svm_linear": "SVM (Linear)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "bagging": "Bagging",
    "hist_gradient_boosting": "Gradient Boosting",
    "neural_net": "Neural Network",
}


def to_report_table(results: pd.DataFrame) -> pd.DataFrame:
    """Project a full metrics frame onto the reported evaluation-table layout,
    in the fixed MODEL_DISPLAY_ORDER with human-readable model names."""
    cols = [c for c in REPORT_COLUMNS if c in results.columns]
    table = results[cols].rename(columns=REPORT_COLUMNS)
    order = [m for m in MODEL_DISPLAY_ORDER if m in set(table["Model"])]
    table = table.set_index("Model").loc[order].reset_index()
    table["Model"] = table["Model"].map(MODEL_DISPLAY_NAMES).fillna(table["Model"])
    return table


def print_report_table(table: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))


TABLE_GOOD_RGB = (0.059, 0.463, 0.431)
TABLE_BAD_RGB = (0.706, 0.325, 0.035)
TABLE_MAX_TINT = 0.42


def _cell_colour(series: pd.Series, value: float, lower_is_better: bool):
    """Blend white toward the good/bad pole by the value's rank within its column."""
    low, high = series.min(), series.max()
    if high == low or pd.isna(value):
        return (1.0, 1.0, 1.0)
    t = (value - low) / (high - low)
    if lower_is_better:
        t = 1.0 - t
    pole = TABLE_GOOD_RGB if t >= 0.5 else TABLE_BAD_RGB
    weight = abs(t - 0.5) * 2 * TABLE_MAX_TINT
    return tuple(1.0 - weight * (1.0 - c) for c in pole)


def save_report_table_png(
    table: pd.DataFrame, title: str, filename: str, degenerate_rows: tuple = (),
) -> Path:
    """Render an evaluation table as a conditional-formatted PNG for the report.

    `degenerate_rows` names models whose values are artefacts rather than results
    (KNN scored on its own training set); they are greyed out and excluded from
    the colour scale so they cannot distort it.
    """
    metrics = [c for c in table.columns if c != "Model"]
    live = table[~table["Model"].isin(degenerate_rows)]

    text = [
        [row["Model"]] + [
            f"{row[c]:.1f}" if c in REPORT_LOWER_IS_BETTER else f"{row[c]:.3f}"
            for c in metrics
        ]
        for _, row in table.iterrows()
    ]

    model_w = 0.11 * max(len(str(v)) for v in table["Model"]) + 0.55
    metric_w = [max(0.95, 0.082 * len(c) + 0.36) for c in metrics]
    total_w = model_w + sum(metric_w)
    col_widths = [model_w / total_w] + [w / total_w for w in metric_w]

    n_rows = len(table) + 1
    fig, ax = plt.subplots(figsize=(total_w, 0.34 * n_rows + 0.55))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)

    mpl_table = ax.table(
        cellText=text,
        colLabels=["Model"] + metrics,
        cellLoc="right",
        colWidths=col_widths,
        bbox=[0, 0, 1, 1],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)

    for (r, c), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#d8dde2")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(fontweight="bold", color="#3d454d")
            if c == 0:
                cell.set_text_props(ha="left")
            continue
        model = table.iloc[r - 1]["Model"]
        if c == 0:
            cell.set_text_props(ha="left", fontweight="medium")
            cell.set_facecolor("#ffffff")
            continue
        if model in degenerate_rows:
            cell.set_facecolor("#f4f5f6")
            cell.set_text_props(color="#9aa2ab", style="italic")
            continue
        column = metrics[c - 1]
        cell.set_facecolor(_cell_colour(
            live[column], table.iloc[r - 1][column], column in REPORT_LOWER_IS_BETTER
        ))

    path = PLOT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def evaluate_holdout(
    models, X_train, X_test, y_train, y_test
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Fit each model on train; return full metric tables for both the held-out
    test set and the training set.

    Returns (test_results, fitted, train_results, fit_seconds).
    """
    fitted, seconds = fit_and_time(models, X_train, y_train)
    test_results = evaluate_fitted(fitted, X_test, y_test, seconds)
    train_results = evaluate_fitted(fitted, X_train, y_train, seconds)
    return test_results, fitted, train_results, seconds


def run_cross_validation(models, X_train, y_train) -> pd.DataFrame:
    """Light stratified CV on a training subsample, for stability estimates.

    Called *after* the grid search so the reported mean +/- std describe the
    tuned models. This is a reporting step, not a selection step: the winning
    hyperparameters were already chosen inside GridSearchCV's own folds.
    """
    if len(X_train) > CV_SUBSAMPLE:
        Xs = X_train.sample(CV_SUBSAMPLE, random_state=RANDOM_STATE)
        ys = y_train.loc[Xs.index]
    else:
        Xs, ys = X_train, y_train

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["balanced_accuracy", "f1", GRID_SCORING]

    rows = []
    for name, model in models.items():
        print(f"  cross-validating {name} ...")
        scores = cross_validate(model, Xs, ys, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name, "cv_rows": len(Xs)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            if metric == GRID_SCORING:
                row[f"{SELECTION_METRIC}_mean"] = 2 * values.mean() - 1
                row[f"{SELECTION_METRIC}_std"] = 2 * values.std()
            else:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(SELECTION_METRIC_CV, ascending=False).reset_index(drop=True)


def plot_confusion(model, X_test, y_test, name: str) -> None:
    """Save a confusion matrix for the given fitted model.

    House style: no title, no colour bar (the counts are printed in the cells, so
    the bar is redundant), plain box.
    """
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["not satisfied", "satisfied"])
    disp.plot(cmap="Blues", values_format=",", colorbar=False)
    disp.ax_.set_xlabel("Predicted label")
    disp.ax_.set_ylabel("True label")
    apply_simple_plot_style(disp.ax_)
    plt.tight_layout()
    path = PLOT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def find_boundary_hits(grid: dict, best_params: dict) -> list[str]:
    """Report tuned values that landed on the edge of their search range.

    A winner at the minimum or maximum of a numeric axis means the search was
    cut off: the true optimum may lie outside the grid, so the reported value is
    partly an artefact of the range we happened to choose. The fix is to widen
    that axis and re-run.

    Only ordered numeric axes are checked. Categorical axes ("gini"/"entropy",
    "sqrt", layer-size tuples) have no meaningful boundary.
    """
    hits = []
    for key, values in grid.items():
        if len(values) < 2:
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        chosen = best_params.get(key)
        short = key.split("__")[-1]
        if chosen == min(values):
            hits.append(f"{short}={chosen} (grid minimum)")
        elif chosen == max(values):
            hits.append(f"{short}={chosen} (grid maximum)")
    return hits


FIXED_HYPERPARAMETERS: dict[str, dict] = {
    "logistic_regression": {
        "solver": "lbfgs", "penalty": "l2 (lbfgs supports no other)",
        "max_iter": 1000, "class_weight": "balanced",
    },
    "svm_linear": {
        "dual": False, "loss": "squared_hinge (forced by dual=False)",
        "max_iter": 5000, "class_weight": "balanced",
    },
    "knn": {
        "weights": "uniform (distance weighting makes the training score a degenerate 1.000)",
        "metric": "minkowski (p=2, Euclidean)", "algorithm": "auto",
    },
    "naive_bayes": {
        "likelihood": "Bernoulli over indicators (no parametric density assumed)",
        "binning strategy": "quantile (invariant to monotone transforms)",
        "class_weight": "not supported by BernoulliNB",
    },
    "neural_net": {
        "activation": "relu", "solver": "adam", "early_stopping": True,
        "max_iter": 100, "class_weight": "not supported by MLPClassifier",
    },
    "decision_tree": {"splitter": "best", "class_weight": "balanced"},
    "bagging": {
        "n_estimators": "50 (fixed: more trees is monotonically non-worse)",
        "estimator": "DecisionTreeClassifier", "class_weight": "balanced (set on the base tree)",
    },
    "random_forest": {
        "n_estimators": "200 (fixed: more trees is monotonically non-worse)",
        "class_weight": "balanced",
    },
    "hist_gradient_boosting": {
        "max_iter": "500 (a cap; early stopping picks the iteration count)",
        "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
        "class_weight": "balanced",
    },
}


def _fmt_value(value) -> str:
    """Compact, readable rendering of a single hyperparameter value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}"


def format_search_space(values: list) -> str:
    """Render a grid axis the way a methods table should read.

    Detects arithmetic and geometric progressions so a long axis collapses to
    "0.0001 to 1 (x10)" instead of spelling out every value.
    """
    if len(values) == 1:
        return _fmt_value(values[0])

    numeric = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )
    if numeric and len(values) >= 3:
        ordered = sorted(values)
        steps = [b - a for a, b in zip(ordered, ordered[1:])]
        if max(steps) - min(steps) < 1e-12:
            return f"{_fmt_value(ordered[0])} to {_fmt_value(ordered[-1])} (step {_fmt_value(steps[0])})"
        if all(v > 0 for v in ordered):
            ratios = [b / a for a, b in zip(ordered, ordered[1:])]
            if max(ratios) - min(ratios) < 1e-9:
                return f"{_fmt_value(ordered[0])} to {_fmt_value(ordered[-1])} (x{_fmt_value(ratios[0])})"

    return ", ".join(_fmt_value(v) for v in values)


def _edge_note(values: list, chosen) -> str:
    """Flag a winner sitting on the edge of an ordered numeric axis."""
    if len(values) < 2:
        return ""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return ""
    if chosen == min(values):
        return "<- grid minimum"
    if chosen == max(values):
        return "<- grid maximum"
    return ""


def print_hyperparameter_table(best_params: dict[str, dict]) -> pd.DataFrame:
    """Print (and return) the methods-section hyperparameter table.

    One row per hyperparameter: what was searched, over what space, and what won.
    Fixed hyperparameters are listed too, because "we did not tune this, and here
    is the value we used" is part of the model specification a reader needs.
    """
    print_section("Hyperparameter tuning summary")

    rows = []
    for key in MODEL_DISPLAY_ORDER:
        display = MODEL_DISPLAY_NAMES.get(key, key)
        first = True

        for param, values in PARAM_GRIDS.get(key, {}).items():
            chosen = best_params.get(key, {}).get(param)
            rows.append({
                "Model": display if first else "",
                "Hyperparameter": param.split("__", 1)[-1],
                "Search Space Tested": format_search_space(values),
                "Optimal Value": _fmt_value(chosen) if chosen is not None else "not searched",
                "Note": _edge_note(values, chosen) if chosen is not None else "",
            })
            first = False

        for param, value in FIXED_HYPERPARAMETERS.get(key, {}).items():
            rows.append({
                "Model": display if first else "",
                "Hyperparameter": param,
                "Search Space Tested": "fixed (not searched)",
                "Optimal Value": _fmt_value(value),
                "Note": "",
            })
            first = False

    table = pd.DataFrame(rows)
    widths = {c: max(len(c), table[c].str.len().max()) for c in table.columns}
    header = "  ".join(c.ljust(widths[c]) for c in table.columns).rstrip()
    print(header)
    print("-" * len(header))
    previous_model = None
    for _, row in table.iterrows():
        if row["Model"] and previous_model is not None:
            print("-" * len(header))
        if row["Model"]:
            previous_model = row["Model"]
        print("  ".join(str(row[c]).ljust(widths[c]) for c in table.columns).rstrip())

    edges = int((table["Note"] != "").sum())
    searched = int((table["Search Space Tested"] != "fixed (not searched)").sum())
    print(
        f"\n{searched} hyperparameters searched across {len(PARAM_GRIDS)} models; "
        f"{edges} winner(s) landed on a grid edge."
    )
    if edges:
        print("A winner on a grid edge means the search was cut off: the optimum may")
        print("lie outside the range, so that value is partly an artefact of the grid.")

    path = OUTPUT_DIR / "hyperparameter_tuning_table.csv"
    table.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return table


def run_grid_search(X_train, y_train) -> dict[str, dict]:
    """Hyperparameter search over every tunable model, on the training set only.

    Each model's grid (PARAM_GRIDS) is searched with GridSearchCV, scored by
    ROC-AUC. The inner StratifiedKFold folds do the validating, so the test set
    is never touched here.

    Runs on a subsample (GRID_SAMPLE) for tractability. We therefore return the
    winning *parameters* rather than search.best_estimator_, because that
    estimator is fit on the subsample only; the caller re-fits the tuned
    pipelines on the full training set.

    Returns {model_name: best_params}, empty for models with no grid.
    """
    print_section("Hyperparameter search (all tunable models, ROC-AUC, CV)")

    if len(X_train) > GRID_SAMPLE:
        Xs = X_train.sample(GRID_SAMPLE, random_state=RANDOM_STATE)
        ys = y_train.loc[Xs.index]
    else:
        Xs, ys = X_train, y_train

    pipes = build_models(Xs)
    grids = {
        name: grid
        for name, grid in PARAM_GRIDS.items()
        if grid and name in pipes
    }
    n_configs = sum(
        int(np.prod([len(v) for v in grid.values()])) for grid in grids.values()
    )
    print(f"Searching on {len(Xs):,} rows, {CV_FOLDS}-fold CV.")
    print(f"{len(grids)} models, {n_configs} configurations, "
          f"{n_configs * CV_FOLDS} fits.")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_params: dict[str, dict] = {}
    rows = []
    all_hits: dict[str, list[str]] = {}
    for name, grid in grids.items():
        print(f"\nSearching {name} ...")
        search = GridSearchCV(pipes[name], grid, scoring=GRID_SCORING, cv=cv, n_jobs=-1)
        search.fit(Xs, ys)
        best_params[name] = search.best_params_
        best_gini = 2 * search.best_score_ - 1
        print(f"  best CV GINI: {best_gini:.4f}")
        print(f"  best params:  {search.best_params_}")

        hits = find_boundary_hits(grid, search.best_params_)
        if hits:
            all_hits[name] = hits
            print(f"  WARNING boundary hit: {'; '.join(hits)}")

        rows.append({
            "model": name,
            "best_cv_gini": best_gini,
            "search_rows": len(Xs),
            "cv_folds": CV_FOLDS,
            "n_configurations": int(np.prod([len(v) for v in grid.values()])),
            "best_params": str(search.best_params_),
            "boundary_hits": "; ".join(hits),
        })

    if not all_hits:
        print("\nNo boundary hits: every tuned value is interior to its grid.")

    tuned = pd.DataFrame(rows).sort_values("best_cv_gini", ascending=False)
    path = OUTPUT_DIR / "hyperparameter_search.csv"
    tuned.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    print_hyperparameter_table(best_params)

    return best_params


def apply_best_params(models: dict, best_params: dict[str, dict]) -> dict:
    """Overwrite each pipeline's hyperparameters with the tuned values.

    The pipelines were built on the full training set, so setting parameters here
    (rather than reusing GridSearchCV's best_estimator_, which saw only the
    search subsample) means the final models are re-fit on all training rows.
    Models without tuned parameters keep their build_models() defaults.
    """
    if not best_params:
        return models

    print_subsection("Applying tuned hyperparameters")
    for name, params in best_params.items():
        if name in models and params:
            models[name].set_params(**params)
            pretty = ", ".join(
                f"{key.split('__')[-1]}={value}" for key, value in sorted(params.items())
            )
            print(f"  {name}: {pretty}")

    untuned = sorted(set(models) - set(best_params))
    if untuned:
        print(f"  (defaults kept for: {', '.join(untuned)})")
    return models


def plot_feature_importance(fitted, X_test, y_test, top_models) -> None:
    """Grouped horizontal bar chart of the top-10 features for the top-3 models.

    Features on the y-axis, permutation importance (drop in ROC-AUC when a
    feature is shuffled) on the x-axis. Permutation importance is the only
    importance measure comparable across model types; computed on a subsample
    for speed. Only the three best models (by test ROC-AUC) are shown.
    """
    print_section("Permutation feature importance (top 3 models)")

    model_names = [m for m in top_models if m in fitted][:3]

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
            fitted[name], Xp, yp, scoring=GRID_SCORING,
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
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation importance (drop in ROC-AUC)")
    plt.legend(title="model")
    plt.tight_layout()
    path = PLOT_DIR / "feature_importance_by_model.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")



def cv_permutation_importance(model, X_train, y_train) -> pd.Series:
    """Fold-averaged permutation importance, measured on held-out training folds.

    For each stratified fold: fit a fresh clone on the in-fold rows, then permute
    each feature on the OUT-OF-FOLD rows and record the drop in score. Averaging
    across folds gives a ranking that never saw the test set, so it is safe to
    select features with.

    Returned in Gini units (the project's SELECTION_METRIC). permutation_importance
    scores with roc_auc, and gini = 2*auc - 1 is affine, so a drop of d in AUC is a
    drop of 2d in Gini.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    per_fold = []

    for fold, (in_fold, out_fold) in enumerate(cv.split(X_train, y_train), start=1):
        X_in, y_in = X_train.iloc[in_fold], y_train.iloc[in_fold]
        X_out, y_out = X_train.iloc[out_fold], y_train.iloc[out_fold]

        estimator = clone(model).fit(X_in, y_in)

        if len(X_out) > TOPK_PERM_SAMPLE:
            X_out = X_out.sample(TOPK_PERM_SAMPLE, random_state=RANDOM_STATE)
            y_out = y_out.loc[X_out.index]

        print(f"  fold {fold}/{CV_FOLDS}: permuting {X_train.shape[1]} features "
              f"on {len(X_out):,} held-out rows ...")
        result = permutation_importance(
            estimator, X_out, y_out, scoring=GRID_SCORING,
            n_repeats=TOPK_PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
        )
        per_fold.append(pd.Series(2 * result.importances_mean, index=X_train.columns))

    return pd.concat(per_fold, axis=1).mean(axis=1).sort_values(ascending=False)


def paired_bootstrap_gini(
    y_true, scores_by_key: dict, reference_key, key_name: str = "n_features",
) -> pd.DataFrame:
    """Percentile CIs for each variant's Gini and for its difference from a reference.

    The bootstrap is PAIRED: one resample of test rows is drawn, and every variant is
    rescored on those same rows. Because the variants' errors are correlated, the
    difference between two of them is pinned down far more tightly than either Gini
    is on its own -- an unpaired standard error would badly overstate the uncertainty
    of the comparison, and would wrongly declare real differences insignificant.

    Used by both the top-k curve (reference = the full-feature model) and the
    feature-group ablation (reference = the model with every group present).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y_true)
    n = len(y)
    keys = list(scores_by_key)

    observed = {k: 2 * roc_auc_score(y, s) - 1 for k, s in scores_by_key.items()}

    levels = {k: [] for k in keys}
    deltas = {k: [] for k in keys}
    for _ in range(PAIRED_BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        gini_b = {k: 2 * roc_auc_score(y_b, scores_by_key[k][idx]) - 1 for k in keys}
        for k in keys:
            levels[k].append(gini_b[k])
            deltas[k].append(gini_b[k] - gini_b[reference_key])

    n_comparisons = max(len(keys) - 1, 1)
    alpha = 0.05
    bonf = 100 * alpha / (2 * n_comparisons)

    rows = []
    for k in keys:
        lo, hi = np.percentile(levels[k], [2.5, 97.5])
        d_lo, d_hi = np.percentile(deltas[k], [2.5, 97.5])
        b_lo, b_hi = np.percentile(deltas[k], [bonf, 100 - bonf])
        rows.append({
            key_name: k,
            "gini_lo": lo, "gini_hi": hi,
            "delta_vs_full": observed[k] - observed[reference_key],
            "delta_vs_full_lo": d_lo, "delta_vs_full_hi": d_hi,
            "delta_bonf_lo": b_lo, "delta_bonf_hi": b_hi,
            "differs_from_full": bool(d_lo > 0 or d_hi < 0),
            "differs_from_full_bonf": bool(b_lo > 0 or b_hi < 0),
        })
    return pd.DataFrame(rows)


def plot_permutation_importance(
    importance: pd.Series, model_name: str, top_n: int = TOPK_IMPORTANCE_PLOT_N,
) -> Path:
    """Horizontal bar chart of the top-n features, coloured by feature block.

    Plots `cv_permutation_importance` output: importance measured on held-out
    CROSS-VALIDATION FOLDS OF THE TRAINING SET, in Gini units. This is a different
    quantity from plot_feature_importance(), which permutes on the TEST set across
    the top three models. The two figures must not be confused; this one is the
    ranking that drives the top-k curve, and it never touches the test set.
    """
    top = importance.head(top_n)[::-1]
    blocks = [_group_membership(f) or "other" for f in top.index]

    unique_blocks = sorted(set(blocks))
    palette = dict(zip(unique_blocks, qualitative_palette("Pastel1", len(unique_blocks))))
    colours = [palette[b] for b in blocks]

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(top) + 1.6))
    ax.barh(range(len(top)), top.to_numpy(), color=colours)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)

    ax.set_xlabel("Permutation importance (GINI lost when shuffled)")
    ax.set_ylabel("Feature")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[b]) for b in unique_blocks]
    ax.legend(handles, unique_blocks, title="Feature block", loc="lower right")
    apply_simple_plot_style(ax)
    fig.tight_layout()

    path = PLOT_DIR / "permutation_importance_top_features.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_importance_comparison(
    agreement: pd.DataFrame, model_names: list[str],
    top_n: int = IMPORTANCE_COMPARISON_TOP_N,
) -> Path:
    """Grouped permutation importance for the statistically tied models, one axes.

    Two bars per feature on a single shared x-axis. IMPORTANCE_PLOT_PRIMARY decides
    which model sorts the features and is drawn on top of each pair; the features
    shown are that model's top-N, in its order.

    A shared axis makes the magnitude difference plain rather than hiding it: the
    models agree on which features matter (Spearman rho ~ 0.82) and disagree on how
    much, so one model's bars run visibly longer on the features it leans on. Where
    the other model's bars break the descending order, that IS the disagreement. The
    caption must state that importances are a ranking, not an additive decomposition
    of Gini, and that magnitudes are estimator-specific.
    """
    ordered_models = sorted(model_names, key=lambda name: name != IMPORTANCE_PLOT_PRIMARY)
    order = agreement[ordered_models[0]].nlargest(top_n).index[::-1]
    top = agreement.loc[order]

    colours = IMPORTANCE_PLOT_COLOURS
    positions = np.arange(len(top))
    height = 0.8 / len(ordered_models)

    fig, ax = plt.subplots(figsize=(
        IMPORTANCE_PLOT_WIDTH_PX / IMPORTANCE_PLOT_DPI,
        IMPORTANCE_PLOT_HEIGHT_PX / IMPORTANCE_PLOT_DPI,
    ))
    for i, name in enumerate(ordered_models):
        offset = ((len(ordered_models) - 1) / 2 - i) * height
        ax.barh(positions + offset, top[name].to_numpy(), height,
                color=colours[i % len(colours)],
                label=MODEL_DISPLAY_NAMES.get(name, name))

    ax.set_yticks(positions)
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Permutation importance (GINI lost when shuffled)")
    ax.set_ylabel("Feature")
    ax.legend(loc="lower right")
    apply_simple_plot_style(ax)
    fig.tight_layout()

    path = PLOT_DIR / "permutation_importance_comparison.png"
    fig.savefig(path, dpi=IMPORTANCE_PLOT_DPI, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_topk_curve(curve: pd.DataFrame, model_name: str) -> Path:
    """Test-set Gini as a function of feature-set size."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(curve["n_features"], curve["gini"], marker="o", linewidth=2, color="#0f766e")
    for _, row in curve.iterrows():
        ax.annotate(f"{row['gini']:.3f}", xy=(row["n_features"], row["gini"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xlabel("Number of features (ranked by cross-validated permutation importance)")
    ax.set_ylabel("Test-set GINI")
    apply_simple_plot_style(ax)
    ax.margins(y=0.14)
    fig.tight_layout()
    path = PLOT_DIR / "topk_feature_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def run_topk_curve(
    model_name: str, best_params: dict, X_train, y_train, X_test, y_test,
    importance: pd.Series | None = None,
) -> pd.DataFrame:
    """Refit `model_name` on its top-k most important features and score on test.

    The ranking comes from cross-validated permutation importance on the TRAINING
    set (cv_permutation_importance), so the test set plays no part in choosing
    which features survive. Each subset is then scored once on the test set.

    The pipeline is rebuilt from scratch for every subset via build_models(), so
    the ColumnTransformer's numeric/categorical column lists match the reduced
    frame -- reusing the full-feature pipeline would reference missing columns.
    """
    print_section(f"Top-k feature-subset curve ({model_name})")
    print("Feature ranking: permutation importance on held-out training folds.")
    print("The test set is used only to score each already-chosen subset.\n")

    if importance is None:
        reference = build_models(X_train)[model_name]
        if best_params.get(model_name):
            reference.set_params(**best_params[model_name])
        importance = cv_permutation_importance(reference, X_train, y_train)
    else:
        print("Reusing the ranking computed for the importance-agreement check.\n")
        importance = importance.sort_values(ascending=False)

    importance.to_csv(OUTPUT_DIR / "topk_feature_ranking.csv", header=["gini_drop"])
    plot_permutation_importance(importance, model_name)

    rows = []
    scores_by_k = {}
    for k in TOPK_VALUES:
        columns = list(X_train.columns) if k is None else list(importance.head(k).index)
        estimator = build_models(X_train[columns])[model_name]
        if best_params.get(model_name):
            estimator.set_params(**best_params[model_name])

        print(f"  refitting on {len(columns):>3} features ...")
        estimator.fit(X_train[columns], y_train)
        y_score = get_scores(estimator, X_test[columns])
        scored = compute_metrics(y_test, estimator.predict(X_test[columns]), y_score)
        scores_by_k[len(columns)] = np.asarray(y_score)
        rows.append({
            "n_features": len(columns),
            "gini": scored["gini"],
            "pr_auc_dissat": scored["pr_auc_dissat"],
            "balanced_accuracy": scored["balanced_accuracy"],
            "top_decile_lift_dissat": scored["top_decile_lift_dissat"],
        })

    curve = pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)
    full_gini = curve.iloc[-1]["gini"]
    curve["gini_pct_of_full"] = 100 * curve["gini"] / full_gini

    print(f"\nPaired bootstrap ({PAIRED_BOOTSTRAP_RESAMPLES:,} resamples of the test rows) ...")
    full_key = max(scores_by_k)
    ci = paired_bootstrap_gini(y_test, scores_by_k, reference_key=full_key)
    curve = curve.merge(ci, on="n_features", how="left")

    print("\n" + "=" * 110)
    print(f"FEATURE-SUBSET CURVE ({model_name}, test set)")
    print("=" * 110)
    show = curve[[
        "n_features", "gini", "gini_pct_of_full", "delta_vs_full",
        "delta_vs_full_lo", "delta_vs_full_hi", "differs_from_full",
        "delta_bonf_lo", "delta_bonf_hi", "differs_from_full_bonf",
    ]]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220):
        print(show.to_string(index=False))
    print("\ndelta_vs_full is the PAIRED difference from the full model: both models are")
    print("scored on the SAME resampled test rows, so their errors are correlated and the")
    print("difference is estimated far more precisely than either Gini is on its own.")
    print("differs_from_full is True when the 95% interval excludes zero. The _bonf columns")
    print("widen each interval so that all comparisons jointly hold at 95% (Bonferroni);")
    print("that is the column to quote, since the curve makes several comparisons at once.")

    print("\nTop 10 features by cross-validated permutation importance:")
    for rank, (name, drop) in enumerate(importance.head(10).items(), start=1):
        print(f"  {rank:2d}. {name:34s} {drop:+.5f} Gini lost when shuffled")

    curve.to_csv(OUTPUT_DIR / "topk_feature_curve.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'topk_feature_curve.csv'}")
    plot_topk_curve(curve, model_name)
    return curve



def _group_membership(column: str) -> str | None:
    """Assign a modelling column to a feature block (None = ungrouped)."""
    if column.startswith("weather_") or column in ("is_rainy", "is_snowy", "is_hot", "is_cold"):
        return "external (weather)"
    if column in ("log_photo_count", "log_tip_compliment_count"):
        return "engagement (photos, tips)"
    if column.startswith("user_") or column.startswith("log_user_"):
        return "user characteristics"
    if column in ("review_year", "review_month", "review_weekday"):
        return "review timing"
    if (column.startswith("attributes.") or column.startswith("ambience_")
            or column.startswith("parking_") or column.startswith("meal_")
            or column == "is_open"):
        return "business attributes"
    if column.startswith("category_"):
        return "restaurant categories"
    if column.startswith("hours_"):
        return "opening hours"
    if column in ("state", "latitude", "longitude"):
        return "location"
    if column == "log_business_review_count":
        return "business popularity"
    return None


def assert_all_features_grouped(columns) -> None:
    """Every modelling column must belong to a block, or the ablation silently
    never tests it. Called before the ablation runs."""
    ungrouped = [c for c in columns if _group_membership(c) is None]
    if ungrouped:
        raise ValueError(
            f"{len(ungrouped)} feature(s) belong to no ablation block and would "
            f"never be tested: {ungrouped}. Extend _group_membership()."
        )


def build_feature_groups(columns) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in columns:
        name = _group_membership(column)
        if name:
            groups.setdefault(name, []).append(column)
    return groups


def run_feature_group_ablation(
    model_name: str, best_params: dict, X_train, y_train, X_test, y_test,
) -> pd.DataFrame:
    """Refit the model with each feature block removed; report the Gini it costs.

    A block that can be deleted without a significant loss of Gini did not add
    predictive value beyond the blocks that remain. Note the "beyond": these are
    marginal contributions, so two blocks carrying the same information can each
    look dispensable while their union is not.
    """
    print_section(f"Feature-group ablation ({model_name})")

    assert_all_features_grouped(X_train.columns)
    groups = build_feature_groups(X_train.columns)
    print(f"{len(groups)} blocks covering all {X_train.shape[1]} features:")
    for name, columns in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:28s} {len(columns):>3} features")

    def fit_and_score(columns):
        estimator = build_models(X_train[columns])[model_name]
        if best_params.get(model_name):
            estimator.set_params(**best_params[model_name])
        estimator.fit(X_train[columns], y_train)
        y_score = get_scores(estimator, X_test[columns])
        return np.asarray(y_score), compute_metrics(
            y_test, estimator.predict(X_test[columns]), y_score)

    print("\n  fitting the full model (all blocks) ...")
    scores_by_key = {}
    rows = []
    full_score, full_metrics = fit_and_score(list(X_train.columns))
    scores_by_key["ALL BLOCKS"] = full_score
    rows.append({"removed_block": "ALL BLOCKS", "n_features_removed": 0,
                 "n_features_used": X_train.shape[1], "gini": full_metrics["gini"]})

    for name, columns in groups.items():
        remaining = [c for c in X_train.columns if c not in set(columns)]
        print(f"  removing {name:28s} (-{len(columns):>2} features) ...")
        score, metrics = fit_and_score(remaining)
        scores_by_key[name] = score
        rows.append({"removed_block": name, "n_features_removed": len(columns),
                     "n_features_used": len(remaining), "gini": metrics["gini"]})

    print(f"\nPaired bootstrap ({PAIRED_BOOTSTRAP_RESAMPLES:,} resamples of the test rows) ...")
    ci = paired_bootstrap_gini(
        y_test, scores_by_key, reference_key="ALL BLOCKS", key_name="removed_block")
    table = (
        pd.DataFrame(rows)
        .merge(ci, on="removed_block", how="left")
        .sort_values("delta_vs_full")
        .reset_index(drop=True)
    )
    table["block_adds_value"] = table["delta_bonf_hi"] < 0

    print("\n" + "=" * 118)
    print(f"FEATURE-GROUP ABLATION ({model_name}, test set)")
    print("=" * 118)
    show = table[[
        "removed_block", "n_features_removed", "gini", "delta_vs_full",
        "delta_vs_full_lo", "delta_vs_full_hi", "delta_bonf_lo", "delta_bonf_hi",
        "block_adds_value",
    ]]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220):
        print(show.to_string(index=False))
    print("\ndelta_vs_full = Gini(without the block) - Gini(all blocks). Negative means")
    print("the block was carrying signal. block_adds_value uses the Bonferroni interval,")
    print("so it holds jointly across all blocks tested.")
    print("\nCaution: these are MARGINAL contributions. Two blocks encoding the same")
    print("information can each look dispensable while removing both would not be.")

    table.to_csv(OUTPUT_DIR / "feature_group_ablation.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'feature_group_ablation.csv'}")
    return table



def run_model_comparison_bootstrap(fitted: dict, results: pd.DataFrame, X_test, y_test):
    """Paired bootstrap of every tuned model against the best one.

    Answers "how many models are genuinely tied for best?" without an arbitrary
    cutoff. All models are scored on the SAME test rows, so a paired bootstrap of
    the difference is far more sensitive than comparing each model's own interval,
    or than comparing the gap to a cross-validation standard deviation (which
    measures fold-to-fold stability, not the uncertainty of a difference).

    Returns (table, indistinguishable) where `indistinguishable` lists the models
    whose Bonferroni-adjusted interval against the best model contains zero -- the
    reference itself included.
    """
    print_section("Pairwise model comparison (paired bootstrap, test set)")

    scores = {}
    for name, model in fitted.items():
        y_score = get_scores(model, X_test)
        if y_score is not None:
            scores[name] = np.asarray(y_score)

    best = results.iloc[0]["model"]
    print(f"Reference (best on test by {SELECTION_METRIC.upper()}): {best}")
    print(f"{PAIRED_BOOTSTRAP_RESAMPLES:,} resamples of {len(y_test):,} test rows, "
          f"{len(scores) - 1} simultaneous comparisons (Bonferroni-adjusted).\n")

    ci = paired_bootstrap_gini(y_test, scores, reference_key=best, key_name="model")
    gini = results.set_index("model")[SELECTION_METRIC]
    table = (
        ci.assign(gini=ci["model"].map(gini))
        .sort_values("gini", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "delta_vs_full": "delta_vs_best",
            "delta_vs_full_lo": "ci95_lo", "delta_vs_full_hi": "ci95_hi",
            "delta_bonf_lo": "bonf_lo", "delta_bonf_hi": "bonf_hi",
            "differs_from_full": "differs_95", "differs_from_full_bonf": "differs_bonf",
        })
    )

    show = table[["model", "gini", "delta_vs_best", "ci95_lo", "ci95_hi",
                  "bonf_lo", "bonf_hi", "differs_95", "differs_bonf"]]
    print("=" * 118)
    print(f"PAIRED DIFFERENCE FROM {best.upper()} (test-set GINI)")
    print("=" * 118)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220):
        print(show.to_string(index=False))

    indistinguishable = table.loc[~table["differs_bonf"], "model"].tolist()
    print(f"\nStatistically indistinguishable from {best} at family-wise 95%: "
          f"{', '.join(indistinguishable)}")
    print(f"  -> {len(indistinguishable)} model(s) qualify for inspection, "
          f"the reference included.")
    print("\nA model whose Bonferroni interval excludes zero is genuinely worse, even")
    print("if its raw Gini looks close. Do not use a cross-validation standard")
    print("deviation as the cutoff: it measures stability, not this difference.")

    path = OUTPUT_DIR / "model_pairwise_bootstrap.csv"
    table.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return table, indistinguishable



def run_importance_agreement(
    model_names: list[str], best_params: dict, X_train, y_train,
) -> pd.DataFrame:
    """Do the statistically tied models rank the features the same way?

    Same method for each: cross-validated permutation importance on held-out
    TRAINING folds, in Gini units. If the rankings agree, the ordering is a
    property of the data; if they disagree, no single model's importances can be
    presented as the project's feature importances.

    Returns a frame of per-model importances (index = feature), also written to
    feature_importance_agreement.csv. The caller can reuse a column to avoid
    recomputing the same ranking for the top-k curve.
    """
    print_section("Feature-importance agreement across the tied models")

    importances = {}
    for name in model_names:
        estimator = build_models(X_train)[name]
        if best_params.get(name):
            estimator.set_params(**best_params[name])
        print(f"\n  {name}:")
        importances[name] = cv_permutation_importance(estimator, X_train, y_train)

    table = pd.DataFrame(importances)
    table["block"] = [_group_membership(f) for f in table.index]

    if len(model_names) < 2:
        print(f"\nOnly one model qualifies ({model_names[0]}), so there is nothing to")
        print("cross-check. Its importances stand on their own.")
    else:
        print("\n" + "=" * 100)
        print("RANK AGREEMENT")
        print("=" * 100)
        for i, first in enumerate(model_names):
            for second in model_names[i + 1:]:
                rho = table[first].corr(table[second], method="spearman")
                top_a = set(table[first].nlargest(10).index)
                top_b = set(table[second].nlargest(10).index)
                print(f"{first} vs {second}:")
                print(f"  Spearman rho over all {len(table)} features: {rho:.3f}")
                print(f"  top-10 overlap: {len(top_a & top_b)}/10")
                if top_a - top_b:
                    print(f"    only {first}: {sorted(top_a - top_b)}")
                if top_b - top_a:
                    print(f"    only {second}: {sorted(top_b - top_a)}")

        print("\nBlock totals (summed importance per model):")
        blocks = table.groupby("block")[model_names].sum()
        with pd.option_context("display.float_format", "{:.5f}".format):
            print(blocks.sort_values(model_names[0], ascending=False).to_string())

        print("\nRead the ORDER as a property of the data and the MAGNITUDE as a")
        print("property of the estimator: a high rank correlation with divergent")
        print("magnitudes means the models agree on what matters, not on how much.")
        print("Importances are a ranking, not an additive decomposition of Gini.")

        plot_importance_comparison(table, list(model_names))

    path = OUTPUT_DIR / "feature_importance_agreement.csv"
    table.to_csv(path)
    print(f"\nSaved: {path}")
    return table



def run_pr_curve_dissatisfied(fitted: dict, X_test, y_test) -> pd.DataFrame:
    """Precision-Recall curve for the DISSATISFIED (minority, y=0) class.

    The satisfied/dissatisfied split is roughly 70/30, so the minority
    dissatisfied class is the actionable one and the whole metric suite is built
    around it. Each model's satisfied-probability score is negated so a high
    score means 'likely dissatisfied', which makes the area under this curve the
    pr_auc_dissat already reported in the evaluation tables (average precision of
    exactly this curve). The dashed line is the no-skill baseline: the
    dissatisfied base rate, which a random ranker reaches at every recall.

    Models are overlaid (same two as the learning curve) so the plot tells the
    same two-model story as the rest of the modelling section.
    """
    print_section("Precision-Recall curve (dissatisfied class)")

    y_true = np.asarray(y_test)
    y_dissat = (y_true == 0).astype(int)
    base_rate = float(y_dissat.mean())
    print(f"Dissatisfied base rate (no-skill precision): {base_rate:.4f}")

    models = [m for m in PR_CURVE_MODELS if m in fitted]
    if not models:
        print("None of the PR_CURVE_MODELS are available; skipping the PR curve.")
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = dict(zip(models, IMPORTANCE_PLOT_COLOURS))

    rows = []
    for name in models:
        y_score = get_scores(fitted[name], X_test)
        if y_score is None:
            print(f"  {name}: no probability/decision scores available, skipped.")
            continue
        dissat_score = -np.asarray(y_score, dtype=float)
        precision, recall, _ = precision_recall_curve(y_dissat, dissat_score)
        ap = average_precision_score(y_dissat, dissat_score)
        ax.plot(recall, precision, linewidth=2.6,
                label=f"{MODEL_DISPLAY_NAMES.get(name, name)} (AP = {ap:.3f})",
                color=palette.get(name))
        rows.append({"model": name, "pr_auc_dissat": ap})
        print(f"  {name:24s} PR-AUC (dissatisfied) = {ap:.4f}")

    ax.axhline(base_rate, color="0.35", linestyle="--", linewidth=1.2,
               label=f"No-skill baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall (dissatisfied)")
    ax.set_ylabel("Precision (dissatisfied)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")
    apply_simple_plot_style(ax)
    fig.tight_layout()
    path = PLOT_DIR / "pr_curve_dissatisfied.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(OUTPUT_DIR / "pr_curve_dissatisfied.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'pr_curve_dissatisfied.csv'}")
    return summary


def run_learning_curve(X_full, y_full, best_params: dict) -> pd.DataFrame:
    """Test-set Gini as a function of TRAINING-SET SIZE, on the full dataset.

    Only the models that scale cheaply are used: HistGradientBoosting (histogram
    binning, linear in rows) and logistic regression. KNN, the linear SVM and the
    MLP are excluded because they cannot be fitted at these sizes -- which is the
    same reason the main comparison is capped at a 100k subsample.

    One stratified test set is held out once and reused at every training size, so
    the curve is not confounded by a moving evaluation target.
    """
    print_section("Learning curve (does more data help?)")

    X_pool, X_lc_test, y_pool, y_lc_test = train_test_split(
        X_full, y_full,
        test_size=LEARNING_CURVE_TEST_SIZE,
        stratify=y_full,
        random_state=RANDOM_STATE + 1,
    )
    print(f"pool {len(X_pool):,} rows | fixed test {len(X_lc_test):,} rows "
          f"| positive rate {y_full.mean():.4f}")
    print("NOTE: this split is independent of the 100k model-comparison split, so")
    print("its numbers are not directly comparable with the model-comparison tables.\n")

    rows = []
    for size in LEARNING_CURVE_SIZES:
        if size > len(X_pool):
            print(f"  skipping n={size:,} (pool has only {len(X_pool):,} rows)")
            continue
        X_s, y_s = stratified_subsample(X_pool, y_pool, size)
        for name in LEARNING_CURVE_MODELS:
            estimator = build_models(X_s)[name]
            if best_params.get(name):
                estimator.set_params(**best_params[name])
            start = perf_counter()
            estimator.fit(X_s, y_s)
            seconds = perf_counter() - start
            scored = compute_metrics(
                y_lc_test, estimator.predict(X_lc_test), get_scores(estimator, X_lc_test))
            print(f"  n={size:>9,}  {name:24s} gini {scored['gini']:.4f}  ({seconds:5.1f}s)")
            rows.append({
                "n_train": size, "model": name, "gini": scored["gini"],
                "pr_auc_dissat": scored["pr_auc_dissat"],
                "balanced_accuracy": scored["balanced_accuracy"],
                "fit_seconds": seconds,
            })

    curve = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("LEARNING CURVE (fixed held-out test set)")
    print("=" * 90)
    wide = curve.pivot(index="n_train", columns="model", values="gini")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(wide.to_string())
    for name in wide.columns:
        first, last = wide[name].iloc[0], wide[name].iloc[-1]
        past_100k = wide[name].loc[wide.index >= 100_000]
        print(f"\n{name}:")
        print(f"  {wide.index[0]:,} -> {wide.index[-1]:,} rows: gini {first:.4f} -> {last:.4f} "
              f"({last - first:+.4f})")
        if len(past_100k) > 1:
            print(f"  from 100,000 rows onward: {past_100k.iloc[0]:.4f} -> {past_100k.iloc[-1]:.4f} "
                  f"({past_100k.iloc[-1] - past_100k.iloc[0]:+.4f})")

    curve.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'learning_curve.csv'}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = dict(zip(LEARNING_CURVE_MODELS, IMPORTANCE_PLOT_COLOURS))
    for name in LEARNING_CURVE_MODELS:
        block = curve[curve["model"] == name]
        ax.plot(block["n_train"], block["gini"], marker="o", linewidth=2.6,
                markersize=6, label=MODEL_DISPLAY_NAMES.get(name, name),
                color=palette.get(name))
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size (log scale)")
    ax.set_ylabel("Test-set GINI (fixed 250k held-out set)")
    ax.legend(loc="lower right")
    apply_simple_plot_style(ax)
    fig.tight_layout()
    path = PLOT_DIR / "learning_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return curve



def export_logistic_coefficients(fitted_logit, top_n: int = 20) -> pd.DataFrame:
    """Signed coefficients and odds ratios for the tuned logistic regression.

    Permutation importance is unsigned: it says a feature matters, never whether
    it is associated with higher or lower satisfaction. Managerial recommendations
    need the sign, so it comes from here.

    Numeric features are standardised inside the pipeline, so a coefficient is the
    log-odds change per one standard deviation. Categorical features are one-hot
    encoded without a dropped reference level, so their coefficients are relative
    to the (L2-shrunk) average level, not to a baseline category -- read them
    against each other, not in absolute terms.
    """
    print_section("Logistic regression coefficients (direction of effect)")

    preprocessor = fitted_logit.named_steps["pre"]
    estimator = fitted_logit.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefficients = estimator.coef_[0]

    table = pd.DataFrame({
        "feature": names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    print(f"{len(table)} encoded features. Odds ratio > 1 = associated with higher")
    print("satisfaction; < 1 = associated with lower satisfaction. Numeric effects")
    print("are per standard deviation.\n")
    header = f"{'feature':<52} {'coef':>9} {'odds ratio':>11}"
    print(header)
    print("-" * len(header))
    for _, row in table.head(top_n).iterrows():
        print(f"{row['feature']:<52} {row['coefficient']:>+9.4f} {row['odds_ratio']:>11.4f}")

    path = OUTPUT_DIR / "coefficients_logistic_regression.csv"
    table.drop(columns="abs_coefficient").to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return table



def models_main(df: pd.DataFrame) -> None:
    print("Building feature set...")
    X_full, y_full = select_features(df)

    X, y = X_full, y_full
    X, y = stratified_subsample(X, y, MODEL_SAMPLE_N)
    if MODEL_SAMPLE_N is not None:
        print(f"Using a stratified subsample of {len(X):,} rows for the model comparison.")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows "
          f"| positive rate {y.mean() * 100:.2f}%")

    print(f"No-information reference (always predict 'satisfied'): "
          f"Balanced Acc. 0.5000, GINI 0.0000, PR-AUC {1 - y.mean():.4f}, TDL 1.0000.")
    print(f"Positive (satisfied) rate {y.mean():.4f}. Plain accuracy is not reported: "
          f"a no-skill rule would score {y.mean():.4f} on it.")

    models = build_models(X_train)

    untuned_test = None
    if DO_UNTUNED_COMPARISON:
        print("\nFitting the UN-TUNED (default) models for the tuning comparison...")
        untuned_fitted, untuned_seconds = fit_and_time(build_models(X_train), X_train, y_train)

        untuned_train = evaluate_fitted(untuned_fitted, X_train, y_train, untuned_seconds)
        untuned_table = to_report_table(untuned_train)
        print_report_table(untuned_table, "EVALUATION: UN-TUNED MODELS on TRAINING data")
        untuned_train.to_csv(OUTPUT_DIR / "eval_untuned_train_full.csv", index=False)
        untuned_table.to_csv(OUTPUT_DIR / "eval_untuned_train.csv", index=False)
        save_report_table_png(
            untuned_table, "Un-tuned models - training data", "eval_untuned_train.png",
        )

        untuned_test = evaluate_fitted(untuned_fitted, X_test, y_test, untuned_seconds)
        untuned_test_table = to_report_table(untuned_test)
        print_report_table(untuned_test_table, "EVALUATION: UN-TUNED MODELS on HELD-OUT TEST data")
        untuned_test.to_csv(OUTPUT_DIR / "eval_untuned_test_full.csv", index=False)
        untuned_test_table.to_csv(OUTPUT_DIR / "eval_untuned_test.csv", index=False)
        save_report_table_png(
            untuned_test_table, "Un-tuned models - held-out test data", "eval_untuned_test.png",
        )
        del untuned_fitted

    best_params: dict[str, dict] = {}
    cv_results = None

    if DO_GRID_SEARCH:
        best_params = run_grid_search(X_train, y_train)
        models = apply_best_params(models, best_params)

    if DO_CV:
        print("\nRunning light cross-validation on the tuned models (training subsample)...")
        cv_results = run_cross_validation(models, X_train, y_train)
        print("\n" + "=" * 100)
        print(f"CROSS-VALIDATION ({CV_FOLDS}-fold, training subsample, tuned models)")
        print("=" * 100)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(cv_results.to_string(index=False))
        cv_results.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    print("\nFitting the TUNED models and evaluating on training and held-out test sets...")
    results, fitted, train_results, _ = evaluate_holdout(models, X_train, X_test, y_train, y_test)

    tuned_train_table = to_report_table(train_results)
    print_report_table(tuned_train_table, "EVALUATION: TUNED MODELS on TRAINING data")
    train_results.to_csv(OUTPUT_DIR / "model_comparison_train.csv", index=False)
    tuned_train_table.to_csv(OUTPUT_DIR / "eval_tuned_train.csv", index=False)
    save_report_table_png(
        tuned_train_table, "Tuned models - training data", "eval_tuned_train.png",
    )

    tuned_test_table = to_report_table(results)
    print_report_table(tuned_test_table, "EVALUATION: TUNED MODELS on HELD-OUT TEST data")
    print("\nCompare against the training table above: scores far higher on train")
    print("than test = overfitting; both low and similar = underfitting.")
    print("Note: Time is the training fit time, so it is identical in both tuned tables.")
    results.to_csv(OUTPUT_DIR / "model_comparison_holdout.csv", index=False)
    tuned_test_table.to_csv(OUTPUT_DIR / "eval_tuned_test.csv", index=False)
    save_report_table_png(
        tuned_test_table, "Tuned models - held-out test data", "eval_tuned_test.png",
    )

    best_name = results.iloc[0]["model"]
    print(f"\nBest model by {SELECTION_METRIC.upper()} (invariant to class imbalance): {best_name}")
    plot_confusion(fitted[best_name], X_test, y_test, best_name)

    if untuned_test is not None:
        delta = (
            results.set_index("model")["gini"] - untuned_test.set_index("model")["gini"]
        ).sort_values(ascending=False)
        print_subsection("Effect of hyperparameter tuning (test-set GINI, tuned - un-tuned)")
        for name, change in delta.items():
            print(f"  {name:24s} {change:+.4f}")
        print(f"\n  mean {delta.mean():+.4f} | median {delta.median():+.4f} | "
              f"improved {int((delta > 0).sum())}/{len(delta)} models")

    if DO_FEATURE_IMPORTANCE:
        plot_feature_importance(fitted, X_test, y_test, results["model"].head(3).tolist())

    if "logistic_regression" in fitted:
        export_logistic_coefficients(fitted["logistic_regression"])

    tied_models = [results.iloc[0]["model"]]
    if DO_MODEL_BOOTSTRAP:
        _, tied_models = run_model_comparison_bootstrap(fitted, results, X_test, y_test)

    agreement = None
    if DO_IMPORTANCE_AGREEMENT:
        agreement = run_importance_agreement(tied_models, best_params, X_train, y_train)

    if DO_TOPK_CURVE:
        if DO_CV:
            topk_model = cv_results.iloc[0]["model"]
            print(f"\nCurve model chosen by cross-validated {SELECTION_METRIC}: {topk_model}")
        else:
            topk_model = results.iloc[0]["model"]
            print(f"\nWARNING: DO_CV is off, so the curve model ({topk_model}) was")
            print("chosen using the test set. Enable DO_CV for a clean selection.")

        precomputed = None
        if agreement is not None and topk_model in agreement.columns:
            precomputed = agreement[topk_model]
        run_topk_curve(topk_model, best_params, X_train, y_train, X_test, y_test,
                       importance=precomputed)

    if DO_GROUP_ABLATION:
        ablation_model = cv_results.iloc[0]["model"] if DO_CV else results.iloc[0]["model"]
        run_feature_group_ablation(
            ablation_model, best_params, X_train, y_train, X_test, y_test)

    if DO_PR_CURVE:
        run_pr_curve_dissatisfied(fitted, X_test, y_test)

    if DO_LEARNING_CURVE:
        run_learning_curve(X_full, y_full, best_params)

    print("\nModel comparison completed.")


def main() -> None:
    """Run the project end to end as one linear pipeline.

        ingestion -> preprocessing -> weather merge -> EDA -> modelling -> results

    Every stage runs on every invocation: the raw CSVs are parsed, cleaned and
    merged, NOAA weather is joined on, and the resulting dataset flows straight
    into EDA and then modelling. Nothing is loaded from a cached dataset.

    Feature drops are sequenced so each one happens after the analysis that
    justifies it. Low-signal features go first, before EDA, so they never enter
    the EDA tables. The redundancy drops happen inside build_model_dataset,
    after eda_main has printed the correlation and VIF diagnostics that motivate
    them.

    The NOAA download itself stays cached (WEATHER_CACHE_PKL): the raw payload is
    fixed historical data, so re-fetching it every run would only add an API
    round-trip. Everything downstream of that payload is rebuilt from scratch.
    """
    tables = load_raw_data()
    processed = process_data(tables)
    enriched = build_weather_enriched(processed)

    enriched = drop_low_signal_features(enriched)

    eda_main(enriched)

    model_df = build_model_dataset(enriched)
    models_main(model_df)

    print("\nDone.")
    print(f"Enriched dataset:  {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")
    print(f"Modelling dataset: {model_df.shape[0]:,} rows x {model_df.shape[1]} columns")


if __name__ == "__main__":
    main()
