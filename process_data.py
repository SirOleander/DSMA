import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR


SAMPLE_SIZE = 3_000_000
RANDOM_STATE = 42


def load_processed_tables() -> dict:
    """Load already-converted pickle files."""
    tables = {
        "business": pd.read_pickle(PROCESSED_DATA_DIR / "business.pkl"),
        "reviews": pd.read_pickle(PROCESSED_DATA_DIR / "reviews.pkl"),
        "users": pd.read_pickle(PROCESSED_DATA_DIR / "users.pkl"),
        "checkin": pd.read_pickle(PROCESSED_DATA_DIR / "checkin.pkl"),
        "tip": pd.read_pickle(PROCESSED_DATA_DIR / "tip.pkl"),
        "photo": pd.read_pickle(PROCESSED_DATA_DIR / "photo.pkl"),
    }

    return tables


def format_clean_table(df: pd.DataFrame) -> str:
    """
    Format a DataFrame as a clean, uniformly left-aligned text table.
    """
    if df.empty:
        return "No data available."

    formatted_df = df.copy()

    for col in formatted_df.columns:
        if pd.api.types.is_float_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}")
        else:
            formatted_df[col] = formatted_df[col].astype(str)

    column_widths = {}

    for col in formatted_df.columns:
        max_content_width = formatted_df[col].map(len).max()
        column_widths[col] = max(len(col), max_content_width)

    header = "  ".join(
        col.ljust(column_widths[col])
        for col in formatted_df.columns
    )

    separator = "  ".join(
        "-" * column_widths[col]
        for col in formatted_df.columns
    )

    rows = []

    for _, row in formatted_df.iterrows():
        rows.append(
            "  ".join(
                str(row[col]).ljust(column_widths[col])
                for col in formatted_df.columns
            )
        )

    return "\n".join([header, separator] + rows)


def print_section_header(title: str) -> None:
    """
    Print a clear section header.
    """
    line = "=" * 100
    print("\n" + line)
    print(title.upper())
    print(line)


def print_subsection_header(title: str) -> None:
    """
    Print a clear subsection header.
    """
    print("\n" + "-" * 100)
    print(title.upper())
    print("-" * 100)


def print_table_and_variable_overview(
    tables: dict,
    title: str
) -> None:
    """
    Print table-level and variable-level overview for a dictionary of DataFrames.

    The output is uniformly left-aligned and variables are grouped by table.
    """
    print_section_header(title)

    table_overview = pd.DataFrame([
        {
            "table": table_name,
            "rows": df.shape[0],
            "columns": df.shape[1],
        }
        for table_name, df in tables.items()
    ])

    print_subsection_header("Table overview")
    print(format_clean_table(table_overview))

    print_subsection_header("Variable overview")

    for table_name, df in tables.items():
        print("\n" + f"[{table_name}]")

        variable_rows = []

        for col in df.columns:
            variable_rows.append({
                "variable": col,
                "dtype": str(df[col].dtype),
                "missing_values": df[col].isna().sum(),
                "missing_percent": round(df[col].isna().mean() * 100, 2),
                "unique_values": df[col].nunique(dropna=True),
            })

        variable_overview = pd.DataFrame(variable_rows)

        print(format_clean_table(variable_overview))

def print_variable_overview_only(
    df: pd.DataFrame,
    title: str,
    table_name: str
) -> None:
    """
    Print only the variable-level overview for a single DataFrame.
    """
    print_section_header(title)

    print("\n" + f"[{table_name}]")

    variable_rows = []

    for col in df.columns:
        variable_rows.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "missing_values": df[col].isna().sum(),
            "missing_percent": round(df[col].isna().mean() * 100, 2),
            "unique_values": df[col].nunique(dropna=True),
        })

    variable_overview = pd.DataFrame(variable_rows)

    print(format_clean_table(variable_overview))

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
    """Sample reviews before merging to keep processing manageable."""
    sample_n = min(SAMPLE_SIZE, len(reviews))

    review_sample = reviews.sample(
        n=sample_n,
        random_state=RANDOM_STATE
    ).copy()

    return review_sample


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

    count_columns = [
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

    for col in count_columns:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0)

    categorical_columns = [
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

    for col in categorical_columns:
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

    return dataset


def add_log_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of skewed count variables."""
    dataset = dataset.copy()

    log_candidates = [
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
        "review_useful",
        "review_funny",
        "review_cool",
        "review_text_length",
    ]

    for col in log_candidates:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
            dataset[f"log_{col}"] = np.log1p(dataset[col].fillna(0).clip(lower=0))

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

        # Keep for EDA, but do not use as model input later
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
    ]

    available_columns = [col for col in model_columns if col in dataset.columns]

    missing_columns = [col for col in model_columns if col not in dataset.columns]
    if missing_columns:
        print("\nColumns not found and therefore skipped:")
        print(missing_columns)

    return dataset[available_columns].copy()

def save_outputs(model_data: pd.DataFrame) -> None:
    """Save processed sample dataset."""
    output_pkl = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"
    output_csv = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.csv"

    model_data.to_pickle(output_pkl)
    model_data.to_csv(output_csv, index=False)

def main() -> None:
    print("Loading processed tables...")
    tables = load_processed_tables()

    business = tables["business"]
    reviews = tables["reviews"]
    users = tables["users"]
    checkin = tables["checkin"]
    tip = tables["tip"]
    photo = tables["photo"]

    business, reviews, users = prepare_core_tables(
        business=business,
        reviews=reviews,
        users=users
    )

    print_table_and_variable_overview(
        tables={
            "business": business,
            "reviews": reviews,
            "users": users,
            "checkin_raw": checkin,
            "tip_raw": tip,
            "photo_raw": photo,
        },
        title="INITIAL DATASET"
    )

    business, reviews, users = select_columns_before_merge(
        business=business,
        reviews=reviews,
        users=users
    )

    review_sample = sample_reviews(reviews)

    checkin_features = create_checkin_features(checkin)
    tip_features = create_tip_features(tip)
    photo_features = create_photo_features(photo)

    print_table_and_variable_overview(
        tables={
            "checkin_features": checkin_features,
            "tip_features": tip_features,
            "photo_features": photo_features,
        },
        title="NEW VARIABLES AFTER AGGREGATION"
    )

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

    print("\nAdding log features...")
    dataset = add_log_features(dataset)

    print("\nSelecting final columns...")
    model_data = select_model_columns(dataset)

    print_variable_overview_only(
        df=model_data,
        title="FINAL VARIABLES FOR EDA AND MODELING",
        table_name="final_eda_dataset"
    )

    print("\nSaving model data...")
    save_outputs(model_data)

    print("\nProcessing completed.")

if __name__ == "__main__":
    main()