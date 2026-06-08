import pandas as pd

from config import PROCESSED_DATA_DIR


SAMPLE_SIZE = 2_000_000
DATA_FILE = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"


def load_data() -> pd.DataFrame:
    df = pd.read_pickle(DATA_FILE)
    print(f"\nLoaded data: {df.shape}")
    return df


def print_basic_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("BASIC DATASET OVERVIEW")
    print("=" * 80)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\nColumn types:")
    print(df.dtypes.value_counts())

    print("\nFirst 5 rows:")
    print(df.head())


def print_target_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)

    target_counts = df["satisfied"].value_counts().sort_index()
    target_percent = df["satisfied"].value_counts(normalize=True).sort_index() * 100

    target_summary = pd.DataFrame({
        "count": target_counts,
        "percent": target_percent.round(2)
    })

    print(target_summary)


def print_missing_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })

    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_percent",
        ascending=False
    )

    if missing.empty:
        print("No missing values.")
    else:
        print(missing)


def print_numeric_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("NUMERIC SUMMARY")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include="number").columns
    summary = df[numeric_cols].describe().T

    selected_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    print(summary[selected_cols].round(3))


def print_categorical_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("CATEGORICAL SUMMARY")
    print("=" * 80)

    categorical_cols = df.select_dtypes(include=["object", "category", "string", "str"]).columns

    # Do not treat IDs as normal categorical variables
    excluded_cols = ["review_id", "business_id", "user_id"]
    categorical_cols = [col for col in categorical_cols if col not in excluded_cols]

    for col in categorical_cols:
        print("\n" + "-" * 80)
        print(f"{col}")
        print("-" * 80)

        print(f"Unique values: {df[col].nunique(dropna=True):,}")
        print(f"Missing percent: {df[col].isna().mean() * 100:.2f}%")

        print("\nTop values:")
        print(
            df[col]
            .value_counts(dropna=False)
            .head(10)
        )


def print_top_locations(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TOP LOCATIONS")
    print("=" * 80)

    if "state" in df.columns:
        print("\nTop 10 states:")
        print(df["state"].value_counts().head(10))

    if "city" in df.columns:
        print("\nTop 10 cities:")
        print(df["city"].value_counts().head(10))


def print_satisfaction_by_category(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SATISFACTION RATE BY CATEGORICAL VARIABLES")
    print("=" * 80)

    categorical_features = [
        "attributes.RestaurantsPriceRange2",
        "attributes.BusinessAcceptsCreditCards",
        "attributes.RestaurantsTakeOut",
        "attributes.RestaurantsDelivery",
        "attributes.OutdoorSeating",
        "attributes.BikeParking",
        "attributes.GoodForKids",
        "attributes.RestaurantsReservations",
        "state",
    ]

    for col in categorical_features:
        if col not in df.columns:
            continue

        print("\n" + "-" * 80)
        print(f"{col}")
        print("-" * 80)

        summary = (
            df.groupby(col)["satisfied"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "satisfaction_rate"})
            .sort_values("count", ascending=False)
        )

        summary["satisfaction_rate"] = (summary["satisfaction_rate"] * 100).round(2)

        print(summary.head(15))


def print_correlation_with_target(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("CORRELATION WITH TARGET")
    print("=" * 80)

    numeric_df = df.select_dtypes(include="number")

    correlations = (
        numeric_df.corr(numeric_only=True)["satisfied"]
        .sort_values(ascending=False)
    )

    # Remove target itself
    correlations = correlations.drop("satisfied", errors="ignore")

    print("\nTop positive correlations:")
    print(correlations.head(15).round(3))

    print("\nTop negative correlations:")
    print(correlations.tail(15).round(3))


def main() -> None:
    df = load_data()

    print_basic_overview(df)
    print_target_distribution(df)
    print_missing_values(df)
    print_numeric_summary(df)
    print_categorical_summary(df)
    print_top_locations(df)
    print_satisfaction_by_category(df)
    print_correlation_with_target(df)

    print("\nEDA terminal summary completed.")


if __name__ == "__main__":
    main()