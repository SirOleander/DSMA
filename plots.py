from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import PROCESSED_DATA_DIR


SAMPLE_SIZE = 2_000_000

DATA_FILE = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"

PROJECT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_pickle(DATA_FILE)
    print(f"Loaded data: {df.shape}")
    return df


def save_plot(filename: str) -> None:
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_target_distribution(df: pd.DataFrame) -> None:
    target_counts = df["satisfied"].value_counts().sort_index()
    target_counts.index = ["Not satisfied", "Satisfied"]

    plt.figure(figsize=(7, 5))
    target_counts.plot(kind="bar")
    plt.title("Distribution of Customer Satisfaction")
    plt.xlabel("Satisfaction Class")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)

    save_plot("01_target_distribution.png")


def plot_review_star_distribution(df: pd.DataFrame) -> None:
    star_counts = df["review_stars"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    star_counts.plot(kind="bar")
    plt.title("Distribution of Review Stars")
    plt.xlabel("Review Stars")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)

    save_plot("02_review_star_distribution.png")


def plot_top_cities(df: pd.DataFrame) -> None:
    top_cities = df["city"].value_counts().head(10).sort_values()

    plt.figure(figsize=(8, 6))
    top_cities.plot(kind="barh")
    plt.title("Top 10 Cities by Number of Restaurant Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("City")

    save_plot("03_top_10_cities.png")


def plot_top_states(df: pd.DataFrame) -> None:
    top_states = df["state"].value_counts().head(10).sort_values()

    plt.figure(figsize=(8, 6))
    top_states.plot(kind="barh")
    plt.title("Top 10 States by Number of Restaurant Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("State")

    save_plot("04_top_10_states.png")


def plot_satisfaction_by_category(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str
) -> None:
    summary = (
        df.groupby(column)["satisfied"]
        .mean()
        .sort_values()
        * 100
    )

    plt.figure(figsize=(8, 5))
    summary.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Satisfaction Rate (%)")
    plt.ylabel(column)

    save_plot(filename)


def plot_correlation_with_target(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number").copy()

    columns_to_drop = [
        "satisfied",
        "review_stars"
    ]

    numeric_df = numeric_df.drop(columns=columns_to_drop, errors="ignore")

    correlations = (
        numeric_df
        .corrwith(df["satisfied"])
        .sort_values()
    )

    selected_correlations = pd.concat([
        correlations.head(10),
        correlations.tail(10)
    ])

    plt.figure(figsize=(9, 7))
    selected_correlations.plot(kind="barh")
    plt.title("Numeric Correlations with Satisfaction")
    plt.xlabel("Correlation with Satisfaction")
    plt.ylabel("Variable")

    save_plot("08_correlation_with_satisfaction.png")


def main() -> None:
    df = load_data()

    plot_target_distribution(df)
    plot_review_star_distribution(df)
    plot_top_cities(df)
    plot_top_states(df)

    plot_satisfaction_by_category(
        df=df,
        column="attributes.RestaurantsPriceRange2",
        title="Satisfaction Rate by Restaurant Price Range",
        filename="05_satisfaction_by_price_range.png"
    )

    plot_satisfaction_by_category(
        df=df,
        column="attributes.OutdoorSeating",
        title="Satisfaction Rate by Outdoor Seating",
        filename="06_satisfaction_by_outdoor_seating.png"
    )

    plot_satisfaction_by_category(
        df=df,
        column="attributes.RestaurantsDelivery",
        title="Satisfaction Rate by Delivery Availability",
        filename="07_satisfaction_by_delivery.png"
    )

    plot_correlation_with_target(df)

    print("\nEDA plots completed.")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()