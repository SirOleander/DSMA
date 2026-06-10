import pandas as pd

from config import PROCESSED_DATA_DIR


SAMPLE_SIZE = 3_000_000

DATA_FILE = PROCESSED_DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.pkl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "unique_city_names.xlsx"


def load_data() -> pd.DataFrame:
    """Load processed Yelp restaurant dataset."""
    return pd.read_pickle(DATA_FILE)


def create_city_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Create overview of unique city-state combinations."""

    city_overview = (
        df.groupby(["city", "state"])
        .size()
        .reset_index(name="review_count")
        .sort_values(["state", "city"])
        .reset_index(drop=True)
    )

    total_reviews = city_overview["review_count"].sum()

    city_overview["review_share_percent"] = (
        city_overview["review_count"] / total_reviews * 100
    ).round(3)

    city_overview_by_count = city_overview.sort_values(
        "review_count",
        ascending=False
    ).reset_index(drop=True)

    city_overview_by_count["rank"] = city_overview_by_count.index + 1

    city_overview_by_count["cumulative_review_count"] = (
        city_overview_by_count["review_count"].cumsum()
    )

    city_overview_by_count["cumulative_share_percent"] = (
        city_overview_by_count["cumulative_review_count"] / total_reviews * 100
    ).round(2)

    city_overview_by_count = city_overview_by_count[
        [
            "rank",
            "city",
            "state",
            "review_count",
            "review_share_percent",
            "cumulative_review_count",
            "cumulative_share_percent",
        ]
    ]

    return city_overview_by_count


def export_to_excel(city_overview: pd.DataFrame) -> None:
    """Export city overview to Excel."""

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        city_overview.to_excel(
            writer,
            sheet_name="unique_cities",
            index=False
        )

        workbook = writer.book
        worksheet = writer.sheets["unique_cities"]

        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))

            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 35)

        worksheet.auto_filter.ref = worksheet.dimensions


def main() -> None:
    print("Loading processed Yelp dataset...")
    df = load_data()

    city_overview = create_city_overview(df)

    print("\nTop 50 city-state combinations by review count:")
    print(city_overview.head(50).to_string(index=False))

    export_to_excel(city_overview)

    print(f"\nExported unique city overview to: {OUTPUT_FILE}")
    print(f"Number of unique city-state combinations: {len(city_overview)}")


if __name__ == "__main__":
    main()