import json

import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


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


def save_dataframe(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame as pickle file."""
    output_path = PROCESSED_DATA_DIR / f"{name}.pkl"
    df.to_pickle(output_path)


def main() -> None:
    check_required_files()

    summary = []

    for name, filename in RAW_FILES.items():
        df = load_json_csv(filename)
        save_dataframe(df, name)

        summary.append({
            "table": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
        })

    summary_df = pd.DataFrame(summary)

    summary_path = PROCESSED_DATA_DIR / "loading_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Raw data loading completed.")
    print(summary_df)


if __name__ == "__main__":
    main()