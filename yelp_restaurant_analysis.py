import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(r"C:\Users\fynne\University\Data Science and Marketing Analytics")
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def load_json_csv(filename):
    """Load a CSV file containing one JSON column and convert it to a DataFrame."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = pd.read_csv(path)
    json_col = raw.columns[0]

    return pd.json_normalize(raw[json_col].apply(json.loads))


# Load raw exported files
business = load_json_csv("restaurant_business_raw.csv")
reviews = load_json_csv("restaurant_reviews_raw.csv")
users = load_json_csv("restaurant_users_raw.csv")
checkin = load_json_csv("restaurant_checkin_raw.csv")
tip = load_json_csv("restaurant_tip_raw.csv")
photo = load_json_csv("restaurant_photo_raw.csv")

print("Loaded data:")
print(f"business: {business.shape}")
print(f"reviews:  {reviews.shape}")
print(f"users:    {users.shape}")
print(f"checkin:  {checkin.shape}")
print(f"tip:      {tip.shape}")
print(f"photo:    {photo.shape}")


# Rename columns to avoid confusion after merging
business = business.rename(columns={
    "name": "business_name",
    "stars": "business_stars",
    "review_count": "business_review_count"
})

reviews = reviews.rename(columns={
    "stars": "review_stars",
    "date": "review_date",
    "text": "review_text",
    "useful": "review_useful",
    "funny": "review_funny",
    "cool": "review_cool"
})

users = users.rename(columns={
    "name": "user_name",
    "review_count": "user_review_count",
    "average_stars": "user_average_stars",
    "useful": "user_useful",
    "funny": "user_funny",
    "cool": "user_cool"
})


# Create target variable
reviews["satisfied"] = (reviews["review_stars"] >= 4).astype(int)

print("\nTarget distribution:")
print(reviews["satisfied"].value_counts(normalize=True).round(3))


# Merge review, business, and user data
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

print("\nMerged dataset:")
print(dataset.shape)

print("\nMissing values after merge:")
print("business_name:", dataset["business_name"].isna().sum())
print("user_average_stars:", dataset["user_average_stars"].isna().sum())


# Save modeling sample
sample_data = dataset.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

sample_path = DATA_DIR / f"restaurant_model_sample_{SAMPLE_SIZE // 1000}k.csv"
sample_data.to_csv(sample_path, index=False)

print(f"\nSaved sample: {sample_path}")
print(f"Sample shape: {sample_data.shape}")