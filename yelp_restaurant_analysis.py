import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(r"C:\Users\fynne\University\Data Science and Marketing Analytics")

def load_json_csv(filename):
    path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print(f"Loading file: {filename}")
    print(f"Full path: {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print("Reading CSV...")
    raw = pd.read_csv(path)

    print(f"CSV loaded successfully.")
    print(f"Raw shape: {raw.shape}")
    print(f"Columns: {list(raw.columns)}")

    json_col = raw.columns[0]
    print(f"Using JSON column: {json_col}")

    print("Converting JSON strings to table format...")
    df = pd.json_normalize(raw[json_col].apply(json.loads))

    print("JSON conversion successful.")
    print(f"Final shape: {df.shape}")
    print(f"First columns: {list(df.columns[:10])}")

    return df


print("Starting data loading...")

business = load_json_csv("restaurant_business_raw.csv")
reviews = load_json_csv("restaurant_reviews_raw.csv")
users = load_json_csv("restaurant_users_raw.csv")
checkin = load_json_csv("restaurant_checkin_raw.csv")
tip = load_json_csv("restaurant_tip_raw.csv")
photo = load_json_csv("restaurant_photo_raw.csv")

print("\n" + "=" * 60)
print("All files loaded successfully.")
print("business:", business.shape)
print("reviews:", reviews.shape)
print("users:", users.shape)
print("checkin:", checkin.shape)
print("tip:", tip.shape)
print("photo:", photo.shape)