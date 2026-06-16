"""
Feature preparation for the Yelp restaurant satisfaction modelling stage.

This is the leakage-safe bridge between the EDA dataset (WEATHER_ENRICHED_PKL)
and the model comparison. It does three things and nothing else:

  1. select_features(): choose the modelling feature set and the target,
     excluding IDs, the raw date, the leakage columns, and (by default) the
     post-hoc review-vote columns; it also drops each raw count that has a
     log_ twin to avoid near-perfect collinearity.
  2. split_data(): a stratified train/test split on the (imbalanced) target.
  3. build_preprocessor(): a scikit-learn ColumnTransformer that imputes,
     scales, and one-hot-encodes. It is returned UNFITTED on purpose so it can
     be fit on the training fold only (inside a Pipeline / cross-validation),
     which is what keeps the evaluation honest.

No model is trained here. Compose the preprocessor with each estimator in a
Pipeline at modelling time, e.g.:

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    pre, *_ = build_preprocessor(X_train)
    clf = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])
    clf.fit(X_train, y_train)              # preprocessor is fit on train only
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import WEATHER_ENRICHED_PKL, ID_COLS, LEAKAGE_COLS

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


def load_enriched() -> pd.DataFrame:
    """Load the weather-enriched review-level dataset."""
    return pd.read_pickle(WEATHER_ENRICHED_PKL)


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
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    return preprocessor, numeric, categorical


def main() -> None:
    print("Loading enriched dataset...")
    df = load_enriched()

    X, y = select_features(df)
    numeric, categorical = split_feature_types(X)

    print("\nFeature set")
    print("-" * 80)
    print(f"Rows:                {len(X):,}")
    print(f"Total features:      {X.shape[1]}")
    print(f"Numeric features:    {len(numeric)}")
    print(f"Categorical features:{len(categorical)}")
    print(f"\nNumeric:     {numeric}")
    print(f"Categorical: {categorical}")

    print("\nTarget balance (satisfied)")
    print("-" * 80)
    balance = y.value_counts(normalize=True).sort_index()
    for value, share in balance.items():
        print(f"  {value}: {share * 100:.2f}%")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nStratified split")
    print("-" * 80)
    print(f"Train rows: {len(X_train):,}  (positive rate {y_train.mean() * 100:.2f}%)")
    print(f"Test rows:  {len(X_test):,}  (positive rate {y_test.mean() * 100:.2f}%)")

    preprocessor, _, _ = build_preprocessor(X_train)
    print("\nPreprocessor built (unfitted). Fit it on X_train only, inside a "
          "Pipeline, at modelling time.")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    main()
