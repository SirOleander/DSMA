"""
Model comparison for the Yelp restaurant satisfaction project.

Compares a baseline against two real classifiers, all leakage-safe via
features.py and all preprocessing fit on training data only:

  - baseline_most_frequent : DummyClassifier, the floor every model must beat
  - logistic_regression    : linear, needs scaling + one-hot (sparse)
  - hist_gradient_boosting : tree ensemble, no scaling, native categorical
                             handling via ordinal-encoded categoricals

The target is imbalanced (~68% satisfied), so the comparison reports F1,
precision, recall, balanced accuracy and PR-AUC alongside accuracy and ROC-AUC,
and the two real models use class_weight="balanced". Accuracy is reported but
should not be the deciding metric.

Evaluation strategy (chosen for a multi-million-row dataset):
  - Primary: train on X_train, evaluate once on the held-out X_test.
  - Optional: light stratified CV on a subsample for stability (DO_CV).

Run:  python models.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
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

import features
from config import PROCESSED_DATA_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROCESSED_DATA_DIR / "model_outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Light cross-validation for stability. Off by default because k-fold on
# millions of rows is slow; when on, it runs on a stratified subsample.
DO_CV = False
CV_FOLDS = 3
CV_SUBSAMPLE = 300_000


# ---------------------------------------------------------------------------
# Tree-specific preprocessor (compact, no one-hot blow-up)
# ---------------------------------------------------------------------------

def build_tree_preprocessor(X: pd.DataFrame):
    """
    Preprocessor for tree models: median-impute numerics (no scaling) and
    ordinal-encode categoricals so HistGradientBoosting can treat them as true
    categoricals natively. Returns (preprocessor, categorical_mask) where the
    mask aligns with the transformed column order [numeric..., categorical...].
    """
    numeric, categorical = features.split_feature_types(X)

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,          # unseen categories -> treated as missing
            encoded_missing_value=np.nan,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    return preprocessor, categorical_mask


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models(X_train: pd.DataFrame) -> dict:
    """Build the candidate models, each as a complete leakage-safe Pipeline."""
    linear_pre, _, _ = features.build_preprocessor(X_train, scale_numeric=True)
    tree_pre, cat_mask = build_tree_preprocessor(X_train)

    return {
        "baseline_most_frequent": Pipeline([
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("pre", linear_pre),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("pre", tree_pre),
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

def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Classification metrics suited to an imbalanced binary target."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def get_scores(model, X) -> np.ndarray | None:
    """Positive-class probabilities if available, else None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Fit each model on train, evaluate on the held-out test set."""
    rows = []
    fitted = {}

    for name, model in models.items():
        print(f"  fitting {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)

        row = {"model": name}
        row.update(compute_metrics(y_test, y_pred, y_score))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return results, fitted


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data and building feature set...")
    df = features.load_enriched()
    X, y = features.select_features(df)
    X_train, X_test, y_train, y_test = features.split_data(X, y)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows "
          f"| positive rate {y.mean() * 100:.2f}%")

    models = build_models(X_train)

    print("\nEvaluating on held-out test set...")
    results, fitted = evaluate_holdout(models, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON (held-out test set)")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(results.to_string(index=False))

    results.to_csv(OUTPUT_DIR / "model_comparison_holdout.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison_holdout.csv'}")

    best_name = results.iloc[0]["model"]
    print(f"\nBest model by ROC-AUC (baseline-proof): {best_name}")
    plot_confusion(fitted[best_name], X_test, y_test, best_name)

    if DO_CV:
        print("\nRunning light cross-validation (subsample)...")
        cv_results = run_cross_validation(models, X_train, y_train)
        print("\n" + "=" * 100)
        print(f"CROSS-VALIDATION ({CV_FOLDS}-fold, subsample)")
        print("=" * 100)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(cv_results.to_string(index=False))
        cv_results.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()
