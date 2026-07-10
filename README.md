# Predicting Restaurant Customer Satisfaction from Yelp Data

Supervised binary classification of Yelp restaurant reviews: is a review
**satisfied** (`review_stars >= 4`) or not, predicted from **structured features
only** — business, user, engagement, contextual, and weather characteristics.
Review text is never used as a model input.

The entire pipeline is a single file, `main.py`, submitted as the report
appendix. It runs end to end with one command and no configuration.

---

## Quick start (for the grader)

```bash
# 1. Create and activate the environment (once)
conda env create -f zenvironment.yaml
conda activate dsma

# 2. Put the six raw CSVs in ./data/raw/   (see "Data" below)

# 3. Run the whole pipeline
python main.py
```

That is the entire interface. There are **no options to set and no stages to
enable** — a single run rebuilds the dataset from the raw CSVs, enriches it with
NOAA weather, runs the exploratory analysis, and then trains and compares the
models, writing every table and figure to disk.

**Expect the run to take roughly 25 minutes** and to use several GB of RAM. The
console prints each stage as it goes, ending with `Done.` and the dataset
dimensions.

> If you only want to confirm the code runs without waiting for the full
> hyperparameter search (the bulk of the 25 minutes), open `main.py` and set
> `DO_GRID_SEARCH = False`. The models then use sensible default hyperparameters
> and the run finishes in a few minutes. This is optional; the default is a full
> run.

---

## Requirements

The `conda env create` step above installs everything. For reference:

- **Python 3.11**
- `pandas`, `numpy`, `matplotlib`, `requests`
- **`scikit-learn >= 1.4`** — a hard requirement (earlier versions lack
  `HistGradientBoostingClassifier(class_weight=...)` and
  `KBinsDiscretizer(encode="onehot-dense")`).
- `colorspace` and `wordcloud` are **optional**: they only affect a few EDA
  figures. If they are missing, the pipeline falls back to matplotlib palettes
  and skips the word cloud — it does **not** fail.

## Data

The Yelp data are **not** included in this repository. Export six CSVs using the
accompanying SQL queries and place them in `data/raw/`:

| File | Contents |
|---|---|
| `restaurant_business_raw.csv` | businesses filtered to the restaurant category |
| `restaurant_reviews_raw.csv` | reviews of those businesses |
| `restaurant_users_raw.csv` | users who wrote them |
| `restaurant_checkin_raw.csv` | check-ins |
| `restaurant_tip_raw.csv` | tips |
| `restaurant_photo_raw.csv` | photos |

**Format matters.** Each CSV must have **exactly one column**, and every row must
hold **one JSON object as text** — the shape produced by `SELECT row_to_json(t)
FROM ... t` exported from pgAdmin. The loader raises immediately if a file has
more than one column. The supplied SQL queries produce that shape; do not rewrite
them as plain column selects.

### Where the data directory is found

`main.py` locates the data automatically, in this order — nothing needs editing
for the normal case:

1. the `DSMA_DATA_DIR` environment variable, if set;
2. `./data` next to `main.py` — **the normal case: put the CSVs here and run**;
3. `../data` beside the repository (the author's own layout);
4. `./data`, created on demand.

To keep the data elsewhere, set the environment variable before running:

```bash
# macOS / Linux
export DSMA_DATA_DIR=/path/to/data
# Windows PowerShell
$env:DSMA_DATA_DIR = "D:\path\to\data"
```

## Internet access

The first run downloads daily weather summaries from NOAA (**no API key
required**), so it needs internet access. The download is then **cached to disk**
and reused on every later run — subsequent runs rebuild everything else from
scratch but never re-fetch the weather, and need no internet. If any city fails
to download, the pipeline **stops with an error** rather than continuing with
missing weather.

## Outputs

Everything is written under `<data directory>/processed/`; **nothing is written
into the repository itself**:

- `model_outputs/` — evaluation tables, the hyperparameter-tuning table, the
  feature-group ablation, the top-k feature curve, the learning curve, the
  precision–recall curve, and the logistic-regression coefficients, all as CSV.
- `model_outputs/plots/` — the same tables rendered at 300 dpi, plus the
  confusion matrix, permutation-importance, feature-subset, learning-curve, and
  precision–recall figures.
- `eda_outputs/plots/` — the exploratory-analysis figures.

## Reproducibility notes

- Every split, subsample, and estimator is seeded (`RANDOM_STATE = 42`).
  Rebuilding the dataset from raw reproduces the same 2,897,032 rows and the same
  model results.
- NOAA occasionally revises historical daily summaries, so a download performed
  much later may differ marginally from an earlier one, and model scores may
  shift in the third decimal. A partial download raises rather than silently
  imputing the gap.
- The model comparison trains all nine models on an identical stratified 100k
  subsample. This is deliberate: k-nearest neighbours, the linear SVM, and the
  neural network cannot be fitted on millions of rows, and training different
  models on different amounts of data would confound the algorithm with the data
  volume.

## Repository layout

```
main.py             the entire pipeline: ingestion -> EDA -> modelling -> results
zenvironment.yaml   conda environment (name: dsma)
README.md           this file
AGENTS.md           implementation contract and design decisions
REQUIREMENTS.md     the academic task specification this project answers
data/               not tracked; see "Data" above
```
