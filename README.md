# Predicting Restaurant Customer Satisfaction from Yelp Data

Supervised binary classification of Yelp restaurant reviews: is a review
**satisfied** (`review_stars >= 4`) or not, predicted from **structured features
only** — business, user, engagement, contextual, and weather characteristics.
Review text is never used as a model input.

The entire pipeline is one file, `main.py`, submitted as the report appendix.

---

## 1. Requirements

```bash
conda env create -f zenvironment.yaml
conda activate dsma
```

Python 3.11, with `pandas`, `numpy`, `matplotlib`, `scikit-learn >= 1.4`, and
`requests`. `colorspace` and `wordcloud` are optional — they only affect EDA
figures, and the pipeline falls back to matplotlib palettes and skips the word
cloud if they are absent.

`scikit-learn >= 1.4` is a hard requirement: earlier versions lack
`HistGradientBoostingClassifier(class_weight=...)` and
`KBinsDiscretizer(encode="onehot-dense")`.

## 2. Data

The Yelp data are **not** in this repository. Export six CSVs from the database
using the accompanying SQL queries and place them in `data/raw/`:

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
FROM ... t` exported from pgAdmin. `load_json_csv` raises immediately if a file
has more than one column. The SQL queries supplied with this project produce that
shape; do not rewrite them as plain column selects.

### Where the data directory is found

`main.py` resolves it in this order, so nothing needs editing:

1. the `DSMA_DATA_DIR` environment variable, if set;
2. `./data` next to `main.py` — the normal case: unzip the CSVs and go;
3. `../data` beside the repository (the author's layout);
4. `./data`, created on demand.

To keep the data elsewhere:

```bash
# macOS / Linux
export DSMA_DATA_DIR=/path/to/data
# Windows PowerShell
$env:DSMA_DATA_DIR = "D:\path\to\data"
```

## 3. Running it

Open `main.py` and set the three stage toggles near the top:

```python
RUN_INGESTION = True    # first run: build the dataset from the raw CSVs
RUN_EDA       = True
RUN_MODELING  = True
```

Then:

```bash
python main.py
```

A first, complete run takes roughly **25 minutes** and needs several GB of RAM.
It downloads daily weather summaries from NOAA (no API key required), so it needs
internet access. If any city fails to download, the pipeline **stops with an
error** rather than continuing with missing weather — see §5.

### Faster re-runs

Ingestion caches the enriched dataset. Afterwards, set `RUN_INGESTION = False`
and the other stages read the cache directly:

| Goal | Toggles |
|---|---|
| Iterate on the EDA | `False, True, False` |
| Iterate on the modelling | `False, False, True` |
| Rebuild everything from raw | `True, True, True` |

With `RUN_INGESTION = False` and no cache present, `main()` raises a clear error
telling you to turn it on once. The NOAA download is cached separately and is
never repeated.

## 4. Outputs

Everything is written under `<DATA_DIR>/processed/`:

- `model_outputs/` — evaluation tables, the hyperparameter tuning table, the
  feature-group ablation, the top-k feature curve, the learning curve, and the
  logistic-regression coefficients, all as CSV.
- `model_outputs/plots/` — the same tables rendered at 300 dpi, plus the
  confusion matrix, permutation importance, feature-subset curve, and learning
  curve figures.
- `eda_outputs/plots/` — EDA figures.

Nothing is written into the repository itself.

## 5. Reproducibility notes

- Every split, subsample, and estimator is seeded (`RANDOM_STATE = 42`). Rebuilding
  the dataset from raw reproduces the same 2,897,032 rows and the same model
  results.
- **Weather is downloaded live.** NOAA occasionally revises historical daily
  summaries, so a download performed much later may differ marginally from the one
  cached here, and model scores may shift in the third decimal. A partial download
  now raises rather than silently median-imputing the gap.
- The comparison trains all nine models on an identical stratified 100k subsample.
  This is deliberate: k-nearest neighbours, the linear SVM, and the neural network
  cannot be fitted on millions of rows, and training different models on different
  amounts of data would confound the algorithm with the data volume.

## 6. Repository layout

```
main.py             the entire pipeline: ingestion -> EDA -> modelling -> results
zenvironment.yaml   conda environment
AGENTS.md           implementation contract and design decisions
REQUIREMENTS.md     the academic task specification this project answers
data/               not tracked; see §2
```
