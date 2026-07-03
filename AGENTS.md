# AGENTS.md

Instructions for AI coding agents (Codex, etc.) working in this repository.
Read this before making changes.

## Project goal

This is a university **Data Science & Marketing Analytics** project. It is a
**supervised binary classification** study that predicts **customer
satisfaction** from Yelp restaurant reviews, using **structured features only**
(no review text). The target is `satisfied` = 1 if `review_stars >= 4`, else 0.

The research question is *which structured factors (business, user, contextual
and weather characteristics) predict satisfaction* — this is deliberately NOT a
sentiment-from-text task. Keep this framing when changing anything.

## Single-file constraint

The entire pipeline lives in **one Python file** (`main.py`) on purpose: it is
submitted as an appendix to the written report. **Do not split it into a
package or multiple modules.** Keep everything in the single file.

## How to run

- Requires Python 3 with: pandas, numpy, scikit-learn (>= 1.8), matplotlib,
  requests.
- Entry point: run `python main.py`.
- Two toggles at the top of `main()` control behaviour:
  - `REBUILD` — rebuild the dataset from raw data instead of loading the cache.
  - `RUN_MODELS` — run the modelling stage (EDA always runs; models are gated).
- The pipeline runs in memory end-to-end. Only two artefacts are cached to disk
  as pickles: the raw EDA dataset and the NOAA weather cache. Model results are
  written as CSVs to the output directory.

## Verification (do this after every change)

There is no automated test suite, and the real data (local CSVs + NOAA network)
is not available in the agent environment. After editing `main.py`, always:

1. `python -m py_compile main.py`
2. `python -m pyflakes main.py`
3. Where feasible, add a small synthetic smoke test in a temp dir to exercise
   the changed function (the pipeline cannot be run end-to-end without the
   user's private data).

Do not claim the full pipeline was run on real data — it cannot be here.

## Pipeline structure (keep this order)

1. Load raw data -> 2. Process/merge/clean -> 2b. NOAA weather enrichment ->
3. EDA (on RAW, untransformed data) -> 4. Build modelling dataset
(EDA-informed transition) -> 5. Modelling -> 6. Results.

Key ordering rules that must be preserved:
- **EDA runs on raw data.** The log transform happens AFTER EDA.
- The EDA -> modelling transition (`build_model_dataset`) does, in order:
  drop identifiers -> drop redundant (multicollinearity/VIF) features ->
  skewness comparison on survivors -> log-and-replace the skewed counts.
- Low-signal features (e.g. `weather_available`) are dropped BEFORE EDA so they
  do not appear in the EDA tables.
- Modelling order: cross-validation and grid search (train only) run BEFORE the
  held-out test evaluation. **The test set is touched last** — never use it for
  tuning or model selection.

## Modelling conventions

- Ten models compared on a stratified 100k subsample (80k train / 20k test):
  majority-class baseline, logistic regression, linear SVM (LinearSVC), KNN,
  Gaussian Naive Bayes, MLP neural net, decision tree, bagging, random forest,
  and HistGradientBoosting (this is the "boosting" model — it is **gradient
  boosting**, not AdaBoost).
- All models are leakage-safe scikit-learn `Pipeline`s. Imputation and encoding
  happen inside the pipeline so they are fit on training folds only.
- Imbalance (~70/30) is handled with stratified splits + `class_weight=
  "balanced"` where supported + balanced metrics. **Do not add SMOTE / resampling.**
- Tree ensembles are regularised (depth + min_samples_leaf) on purpose to curb
  overfitting — do not remove these constraints.
- Metrics include: accuracy, balanced accuracy, precision/recall/F1 for BOTH
  classes, ROC-AUC, PR-AUC, Gini (= 2*AUC - 1), and top-decile lift (including
  for the dissatisfied minority).

## Leakage rules (critical — do not violate)

- **Never use `review_text`** as a feature (written at the same time as the
  rating -> leakage). Only `review_text_length` is kept, as a proxy.
- **Never use the review vote counts** (`review_useful/funny/cool`) as features
  (they accrue after posting -> look-ahead leakage).
- Do not use `review_stars`, `business_stars`, or `user_average_stars` as model
  inputs (the target derives from the rating).
- Imputation must be fit on training data only, never on the full dataset.

## Feature decisions (already made — keep unless asked)

- Multicollinearity/VIF drops: photo sub-types (keep `photo_count`, their exact
  sum), `weather_tmin` (temp_range = tmax - tmin), `user_funny`/`user_cool`
  (keep `user_useful`), `checkin_count`/`tip_count` (keep `business_review_count`).
- Missing values: counts -> 0 and categoricals -> "Unknown" during cleaning
  (known-meaning fills); weather/user NaNs -> median imputation inside the
  pipeline.
- Final model feature set is ~37 features (20 numeric + 17 categorical).

## Style

- Keep the code readable and well-commented; this file is read by graders.
- Prefer minimal, targeted diffs. Do not refactor broadly without being asked.
- Preserve the existing section headers and the feature-change reporting prints.
- Explain WHY in comments, not just what.

## Central finding (context for any modelling changes)

Performance is bounded by a **feature ceiling**, not by model choice, tuning, or
data volume: nine diverse models cluster at test ROC-AUC ~0.61-0.70, with small
train-test gaps after regularisation. HistGradientBoosting is best (~0.70).
More data or training will not raise this ceiling; only richer features would.
Do not "fix" the modest scores by adding leakage or resampling.
