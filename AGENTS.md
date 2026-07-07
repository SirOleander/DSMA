# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code, etc.) working in this repository.

Read this file before making any changes. Also read `REQUIREMENTS.md` before auditing academic completeness or changing the analytical scope.

## Relationship between files

This repository uses two separate instruction documents:

- `AGENTS.md` = repository and implementation instructions for AI coding agents.
- `REQUIREMENTS.md` = academic task overview and coverage rubric for the term paper.

When reviewing or editing the code, use both:

1. Use `REQUIREMENTS.md` to check whether the project covers the lecturer's task.
2. Use this `AGENTS.md` file to understand how the implementation is intentionally structured and what must not be changed without explicit permission.

If these two documents appear to conflict, do not silently rewrite the project. Report the conflict clearly and ask the user before making broad changes.

## Project goal

This is a university **Data Science & Marketing Analytics** project. It is a
**supervised binary classification** study that predicts **customer
satisfaction** from Yelp restaurant reviews, using **structured features only**
(no review text as a model input).

The target is:

```text
satisfied = 1 if review_stars >= 4
satisfied = 0 otherwise
```

The research question is which structured factors — business, user, contextual,
engagement, and weather characteristics — predict satisfaction.

This is deliberately **not** a sentiment-from-review-text task. Keep this framing when changing anything.

## Academic coverage expectation

Before judging whether the project is complete, compare the implementation against `REQUIREMENTS.md`.

At minimum, check whether the code and outputs address:

- restaurant-category filtering
- clear unit of analysis
- careful joins across Yelp tables with different observation levels
- KPI and target construction
- data cleaning and feature engineering
- EDA on the appropriate dataset
- modelling dataset construction
- multiple model comparison
- suitable evaluation metrics
- leakage prevention
- class imbalance handling
- model interpretation
- managerial usefulness for restaurant owners or investors
- documented limitations

Do not assume that strong code alone means the term paper is complete. The project must also support the written analysis.

## Single-file constraint

The entire pipeline lives in **one Python file** (`main.py`) on purpose: it is
submitted as an appendix to the written report. **Do not split it into a
package or multiple modules.** Keep everything in the single file unless the user explicitly asks otherwise.

## How to run

- Requires Python 3 with: pandas, numpy, scikit-learn, matplotlib, requests, colorspace.
  - `colorspace` is a hard import (used for the diagnostic plot palettes).
  - `wordcloud` is optional: it is imported lazily and skipped if not installed.
- Entry point: run `python main.py`.
- The project is one linear pipeline, gated by three stage toggles at the top of
  the file (not inside `main()`):
  - `RUN_INGESTION` — parse the raw CSVs, clean, merge, and enrich with NOAA
    weather, rebuilding the cached enriched dataset from raw. Turn OFF to skip
    ingestion and load the cached dataset instead (fast reruns).
  - `RUN_EDA` — run EDA on the raw, untransformed enriched dataset.
  - `RUN_MODELING` — run the EDA-informed transition, model training and
    comparison, and exported results.
  - Each stage hands off through the cached enriched dataset, so EDA or modelling
    can run straight from the cache. If `RUN_INGESTION` is off and no cache
    exists, `main()` raises a clear error telling you to turn it on once.
  - Note: ingestion rebuilds the dataset from raw, but the NOAA weather download
    stays cached, so it is not re-downloaded.
- The pipeline runs in memory end-to-end.
- Only two artefacts are cached to disk as pickles:
  - the weather-enriched dataset (the single dataset feeding EDA and modelling)
  - the NOAA weather cache
- Model results are written as CSVs to the output directory.

## Verification after every code change

There is no automated test suite, and the real data — local CSVs plus NOAA network access — may not be available in the agent environment.

After editing `main.py`, always run:

```bash
python -m py_compile main.py
python -m pyflakes main.py
```

Where feasible, add a small synthetic smoke test in a temporary directory to exercise the changed function.

The pipeline cannot necessarily be run end-to-end without the user's private data. Do **not** claim the full pipeline was run on real data unless it actually was.

## Pipeline structure

Keep this order:

1. Load raw data
2. Process, merge, and clean
3. NOAA weather enrichment
4. EDA on raw, untransformed data
5. Build modelling dataset through the EDA-informed transition
6. Modelling
7. Results and exported artefacts

## Key ordering rules

The following ordering rules must be preserved:

- **EDA runs on raw data.**
- The log transform happens **after** EDA.
- The raw-vs-log skewness comparison is produced **in the EDA** as skewness
  evidence (`run_skewness_comparison`), on the features that survive screening.
  The actual log-and-replace transform still happens later, at the modelling
  stage, so the transform itself stays after EDA.
- The EDA feature screening removes excluded variables cleanly: leakage/ID/date/
  post-review columns and then the correlation/VIF casualties are physically
  dropped from a single working frame, so the surviving frame is exactly the
  reduced EDA feature set the summaries describe.
- The EDA-to-modelling transition happens in `build_model_dataset`.
- That transition should follow this order:
  1. drop identifiers
  2. drop redundant multicollinearity/VIF features
  3. log-and-replace the skewed count variables
- Low-signal features such as `weather_available` are dropped before EDA so they do not appear in the EDA tables.
- Cross-validation and grid search on the training set happen before held-out test evaluation.
- The test set is touched last. Never use the test set for tuning or model selection.

## Modelling conventions

The project compares ten models on a stratified 100k subsample, using an 80k train / 20k test split:

1. majority-class baseline
2. logistic regression
3. linear SVM / LinearSVC
4. KNN
5. Gaussian Naive Bayes
6. MLP neural network
7. decision tree
8. bagging
9. random forest
10. HistGradientBoosting

HistGradientBoosting is the boosting model. It is **gradient boosting**, not AdaBoost.

All models should be leakage-safe scikit-learn `Pipeline`s. Imputation and encoding must happen inside the pipeline so they are fit on training folds only.

Imbalance is expected to be around 70/30 and is handled with:

- stratified splits
- `class_weight="balanced"` where supported
- balanced metrics

Do **not** add SMOTE or other resampling unless explicitly requested.

Tree ensembles are regularised on purpose to curb overfitting. Do not remove depth or `min_samples_leaf` constraints without a clear reason.

## Required metrics

Model results should include:

- accuracy
- balanced accuracy
- precision for both classes
- recall for both classes
- F1 for both classes
- ROC-AUC
- PR-AUC
- Gini coefficient, defined as `2 * AUC - 1`
- top-decile lift
- top-decile lift for the dissatisfied minority, where applicable

Accuracy alone is not enough because the target may be imbalanced.

## Leakage rules

These rules are critical. Do not violate them.

- Never use `review_text` as a model feature. It is written at the same time as the rating and would create leakage.
- Only `review_text_length` may be kept as a structured proxy.
- Never use review vote counts such as `review_useful`, `review_funny`, or `review_cool` as features. They accrue after posting and create look-ahead leakage.
- Do not use `review_stars`, `business_stars`, or `user_average_stars` as model inputs because the target is derived from ratings.
- Imputation must be fit on training data only, never on the full dataset.
- Do not use future information to predict past reviews. Where feasible, features should reflect information available before or at the review date.
- If a feature cannot be made fully time-safe, document the limitation rather than hiding it.

## Feature decisions already made

Keep these decisions unless the user explicitly asks to revisit them:

- Drop photo sub-types during multicollinearity/VIF pruning and keep `photo_count`, because the sub-types sum exactly to the total.
- Drop `weather_tmin`; keep temperature range based on max/min where used.
- Drop `user_funny` and `user_cool`; keep `user_useful`.
- Drop `checkin_count` and `tip_count`; keep `business_review_count`.
- Fill known count missings with 0 during cleaning.
- Fill categorical missing values with `"Unknown"` during cleaning.
- Use median imputation inside the modelling pipeline for weather/user numerical NaNs.
- Final model feature set is expected to contain around 37 features, roughly 20 numeric and 17 categorical.

## EDA requirements

EDA should be run on the raw, untransformed modelling-relevant dataset, before log transforms.

The EDA should support the written term paper by producing interpretable summaries such as:

- satisfaction distribution
- class balance
- rating-related summaries
- distributions of key business, user, contextual, weather, and engagement variables
- comparisons between satisfied and dissatisfied reviews
- checks that support later feature engineering decisions
- evidence for skewness, missingness, and dropped features

Do not move EDA after modelling transformations unless explicitly asked.

## Interpretation and managerial usefulness

The code should produce outputs that support interpretation in the written report.

Useful outputs include:

- model comparison tables
- confusion matrix information
- feature importance or coefficient tables
- performance metrics for both satisfied and dissatisfied classes
- evidence for the feature ceiling
- concise artefacts that can be translated into recommendations for restaurant owners or investors

Avoid purely technical changes that make the project harder to explain in the written paper.

## Style

- Keep the code readable and well-commented.
- This file is read by graders, so clarity matters.
- Prefer minimal, targeted diffs.
- Do not refactor broadly without being asked.
- Preserve existing section headers and feature-change reporting prints where possible.
- Explain why important decisions are made, not only what the code does.

## Central finding

The current interpretation is that performance is bounded by a **feature ceiling**, not mainly by model choice, tuning, or data volume.

Diverse models cluster at modest test ROC-AUC levels, with small train-test gaps after regularisation. HistGradientBoosting is currently the best-performing model.

Do not try to “fix” modest performance by adding leakage, review text, target-derived variables, or resampling. Better performance would require genuinely richer non-leaky features.

## Expected agent behaviour

When asked to review the project, first produce a coverage audit rather than immediately rewriting code.

A good audit should include:

- what is already covered
- what is partially covered
- what is missing
- what is risky
- what should be fixed first
- which changes are code changes versus report-writing changes
- whether any requested change would violate leakage or project-framing rules

Only after the audit should you make targeted edits requested by the user.
