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

- Requires Python 3.11 with pandas, numpy, matplotlib, requests, and **scikit-learn >= 1.4** (earlier versions lack `HistGradientBoostingClassifier(class_weight=...)` and `KBinsDiscretizer(encode="onehot-dense")`). See `zenvironment.yaml`; see `README.md` for the end-user runbook.
  - The full modelling stage searches 187 hyperparameter configurations (561 fits at 3-fold). A complete `RUN_INGESTION + RUN_EDA + RUN_MODELING` run takes ~25 minutes. Set `DO_GRID_SEARCH = False` for fast reruns.
  - `colorspace` is **optional**: it supplies the EDA palettes and `qualitative_palette` falls back to matplotlib colormaps when it is absent. A missing plotting dependency must never stop the modelling pipeline — do not restore the hard import.
  - `wordcloud` is optional: it is imported lazily and skipped if not installed.
- **`DATA_DIR` is not hard-coded.** `_resolve_data_dir` checks `$DSMA_DATA_DIR`, then `./data` beside `main.py`, then `../data` beside the repo, accepting a candidate only if it already contains `raw/`. Do not replace this with an absolute path: the file is submitted as a report appendix and must run on a grader's machine.
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
- Model results are written as CSVs to the output directory:
  `hyperparameter_search.csv` (winning parameters and best CV ROC-AUC per model),
  `model_comparison_cv.csv`, `model_comparison_train.csv`, `model_comparison_holdout.csv`.

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
- Inside the modelling stage the order is **tune → cross-validate → test**:
  1. `run_grid_search` picks hyperparameters using inner CV folds on the training set.
  2. `apply_best_params` writes them onto the pipelines.
  3. `run_cross_validation` then reports fold-to-fold stability of the **tuned** models. Running it before tuning would describe models that are never reported.
  4. Only then is the held-out test set opened.
- Cross-validation and grid search on the training set happen before held-out test evaluation.
- The test set is touched last. Never use the test set for tuning or model selection.

## Modelling conventions

The project compares nine models on a stratified 100k subsample, using an 80k train / 20k test split:

1. logistic regression
2. linear SVM / LinearSVC
3. KNN
4. Bernoulli Naive Bayes over quantile-binned features
5. MLP neural network
6. decision tree
7. bagging
8. random forest
9. HistGradientBoosting

There is deliberately **no** `DummyClassifier` baseline row (removed at the user's request). Because of that, `models_main` prints an explicit no-information reference line — the accuracy an always-predict-satisfied rule achieves, equal to the positive rate — before the comparison tables. Keep that line: under a ~70/30 prior the accuracy and F1 columns are uninterpretable without it, and it is the class-imbalance check `REQUIREMENTS.md` §14 requires. Model selection uses ROC-AUC, which is insensitive to the class prior.

HistGradientBoosting is the boosting model. It is **gradient boosting**, not AdaBoost.

The SVM is linear on purpose. A true kernel SVM (`SVC(kernel="rbf")`) is **not** feasible here: libsvm training is between O(n²) and O(n³) in rows, so it is out of reach at 80k rows before any grid search is applied. `REQUIREMENTS.md` §12 permits an SVM only "if computationally feasible", and the nonlinearity a kernel would buy is already covered by the tree ensembles. Do not add `SVC`, and do not add a kernel approximation (`Nystroem` / `RBFSampler`) unless the user explicitly asks: this was considered and deliberately declined.

**Naive Bayes is not Gaussian, on purpose.** `GaussianNB` fits a mean and variance to every column, including one-hot city indicators and binary attribute flags that take only the values 0 and 1, plus zero-inflated counts. That likelihood is badly misspecified and it was the weakest model in the comparison. `build_nb_preprocessor` instead cuts continuous numerics into **quantile bins**, passes 0/1 flags straight through, one-hot encodes the categoricals, and feeds `BernoulliNB`. Nothing assumes a parametric density — the scikit-learn equivalent of R's `usekernel = TRUE`. It also gives Naive Bayes two real hyperparameters (`n_bins`, and `alpha`, which is R's `fL` Laplace correction) where `GaussianNB` had only the cosmetic `var_smoothing`. Do not revert it to `GaussianNB`.

Many continuous features are zero-inflated (`weather_snow` 76% zeros, `log_tip_compliment_count` 60%, `log_user_fans` 48%) or near-discrete (`hours_open_days_count` has 8 distinct values), so they cannot be cut into 20 distinct quantiles. scikit-learn drops the degenerate zero-width bins; a module-level `warnings.filterwarnings` silences the resulting flood (525 per grid search) without changing the behaviour. This also means `n_bins` hitting its grid maximum does **not** imply more bins would help — the affected features already receive fewer.

Note that quantile binning is invariant to monotone transforms, so this model is unaffected by the upstream log transform. The log transform stays regardless: it exists for the linear and distance-based models (logit, SVM, KNN, MLP). The four tree models are monotone-invariant and do not need it either.

**KNN uses `weights="uniform"`, fixed, never tuned.** With `weights="distance"` a training point is its own nearest neighbour at distance 0, so it receives infinite weight and the model reproduces the training labels exactly — training ROC-AUC = 1.000 on every metric. That is a definitional artefact, not overfitting, and it makes the training row uninterpretable. Measured cost of pinning uniform: **0.0175 GINI on the test set** (0.2721 → 0.2546), on the 7th-best of nine models. Do not add `weights` back to `PARAM_GRIDS`.

All models should be leakage-safe scikit-learn `Pipeline`s. Imputation and encoding must happen inside the pipeline so they are fit on training folds only.

Imbalance is expected to be around 70/30 and is handled with:

- stratified splits
- `class_weight="balanced"` where supported
- balanced metrics

Do **not** add SMOTE or other resampling unless explicitly requested.

Tree ensembles are regularised on purpose to curb overfitting. Do not remove depth or `min_samples_leaf` constraints without a clear reason.

## Hyperparameter tuning

Every model has a grid in `PARAM_GRIDS`. `run_grid_search` searches them with `GridSearchCV`, scored by ROC-AUC, on the training set only, and **returns** the winning parameters. `apply_best_params` then writes them onto the pipelines before anything is fit.

That return path matters. Earlier the search printed `best_params_` and discarded it, so the reported models were the untuned defaults.

Rules to preserve:

- The search runs on a subsample (`GRID_SAMPLE`), so use `best_params_`, **not** `search.best_estimator_` — that estimator saw only the subsample. The tuned pipelines are re-fit on the full training set.
- The inner `StratifiedKFold` folds of `GridSearchCV` *are* the validation step. There is no separate validation split, and adding one would be redundant.
- With `DO_GRID_SEARCH = False`, models fall back to the documented default hyperparameters hard-coded in `build_models`. Keep those defaults meaningful.
- `class_weight="balanced"` is the documented imbalance strategy, not a hyperparameter. Do not add it to a grid.
- Do not tune `n_estimators` for the random forest or bagging: more trees is monotonically non-worse in expectation, so searching it only buys compute.
- Tree grids must keep `max_depth` and `min_samples_leaf` bounded. Never offer `max_depth=None` — the unconstrained ensembles reached train ROC-AUC = 1.0.
- **`print_hyperparameter_table` prints the methods-section table** at the end of `run_grid_search` (also saved as `hyperparameter_tuning_table.csv`): one row per hyperparameter, giving its search space, the winning value, and a grid-edge flag. It also lists the **fixed** hyperparameters from `FIXED_HYPERPARAMETERS` — "we did not tune this, and here is the value" is part of the model specification. Keep `FIXED_HYPERPARAMETERS` in step with `build_models`; if you change a fixed value in one, change it in the other.
- **Every grid must bracket its winner.** `run_grid_search` calls `find_boundary_hits` and prints a warning whenever a tuned value lands on the minimum or maximum of a numeric axis. A boundary winner means the search was cut off and the "optimum" is partly an artefact of the range. Widen that axis and re-run; do not report boundary-hit values. The grids were already widened once for exactly this reason.
- `HistGradientBoostingClassifier` sets `early_stopping=True` explicitly. The sklearn default `'auto'` silently enables it whenever n > 10,000, which is always true here, making `max_iter` a mere upper bound. Do not tune `max_iter`; tune learning rate, leaf count, and L2 instead.

Some parameters are genuinely not tunable and should not be added: `svm_linear` keeps `dual=False`, which forces `loss="squared_hinge"`; `logistic_regression` uses lbfgs, which supports L2 only (L1 needs `saga`, too slow on this one-hot matrix); `naive_bayes` has one knob.

If the search becomes too slow, swap `GridSearchCV` for `RandomizedSearchCV` rather than silently shrinking the grids.

## Required metrics

Model results should include:

- balanced accuracy
- precision for both classes
- recall for both classes (recall on the dissatisfied class is reported as `Specificity`)
- F1 for both classes
- PR-AUC for the dissatisfied minority (`pr_auc_dissat`)
- Gini coefficient, `2 * AUC - 1` — this is how ROC-AUC is reported
- top-decile lift
- top-decile lift for the dissatisfied minority

Deliberately **absent**: plain accuracy, ROC-AUC as a reported column, and satisfied-class PR-AUC. See the metric rules below.

The three report tables do **not** carry every required metric: `precision_dissat`, `f1_dissat`, `f1_macro`, and `top_decile_lift_dissat` live only in the full `model_comparison_*.csv` files. Do not delete those files — they are the only place the per-class dissatisfied metrics exist.

### Reported evaluation tables

`models_main` writes three tables in the report layout (`to_report_table`), each as a CSV *and* a conditional-formatted PNG in `PLOT_DIR`: `eval_untuned_train`, `eval_tuned_train`, `eval_tuned_test`. Columns are **Balanced Acc., PR-AUC, GINI, TDL, Precision, Recall, F1, Specificity, Time (s)**.

The metric choices are deliberate and imbalance-driven. Do not "simplify" them back:

- **Balanced accuracy replaces plain accuracy.** Accuracy is corrupted by the 70/30 prior. It is still computed in the full CSVs (the rubric asks for it, and the no-information reference needs it) but never appears in a report table and never ranks anything.
- **`PR-AUC` is `pr_auc_dissat`, the DISSATISFIED-class average precision.** Average precision is *not* prior-invariant: its no-skill floor equals the rate of the class being scored. Satisfied-class PR-AUC therefore has a 0.697 floor and looks impressive on random scores; the dissatisfied version has a 0.303 floor and is the informative one. The satisfied-class `pr_auc` is **not computed at all**.
- **`SELECTION_METRIC` = `gini`** and is the single source of truth for ranking, CV sorting, best-model choice, and top-3 permutation importance. Gini = `2 * roc_auc - 1` inherits ROC-AUC's invariance to the class prior (no-skill = 0 at any imbalance), so it cannot be gamed by predicting the majority class and stays comparable across subsamples. A common error is to swap it for PR-AUC "because of imbalance" — that reasoning applies to accuracy, not to ROC-AUC/Gini.
- **ROC-AUC is never reported, but cannot be deleted.** `gini` is computed *from* it, and scikit-learn has no `"gini"` scorer string. `GRID_SCORING = "roc_auc"` is therefore the internal implementation of `SELECTION_METRIC` inside `GridSearchCV`, `cross_validate`, and `permutation_importance` — legitimate because gini is a strictly increasing function of ROC-AUC, so maximising one maximises the other and the argsort is identical. It never leaves the module as a reported number.
- **Plain accuracy is not computed.** `compute_metrics` no longer returns it, and `accuracy` was removed from the CV scoring list. With a 69.71% positive rate it is uninterpretable and it inverts the true ranking. `models_main` prints a no-information reference line instead, which is the imbalance check `REQUIREMENTS.md` §14 asks for. Do not re-add `accuracy_score`.
- **`GINI` carries the ROC-AUC information; ROC-AUC is not a table column.** `gini = 2 * roc_auc - 1` is a linear rescaling, so the two are perfectly collinear and only one belongs in a table. Gini is the conventional headline in marketing analytics and is the one shown. ROC-AUC stays in the full CSVs, remains `SELECTION_METRIC`, and is recoverable exactly as `roc_auc = (gini + 1) / 2`. `REQUIREMENTS.md` §14 asks for ROC-AUC, so the paper must state that relation once. Do not re-add a ROC-AUC column beside GINI.
- `Specificity` is recall on the dissatisfied class — the same quantity as `recall_dissat`, renamed to the reporting convention.
- `Time (s)` is training fit **seconds** for the whole pipeline. **Lower is better**; `REPORT_LOWER_IS_BETTER` makes the colour scale invert for it. It flatters KNN, which barely fits and pays its cost at prediction time.
- Rows use the fixed `MODEL_DISPLAY_ORDER`, not score order, so a model can be compared across the three tables along one row. Do not re-sort them by score.
- The PNGs use a teal↔amber diverging scale, **not** red↔green: under simulated deuteranopia red/green separate by only ΔE 10.4 (below the ΔE ≥ 12 floor) while teal/amber hold ΔE 62.5. Colour is redundant — every value is printed in its cell.

`save_report_table_png` accepts a `degenerate_rows` argument that greys a row out and excludes it from the colour scale. It is currently unused: KNN's `weights="uniform"` pin removed the only degenerate row. Keep the argument for any future model whose in-sample score is definitional rather than earned.

**Tuning lowers training-set scores for the tree ensembles** (HGB train AUC fell 0.768 → 0.724). That is the regularisation working, not a regression. Tuned-vs-untuned is only meaningful on held-out data; comparing them on the training set rewards the *less* regularised model.

**Never rank models by accuracy in this project.** The positive rate is 69.71%, so an always-predict-satisfied rule scores 0.6971 accuracy. Models that carry `class_weight="balanced"` deliberately trade accuracy for recall on the dissatisfied minority and therefore score *below* that line — the best model by ROC-AUC among them had 0.6368 accuracy. Models with no `class_weight` (KNN, Naive Bayes, the MLP) score above it by mostly predicting the majority class, with recall on dissatisfied reviews around 0.11–0.34. Accuracy is not comparable between the two groups, and ranking by it inverts the true order. Rank by ROC-AUC (insensitive to the class prior) and report balanced accuracy alongside it.

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
- Final model feature set contains **97 predictors: 19 continuous, 59 binary indicators, 19 categorical.**
- **`DROP_FOR_REDUNDANCY` must agree with what the EDA's correlation/VIF screening actually removes.** They drifted apart once: the EDA screened out 5 `hours_*` numerics and 4 binaries that the models were still being trained on, so the EDA reported a reduced feature set the models never used. The list now carries both provenances, labelled: structural drops justified by an exact algebraic identity (`photo_count` is the exact sum of its five sub-types; `weather_temp_range = tmax - tmin`), and the EDA's own VIF casualties. If the screening changes, update the list. The structural drops are *stronger* than VIF: the iterative loop stops once two photo sub-types are gone, even though the identity still holds.
- The EDA reports 24 continuous survivors against modelling's 19. The difference is fully accounted for: the EDA counts the three photo sub-types that the structural rule removes, and treats `review_month`/`review_weekday` as numeric where modelling encodes them as categorical (19 + 3 + 2 = 24). Say so in the paper rather than letting the two numbers sit unexplained.

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

## NOAA weather download

`download_weather_for_top_cities` **raises** if any city returns no data. It previously caught the per-city exception, printed a line, and continued: the affected reviews got `NaN` weather, which the modelling pipelines median-imputed without complaint. A transient NOAA outage therefore produced a quietly different dataset and quietly different results, with no error anywhere. Do not soften this back into a warning — a rerun that cannot reproduce the cached numbers is worse than a rerun that fails.

The download is cached (`WEATHER_CACHE_PKL`) and is never repeated once written. Note that NOAA occasionally revises historical daily summaries, so a fresh download much later may differ marginally from the cache; say so in the paper's limitations.

## Business attribute levels

Yelp stores the nested business attributes as **Python-2 repr strings**, so the same level arrives under two spellings: `u'quiet'` and `'quiet'`. Left untouched they become two separate one-hot columns, splitting one effect across two coefficients. `normalize_attribute_levels` strips the prefix and quotes; it collapsed 13 duplicate levels across `Alcohol`, `WiFi`, `NoiseLevel`, and `RestaurantsAttire`.

It is **idempotent** and called twice on purpose: inside `process_data` so newly built caches are clean, and in `main()` after loading a cached dataset so caches written before the fix are repaired on load. Do not remove either call.

## Learning curve

`run_learning_curve` fits HistGradientBoosting and logistic regression on training sets of increasing size drawn from the **full ~2.9M-row dataset**, scoring each on one fixed 250k held-out set. It exists to test the data-volume half of the feature-ceiling claim, which was previously asserted rather than measured.

- Only these two models are used: they scale linearly in rows. KNN, the linear SVM, and the MLP cannot be fitted at these sizes — the same reason the model comparison is capped at a 100k subsample.
- It uses **its own stratified split** (`RANDOM_STATE + 1`) of the full data. It is a separate experiment, and its Gini values are **not** directly comparable with the model-comparison tables. Say so when reporting it.
- The test set is held out once and reused at every training size, so the curve is not confounded by a moving evaluation target.

## Feature-group ablation

`run_feature_group_ablation` removes one block of features at a time, refits the CV-best model, and measures the Gini it costs, with the same paired bootstrap the top-k curve uses. This is what answers `REQUIREMENTS.md` §17 subquestions 2–5 — whether engagement, user, timing, and weather features add predictive value. Permutation importance cannot answer them: features inside a correlated block mask one another, so each looks dispensable while the block may not be.

- `assert_all_features_grouped` raises if any modelling column belongs to no block. An unassigned feature would never be ablated, and the omission would be silent. Extend `_group_membership` rather than letting it slide.
- Results are **marginal** contributions: two blocks encoding the same information can each look dispensable while removing both would not be. State that caveat when reporting.
- `block_adds_value` uses the **Bonferroni** interval, because the ablation makes nine simultaneous comparisons against the full model.

## Direction of effect

`export_logistic_coefficients` writes signed coefficients and odds ratios for the tuned logistic regression (`coefficients_logistic_regression.csv`). Permutation importance is **unsigned** — it says a feature matters, never whether it is associated with higher or lower satisfaction. `REQUIREMENTS.md` §15–16 need the sign, so managerial recommendations must come from here, not from the importance ranking.

Numeric coefficients are per standard deviation (features are standardised inside the pipeline). Categorical features are one-hot encoded with no dropped reference level, so their coefficients are relative to the L2-shrunk average level, not to a baseline category — read them against each other.

## Top-k feature-subset curve

`run_topk_curve` refits the best model on its k most important features for k in `TOPK_VALUES` and scores each subset on the test set. It exists to turn the feature-ceiling claim into a measurement: if a small k recovers nearly the full-model Gini, the signal concentrates in a nameable handful of features (the managerial short-list) and the remaining features add almost nothing.

**The leakage rule is the whole point of the design. Do not weaken it.**

- The feature ranking comes from `cv_permutation_importance`, which fits a clone on each stratified training fold and permutes features on the **held-out fold**. The test set never participates in ranking. Ranking features by their test-set importance and then reporting test-set performance would be circular — the test set would have chosen the features.
- Note that `plot_feature_importance` *does* compute importance on the test set. That is legitimate for **reporting**, because nothing is refitted from it. It must never be reused to select features.
- The curve model is chosen from `cv_results` (training-set cross-validation), not from the test-set table, so the test set selects neither the model nor its features.
- The curve is **descriptive**. Do not pick the best k off the curve and report that k's test score as the headline model — that reintroduces selection on the test set. The honest framing is "performance as a function of feature-set size."
- Importance is reported in **Gini units**: `permutation_importance` scores with `roc_auc`, and gini = 2·auc − 1 is affine, so an AUC drop of d is a Gini drop of 2d.

Each subset rebuilds its pipeline through `build_models(X_train[columns])`. Reusing the full-feature pipeline would fail, because its `ColumnTransformer` names columns absent from the reduced frame.

Outputs: `topk_feature_curve.csv`, `topk_feature_ranking.csv`, `plots/topk_feature_curve.png`, `plots/permutation_importance_top_features.png`.

**There are two importance figures and they measure different things. Do not conflate them.**

| Figure | Function | Permuted on | Models |
|---|---|---|---|
| `permutation_importance_top_features.png` | `plot_permutation_importance` | held-out **training** folds | the CV-best model only |
| `feature_importance_by_model.png` | `plot_feature_importance` | the **test** set | top three by test Gini |

The first is the ranking that drives the top-k curve and is safe to select features with. The second is descriptive only — nothing may be refitted from it. The paper should cite the first.

Importances are a **ranking, not an additive decomposition**: they do not sum to the model's Gini. Magnitudes are estimator-specific (random forest and HistGradientBoosting agree on order, Spearman ρ = 0.82, but HGB assigns the user-characteristics block 0.112 Gini against the forest's 0.018). Report order as a property of the data and magnitude as a property of the model.

**PCA was considered and rejected.** It produces linear combinations of all features rather than selecting a subset, so it cannot answer "which features matter"; it destroys the interpretability that `REQUIREMENTS.md` §15–16 demand; and it is poorly suited to a matrix that is mostly binary indicators. It would also break the gradient-boosting pipeline, which passes categoricals natively rather than one-hot. Do not add it.

## Central finding

Performance is bounded by a **feature ceiling with respect to model choice and tuning** — but **NOT with respect to data volume.** The learning curve refutes the data-volume half of the claim, and the earlier phrasing ("not mainly model choice, tuning, or data volume") was wrong. Do not restore it.

Measured on the held-out test set with tuned models (97 features, 100k subsample, 80k train / 20k test). Reported in GINI, the project's `SELECTION_METRIC`:

| Model | Test GINI | Test PR-AUC (dissat) | Balanced Acc. |
|---|---|---|---|
| random forest | **0.342** | 0.490 | 0.622 |
| HistGradientBoosting | 0.339 | 0.488 | 0.621 |
| bagging | 0.329 | 0.482 | 0.616 |
| MLP neural network | 0.322 | 0.481 | 0.573 |
| logistic regression | 0.296 | 0.460 | 0.605 |
| linear SVM | 0.296 | 0.460 | 0.606 |
| decision tree | 0.254 | 0.433 | 0.588 |
| KNN | 0.253 | 0.438 | 0.517 |
| Bernoulli Naive Bayes | 0.245 | 0.434 | 0.588 |

Random forest is best on both the test set (0.342) and cross-validation (0.3382 ± 0.0015), so test and CV now agree. Its margin over HistGradientBoosting (0.3322 ± 0.0032) is under two CV standard deviations — **the top two ensembles remain statistically indistinguishable.** Do not declare a winner more strongly than that.

Its margin over a tuned logistic regression is **0.046 GINI**, and the spread across all nine models is 0.097 GINI.

### What the diagnostics establish

- **Tuning buys almost nothing.** Tuned minus un-tuned on the *test* set: mean −0.0004, median −0.0001, only 4 of 9 models improved. Random forest gained +0.0196 and the decision tree +0.0154; the rest were flat or slightly worse. (KNN lost 0.0363, partly an artefact: the grid searches on a 30k subsample but the final fit uses 80k rows, and the optimal `n_neighbors` grows with sample size. Tuning a sample-size-dependent hyperparameter on a subsample is a known wrinkle — mention it rather than hide it.)
- **Tuning drives almost every model toward simplicity.** The search repeatedly selects the most-regularised value available. Winners keep landing on grid boundaries even after the grids are widened, which suggests a nearly flat objective surface rather than ranges that are too small.
- **The feature set saturates at ~25 features.** The top-k curve: k=25 matches the full 97-feature model to within 0.0034 GINI (Bonferroni CI includes zero), while k=5 and k=10 are decisively worse. 81 of the ranked features add nothing measurable.
- **BUT MORE DATA DOES HELP, for flexible models.** The learning curve, on a fixed 250k held-out set:

  | Train rows | HistGradientBoosting | logistic regression |
  |---|---|---|
  | 10,000 | 0.2874 | 0.2742 |
  | 100,000 | 0.3457 | 0.3008 |
  | 1,000,000 | **0.3736** | 0.3034 |

  From 100k to 1M rows, HGB gains **+0.0279 GINI** and is still climbing at 1M. Logistic regression gains +0.0026 and is flat from 100k onward. **The +0.0279 that data volume buys HGB is larger than the +0.0046 that the best model choice buys over logistic regression.** For this project, at these sizes, data volume dominates model choice.

### How to state the finding

The ceiling is a **capacity** ceiling for the linear models — they saturate by 100k rows and cannot use more — and a **feature-set** ceiling in the sense that 25 features carry all the available signal. It is *not* a data ceiling for the flexible models: the 100k-subsample comparison **understates** what the tree ensembles can achieve. The comparison remains internally fair (all models see identical data), but its absolute Gini values should not be presented as the maximum attainable.

Note the learning curve uses its own split of the full dataset, so its Gini levels are not directly comparable with the model-comparison tables. The *trend* is what carries the argument.

Do not compare these numbers against pre-2026-07 results: those predate the tuning rewiring, the bagging `class_weight` fix, the 106→97 feature reconciliation, and the attribute-level normalisation.

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
