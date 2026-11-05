# Milestone 2 PowerPoint Evidence Audit

Audit date: October 20, 2026. Scope: repository files and retained successful October 13 training runs, plus October 13/14 comparison outputs. This audit did not fit, tune, or retrain any model, modify source data/code/configuration, or overwrite experiment artifacts. Newly derived descriptive statistics use complete saved customer assignments and source records; they are explicitly identified. All charts are new 300-DPI PNG renderings of verified saved values.

**Evidence labels:** Verified = directly supported by source/artifacts or an audit consistency check; Derived = arithmetic from complete existing records; Inferred = interpretation, not an experimental measurement; Missing = not retained or unavailable. No presentation file was found, so slide corrections address only the claims supplied in the request; unseen slide content is unverifiable.

## Evidence register

| experiment | source | status |
| --- | --- | --- |
| logistic_regression | [experiments/outputs/logistic_regression/run_20261013_185446_609612_1216ba4a/run_metadata.json](experiments/outputs/logistic_regression/run_20261013_185446_609612_1216ba4a/run_metadata.json) | complete |
| decision_tree | [experiments/outputs/decision_tree/run_20261013_185500_256885_7761ad21/run_metadata.json](experiments/outputs/decision_tree/run_20261013_185500_256885_7761ad21/run_metadata.json) | complete |
| random_forest | [experiments/outputs/random_forest/run_20261013_185521_750053_673e200e/run_metadata.json](experiments/outputs/random_forest/run_20261013_185521_750053_673e200e/run_metadata.json) | complete |
| gradient_boosting | [experiments/outputs/gradient_boosting/run_20261013_185624_058701_3f073f00/run_metadata.json](experiments/outputs/gradient_boosting/run_20261013_185624_058701_3f073f00/run_metadata.json) | complete |
| kmeans | [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/run_metadata.json](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/run_metadata.json) | complete |

## 1. Dataset dimensions and preparation

| stage | rows | columns | meaning |
| --- | --- | --- | --- |
| Raw source | 2240 | 29 | 29 total columns = 28 non-target columns, including ID and constants |
| Milestone 1 processed CSV | 2237 | 33 | 33 total columns, not 33 model predictors |
| Classification before encoding | 2237 | 28 | 26 numeric + 2 categorical predictors |
| Training before encoding | 1789 | 28 | Response stored separately |
| Testing before encoding | 448 | 28 | Response stored separately |
| Final classification after encoding | 2237 | 39 | 26 numeric + all education/marital indicator columns |
| K Means input | 2237 | 16 | Numerical features; full dataset |

| partition | rows | nonresponders | responders | response_rate |
| --- | --- | --- | --- | --- |
| Full | 2237 | 1903 | 334 | 0.14930711 |
| Train | 1789 | 1522 | 267 | 0.14924539 |
| Test | 448 | 381 | 67 | 0.14955357 |

### Resolving 28 versus 33

Verified: the raw file has **29 total columns**: 28 non-target columns including ID and two constants. Milestone 1 adds four columns (`Customer_Age`, `Total_Spending`, `Number_of_Children`, `Past_Campaign_Responses`), producing **33 total columns**. These are not 33 usable predictors. Modeling drops ID and Response (33→31), removes two constants (31→29), replaces `Dt_Customer` with tenure (no net change), and replaces `Year_Birth` with the already-existing `Customer_Age` (29→28). Final one-hot encoding expands the 2 categorical variables into 13 indicators, giving **39 transformed features**.

Sources: [docs/milestone1/milestone1_analysis.py](docs/milestone1/milestone1_analysis.py) (`clean_data`, `engineer_features`), [experiments/common/preprocessing.py](experiments/common/preprocessing.py) (`features`, `preprocessor`); saved model inspection: [support JSON](milestone2_ppt_support/saved_model_inspection.json).

### Complete classification inputs

`Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`, `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `AcceptedCmp3`, `AcceptedCmp4`, `AcceptedCmp5`, `AcceptedCmp1`, `AcceptedCmp2`, `Complain`, `Customer_Age`, `Total_Spending`, `Number_of_Children`, `Past_Campaign_Responses`, `Customer_Tenure_Days`.

`Education` and `Marital_Status` are categorical; the other 26 are numeric. No K Means labels are included. ID is excluded as identifier; Response is the target; `Z_CostContact` and `Z_Revenue` are excluded because constant. Raw birth year and registration date are replaced by age and tenure. No other original predictive columns are silently dropped.

### Cleaning, missingness and timing

Verified raw missing values: {'Income': 24}. All other raw columns have zero missing entries. Raw Income also contains one `666666` value treated by project code as a sentinel (the data-entry interpretation itself is an assumption). Milestone 1 removes three records with Year_Birth < 1926; no duplicate records are removed. It retains the income-affected records, replaces the sentinel with missing, and fills 25 incomes with global cleaned-data median **51,371**. The saved processed CSV has **zero missing values in all 33 columns**.

Milestone 2 restores those **25 Income cells to NaN in memory**, aligned by ID to the raw file; it does not restore deleted customer records. Numeric `SimpleImputer(strategy="median")` then learns from each fold training subset, and final training learns from all 1,789 training records. Categoricals use most-frequent imputation and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` with all categories retained (no reference category dropped). Logistic Regression additionally uses numeric `StandardScaler` (centering and unit variance); trees do not scale. Numeric/categorical imputation, encoding and LR scaling all live inside the cloned pipeline fitted within each fold. K Means uses full-data median imputation/scaling independently.

The saved `data_audit.missing` values describe the CSV **before restoration**, so Income=0 there and `restored_missing_income=25` are compatible. After restoration there are 25 missing income cells; saved fitted imputers provide finite replacements. Final saved models have 39 features; fold encoders can omit categories absent from that training fold, and per-fold transformed dimensions are not saved.

Age in Milestone 1 is `2026-Year_Birth`; classification and clustering overwrite it as `2014-Year_Birth`. Tenure is `2014-12-31-Dt_Customer`. This fixed date is not a verified offer date. Three birth-year exclusions use a 2026 rule despite the later 2014 modeling reference.

Sources: [experiments/common/data_loader.py](experiments/common/data_loader.py) (`load_data`), [experiments/common/data_validation.py](experiments/common/data_validation.py) (`validate`), [experiments/common/evaluation.py](experiments/common/evaluation.py) (`run_classifier`), [docs/milestone1/reports/data_quality_summary.json](docs/milestone1/reports/data_quality_summary.json).

Cleaned dataset dimensions are 2,237×33; model matrix dimensions are 1,789×28 train and 448×28 test before encoding, and 1,789×39 and 448×39 under the final fitted encoders. Labels have 1,789 and 448 entries respectively.

## 2. Classification experiment configuration

| model | estimator | explicit_parameters | numeric_scaling | class_weight | seed | train_rows | test_rows | cv | features_before_encoding | features_after_encoding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | sklearn.linear_model._logistic.LogisticRegression | {"class_weight": "balanced", "max_iter": 2000, "random_state": 42, "solver": "lbfgs"} | True | balanced | 42 | 1789 | 448 | shared shuffled 5-fold StratifiedKFold | 28 | 39 |
| Decision Tree | sklearn.tree._classes.DecisionTreeClassifier | {"class_weight": "balanced", "max_depth": 5, "min_samples_leaf": 10, "random_state": 42} | False | balanced | 42 | 1789 | 448 | shared shuffled 5-fold StratifiedKFold | 28 | 39 |
| Random Forest | sklearn.ensemble._forest.RandomForestClassifier | {"class_weight": "balanced", "min_samples_leaf": 3, "n_estimators": 300, "n_jobs": -1, "random_state": 42} | False | balanced | 42 | 1789 | 448 | shared shuffled 5-fold StratifiedKFold | 28 | 39 |
| Gradient Boosting | sklearn.ensemble._gb.GradientBoostingClassifier | {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 200, "random_state": 42} | False | None | 42 | 1789 | 448 | shared shuffled 5-fold StratifiedKFold | 28 | 39 |

All four estimators are scikit-learn classifiers. Explicit parameters above come from each saved run config, and fitted artifacts were loaded for inspection. Full effective estimator parameters, including defaults, are in [saved_model_inspection.json](milestone2_ppt_support/saved_model_inspection.json). No resampling/SMOTE or sample weights appear in the fitting code; class weights are balanced for LR, DT and RF; Gradient Boosting has no class balancing configured.

Verified: stratified 80/20 split, seed 42, same persisted train/test row indices and identical five shuffled stratified training folds for every classifier. The saved split identity is `287480a9a9870a43ec7ed3f984324cd844fdec9d9bacdbe40519fb04640b5094`. CV validation sizes: [358, 358, 358, 358, 357]. Config settings are fixed; no GridSearchCV, RandomizedSearchCV or parameter-search loop exists in these entry points. Manual fixed configuration is supported; how people originally chose those values is not recorded. All four use identical 28 underlying features.

Source: [experiments/outputs/shared/287480a9a9870a43ec7ed3f984324cd844fdec9d9bacdbe40519fb04640b5094/split.json](experiments/outputs/shared/287480a9a9870a43ec7ed3f984324cd844fdec9d9bacdbe40519fb04640b5094/split.json), `shared_split` in [experiments/common/split_manager.py](experiments/common/split_manager.py), each `experiments/<model>/train.py` and saved `config.json`.

## 3. Classification performance

All final test columns below use each model’s OOF-selected threshold; CV F1 uses **0.50**. SD is the sample SD across five folds (`ddof=1`), not a confidence interval. ROC AUC is reported because it already exists in the historical JSON files.

| model | cv_f1 | cv_f1_sample_sd | test_f1 | precision | recall | accuracy | macro_f1 | roc_auc | average_precision | threshold | default_test_f1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.57562116 | 0.02263395 | 0.56790123 | 0.48421053 | 0.68656716 | 0.84375000 | 0.73626669 | 0.89136992 | 0.61603840 | 0.60000000 | 0.54255319 | 332 | 49 | 21 | 46 |
| Decision Tree | 0.42782230 | 0.04101992 | 0.41481481 | 0.41176471 | 0.41791045 | 0.82366071 | 0.65550202 | 0.77776472 | 0.40017689 | 0.75000000 | 0.44025157 | 341 | 40 | 39 | 28 |
| Random Forest | 0.56280108 | 0.03006107 | 0.58741259 | 0.55263158 | 0.62686567 | 0.86830357 | 0.75452967 | 0.88643397 | 0.62049038 | 0.50000000 | 0.58741259 | 347 | 34 | 25 | 42 |
| Gradient Boosting | 0.51071901 | 0.04382776 | 0.52777778 | 0.49350649 | 0.56716418 | 0.84821429 | 0.71867612 | 0.89121323 | 0.62678523 | 0.25000000 | 0.48979592 | 342 | 39 | 29 | 38 |

### Threshold verification

Verified: 17 thresholds from 0.10 through 0.90 inclusive in 0.05 steps are evaluated against training **out-of-fold probabilities**. Selection maximizes pooled positive-class OOF F1, breaking ties by higher precision, then closeness to 0.50. Each classifier chooses separately. Selected thresholds are frozen before predicting the 448 test customers. The test set used for final evaluation is the saved common holdout, not the threshold-development set. Audit recomputation from complete saved predictions confirmed selected precision/recall/F1/accuracy, confusion matrices, threshold decisions, and all threshold-grid F1 values.

Threshold-development performance is optimistic for the threshold it selects because the same OOF labels choose and score it; it is not an independent estimate. Test threshold-selection leakage is **not evident** in the implemented workflow. Subsequent selection or tuning using the already-viewed test leaderboard could bias future claims. CV-at-0.50 and test-at-selected-threshold are different evaluation policies and should be labeled separately.

Existing precision–recall evidence: each successful run has `precision_recall_curve.png`, `test_metrics.json` average precision, `threshold_analysis.csv`, `oof_predictions.csv`, and `test_predictions.csv`. Average precision and ROC AUC use probabilities, not the selected binary threshold. Historical ROC AUC values/plots persist, although current `evaluation.metrics` and `compare_classifiers.py` omit ROC AUC; an exact rerun of current reporting will not regenerate those historical fields.

Sources: `metrics`, `run_classifier` in [experiments/common/evaluation.py](experiments/common/evaluation.py); each run’s `cv_metrics.json`, `test_metrics.json`, `threshold_development.json`, and CSVs. [Audit verification](milestone2_ppt_support/audit_verification.txt).

### Model recommendation

Highest cv_f1: **Logistic Regression** (0.57562116); Highest test_f1: **Random Forest** (0.58741259); Highest precision: **Random Forest** (0.55263158); Highest recall: **Logistic Regression** (0.68656716).

LR is the provisional CV-based recommendation; RF is the strongest observed test-F1/precision alternative. The small CV F1 gap is 0.01282008, and RF’s test F1 advantage over LR is 0.01951135. No repeated splits, confidence intervals or paired significance test establish a reliable winner. Only 67 positives are in the holdout. The saved majority baseline has accuracy 0.85044643 but positive-class F1=0; accuracy alone is misleading. Operational choice requires contact cost, benefit and capacity, none measured here.

## 4. Feature importance and interpretation

### Logistic Regression: top signed coefficients

Largest positive coefficients:

| feature | importance |
| --- | --- |
| Customer_Tenure_Days | 1.07615632 |
| Education_PhD | 1.05221394 |
| Marital_Status_Alone | 0.61527298 |
| Marital_Status_Widow | 0.52146365 |
| MntMeatProducts | 0.49955675 |
| Past_Campaign_Responses | 0.45589273 |
| AcceptedCmp3 | 0.41040797 |
| Education_Master | 0.34216928 |
| NumDealsPurchases | 0.33755542 |
| Marital_Status_Divorced | 0.33514121 |

Most negative coefficients:

| feature | importance |
| --- | --- |
| Education_Basic | -1.37431252 |
| Recency | -0.97927197 |
| Marital_Status_Together | -0.97241910 |
| NumStorePurchases | -0.84583908 |
| Marital_Status_Married | -0.84441359 |
| Teenhome | -0.49270595 |
| Education_2n Cycle | -0.27743827 |
| MntWines | -0.25629726 |
| Number_of_Children | -0.17894973 |
| Marital_Status_YOLO | -0.13206821 |

Numeric coefficients refer to a one-training-standard-deviation change after median imputation and StandardScaler; one-hot coefficients refer to indicator columns, which are not standardized. All categories are retained, and regularization/correlated variables complicate reference-category comparisons. Coefficients are conditional log-odds associations, not causal effects or raw-unit changes.

### Random Forest top 10

| feature | importance |
| --- | --- |
| Customer_Tenure_Days | 0.10288718 |
| Recency | 0.10263483 |
| Past_Campaign_Responses | 0.09791355 |
| Total_Spending | 0.06354434 |
| Income | 0.05585407 |
| MntMeatProducts | 0.05470532 |
| MntGoldProds | 0.05282987 |
| MntWines | 0.04569582 |
| NumStorePurchases | 0.03715905 |
| NumCatalogPurchases | 0.03585339 |

Source: [experiments/outputs/random_forest/run_20261013_185521_750053_673e200e/feature_importance.csv](experiments/outputs/random_forest/run_20261013_185521_750053_673e200e/feature_importance.csv). Method: fitted estimator `feature_importances_` (normalized impurity-based importance), verified against loaded artifact.

### Gradient Boosting top 10

| feature | importance |
| --- | --- |
| Past_Campaign_Responses | 0.27192591 |
| Recency | 0.16228250 |
| Customer_Tenure_Days | 0.12234533 |
| MntMeatProducts | 0.05422305 |
| Income | 0.04495600 |
| NumStorePurchases | 0.03433519 |
| AcceptedCmp3 | 0.03342984 |
| NumCatalogPurchases | 0.02864796 |
| NumWebVisitsMonth | 0.02427958 |
| MntFishProducts | 0.02207358 |

Source: [experiments/outputs/gradient_boosting/run_20261013_185624_058701_3f073f00/feature_importance.csv](experiments/outputs/gradient_boosting/run_20261013_185624_058701_3f073f00/feature_importance.csv). Method: fitted estimator `feature_importances_` (normalized impurity-based importance), verified against loaded artifact.

Each classifier also has `permutation_importance.csv` (five repeats, seed 42, scoring F1 at default 0.50), evaluated on the first saved training validation fold after fitting its training portion; it does not use the final test set. Existing `model_behavior.md`, coefficient/importance PNGs, classification reports and a Decision Tree visualization provide additional interpretation. Totals and their components overlap: spending totals, child totals and historical campaign totals can redistribute importance. Neither coefficients, impurity nor permutation values prove causality. Sources: `run_classifier` interpretation block and each successful run’s outputs.

## 5. K Means clustering configuration

Verified **16 numeric inputs**: `Income`, `Kidhome`, `Teenhome`, `Customer_Age`, `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`, `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`.

Six product-spending features represent spending; four purchase counts represent transactions/deals/channels; website visits represent engagement; Income, age and household counts represent demographics; Recency represents purchase timing. Income and Recency are included. ID, Response, AcceptedCmp1–5 and Past_Campaign_Responses are excluded; aggregate Total_Spending and Number_of_Children are also not clustering inputs. Features are manually listed, not automatically selected.

K Means is scikit-learn `KMeans`, Euclidean-distance clustering, numeric median imputation and `StandardScaler` on **all 2,237 customers**. Explicit code settings: n_clusters per candidate, n_init=20, random_state=42. Saved artifact effective settings: `{"algorithm": "lloyd", "copy_x": true, "init": "k-means++", "max_iter": 300, "n_clusters": 2, "n_init": 20, "random_state": 42, "tol": 0.0001, "verbose": 0}`. Fitted selected model iterations: **5**. Initialization is k-means++; default max_iter=300 and tol=0.0001 are effective artifact parameters, not explicitly configured in JSON. Twenty initializations choose a low-inertia solution for a given K; this does not guarantee global optimality.

Clustering is independent from supervised classification. It never calls `shared_split`; Response is attached only after fitting/selection. Classifiers do not consume cluster assignments, and no cluster-specific classifiers appear. Sources: [experiments/kmeans/train.py](experiments/kmeans/train.py), [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/config.json](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/config.json), [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/model.joblib](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/model.joblib).

## 6. Selecting the number of clusters

| k | silhouette | inertia | min_cluster_size | max_cluster_size | eligible | cluster_sizes |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.30376038 | 24736.51577595 | 933 | 1304 | True | 1304, 933 (cluster IDs 0, 1) |
| 3 | 0.23406931 | 21465.75968904 | 596 | 1008 | True | Not saved in full; see min/max |
| 4 | 0.16508608 | 20052.68985558 | 537 | 579 | True | Not saved in full; see min/max |
| 5 | 0.15659713 | 19062.97437860 | 261 | 571 | True | Not saved in full; see min/max |
| 6 | 0.15940521 | 18133.45061165 | 176 | 540 | True | Not saved in full; see min/max |
| 7 | 0.15200715 | 17454.38542814 | 146 | 537 | True | Not saved in full; see min/max |
| 8 | 0.15380181 | 16908.04814027 | 143 | 536 | True | Not saved in full; see min/max |

Verified K=2 has the highest silhouette **0.3037603760365297**, correctly rounded to **0.3038**. K=3 is next at 0.23406930625174138 (gap 0.06969107); none matches K=2 under this rule. All candidates pass the minimum-cluster-size constraint (5%, requiring at least 112 of 2,237 customers). The selector maximizes silhouette among eligible candidates; exact ties favor lower K. Interpretability is discussed afterward and is **not a numerical selection criterion**.

Inertia values were saved, but no elbow-selection algorithm or original elbow plot is evident. This audit creates an inertia chart from those saved values; it does not imply that elbow analysis originally selected K. Full per-cluster sizes/labels for K=3–8 were not retained, only min/max sizes; they cannot be reconstructed without fitting and are deliberately not invented.

K=2 is supported as the best candidate **among 2–8 for these features, scaling and silhouette constraint**. It is not proven globally or commercially optimal. No stability/resampling study is saved. Sources: [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/cluster_metrics.csv](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/cluster_metrics.csv), [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/selection.json](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/selection.json) and `main` in K Means training.

## 7. Customer cluster profiles

The complete CSV includes all 16 feature means, sizes and response rates plus new explicitly derived summaries. Income means in the original profiles omit restored missing/sentinel incomes; K Means fitting instead imputes them. Median income below is newly derived from the complete observed-income records after the same restoration, not a previously saved statistic. Total spending is newly derived by summing all six spending columns; counts and shares are derived from complete saved assignments.

| cluster | size | share | Income | median_income_observed | income_missing_excluded | mean_total_spending | Recency | Kidhome | Teenhome | accepted | not_accepted | response_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1304 | 0.58292356 | 38256.83281493 | 37780.50000000 | 18 | 175.69095092 | 48.61963190 | 0.71319018 | 0.55061350 | 131 | 1173 | 0.10046012 |
| 1 | 933 | 0.41707644 | 70987.69114471 | 70889.50000000 | 7 | 1206.80278671 | 49.78242229 | 0.06859593 | 0.44480171 | 203 | 730 | 0.21757771 |

Channel/engagement comparisons:

| cluster | NumDealsPurchases | NumWebPurchases | NumCatalogPurchases | NumStorePurchases | NumWebVisitsMonth | Customer_Age |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.56288344 | 2.89800613 | 0.83742331 | 3.82668712 | 6.47852761 | 43.71855828 |
| 1 | 1.99678457 | 5.74919614 | 5.21329046 | 8.54448017 | 3.69882101 | 47.02679528 |

The supplied counts **1,304** and **933**, rounded incomes **38,257** and **70,988**, and response rates **10.0%** and **21.8%** are verified. A dollar sign is not established by the repository’s data dictionary; use “income (source currency units)” unless currency is independently confirmed.

Derived from unrounded counts: absolute response-rate difference = 0.117117583624, or **11.71175836 percentage points**. Response-rate ratio = **2.1658116721**.

“Lower Spending Customers” and “Higher Spending Customers” are empirically supported by all six product-spending means and total spending, with higher web/catalog/store purchase frequencies in Cluster 1. Cluster 0 has more young children and more website visits despite fewer purchases; that is a descriptive pattern, not proof of why purchasing differs. The response comparison is retrospective and in-sample for segmentation; it does not establish incremental campaign lift, targeting profitability or a causal income effect. Sources: [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/cluster_profiles.csv](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/cluster_profiles.csv), [experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/cluster_assignments.csv](experiments/outputs/kmeans/run_20261013_185820_218800_afad3148/cluster_assignments.csv), [derived complete profiles](milestone2_ppt_support/cluster_profile_comparison.csv).

## 8. Reproducibility and project architecture

| Component | Verified implementation |
|---|---|
| Data loading/validation | `common/data_loader.py::load_data`, `data_validation.py::validate` |
| Feature preparation | `common/preprocessing.py::features`, `preprocessor` |
| Persisted split/CV | `common/split_manager.py::shared_split`, shared split.json and assignments.csv |
| Classifier evaluation | `common/evaluation.py::run_classifier`, `metrics` |
| Saving/metadata | `common/result_saver.py::new_run`, `metadata`, `save_json` |
| Independent entry points | each model’s train.py builds its estimator; K Means separate |
| Saved pipelines | all five successful run model.joblib files loaded in this audit |
| Saved-result comparison | `comparison/compare_classifiers.py`, no fitting |
| Existing verification | `verify_outputs.py` checks artifacts and writes EXECUTION_SUMMARY.md |
| Cluster explanation | `kmeans/explain_results.py` reads saved results, no fitting |

Saved metadata records Python 3.14.3, scikit-learn 1.9.1, seed 42, dataset hashes, configs and split IDs. requirements.txt pins six numerical/model libraries, but is not a full environment lock (e.g. Pillow used by verification is not pinned). Current raw and processed hashes match retained successful metadata. No source-code commit/hash is captured in run metadata, so exact historical code equivalence is not proven. Current evaluation no longer emits historical ROC AUC and K Means current code calls a per-run explanation absent from this older run; complete artifact-for-artifact reproduction is therefore not established. Absolute Windows paths in saved configurations require adjustment on another machine.

The root `outputs/logistic_regression/run_20261013_185401_860960_f3ed22a6/` is a retained **failed** attempt: its error.txt records a FileNotFoundError from a wrongly resolved dataset path. It is excluded from performance tables. The successful model runs are under `experiments/outputs/`. Both saved comparison CSVs agree on their shared numeric results; the older comparison retains ROC AUC columns while the later one omits them.

Existing EXECUTION_SUMMARY.md and the handoff document state verification passed. This is retained documentary evidence, not a timestamped standalone verification log. The audit did not run verify_outputs.py because it overwrites the existing execution summary. Instead these read-only checks **actually ran successfully**:

- Logistic Regression: saved pipeline loaded; selected-threshold classes, TN/FP/FN/TP and four primary test metrics match complete saved predictions; all 17 threshold F1 scores match saved OOF predictions; selected threshold and feature importance match artifacts. Load warnings: 0.
- Decision Tree: saved pipeline loaded; selected-threshold classes, TN/FP/FN/TP and four primary test metrics match complete saved predictions; all 17 threshold F1 scores match saved OOF predictions; selected threshold and feature importance match artifacts. Load warnings: 0.
- Random Forest: saved pipeline loaded; selected-threshold classes, TN/FP/FN/TP and four primary test metrics match complete saved predictions; all 17 threshold F1 scores match saved OOF predictions; selected threshold and feature importance match artifacts. Load warnings: 0.
- Gradient Boosting: saved pipeline loaded; selected-threshold classes, TN/FP/FN/TP and four primary test metrics match complete saved predictions; all 17 threshold F1 scores match saved OOF predictions; selected threshold and feature importance match artifacts. Load warnings: 0.
- Current raw/processed SHA256 match saved metadata; 22 existing experiment PNGs decode successfully.
- Saved K Means labels, IDs, Response values, cluster counts and response rates match full assignments. No models fitted or retrained.
- Existing verify_outputs.py was inspected but not executed because it overwrites experiments/EXECUTION_SUMMARY.md. The independent checks above ran during this audit.
- Both saved comparison CSVs agree with source CV/test F1. All five saved fold F1 scores and their mean/sample SD match OOF records; fold train/validation indices are disjoint and exhaustive. All 16 saved cluster feature means match complete source records under the documented 2014 age convention.

No predictions were regenerated from models, and no training reproducibility run was performed. Artifact loadability and consistency are verified; deterministic numerical reproduction across environments is not claimed.

## 9. Methodological risks and limitations

| Category | Issue | Evidence-based explanation | Source |
| --- | --- | --- | --- |
| Potential Issue | Historical feature timing | AcceptedCmp1–5 and their sum are used; campaign chronology is not independently verified. Spending/recency/complaints must also precede the offer. | preprocessing.features; run feature lists |
| Confirmed Issue | Timing convention inconsistency | Milestone 1 age uses 2026; model age uses 2014; cleaning excludes birth years using a 2026 age cutoff. Fixed 2014-12-31 is not a verified prediction timestamp. | milestone1_analysis.clean_data/engineer_features; preprocessing.features |
| Not Evident From Available Evidence | Fold preprocessing leakage | Imputation/encoding/scaling fit inside cloned fold pipelines; previously global income imputations are restored before fitting. Unsupervised full-data clustering is independent. | data_loader.load_data; evaluation.run_classifier |
| Not Evident From Available Evidence | Test-based threshold optimization | Thresholds use training OOF predictions; test threshold is already frozen. | evaluation.run_classifier; threshold_analysis/OOF CSVs |
| Confirmed Issue | Threshold-development optimism | Same pooled OOF outcomes choose and score threshold; development performance is not independent validation. | threshold_development.json; evaluation.run_classifier |
| Confirmed Issue | Class imbalance | 334/2237 positives; only 67 positive test cases; accuracy alone favors the majority. | target counts; test_metrics baselines |
| Potential Issue | Model ranking uncertainty | Small observed CV differences, one holdout, no repeated splits or paired significance evidence; SD is not a confidence interval. | cv_metrics.json; split.json |
| Potential Issue | Feature importance interpretation | Correlated totals/components and impurity cardinality bias can redistribute apparent importance; causal claims unsupported. | feature lists; feature_importance.csv; permutation_importance.csv |
| Potential Issue | Segmentation overinterpretation | Moderate silhouette, full-data descriptive responses, correlated spending dimensions and untested stability limit generalization/business claims. | cluster_metrics.csv; cluster_profiles.csv |
| Confirmed Issue | Reporting/code drift | Historical ROC AUC persists but current code omits it; root README says planning phase; handoff lists TEAM_HANDOFF_SUMMARY.md but actual filename is Oct14_HANDOFF_SUMMARY.md. | test_metrics.json; evaluation.py; README.md; Oct14_HANDOFF_SUMMARY.md |
| Potential Issue | Repeated future use of the holdout | Comparing reported models is descriptive; future tuning/selection using these viewed test scores would compromise fresh evaluation. | comparison summary; common test split |
| Not Evident From Available Evidence | Cluster labels in classifiers | No cluster inputs or cluster-specific training exist in inspected successful workflows. | features metadata; train.py; evaluation.py |

Additional discrepancies/limits: 28 “original features” counts non-target raw columns including nonpredictive ID/constants; 33 describes total processed columns. Income=0 missing in saved audit precedes the restoration of 25 cells. Stored classification training metrics use threshold 0.50, not necessarily each selected threshold. Old comparison includes ROC AUC while current comparison omits it. Missing full K=3–8 cluster sizes are not evidence of equal-sized clusters. The current K Means source would produce per-run results_explanation.md, absent from the retained run; the separate kmeans/RESULTS_EXPLANATION.md exists. These differences are reported rather than silently normalized.

## 10. Presentation-ready tables and visualizations

All support is in [milestone2_ppt_support](milestone2_ppt_support/). CSVs preserve substantially more precision than slide labels. Generated figures use consistent English labels and 300 DPI.

| file | purpose |
| --- | --- |
| audit_verification.txt | Table/data or verification |
| classification_performance_comparison.csv | Table/data or verification |
| cluster_profile_comparison.csv | Table/data or verification |
| cluster_response_rate_comparison.png | Presentation figure |
| cluster_size_comparison.png | Presentation figure |
| confusion_matrix_decision_tree.png | Presentation figure |
| confusion_matrix_gradient_boosting.png | Presentation figure |
| confusion_matrix_logistic_regression.png | Presentation figure |
| confusion_matrix_random_forest.png | Presentation figure |
| dataset_hash_verification.csv | Table/data or verification |
| dataset_summary.csv | Table/data or verification |
| feature_importance_decision_tree.csv | Table/data or verification |
| feature_importance_decision_tree.png | Presentation figure |
| feature_importance_gradient_boosting.csv | Table/data or verification |
| feature_importance_gradient_boosting.png | Presentation figure |
| feature_importance_logistic_regression.csv | Table/data or verification |
| feature_importance_logistic_regression.png | Presentation figure |
| feature_importance_random_forest.csv | Table/data or verification |
| feature_importance_random_forest.png | Presentation figure |
| inertia_comparison.png | Presentation figure |
| k_selection_comparison.csv | Table/data or verification |
| model_configuration_comparison.csv | Table/data or verification |
| precision_recall_comparison.png | Presentation figure |
| saved_model_inspection.json | Table/data or verification |
| silhouette_comparison.png | Presentation figure |
| target_distribution.csv | Table/data or verification |
| test_f1_comparison.png | Presentation figure |

Existing PR curves remain in each original classifier run. The new inertia plot uses existing inertia; no new elbow decision was made. Unavailable: full candidate cluster sizes for K=3–8, historical per-fold fitted pipelines/dimensions, a verified source-currency definition, confidence intervals, stability tests and the actual presentation deck. No figures requiring these missing data were fabricated. The support builder is retained for transparency; it only reads repository evidence and writes the audit/support outputs.

## 11. Slide-specific corrections

Actual slide text/figures were not supplied or found. “Correct” applies only to the numeric claim given in the request; topic-only claims remain unverifiable. Suggested replacements are fully evidence based.

| Slide | Current supplied claim | Status | Recommended PowerPoint wording | Evidence |
| --- | --- | --- | --- | --- |
| 6 | 2,237 records, 33 features, 25 missing income records | Needs Clarification | 2,237 customers; 33 processed columns; 28 classification inputs before encoding, 39 after encoding. 24 originally missing incomes + 1 sentinel restored and fold-imputed. | Section 1; raw/processed CSV; preprocessing; data_loader |
| 8 | Threshold optimization procedure (exact slide wording unavailable) | Unverifiable | Choose a separate threshold per model using training OOF F1 over 0.10–0.90 (step 0.05); freeze before final common-holdout evaluation. | evaluation.run_classifier; threshold_analysis.csv; threshold_development.json |
| 9 | Classification hyperparameters (exact slide wording unavailable) | Unverifiable | LR: balanced, lbfgs, max_iter 2000; DT: depth 5, leaf 10, balanced; RF: 300 trees, leaf 3, balanced; GB: 200 trees, rate .05, depth 3. Seed 42 throughout. | Section 2 table; saved config.json |
| 10 | CV F1, Test F1, Precision, Recall (slide numbers unavailable) | Unverifiable | Use Section 3 table: CV F1 at .50; Test F1/precision/recall at model-specific OOF-selected thresholds. | cv_metrics.json; test_metrics.json; validated prediction CSVs |
| 11 | Logistic Regression versus Random Forest recommendation | Needs Clarification | LR has highest mean training CV F1 (0.5756); RF has highest observed test F1 (0.5874). Evidence does not establish a statistically reliable winner. | Section 3; current comparison summary |
| 12 | 16 K Means features; K=2–8; best silhouette .3038 | Correct | 16 standardized numerical inputs, all 2,237 customers; K=2 wins silhouette among K=2–8 subject to every cluster >=5%; silhouette 0.3038. | K Means config/train.py; cluster_metrics.csv; selection.json |
| 13 | Cluster 0:1304/income 38257/10.0%; Cluster 1:933/income 70988/21.8% | Correct | Lower-/higher-spending groups: 1,304 vs 933 customers; mean income 38,257 vs 70,988 source units; observed responses 10.0460% vs 21.7578%. Associations, not causal lift. | cluster_profiles.csv; cluster_assignments.csv; Section 7 |
| 14 | Reproducibility and output verification | Needs Clarification | Saved configs, seed, hashes, common split/folds, predictions and pipelines support inspection. Audit consistency checks passed; no retraining reproduction performed; current code differs from historical output schema. | Section 8; audit_verification.txt; metadata |
| 16 | Outstanding issues and future work (exact slide wording unavailable) | Unverifiable | Confirm pre-offer timing/date assumptions; acknowledge imbalance, small holdout and ranking uncertainty; assess cluster stability; use fresh/nested validation for future tuning and business-cost policy. | Section 9; repository limitations |

For Slide 10 insert the consolidated performance table from Section 3 or its CSV; do not mix default and selected-threshold precision/recall. For Slide 13 avoid an unsupported dollar currency label and include the 11.7118 percentage-point descriptive difference.

## 12. Final summary for presentation revision

**Confirmed results.** Four independently trained classifiers share 1,789 training and 448 test customers, 28 original model inputs and 39 encoded inputs under final encoders. LR leads default-threshold training CV F1; RF leads observed selected-threshold test F1 and precision; LR leads recall. Independent K Means uses 16 standardized inputs across 2,237 customers; K=2 has the best tested silhouette, 0.30376038.

**Technical corrections.** Distinguish total columns from predictors; 24 missing raw incomes plus one sentinel from the 25 restored cells; CV threshold .50 from selected-threshold test metrics; training OOF threshold selection from test tuning; code/artifact consistency from retraining reproducibility. Historical ROC AUC is real saved evidence despite its removal from current reporting.

**Business insights.** Higher-spending customers have substantially higher purchase counts and a 21.7578% observed acceptance rate versus 10.0460% in the lower-spending group. RF offers the highest observed precision; LR recovers the most actual responders. These facts motivate controlled targeting tests, not claims of incremental profit.

**Priority figures.** Test F1, precision/recall, four confusion matrices, silhouette by K, and cluster counts/response rates. Use coefficient and RF/GB importance charts when discussing associations. Inertia is optional context, not the original selection rule.

**Prioritized edits before presenting:**

1. Correct Slides 6, 8 and 10 with verified feature counts, restoration/imputation and threshold labels.
2. Populate Slide 9 from saved configurations and retain exact numeric CSVs as backup.
3. Qualify Slide 11: LR provisional CV choice; RF strongest observed test-F1 alternative; no established winner.
4. Keep Slides 12–13 numeric results, add candidate-range/minimum-size and descriptive-association qualifications.
5. Revise Slide 14 to state the checks that actually ran, and disclose historical/current-code schema drift.
6. Use Slide 16 for historical timing validation, date-reference consistency, fresh evaluation for future tuning, cluster stability, and cost/capacity-driven campaign decisions.

No model changes or new experiments are required to make these evidence-based corrections.
