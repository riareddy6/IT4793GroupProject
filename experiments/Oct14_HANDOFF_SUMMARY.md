# Milestone 2: Team Handoff Summary

**Project:** Marketing Offer Acceptance Prediction — IT 4793  
**Status as of:** October 14, 2026  
**Completed:** Four independent classification experiments, one independent customer segmentation experiment, saved-result comparison, output verification, and supporting documentation.

## 1. What has been completed

A modular experiment framework has been implemented under `experiments/`. Each model has its own training entry point, configuration, and separate run outputs. Shared utilities keep data preparation and classification evaluation consistent. Original datasets and Milestone 1 artifacts have been preserved.

The completed classifiers are Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. All predict `Response`, where 1 means offer acceptance. K Means performs customer segmentation separately.

**The classifiers do not use data transformed by K Means, cluster labels, or separate models within each cluster.** Both branches start from the Milestone 1 processed CSV:

```text
Milestone 1 processed dataset
├── Shared classification preparation and saved split
│   ├── Logistic Regression
│   ├── Decision Tree
│   ├── Random Forest
│   └── Gradient Boosting
│       └── Separate saved results → classifier comparison
└── Numerical segmentation preparation
    └── K Means → customer assignments, profiles, and interpretation
```

## 2. Directory guide

All paths below are relative to the repository root unless stated otherwise.

```text
experiments/
├── common/                  Shared data, validation, evaluation, and saving utilities
├── logistic_regression/     Logistic Regression train.py and config.json
├── decision_tree/           Decision Tree train.py and config.json
├── random_forest/           Random Forest train.py and config.json
├── gradient_boosting/       Gradient Boosting train.py and config.json
├── kmeans/                  K Means training, configuration, and interpretation
├── comparison/              Comparison of saved classification results
├── outputs/                 Shared partitions and separate timestamped runs
├── verify_outputs.py        Saved-output checks and execution-summary generation
├── requirements.txt         Pinned experiment dependencies
├── README.md                Reproduction instructions and methodological details
├── EXECUTION_SUMMARY.md      Verified execution results and run references
└── TEAM_HANDOFF_SUMMARY.md   This team handoff report
```

### Shared utilities: `experiments/common/`

| File | Responsibility |
|---|---|
| `data_loader.py` | Read configuration and processed data; restore original missing-income markers in memory using customer IDs from the raw file. |
| `data_validation.py` | Validate binary labels and unique IDs; record shape, types, missingness, constants, target counts, and birth-year range. |
| `preprocessing.py` | Prepare features and construct fold-local imputation, one-hot encoding, and optional numerical scaling. |
| `split_manager.py` | Create or reuse the same stratified train/test split and five CV folds for all classifiers. |
| `evaluation.py` | Fit the supplied classifier, evaluate CV and final test performance, select an OOF threshold, and save predictions, plots, and explanations. |
| `result_saver.py` | Create unique run directories and save configuration, data hashes, software versions, seed, and timestamps. |

Each model's `train.py` constructs its own estimator. Its `config.json` specifies data paths and model settings. The shared evaluation utility evaluates one supplied classifier; there is no single script that trains all models. `__init__.py` files make directories importable as Python packages.

`comparison/compare_classifiers.py` reads saved results and checks compatibility without retraining. `verify_outputs.py` checks required artifacts and generates the execution summary.

## 3. Data preparation and validation design

**Source:** `docs/milestone1/processed_data/marketing_campaign_processed.csv`  
**Processed dataset:** 2,237 rows and 33 columns; 334 responders and 1,903 non-responders.  
**Classification inputs:** 28 features before one-hot encoding.  
**Shared partition:** 1,789 training observations and 448 independent test observations.

Milestone 1 supplied the cleaned customer records and engineered spending, household, age, and historical acceptance features. Milestone 2 reuses these records rather than repeating the original cleaning and EDA.

The following modeling preparation is applied:

- Restore 25 originally missing/sentinel income values in memory, then impute using the applicable training fold's median. The source CSV is unchanged.
- Impute categorical values using the training mode and one-hot encode nominal categories, handling unseen categories.
- Exclude `ID` and `Response` from predictors. Remove the two verified constant columns, `Z_CostContact` and `Z_Revenue`.
- Replace birth year with age referenced to 2014 and registration date with tenure at the fixed reference date, December 31, 2014. This is a documented convention, not a verified offer date.
- Scale numerical inputs for Logistic Regression. Trees use the same feature information without scaling.
- Retain historical campaign acceptance variables for classification under the assumption they were available before the current offer. Their actual timing still requires confirmation.

All classifiers use random seed 42, a stratified 80/20 partition, and the same shuffled five-fold StratifiedKFold assignments. Imputers, encoders, and scalers are fitted only on the applicable training observations.

Thresholds from 0.10 through 0.90 in 0.05 steps are evaluated using training out-of-fold (OOF) probabilities. The threshold maximizing positive-class F1 is selected; ties favor precision and proximity to 0.50. These threshold-selection scores are development estimates. Final test probabilities are obtained after the model and threshold are frozen. Both default-threshold and selected-threshold test metrics are saved.

## 4. Model settings and actual results

| Classifier | Main settings; all use random_state=42 |
|---|---|
| Logistic Regression | max_iter=2000, balanced class weights, lbfgs solver |
| Decision Tree | max_depth=5, min_samples_leaf=10, balanced class weights |
| Random Forest | 300 trees, min_samples_leaf=3, balanced class weights, n_jobs=-1 |
| Gradient Boosting | 200 trees, learning_rate=0.05, max_depth=3 |

The table below uses saved results. CV F1 uses threshold 0.50; final precision, recall, and F1 use the threshold selected from training OOF predictions.

| Model | CV F1 mean ± sample SD | Test F1 at 0.50 | Selected threshold | Final test precision | Final test recall | Final test F1 | Test average precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5756 ± 0.0226 | 0.5426 | 0.60 | 0.4842 | 0.6866 | 0.5679 | 0.6160 |
| Decision Tree | 0.4278 ± 0.0410 | 0.4403 | 0.75 | 0.4118 | 0.4179 | 0.4148 | 0.4002 |
| Random Forest | 0.5628 ± 0.0301 | 0.5874 | 0.50 | 0.5526 | 0.6269 | 0.5874 | 0.6205 |
| Gradient Boosting | 0.5107 ± 0.0438 | 0.4898 | 0.25 | 0.4935 | 0.5672 | 0.5278 | 0.6268 |

**Provisional recommendation:** Logistic Regression has the strongest mean training CV F1 and offers signed coefficients for explanation. Random Forest is a strong alternative and achieved the highest final test F1. The small CV difference is not evidence of a statistically established winner. Higher precision reduces wasted contacts; higher recall reaches more actual responders. Contact costs, revenue, and campaign capacity should guide the eventual targeting policy.

Additional saved evaluations include accuracy, macro F1, confusion matrices, classification reports, training performance, and majority/stratified dummy baselines. No warnings were captured in the four successful classifiers' main CV and final training fits.

## 5. Independent K Means results

K Means uses 16 standardized numerical income, age, household, spending, purchasing, and recency features. `Response`, `ID`, and all campaign acceptance features are excluded from fitting and cluster-count selection.

Cluster counts 2–8 were evaluated with `n_init=20` and seed 42. The selected solution is **k=2**, with silhouette **0.3038**, the highest eligible score. Every cluster must contain at least 5% of customers.

| Segment | Customers | Share | Mean income | Observed response rate | Interpretation |
|---|---:|---:|---:|---:|---|
| Cluster 0 | 1,304 | 58.3% | 38,257 | 10.0% | Lower spending and purchase frequency; more young children on average |
| Cluster 1 | 933 | 41.7% | 70,988 | 21.8% | Higher spending and purchase frequency |

Income profile means exclude the restored missing values; clustering uses median-imputed incomes. Currency units follow the source data. Response rates were calculated only after segmentation and are descriptive associations, not independent estimates of future campaign success. Cluster identifiers are arbitrary.

The separate [K Means explanation](kmeans/RESULTS_EXPLANATION.md) provides the candidate scores, detailed profiles, interpretation, and limitations.

## 6. Where the results are stored

### Shared classification partition

Directory: `experiments/outputs/shared/287480a9a9870a43ec7ed3f984324cd844fdec9d9bacdbe40519fb04640b5094/`

| File | Contents |
|---|---|
| `data_audit.json` | Data validation and restored-missing-income counts |
| `split.json` | Machine-readable training, test, and CV indices |
| `assignments.csv` | Customer-level partition and validation-fold membership |

In `assignments.csv`, `row_index` is the zero-based processed-data row position, `ID` is the customer identifier, and `partition` is `train` or `test`. For training customers, `fold` 0–4 identifies the fold in which they serve as validation observations; they train in the other four folds. Test customers have `fold=-1` and do not participate in CV or threshold selection.

### Successful experiment runs

The following paths are relative to **`experiments/outputs/`**. Links open each run's metadata or comparison table.

| Experiment | Saved run |
|---|---|
| Logistic Regression | [logistic_regression/run_20261013_185446_609612_1216ba4a/](outputs/logistic_regression/run_20261013_185446_609612_1216ba4a/run_metadata.json) |
| Decision Tree | [decision_tree/run_20261013_185500_256885_7761ad21/](outputs/decision_tree/run_20261013_185500_256885_7761ad21/run_metadata.json) |
| Random Forest | [random_forest/run_20261013_185521_750053_673e200e/](outputs/random_forest/run_20261013_185521_750053_673e200e/run_metadata.json) |
| Gradient Boosting | [gradient_boosting/run_20261013_185624_058701_3f073f00/](outputs/gradient_boosting/run_20261013_185624_058701_3f073f00/run_metadata.json) |
| K Means | [kmeans/run_20261013_185820_218800_afad3148/](outputs/kmeans/run_20261013_185820_218800_afad3148/run_metadata.json) |
| Current classifier comparison | [comparison/run_20261014_235704_874397_c68de835/](outputs/comparison/run_20261014_235704_874397_c68de835/model_comparison.csv) |

Run names use UTC timestamps and configuration identifiers. Each training invocation creates a separate directory rather than overwriting earlier results. Use the successful paths above for reporting. The repository-root `outputs/logistic_regression/run_20261013_185401_860960_f3ed22a6/` is a retained failed path-resolution attempt, not a successful model result.

### Files to use within a classification run

| Need | Files |
|---|---|
| Trained pipeline and selected threshold | `model.joblib` |
| Exact parameters, environment, seed, data hashes, and execution time | `config.json`, `run_metadata.json` |
| CV mean, sample SD, and per-fold metrics | `cv_metrics.json` |
| Threshold-development results | `threshold_analysis.csv`, `threshold_development.json`, `oof_predictions.csv` |
| Final test metrics and individual predictions | `test_metrics.json`, `test_predictions.csv`, `classification_report.txt` |
| Evaluation figures | `confusion_matrix.png`, `precision_recall_curve.png` |
| Model explanations | `feature_importance.csv`, `feature_importance.png`, `permutation_importance.csv`, `model_behavior.md` |
| Captured training warnings | `warnings.json` |
| Readable fitted tree, for Decision Tree only | `decision_tree.png` |

Feature importance distinguishes signed Logistic Regression coefficients from tree-based impurity importance. Permutation importance is calculated on a held-out training fold, separately from the independent test set. Correlated features can redistribute or suppress importance.

K Means stores `model.joblib`, `config.json`, `run_metadata.json`, `selection.json`, `cluster_metrics.csv`, `cluster_assignments.csv`, and `cluster_profiles.csv`. Its figures are `silhouette_scores.png`, `cluster_size_distribution.png`, and `cluster_response_rates.png`. The standalone narrative is `experiments/kmeans/RESULTS_EXPLANATION.md`.

The comparison run stores `model_comparison.csv`, `model_comparison.png`, and `model_selection_summary.md`, plus configuration and metadata. It reads saved classifier results and does not fit models.

## 7. How to reproduce or inspect the work

Run from the repository root. The current setup uses Python 3.14 and local dependencies because the pre-existing virtual environment points to an inaccessible interpreter.

```powershell
# Install dependencies when setting up a new checkout.
python -m pip install --target .experiment_dependencies -r experiments/requirements.txt
$env:PYTHONPATH="$PWD\.experiment_dependencies"

# Establish or reuse the shared classification partition.
python -m experiments.common.split_manager

# Each model is independently executable.
python -m experiments.logistic_regression.train
python -m experiments.decision_tree.train
python -m experiments.random_forest.train
python -m experiments.gradient_boosting.train
python -m experiments.kmeans.train

# Read saved results without retraining.
python -m experiments.comparison.compare_classifiers
python -m experiments.kmeans.explain_results
python -m experiments.verify_outputs
```

Each training entry point accepts `--config PATH`. The comparison defaults to the latest completed run per classifier and checks compatibility; `--runs LR_RUN DT_RUN RF_RUN GB_RUN` selects explicit run directories in that order. The explanation command accepts `--run PATH` for a specific K Means run. Keep the paths in this report when reproducing the exact reported comparison rather than a newer experiment.

Verification has checked artifact existence and nonempty content, JSON/CSV parsing, image decoding, saved-model loading, prediction/threshold consistency, identical split identifiers, and cluster population totals. The saved classifier feature lists also confirm that cluster labels are absent.

## 8. Remaining work for the team

1. Confirm that campaign history, spending, purchases, recency, and complaints would all be known before the offer being predicted. Confirm a defensible date reference.
2. Review coefficients, tree rules, and permutation importance, accounting for correlated aggregate and component features.
3. Review segment profiles with the business team and assess cluster stability across resampling and alternative feature choices.
4. Define campaign contact costs, expected return, and capacity before choosing an operational threshold or segment-specific offer.
5. Use nested CV or fresh holdout data for further model tuning; the reported test set has already been evaluated.
6. Use the successful run paths and saved figures above when preparing the Milestone 2 presentation and report.

For further details, see [README](README.md), [verified execution summary](EXECUTION_SUMMARY.md), and [K Means interpretation](kmeans/RESULTS_EXPLANATION.md).
