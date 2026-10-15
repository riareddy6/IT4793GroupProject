# Verified Milestone 2 execution

Original project files were preserved. Created `experiments/` with six shared utility modules, five independent training modules and JSON configurations, result comparison, pinned requirements, documentation, verification and separate timestamped outputs. Local dependencies are in `.experiment_dependencies/`.

## Exact execution commands

From repository root in PowerShell:

```powershell
$env:PYTHONPATH="$PWD\.experiment_dependencies"
python -m experiments.common.split_manager
python -m experiments.logistic_regression.train
python -m experiments.decision_tree.train
python -m experiments.random_forest.train
python -m experiments.gradient_boosting.train
python -m experiments.kmeans.train
python -m experiments.comparison.compare_classifiers
python -m experiments.verify_outputs
```

## Classification results

| Model | CV F1 mean ± SD (.50) | Test F1 (.50) | Threshold | Test precision | Test recall | Test F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5756 ± 0.0226 | 0.5426 | 0.60 | 0.4842 | 0.6866 | 0.5679  | 0.6160 |
| decision_tree | 0.4278 ± 0.0410 | 0.4403 | 0.75 | 0.4118 | 0.4179 | 0.4148  | 0.4002 |
| random_forest | 0.5628 ± 0.0301 | 0.5874 | 0.50 | 0.5526 | 0.6269 | 0.5874  | 0.6205 |
| gradient_boosting | 0.5107 ± 0.0438 | 0.4898 | 0.25 | 0.4935 | 0.5672 | 0.5278  | 0.6268 |

Provisional recommendation by training CV F1: **logistic_regression**. See the [selection summary](outputs/comparison/run_20261014_235704_874397_c68de835/model_selection_summary.md) for marketing and interpretability tradeoffs.

All four classifiers use 1,789 training observations, 448 independent test observations, the same persisted folds, and identical underlying input features (counts listed below). One-hot encoding changes the transformed feature count. Numerical scaling is enabled for Logistic Regression and K Means. Exact estimator parameters are in each run config.

## Saved runs

- `logistic_regression`: [run_20261013_185446_609612_1216ba4a](outputs/logistic_regression/run_20261013_185446_609612_1216ba4a/run_metadata.json), 2237 rows, 28 input features.
  Training/test: 1789/448; captured warnings: 0.
- `decision_tree`: [run_20261013_185500_256885_7761ad21](outputs/decision_tree/run_20261013_185500_256885_7761ad21/run_metadata.json), 2237 rows, 28 input features.
  Training/test: 1789/448; captured warnings: 0.
- `random_forest`: [run_20261013_185521_750053_673e200e](outputs/random_forest/run_20261013_185521_750053_673e200e/run_metadata.json), 2237 rows, 28 input features.
  Training/test: 1789/448; captured warnings: 0.
- `gradient_boosting`: [run_20261013_185624_058701_3f073f00](outputs/gradient_boosting/run_20261013_185624_058701_3f073f00/run_metadata.json), 2237 rows, 28 input features.
  Training/test: 1789/448; captured warnings: 0.
- `kmeans`: [run_20261013_185820_218800_afad3148](outputs/kmeans/run_20261013_185820_218800_afad3148/run_metadata.json), 2237 rows, 16 input features.
- Comparison: [run_20261014_235704_874397_c68de835](outputs/comparison/run_20261014_235704_874397_c68de835/model_comparison.csv).

## Customer segmentation

Standalone interpretation: [K Means results](kmeans/RESULTS_EXPLANATION.md). The classifiers use the processed data directly; cluster labels are not classification inputs.

Selected k=2; 2,237 customers; 16 standardized numerical inputs. All candidate k values, silhouettes, inertia and size constraints are saved. Response was excluded from fitting and selection.

 cluster       Income  Kidhome  Teenhome  Customer_Age   Recency  MntWines  MntFruits  MntMeatProducts  MntFishProducts  MntSweetProducts  MntGoldProds  NumDealsPurchases  NumWebPurchases  NumCatalogPurchases  NumStorePurchases  NumWebVisitsMonth  size  response_rate
       0 38256.832815 0.713190  0.550613     43.718558 48.619632  97.01227   6.039877        35.412577         9.089724          6.089724     22.046779           2.562883         2.898006             0.837423           3.826687           6.478528  1304       0.100460
       1 70987.691145 0.068596  0.444802     47.026795 49.782422 593.28403  54.545552       350.712755        77.262594         56.390139     74.607717           1.996785         5.749196             5.213290           8.544480           3.698821   933       0.217578

Interpretation from original-unit profiles:

 cluster  size  response_rate       Income  MntWines  MntMeatProducts  NumWebPurchases  Kidhome  Teenhome
       0  1304       0.100460 38256.832815  97.01227        35.412577         2.898006 0.713190  0.550613
       1   933       0.217578 70987.691145 593.28403       350.712755         5.749196 0.068596  0.444802

Cluster 0 contains 1,304 customers with lower mean income (38,257), lower wine/meat spending and fewer web purchases; it has more young children on average. Cluster 1 contains 933 customers with higher mean income (70,988), substantially higher spending and more purchases. Their observed response rates are 10.0% and 21.8%. This two-group solution is interpretable as lower- and higher-spending customer segments; these are descriptive labels, not validated targeting rules.

## Warnings, limitations and next steps

The original virtual environment was inaccessible; dependencies were installed locally. Pip reported conflicts with globally installed streamlit/sklearn-compat; these experiments do not import those packages and use the isolated local dependency path. An initial path-resolution error was corrected and its failed run retained. Initial sandbox dependency-read failures occurred before model training. Failed records: outputs\logistic_regression\run_20261013_185401_860960_f3ed22a6\error.txt

Threshold OOF results are development estimates; the chosen threshold is evaluated once on frozen test probabilities. Models are provisionally ranked by training CV F1, not test accuracy. Historical feature timing and the fixed 2014 date convention need verification. Existing globally imputed incomes were restored in memory; source files were untouched. Correlated aggregates complicate importance. Segments and their response rates are descriptive; cluster stability and operational interpretability remain to be reviewed.

Next: validate pre-offer data availability, review segment profiles with marketing stakeholders, define cost/capacity objectives, and use nested CV or fresh holdout data for subsequent tuning. See [README](README.md) for preprocessing, configurations, reproduction and limitations.

Historical run artifacts are preserved. Current summaries and new comparisons omit the removed ranking metric and its curve; classifiers were not retrained for this reporting change.

Verification: all required artifacts exist and are nonempty; JSON/CSV parsing, image decoding, joblib loading, prediction-threshold consistency, identical split IDs, and cluster totals passed.