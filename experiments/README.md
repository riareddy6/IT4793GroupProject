# Milestone 2 independent experiments

Run commands from the repository root using Python 3.14. Dependencies were installed locally in `.experiment_dependencies` because the pre-existing `.venv` points to an inaccessible Python executable. No existing dataset, experiment, notebook, website, or report was edited.

```powershell
python -m pip install --target .experiment_dependencies -r experiments/requirements.txt
$env:PYTHONPATH="$PWD\.experiment_dependencies"
python -m experiments.common.split_manager
python -m experiments.logistic_regression.train
python -m experiments.decision_tree.train
python -m experiments.random_forest.train
python -m experiments.gradient_boosting.train
python -m experiments.kmeans.train
python -m experiments.comparison.compare_classifiers
```

Each training entry point owns its estimator construction and accepts `--config PATH`. It runs independently. The shared evaluation utility fits only that supplied estimator. Model artifacts contain a fitted pipeline, feature names, and (for classifiers) the selected threshold. Apply the same deterministic `features()` transformation to new customer rows before pipeline prediction. The comparison reads completed artifacts and never fits a model. `--runs LR_RUN DT_RUN RF_RUN GB_RUN` pins explicit comparison inputs; absent that, the latest completed run per classifier is used and compatibility checked.

## Repository inspection and preprocessing

The existing `docs/milestone1/milestone1_analysis.py`, notebook and reports contain cleaning and EDA, with no saved classification or clustering experiments or partitions. The IDE's old `docs/milestone1_analysis.py` path has moved into `docs/milestone1`. Existing staged moves and edits were preserved.

The cleaned CSV has 2,237 rows and 33 columns. Education and marital status are unencoded categorical strings; numeric features are unscaled. Milestone 1 removed three implausible birth years, replaced one income sentinel (666666), filled 24 missing incomes plus the sentinel with a global median, and added age, total spending, children count and historical acceptance count. The saved data audit records exact dtypes, missingness, class distribution and constant columns.

For Milestone 2, the 25 originally missing/sentinel incomes are restored to missing **in memory by customer ID**, so fold-local median imputation is possible while keeping the processed CSV unchanged. Both source hashes identify this data version. Numerical imputation uses training medians, categorical imputation uses the training mode and nominal encoding uses one-hot encoding with unknown categories ignored. Logistic Regression scales numerical features; trees use the same information without scaling. Each CV fold fits its own preprocessing pipeline.

`ID` is retained only for traceability. Constant Z_CostContact and Z_Revenue are removed after checking their variation. Birth year is represented as age in 2014, replacing the misleading 2026 age derived in Milestone 1; this affine transformation preserves birth-year information. Customer registration date is parsed explicitly as DD-MM-YYYY and converted to tenure at the fixed reference 2014-12-31. This reference is a reproducible historical convention, not a verified offer date. No rows are additionally dropped; rare marital categories and legitimate high incomes are retained. Derived spending and household totals remain alongside their components; resulting correlations affect coefficient and importance interpretation.

AcceptedCmp1 through AcceptedCmp5 and Past_Campaign_Responses are retained for classification on the assumption they predate the current Response campaign. Their availability and the timing of spending, recency, complaints and purchase counts must be verified before operational prediction; the dataset alone does not establish temporal ordering. Response is excluded from all inputs. Clustering uses 16 explicit numerical income, age, household, spending, purchase and recency features and excludes every campaign acceptance variable and its derived total. It omits redundant spending/children totals to avoid doubling their weight.

## Evaluation and preserved runs

All classifiers use seed 42, a stratified 80/20 split and identical shuffled five-fold StratifiedKFold assignments. Shared split indices and customer IDs are under `outputs/shared/<identity>/`. Configuration changes to the initial shared evaluation settings are rejected to avoid misleading comparisons. Full dataset and raw-file SHA256 hashes bind splits to their data version.

Fixed model configurations are in the individual JSON files. Logistic Regression: max_iter 2000, balanced weights, lbfgs. Tree: depth 5, leaf minimum 10, balanced weights. Forest: 300 trees, leaf minimum 3, balanced weights, n_jobs -1. Gradient Boosting: 200 trees, learning rate .05, depth 3, no class_weight. All use seed 42.

CV mean and sample standard deviation describe performance at threshold .50. Thresholds .10 through .90 in .05 steps are assessed using training OOF probabilities; maximum positive-class F1 selects the threshold, with precision and proximity to .50 breaking ties. Those scores are **development performance**, not independent validation. Selection is frozen before obtaining test probabilities once; default and selected thresholds share those frozen test probabilities. Training metrics help assess overfitting. The test report and confusion-matrix chart use the selected threshold; default metrics are separately saved. Most-frequent (majority) and seeded stratified DummyClassifier baselines provide context.

Every run gets a unique UTC timestamp, microseconds and config hash; no previous run is overwritten. Failure writes traceback and failed metadata and does not affect earlier results. Convergence and other captured training warnings are saved. Coefficients/impurity importance use transformed feature names; five-repeat permutation F1 importance uses a held-out training fold, separate from the test set, and original feature names. Correlated features can suppress permutation importance. Tree visualization shows the full depth-limited fitted tree.

K Means uses StandardScaler, n_init 20 and seed 42 on all customers, independently of supervised partitions. Counts 2–8 are assessed without Response. Select maximum silhouette subject to a minimum 5% cluster size, favoring smaller k on ties. Original-unit profiles support interpretation and business review; Response rates are calculated only afterward and are descriptive in-sample associations. Euclidean distance and scaling assumptions should be reviewed, and mixed categories are intentionally excluded in this first segmentation.

See `EXECUTION_SUMMARY.md` for verified results and exact artifact paths. Next steps: confirm temporal feature availability, define campaign contact costs/capacity, examine segment interpretability with stakeholders, and use nested CV or a new holdout for further tuning. Do not repeatedly choose configurations using this already-reported test set.

## Current reporting scope and standalone segmentation explanation

Current classification evaluation and comparison omit ROC AUC and ROC curves. Older saved runs remain unchanged as historical records; regenerating a comparison excludes those legacy metric fields without retraining classifiers. Precision-Recall curves and Average Precision remain included.

The four classifiers consume the Milestone 1 processed CSV directly with shared classification preprocessing; they never read K Means cluster assignments. K Means is an independent segmentation analysis, not a preprocessing stage for classification.

Read [K Means results and interpretation](kmeans/RESULTS_EXPLANATION.md). Regenerate it from saved results without fitting:

```powershell
python -m experiments.kmeans.explain_results
```

Future K Means training runs also save `results_explanation.md` in their own run directory.
