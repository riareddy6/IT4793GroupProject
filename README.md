# IT4793 Customer Personality Analysis and Marketing Response Prediction

This project combines customer exploration, four classifiers predicting marketing acceptance (`Response`) and independent K-Means segmentation. Milestone 3 integrates verified model evaluation, a static interactive dashboard, reproducible workflows, tests and function-level documentation.

The raw Kaggle Customer Personality Analysis file contains **2,240 observations × 29 columns**. Historical cleaning retains **2,237 × 33**, including engineered fields: **334 responders** and **1,903 nonresponders**. Classifiers share **1,789 training / 448 test** records, five stratified training folds and seed **42**. There are 28 underlying model predictors (39 encoded columns in final fitted pipelines). Positive-class **F1** is the primary KPI.

## Verified findings

Random Forest has the highest observed selected-threshold test F1 (**0.5874**); Logistic Regression has the strongest mean CV F1 (**0.5756**) and selected-threshold test recall (**68.66%**). CV uses threshold 0.50; test scores use training-OOF-selected thresholds. Differences do not establish a statistically reliable winner. K-Means selects K=2, silhouette **0.3038**, sizes **1,304 / 933**, observed response rates **10.05% / 21.76%**. The task reference 17.2% does not match complete saved records.

## Structure

```text
data/                         Original immutable customer TSV
docs/milestone1/              Cleaning, feature engineering, processed data, EDA
experiments/common/           Loading, transforms, split, metrics, plots, persistence
experiments/<model>/           Four classifier and independent K-Means entry points
experiments/outputs/           Retained and fresh timestamped runs; shared folds
experiments/milestone3/        Pinned report sources, verification/export, documentation
docs/milestone3/              Generated CSV/JSON/PNG and validation evidence
docs/index.html               Dashboard (dashboard.css / dashboard.js)
tests/                        Focused automated regression tests
milestone2_ppt_support/        Retained historical audit sources and artifacts
```

## Installation

Run from the repository root in PowerShell. Python **3.14** with the pinned library versions matches the historical fitted models:

```powershell
python -m venv .venv-m3
.\.venv-m3\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The existing `.venv` may point to an unavailable interpreter; use a new environment. This checkout also supports the existing isolated dependency directory:

```powershell
$env:PYTHONPATH="$PWD\.experiment_dependencies"
```

Root requirements include the pinned experimental stack plus seaborn for existing EDA, Pillow for image verification and threadpoolctl for clustering. No Streamlit, web framework, JavaScript package or CDN is required.

## Reproduce evaluation and launch the dashboard

```powershell
# Verify pinned historical runs; regenerate complete evaluation, CSVs and 300-DPI figures.
python -m experiments.milestone3.build

# Check focused regression tests and regenerate the function reference.
python -m unittest discover -s tests -v
python -m experiments.milestone3.documentation

# Optional development QA (Node 24+, no npm packages).
node --test tests/dashboard.test.cjs
python -m scripts.check_static

# Serve docs, then open http://127.0.0.1:8000
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Stop the server with Ctrl+C. A static host can serve `docs/` unchanged. The report builder loads trusted local joblib models and verifies their predictions; it does **not** refit or tune models. Sources are pinned in `experiments/milestone3/sources.json`, rather than implicitly changing when new runs appear. Dashboard threshold exploration displays saved **training OOF** scores; it never tunes on test labels.

## Run the original pipelines

```powershell
# Explicitly regenerates named Milestone 1 outputs; raw source is unchanged.
python docs/milestone1/milestone1_analysis.py

python -m experiments.common.split_manager
python -m experiments.logistic_regression.train
python -m experiments.decision_tree.train
python -m experiments.random_forest.train
python -m experiments.gradient_boosting.train
python -m experiments.kmeans.train

# Compare newest completed compatible classifier runs and inspect artifacts.
python -m experiments.comparison.compare_classifiers
python -m experiments.kmeans.explain_results
python -m experiments.verify_outputs
```

Each training command supports `--config PATH` and creates a **new** timestamped run. No retained training run is overwritten. Comparison supports `--runs LR_RUN DT_RUN RF_RUN GB_RUN` for explicit sources. K-Means explanation accepts `--run PATH --output PATH`.

For a comprehensive replay that validates Milestone 1 in a temporary directory, all four classifier entry points, K-Means, permutation importance and silhouette:

```powershell
python -m experiments.milestone3.validate_workflow
```

This creates new timestamped training/comparison runs and refreshes the original execution summary. `--skip-training` skips the five full training entry points but still validates EDA and interpretation. Results are written to `docs/milestone3/workflow_validation.json` only after checks pass. The historical audit builder is retained for provenance and is not required by the production/reporting workflow.

## Output locations and documentation

- Model pipelines, probabilities, OOF scores, threshold tables and importance: `experiments/outputs/<model>/run_*/`.
- Shared train/test/CV membership: `experiments/outputs/shared/<split_id>/`.
- Unified report: [evaluation methods and verified results](docs/MODEL_EVALUATION.md).
- Presentation assets: `docs/milestone3/` (comparison/profile CSVs; combined ROC/PR/F1, four confusion/threshold/importance plots, cluster plot; dashboard JSON and provenance hashes).
- [Initial audit](docs/MILESTONE3_AUDIT.md), [architecture](docs/ARCHITECTURE.md), [function reference](docs/FUNCTION_REFERENCE.md), [dashboard guide](docs/DASHBOARD_GUIDE.md), [business recommendations](docs/BUSINESS_RECOMMENDATIONS.md), [completion and QA](docs/MILESTONE3_COMPLETION.md).
- Original [Milestone 2 handoff](experiments/Oct14_HANDOFF_SUMMARY.md) and [evidence audit](milestone2_ppt_audit.md) remain historical records.

## Reproducibility and limits

Preprocessing is fitted within training folds. Originally missing/sentinel Income values are restored before median imputation. Thresholds are selected solely from training OOF probabilities. Probability-ranking metrics are separate from threshold-dependent decisions. K-Means uses all customers independently, excluding Response from fitting/selection, and its labels never feed the classifiers.

Historical model parameters, features, split, thresholds and cluster assignments are preserved. Fixed 2014 modeling dates and 2026 birth-year cleaning rules remain disclosed conventions; pre-offer data availability needs confirmation. No new hyperparameter search, significance test, cluster stability experiment or revenue/ROI improvement is claimed. Future tuning needs training-only validation/nested CV and fresh confirmation outcomes.

## Team

| Member | Responsibility |
|---|---|
| Calvin Abraham | Team Lead, Business Insights & Presentation |
| Stephen Newton | Data Preparation & Exploratory Analysis |
| Ria Reddy | Predictive Modeling & Model Evaluation |
| Meilun Wu | Documentation & Report Editing |
| Brooklynn Daniels | Analysis & Project Website |

Instructor: Arindam Khanda. Data source: Kaggle **Customer Personality Analysis**, retained as `data/marketing_campaign.csv`.
