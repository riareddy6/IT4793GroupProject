# Milestone 3 completion report

Implementation and validation date: November 5, 2026. This is a repository deliverable; no external website was published. Historical raw/processed data and pinned Milestone 2 training artifacts are preserved.

## A. Completed work

- Audited repository source, configurations, historical evidence, website placeholders and evaluation risks before editing. See [initial audit](MILESTONE3_AUDIT.md).
- Restored ROC AUC in the common training metrics and per-model ROC export. Standardized fixed-label confusion matrices and validated metric inputs.
- Added a pinned report builder that verifies source hashes, IDs, common split/folds, saved probabilities, OOF scores, selected thresholds, fitted importance, K selection and cluster profiles before exporting results.
- Exported comparison/profile CSVs, dashboard JSON, artifact hashes and **16** presentation-ready 300-DPI figures: combined ROC/PR/F1, four confusion matrices, four threshold curves, four importance charts and one cluster chart.
- Replaced the existing website's pending results with actual evidence using the same HTML/CSS/JavaScript framework, responsive navigation, model selection, training-OOF threshold slider, segment profiles, recommendation hypotheses, CSV download and invalid/missing-data recovery messages.
- Added architecture, function reference, model evaluation, dashboard and business documentation; updated the root and experiment READMEs.
- Added focused regression tests and a full-workflow replay harness. No hyperparameter tuning or test-threshold search was introduced.

## B. Verified model results

All classification metrics below are positive-class metrics. CV uses threshold 0.50; holdout decisions use frozen training-OOF-selected thresholds. Probability metrics use continuous scores.

| Model | Threshold | CV F1 | Test accuracy | Precision | Recall | Test F1 | ROC AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.60 | 0.5756 | 0.8438 | 0.4842 | 0.6866 | 0.5679 | 0.8914 | 0.6160 |
| Decision Tree | 0.75 | 0.4278 | 0.8237 | 0.4118 | 0.4179 | 0.4148 | 0.7778 | 0.4002 |
| Random Forest | 0.50 | 0.5628 | 0.8683 | 0.5526 | 0.6269 | 0.5874 | 0.8864 | 0.6205 |
| Gradient Boosting | 0.25 | 0.5107 | 0.8482 | 0.4935 | 0.5672 | 0.5278 | 0.8912 | 0.6268 |

Random Forest is the strongest observed test-F1 model; Logistic Regression remains the training-CV analytical benchmark and highest-recall alternative. No statistically established winner is claimed. Values are computed in [model_comparison.csv](milestone3/model_comparison.csv); this rounded completion table summarizes that generated evidence.

K-Means: K=2, silhouette 0.3037603760; clusters contain 1,304 and 933 customers. Response counts are 131 and 203, producing **10.0460% and 21.7578%**. The task reference 17.2% is inconsistent with saved assignments and is not used. Labels and source records were not changed.

## C. Refactoring and preservation

Extracted common plot saving, completed-run discovery and threshold selection; centralized shared settings; removed dynamic pathlib importing and duplicated threshold-selection/discovery blocks. K-Means no longer imports the training/evaluation module solely for matplotlib saving. Comparison now derives metrics from saved probabilities instead of omitting historical ROC fields. Old website placeholder markup/styles were replaced by working views, without adding a new framework or web dependency.

All existing Python functions, including Milestone 1 and historical audit helpers, received behavior-specific documentation. No historical training source artifacts were deleted. Empty notebook scaffolding and failed-run records remain documented historical evidence. No gratuitous reorganization, estimator changes or feature-definition changes were performed.

## D. Documentation coverage

All **66** project Python functions, nested helpers and tests have been reviewed/documented and checked through AST inspection for purpose, Args, Returns, Processing and Raises, with **no coverage exceptions**. The generated [function reference](FUNCTION_REFERENCE.md) records signatures and statically observed call dependencies. Installed libraries and the empty notebook contain no owned active functions to document. JavaScript presentation helpers have descriptive JSDoc comments.

## E. Validation results

Executed successfully:

- Nine focused Python unittest cases: metrics and ROC/AP agreement, confusion orientation, threshold inclusivity/tie-breaking, invalid inputs, fold-local imputation, unseen categories, original data/split/fold coverage, config validation, strict serialization, generated-bundle consistency and every-function documentation coverage.
- Full Milestone 1 pipeline redirected to temporary output: 2,237 processed records, eight EDA figures, regenerated CSV hash exactly matches historical processed source.
- Four classifier training entry points: historical test IDs, selected thresholds and prediction classes match; maximum probability differences are zero for LR/DT/GB and 3.33e-16 for RF.
- K-Means training entry point: assignments exactly match original records.
- All four saved permutation importance means and SDs independently reproduced on the first training validation fold; selected silhouette independently recomputed.
- Classifier comparison, artifact verification and K-Means explanation entry points executed successfully. Original source-run hashes verified unchanged before and after replay.
- Verified reporting builder executed; combined curves use sklearn roc_curve and precision_recall_curve. Node syntax check passed for dashboard.js.
- Four Node logic tests against a minimal DOM double passed: actual data rendering, all four model selections/OOF slider endpoints, missing-data recovery and malformed-metric rejection. These are logic tests, not browser rendering tests.
- Static QA compiled 34 Python files, imported 29 experiment modules, decoded all 16 generated PNGs and checked documentation/dashboard resource links. Combined ROC and RF confusion figures were visually inspected as standalone PNGs.
- Local HTTP server returned status 200 for the dashboard homepage. No external hosting was performed.

Full measured replay evidence is in [workflow_validation.json](milestone3/workflow_validation.json). Static/logic QA evidence is in [quality_checks.json](milestone3/quality_checks.json). Reproduce supplemental checks with `node --test tests/dashboard.test.cjs` and `python -m scripts.check_static` (Node 24+ is optional for development QA, not required to view the dashboard).

**Browser verification limitation:** the browser access host was unavailable. The native Windows fallback then stopped because the tool could not reliably determine the current browser URL for policy enforcement. No further UI input was attempted. Actual browser rendering, dropdown/slider operation and mobile layout are therefore not claimed as verified. Static/logic checks supplement but do not replace that outstanding browser QA.

The original audit builder is syntax-checked/documented but deliberately not executed because it overwrites the user's earlier audit package. The empty notebook has no cells to execute.

## F. Significant files

| Created / modified | Files |
|---|---|
| Shared source | `experiments/common/settings.py`, `artifacts.py`, `visualization.py`; updated loading, validation, preprocessing, split, persistence and evaluation modules |
| Entry points | Four classifier train.py files, K-Means train/explanation, comparison and artifact verification; documentation additions preserve established behavior |
| Milestone 1 | `docs/milestone1/milestone1_analysis.py` docstrings; processed CSV and historical EDA unchanged |
| Reporting and QA | `experiments/milestone3/{sources.json,build.py,documentation.py,validate_workflow.py}`, `tests/test_pipeline.py`, `tests/dashboard.test.cjs`, `scripts/check_static.py` |
| Dashboard | `docs/index.html`, `docs/dashboard.css`, `docs/dashboard.js` |
| Documentation | Root/experiment READMEs; `docs/{MILESTONE3_AUDIT,ARCHITECTURE,FUNCTION_REFERENCE,MODEL_EVALUATION,DASHBOARD_GUIDE,BUSINESS_RECOMMENDATIONS,MILESTONE3_COMPLETION}.md` |
| Dependencies | Root `requirements.txt`, `.gitignore`; seaborn installed in existing local dependency directory |
| Evidence | `docs/milestone3/`, fresh timestamped replay runs under `experiments/outputs/`, refreshed `experiments/EXECUTION_SUMMARY.md` |
| Historical audit | `milestone2_ppt_support/build_audit.py`: docstrings only; previous report/CSV/PNG/ZIP retained |

No tracked source/data artifacts were removed. Temporary documentation-migration tooling was removed after use.

## G. Remaining issues

1. Actual browser and mobile visual QA remains unverified for the environment reason above.
2. Domain confirmation of feature timing and reference date remains required; existing historical convention is preserved and disclosed.
3. Hyperparameter search, repeated-split uncertainty/significance, cluster stability, probability calibration and external validation are not existing project results and were not fabricated.
4. Contact costs, capacity, incremental response and revenue are unmeasured; recommendations require controlled campaign testing.

The repository now provides the requested evaluation, dashboard implementation, documentation and reproducibility artifacts. The limitations above remain explicitly identified rather than treated as completed experimental evidence.
