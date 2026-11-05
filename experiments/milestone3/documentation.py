"""Generate evidence-based evaluation prose and a complete AST function reference."""
import ast
import inspect
from pathlib import Path
from experiments.common.data_loader import PROJECT


def source_files():
    """List all project Python sources while excluding installed dependencies.

    Args:
        None: Uses the repository root derived from this module.

    Returns:
        list[Path]: Sorted experiment, report, test and tooling Python files.

    Processing:
        Search owned source directories and the historical audit support script.

    Raises:
        OSError: If source directories are inaccessible. No files are changed.
    """
    files = []
    for directory in ('experiments', 'docs', 'tests', 'scripts', 'milestone2_ppt_support'):
        files.extend((PROJECT/directory).rglob('*.py'))
    return sorted(p for p in files if '__pycache__' not in p.parts)


def write_reference():
    """Generate the function catalog and reject incomplete function docstrings.

    Args:
        None: Reads Python sources returned by source_files.

    Returns:
        int: Total documented functions, including nested and test functions.

    Processing:
        Parse ASTs without importing executable scripts; include signatures,
        docstrings and statically observed call dependencies for each function.

    Raises:
        ValueError: If any function lacks Args, Returns, Processing or Raises.
        OSError: If source reads or docs/FUNCTION_REFERENCE.md writing fails.
    """
    lines = ['# Function reference', '', 'Generated with `python -m experiments.milestone3.documentation`. Includes every project Python function, nested helper and test; installed dependencies are excluded. Purpose is the opening docstring sentence. Call lists are syntactic dependencies, not a runtime call graph.', '']
    count = 0
    for path in source_files():
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        functions = sorted((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), key=lambda n: n.lineno)
        if not functions:
            continue
        relative = path.relative_to(PROJECT).as_posix()
        lines.extend([f'## {relative}', ''])
        for node in functions:
            doc = ast.get_docstring(node) or ''
            if not all(section in doc for section in ('Args:', 'Returns:', 'Processing:', 'Raises:')):
                raise ValueError(f'Incomplete documentation: {relative}:{node.lineno} {node.name}')
            calls = sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
            lines.extend([f'### `{node.name}({ast.unparse(node.args)})`', '', f'[Source]({"../" + relative}#L{node.lineno})', '',
                          inspect.cleandoc(doc), '', '**Dependencies / calls:** ' + ', '.join(f'`{c}`' for c in calls) + '.', ''])
            count += 1
    lines.insert(4, f'Total documented functions: **{count}**. No docstring coverage exceptions.')
    (PROJECT/'docs/FUNCTION_REFERENCE.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Documented {count} functions')
    return count


def write_evaluation(bundle, path):
    """Write model evaluation and marketing interpretation from verified values.

    Args:
        bundle (dict): Verified overview, comparison, model and clustering evidence.
        path (Path): Destination MODEL_EVALUATION.md document.

    Returns:
        None: Writes metric definitions, actual results, charts and limitations.

    Processing:
        Derive metric winners, comparison rows and confusion-based decisions
        from the evidence bundle; label CV, holdout and OOF development policies.

    Raises:
        KeyError: If required evidence is absent.
        OSError: If documentation cannot be written.
    """
    o = bundle['overview']; rows = bundle['comparison']
    best_f1 = max(rows, key=lambda r: r['test_f1'])
    best_p = max(rows, key=lambda r: r['test_precision'])
    best_r = max(rows, key=lambda r: r['test_recall'])
    best_cv = max(rows, key=lambda r: r['cv_f1'])
    lines = ['# Model evaluation', '', 'Generated from pinned, verified experiment artifacts. No model is tuned or retrained by this report command.', '',
             '## Data and experimental design', '',
             f"Raw Kaggle Customer Personality Analysis: 2,240 rows, 29 columns. Processed: {o['rows']:,} rows, {o['columns']} columns. Response=1 means acceptance of the current campaign: {o['responders']} responders and {o['nonresponders']} nonresponders ({o['prevalence']:.2%} positive).",
             f"Shared stratified 80/20 split: {o['train_rows']} train, {o['test_rows']} test; {o['cv_folds']} shuffled stratified training folds; seed 42. Split ID: `{bundle['split_id']}`. All four classifiers use identical underlying 28 predictors (39 transformed columns in the final fitted models).", '',
             'Income missing/sentinel cells are restored from raw IDs before modeling. Median numeric imputation, most-frequent categorical imputation and one-hot encoding are fitted within each training fold. Only Logistic Regression scales numeric predictors. IDs, target and two constants are excluded; birth year/date become age/tenure using 2014-12-31. No feature selection, SMOTE or cluster labels are used.', '',
             '## Algorithms and exact configured parameters', '']
    for row in rows:
        lines.append(f"- **{row['label']}**: `{bundle['models'][row['model']]['parameters']}`.")
    lines += ['', '## Evaluation policies and definitions', '',
              '- **Primary KPI: positive-class F1**, the harmonic mean of precision and recall. It balances missed responders against wasted contacts without being dominated by true negatives. It does not encode financial costs.',
              '- **Accuracy**: (TP+TN)/N, overall correct fraction. Majority-only test accuracy is 381/448 = 85.04%, with positive-class F1=0; accuracy alone is misleading.',
              '- **Precision**: TP/(TP+FP), the fraction of targeted customers who actually respond.',
              '- **Recall**: TP/(TP+FN), the fraction of actual responders reached.',
              '- **ROC AUC**: probability-ranking discrimination over all thresholds, based on false-positive rate versus true-positive rate; random ranking is 0.50.',
              '- **Average Precision (AP)**: sklearn average_precision_score, the sum of precision weighted by increments in recall. This is not trapezoidal PR AUC. The PR no-skill reference is test positive prevalence.',
              '- **Confusion matrix**: rows actual [0,1], columns predicted [0,1], yielding [[TN,FP],[FN,TP]]. FP means contacting a nonresponder; FN means failing to contact an observed responder.',
              '- **CV**: fold metrics at threshold 0.50; mean and sample SD (ddof=1), not confidence intervals.',
              '- **Threshold development**: 17 thresholds, 0.10–0.90 by 0.05, evaluated on pooled training OOF probabilities. Maximize F1, then precision, then proximity to 0.50. This selection reuses OOF labels, so development scores are optimistic.',
              '- **Holdout**: thresholds frozen before test evaluation. Test accuracy/precision/recall/F1 below use selected thresholds. ROC AUC and AP use probabilities and do not change with threshold.', '',
              '## Verified comparison', '',
              '| Model | Threshold | CV F1 ± SD (0.50) | Test accuracy | Test precision | Test recall | Test F1 | Test ROC AUC | Test AP |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['threshold']:.2f} | {r['cv_f1']:.4f} ± {r['cv_f1_sd']:.4f} | {r['test_accuracy']:.4f} | {r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1']:.4f} | {r['test_roc_auc']:.4f} | {r['test_average_precision']:.4f} |")
    lines += ['', '## Interpretation and campaign decisions', '',
              f"Highest observed test F1: **{best_f1['label']} ({best_f1['test_f1']:.4f})**. Highest selected-threshold precision: **{best_p['label']} ({best_p['test_precision']:.2%})**. Highest selected-threshold recall: **{best_r['label']} ({best_r['test_recall']:.2%})**. The training-CV benchmark remains **{best_cv['label']} ({best_cv['cv_f1']:.4f})**. The test leaderboard is descriptive, not a new tuning criterion.", '']
    for r in rows:
        cm = bundle['models'][r['model']]['confusion_matrix']
        lines.append(f"- {r['label']}: contact {cm[0][1]+cm[1][1]} customers; {cm[1][1]} observed responders reached, {cm[0][1]} nonresponders contacted, {cm[1][0]} responders missed, {cm[0][0]} nonresponders correctly excluded.")
    lines += ['', 'Higher precision suits limited contact capacity; higher recall suits broader outreach when missing a responder is expensive. ROC and PR curves show ranking quality and tradeoffs beyond a single cutoff. PR emphasizes positive-class retrieval under imbalance. AP and ROC AUC can rank models differently from F1. Evaluate future contact-cost, benefit and capacity policies on training/development data, then validate on fresh outcomes.', '',
              '## Figures and interpretation methods', '',
              '![ROC](milestone3/roc_comparison.png)', '![Precision Recall](milestone3/pr_comparison.png)', '![F1](milestone3/f1_comparison.png)', '',
              'The dashboard permits model selection and a training-OOF threshold slider. The slider never recomputes or chooses thresholds using holdout labels. Every model has confusion, threshold and importance PNGs in `docs/milestone3/`.', '',
              'Logistic Regression signed coefficients apply to standardized numeric inputs and unscaled category indicators. Tree importance is normalized impurity reduction and can favor high-cardinality predictors. Saved permutation importance uses the first training validation fold, F1 at 0.50, five repeats and seed 42; it never uses final test records. Correlated totals/components can redistribute importance. None of these methods establishes causality.', '',
              '## Segmentation verification', '',
              f"Selected K={bundle['clusters']['k']}; silhouette={bundle['clusters']['silhouette']:.4f}, highest among eligible K=2–8, minimum cluster share 5%. All 16 numeric inputs are median-imputed and standardized on the full customer set. Response and campaign histories are excluded from fitting/selection. Classification does not consume these clusters.", '']
    for p in bundle['clusters']['profiles']:
        lines.append(f"- Cluster {int(p['cluster'])}: {int(p['size'])} customers, {int(p['responders'])} responders, observed rate {p['response_rate']:.4%}; mean income {p['Income']:,.0f}, mean total spending {p['Total_Spending']:,.1f}.")
    lines += ['', 'The task reference of approximately 17.2% for the second cluster does not match complete saved customer assignments. The rates above are recalculated from those assignments; no source records or cluster labels were changed. Income profile means exclude originally missing incomes; clustering uses imputed values.', '',
              '## Limitations and reproduction', '',
              'Only 67 positives are in the holdout. No repeated-split uncertainty, paired significance test or clustering stability experiment establishes a reliable winner. Fixed hyperparameters were not searched. Pre-offer availability of all campaign/spending features and the fixed 2014 date require confirmation; Milestone 1 excluded births before 1926 using a 2026 convention. Retained for comparability. Segments are full-data descriptive associations; neither future response nor revenue/ROI improvement has been measured. No operational decision should treat feature effects as causal.', '',
              'Run `python -m experiments.milestone3.build` from the repository root after installation. It checks source hashes, IDs, common split/folds, saved-model predictions, threshold-grid scores and selection, fitted feature importance, cluster assignments and profiles. Sources are pinned in `experiments/milestone3/sources.json`; hashes are exported to `milestone3/provenance.json`. It recomputes reporting from saved probabilities without refitting. Training commands are in the root README. Further tuning must use training-only validation, nested CV or fresh holdout data.']
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    write_reference()
