"""Verify frozen experiments and export presentation figures and dashboard data."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from experiments.common.data_loader import PROJECT, load_config, load_data
from experiments.common.preprocessing import features
from experiments.common.result_saver import digest, save_json
from experiments.common.evaluation import metrics, select_threshold
from experiments.common.settings import MODELS, LABELS, THRESHOLDS
from experiments.common.visualization import plt, figure


def read_json(path):
    """Read UTF-8 JSON from a project artifact.

    Args:
        path (Path): Existing JSON file.

    Returns:
        dict | list: Decoded JSON value.

    Processing:
        Read text and parse with the standard JSON decoder.

    Raises:
        OSError: If the file cannot be read.
        JSONDecodeError: If the content is invalid.
    """
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    """Reject inconsistent artifacts even when Python assertions are disabled.

    Args:
        condition (bool): Required invariant.
        message (str): Diagnostic describing the failed invariant.

    Returns:
        None: Continues on a true condition.

    Processing:
        Raise an explicit validation error on a false condition.

    Raises:
        ValueError: If condition is false. No file side effects.
    """
    if not condition:
        raise ValueError(message)


def verify_classifier(name, run, data, config, split, split_id):
    """Verify one historical classifier against records, folds and fitted state.

    Args:
        name (str): Classifier identifier in MODELS.
        run (Path): Completed run directory containing predictions and model.
        data (pd.DataFrame): Processed records with restored missing Income.
        config (dict): Shared resolved data configuration.
        split (dict): Persisted train/test indices and relative CV fold indices.
        split_id (str): Expected shared split directory identifier.

    Returns:
        tuple[dict, dict]: Comparison row and dashboard model evidence.

    Processing:
        Check hashes, ID alignment, saved predictions, fold metrics, OOF grid,
        threshold selection, fitted probabilities and estimator importance.

    Raises:
        ValueError: If any saved artifact disagrees with recomputed evidence.
        OSError: If artifacts are missing. Loads trusted local joblib only;
            does not fit models or modify the source run.
    """
    meta, cv, test = [read_json(run / f) for f in ('run_metadata.json', 'cv_metrics.json', 'test_metrics.json')]
    require(meta['status'] == 'complete' and meta['split_id'] == split_id, f'{name}: status/split mismatch')
    for key, path_key in (('dataset_sha256', 'dataset'), ('raw_sha256', 'raw_dataset')):
        require(meta[key] == digest(Path(config[path_key])), f'{name}: {key} mismatch')
    pred, oof = [pd.read_csv(run / f) for f in ('test_predictions.csv', 'oof_predictions.csv')]
    for frame, indices in ((pred, split['test']), (oof, split['train'])):
        require(frame.row_index.tolist() == indices, f'{name}: row order mismatch')
        require(frame.ID.tolist() == data.iloc[indices].ID.tolist(), f'{name}: ID mismatch')
        require(frame.true_label.tolist() == data.iloc[indices].Response.tolist(), f'{name}: labels mismatch')
    train_set, test_set = set(split['train']), set(split['test'])
    require(not train_set & test_set and train_set | test_set == set(range(len(data))), 'Invalid partition')
    validation = []
    for i, fold in enumerate(split['folds']):
        a, b = set(fold['train']), set(fold['validation'])
        require(not a & b and a | b == set(range(len(oof))), 'Invalid CV fold')
        validation.extend(fold['validation'])
        part = oof.iloc[fold['validation']]
        actual = metrics(part.true_label, part.oof_probability)
        for key, value in cv['folds'][i].items():
            require(np.allclose(actual[key], value, atol=1e-12), f'{name}: fold {i} {key} mismatch')
    require(sorted(validation) == list(range(len(oof))), 'OOF coverage must be exactly once')
    for key in cv['mean']:
        values = [f[key] for f in cv['folds']]
        require(np.isclose(np.mean(values), cv['mean'][key]), f'{name}: CV mean mismatch')
        require(np.isclose(np.std(values, ddof=1), cv['std'][key]), f'{name}: CV SD mismatch')
    grid = pd.DataFrame([dict(threshold=t, **{k: v for k, v in metrics(oof.true_label, oof.oof_probability, t).items()
                                             if k != 'confusion_matrix'}) for t in THRESHOLDS])
    saved_grid = pd.read_csv(run / 'threshold_analysis.csv')
    for key in saved_grid:
        require(np.allclose(grid[key], saved_grid[key], atol=1e-12), f'{name}: threshold grid {key} mismatch')
    threshold = select_threshold(grid)
    require(threshold == test['selected_threshold'], f'{name}: OOF threshold mismatch')
    selected = metrics(pred.true_label, pred.predicted_probability, threshold)
    for key, value in test['selected'].items():
        require(np.allclose(selected[key], value, atol=1e-12), f'{name}: selected {key} mismatch')
    require(np.array_equal(pred.predicted_class, pred.predicted_probability >= threshold), 'Saved classes mismatch')
    artifact = joblib.load(run / 'model.joblib')
    x = features(data, config)
    require(x.columns.tolist() == artifact['feature_columns'] == meta['features'], f'{name}: input features mismatch')
    require(artifact['threshold'] == threshold, f'{name}: artifact threshold mismatch')
    pipeline = artifact['pipeline']
    require(np.allclose(pipeline.predict_proba(x.iloc[split['test']])[:, 1], pred.predicted_probability, atol=1e-12),
            f'{name}: fitted probabilities mismatch')
    importance = pd.read_csv(run / 'feature_importance.csv')
    model = pipeline.named_steps['model']
    values = model.coef_[0] if hasattr(model, 'coef_') else model.feature_importances_
    require(importance.feature.tolist() == pipeline.named_steps['preprocessor'].get_feature_names_out().tolist(), 'Importance names mismatch')
    require(np.allclose(values, importance.importance), 'Importance values mismatch')
    permutation = pd.read_csv(run / 'permutation_importance.csv')
    require(permutation.feature.tolist() == x.columns.tolist(), 'Permutation names mismatch')
    row = dict(model=name, label=LABELS[name], threshold=threshold, cv_f1=cv['mean']['f1'], cv_f1_sd=cv['std']['f1'],
               **{'test_' + k: v for k, v in selected.items() if k != 'confusion_matrix'})
    fpr, tpr, _ = roc_curve(pred.true_label, pred.predicted_probability)
    precision, recall, _ = precision_recall_curve(pred.true_label, pred.predicted_probability)
    detail = dict(label=LABELS[name], threshold=threshold, confusion_matrix=selected['confusion_matrix'],
                  roc=dict(fpr=fpr.tolist(), tpr=tpr.tolist()),
                  pr=dict(precision=precision.tolist(), recall=recall.tolist()),
                  threshold_grid=grid.to_dict('records'), importance=importance.to_dict('records'),
                  permutation=permutation.to_dict('records'), parameters=read_json(run/'config.json')['parameters'])
    return row, detail


def verify_clusters(run, data, config):
    """Reconcile K-Means assignments, profiles, selected K and fitted pipeline.

    Args:
        run (Path): Completed K-Means run directory.
        data (pd.DataFrame): Full restored-income records in original order.
        config (dict): Shared configuration used for data hash verification.

    Returns:
        dict: Selected K, silhouette, candidate scores and enriched profiles.

    Processing:
        Verify hashes, recompute original-unit means and response rates,
        compare fitted assignments and check silhouette-based K selection.

    Raises:
        ValueError: If artifacts disagree. OSError if files cannot be read.
        Loads a trusted local model without refitting or changing labels.
    """
    meta = read_json(run/'run_metadata.json')
    require(meta['status'] == 'complete', 'Incomplete clustering run')
    for key, cfg in (('dataset_sha256', 'dataset'), ('raw_sha256', 'raw_dataset')):
        require(meta[key] == digest(Path(config[cfg])), f'Clustering {key} mismatch')
    assignments = pd.read_csv(run/'cluster_assignments.csv')
    profiles = pd.read_csv(run/'cluster_profiles.csv')
    candidates = pd.read_csv(run/'cluster_metrics.csv')
    c = read_json(run/'config.json')
    x = features(data, c, clustering=True)
    require(assignments.ID.tolist() == data.ID.tolist(), 'Cluster IDs mismatch')
    require(assignments.Response.tolist() == data.Response.tolist(), 'Cluster labels mismatch')
    pipeline = joblib.load(run/'model.joblib')['pipeline']
    require(np.array_equal(pipeline.predict(x), assignments.cluster), 'Fitted cluster assignments mismatch')
    grouped = x.assign(cluster=assignments.cluster.to_numpy()).groupby('cluster')
    actual = grouped.mean()
    actual['size'] = grouped.size()
    actual['response_rate'] = data.assign(cluster=assignments.cluster.to_numpy()).groupby('cluster').Response.mean()
    for key in actual:
        require(np.allclose(profiles.set_index('cluster')[key], actual[key], equal_nan=True), f'Cluster {key} mismatch')
    eligible = candidates[candidates.eligible].sort_values(['silhouette', 'k'], ascending=[False, True])
    selected = eligible.iloc[0]
    require(int(selected.k) == meta['selected_k'] == read_json(run/'selection.json')['k'], 'Selected K mismatch')
    require(read_json(run/'selection.json')['response_used_for_selection'] is False, 'Response used for clustering selection')
    profiles['share'] = profiles['size'] / len(data)
    profiles['responders'] = data.assign(cluster=assignments.cluster.to_numpy()).groupby('cluster').Response.sum().values
    spending = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    profiles['Total_Spending'] = profiles[spending].sum(axis=1)
    return dict(k=int(selected.k), silhouette=float(selected.silhouette),
                profiles=profiles.to_dict('records'), candidates=candidates.to_dict('records'))


def create_plots(table, details, clusters, output):
    """Export combined evaluation and customer-segmentation figures.

    Args:
        table (pd.DataFrame): Verified model comparison rows.
        details (dict): Curves, confusion matrices, OOF grids and importance.
        clusters (dict): Verified cluster profiles and candidate scores.
        output (Path): Existing output directory for 300-DPI PNGs.

    Returns:
        None: Writes ROC, PR, F1, confusion, threshold, importance and cluster PNGs.

    Processing:
        Plot probability curves separately from selected-threshold outcomes;
        label confusion cells TN/FP/FN/TP and OOF curves as development scores.

    Raises:
        OSError: If image export fails. Closes each generated figure.
    """
    colors = ['#246b87', '#b65a47', '#218178', '#92712a']
    for kind in ('roc', 'pr'):
        plt.figure(figsize=(8, 6))
        for color, row in zip(colors, table.to_dict('records')):
            d = details[row['model']]
            if kind == 'roc':
                plt.plot(d['roc']['fpr'], d['roc']['tpr'], color=color, label=f"{d['label']} AUC={row['test_roc_auc']:.4f}")
            else:
                plt.step(d['pr']['recall'], d['pr']['precision'], where='post', color=color,
                         label=f"{d['label']} AP={row['test_average_precision']:.4f}")
        if kind == 'roc':
            plt.plot([0, 1], [0, 1], '--', color='gray', label='Random ranking')
            plt.xlabel('False positive rate'); plt.ylabel('True positive rate (recall)')
        else:
            plt.axhline(details['test_prevalence'], linestyle='--', color='gray', label=f"Test prevalence {details['test_prevalence']:.3f}")
            plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.xlim(0, 1); plt.ylim(0, 1.02); plt.legend(fontsize=9)
        plt.title('Held-out test ' + ('ROC curves' if kind == 'roc' else 'precision–recall curves (AP, not trapezoidal PR AUC)'))
        figure(output/f'{kind}_comparison.png')
    plt.figure(figsize=(9, 5))
    best = table.test_f1.idxmax()
    bars = plt.bar(table.label, table.test_f1, color=['#218178' if i == best else '#8da5b1' for i in table.index])
    plt.bar_label(bars, fmt='%.4f'); plt.ylim(0, 1); plt.ylabel('Positive-class F1')
    plt.title('Held-out test F1 at frozen OOF-selected thresholds'); figure(output/'f1_comparison.png')
    for name in MODELS:
        d = details[name]
        plt.figure(figsize=(6, 5)); plt.imshow(d['confusion_matrix'], cmap='Blues')
        plt.xticks([0, 1], ['Predicted no response', 'Predicted response'], rotation=8)
        plt.yticks([0, 1], ['Actual no response', 'Actual response'])
        for i, j, label in ((0, 0, 'TN'), (0, 1, 'FP'), (1, 0, 'FN'), (1, 1, 'TP')):
            value = d['confusion_matrix'][i][j]
            plt.text(j, i, f'{label}\n{value}', ha='center', va='center', fontsize=18, color='white' if value > 200 else 'black')
        plt.title(f"{d['label']} — threshold {d['threshold']:.2f}"); figure(output/f'confusion_{name}.png')
        grid = pd.DataFrame(d['threshold_grid'])
        plt.figure(figsize=(7, 5))
        for key in ('precision', 'recall', 'f1'):
            plt.plot(grid.threshold, grid[key], marker='.', label=key.title())
        plt.axvline(d['threshold'], color='gray', linestyle='--', label='Selected using training OOF')
        plt.xlabel('Probability threshold'); plt.ylabel('Training OOF development score')
        plt.ylim(0, 1); plt.legend(); plt.title(d['label'] + ' — threshold development'); figure(output/f'threshold_{name}.png')
        fi = pd.DataFrame(d['importance']).assign(magnitude=lambda x: x.importance.abs()).nlargest(12, 'magnitude').sort_values('importance')
        plt.figure(figsize=(9, 6)); plt.barh(fi.feature, fi.importance, color='#246b87')
        method = 'Signed coefficients (scaled numeric / indicator)' if name == 'logistic_regression' else 'Impurity importance'
        plt.xlabel(method); plt.title(d['label'] + ' — associations, not causal effects'); figure(output/f'importance_{name}.png')
    profiles = pd.DataFrame(clusters['profiles'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['Cluster ' + str(int(v)) for v in profiles.cluster]
    for ax, field, title in zip(axes, ('size', 'response_rate'), ('Customer counts', 'Observed response rate')):
        bars = ax.bar(labels, profiles[field], color=['#246b87', '#218178'])
        ax.bar_label(bars, fmt='%.0f' if field == 'size' else '%.3f'); ax.set_title(title)
        ax.set_ylim(0, profiles[field].max() * 1.25)
    figure(output/'cluster_comparison.png')


def build(manifest_path):
    """Build the entire verified report bundle from pinned experiment sources.

    Args:
        manifest_path (Path): JSON with repository-relative runs and output path.

    Returns:
        Path: Generated output directory.

    Processing:
        Validate source artifacts, export comparison/profile CSVs, dashboard JSON,
        source hashes, figures and generated evaluation interpretation.

    Raises:
        ValueError: If source evidence is inconsistent.
        OSError: If inputs are missing or outputs cannot be written. Overwrites
            only the manifest's generated report directory, never training runs.
    """
    manifest = read_json(manifest_path)
    runs = {name: PROJECT/path for name, path in manifest['runs'].items()}
    config = load_config(runs[MODELS[0]]/'config.json')
    data, audit = load_data(config)
    split_id = read_json(runs[MODELS[0]]/'run_metadata.json')['split_id']
    split = read_json(PROJECT/'experiments/outputs/shared'/split_id/'split.json')
    rows, details = [], {}
    for name in MODELS:
        row, detail = verify_classifier(name, runs[name], data, config, split, split_id)
        rows.append(row); details[name] = detail
    table = pd.DataFrame(rows)
    clusters = verify_clusters(runs['kmeans'], data, config)
    output = PROJECT / manifest['output']
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/'model_comparison.csv', index=False, float_format='%.12g')
    pd.DataFrame(clusters['profiles']).to_csv(output/'cluster_profiles.csv', index=False)
    overview = dict(rows=len(data), columns=len(data.columns), responders=int(data.Response.sum()),
                    nonresponders=int((data.Response == 0).sum()), prevalence=float(data.Response.mean()),
                    train_rows=len(split['train']), test_rows=len(split['test']), cv_folds=len(split['folds']),
                    test_prevalence=float(data.iloc[split['test']].Response.mean()), restored_missing_income=audit['restored_missing_income'])
    details['test_prevalence'] = overview['test_prevalence']
    create_plots(table, details, clusters, output)
    bundle = dict(overview=overview, comparison=rows, models={n: details[n] for n in MODELS}, clusters=clusters,
                  sources=manifest['runs'], split_id=split_id,
                  best_test_f1=table.loc[table.test_f1.idxmax(), 'model'], best_cv_f1=table.loc[table.cv_f1.idxmax(), 'model'])
    save_json(output/'dashboard_data.json', bundle)
    hashes = {p.relative_to(PROJECT).as_posix(): digest(p) for run in runs.values() for p in run.iterdir() if p.is_file()}
    save_json(output/'provenance.json', dict(manifest=str(Path(manifest_path).relative_to(PROJECT)), split_id=split_id,
                                            source_sha256=hashes, checks='IDs, hashes, probabilities, fold metrics, thresholds, importance, cluster profiles'))
    from experiments.milestone3.documentation import write_evaluation
    write_evaluation(bundle, PROJECT/'docs/MODEL_EVALUATION.md')
    print(f'Verified report bundle: {output}')
    return output


def main():
    """Run the verified report builder from the command line.

    Args:
        None: Reads optional --manifest Path, default sources.json beside module.

    Returns:
        None: Writes report assets through build and prints their directory.

    Processing:
        Parse arguments and invoke the pinned-source report workflow.

    Raises:
        SystemExit: For invalid CLI arguments; build errors propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('sources.json'))
    build(parser.parse_args().manifest)


if __name__ == '__main__':
    main()
