"""Replay original pipelines without overwriting retained Milestone 1/2 evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
from experiments.common.artifacts import latest
from experiments.common.data_loader import PROJECT, load_config, load_data
from experiments.common.preprocessing import features
from experiments.common.result_saver import digest, save_json
from experiments.common.settings import MODELS, SEED, PERMUTATION_REPEATS
from experiments.common.split_manager import shared_split
from experiments.milestone3.build import read_json, require


def validate_milestone1():
    """Execute Milestone 1 in a temporary directory and compare processed data.

    Args:
        None: Uses the original raw source and historical processed CSV.
    Returns:
        dict: Dimensions and exact processed-data hash agreement.
    Processing:
        Import the existing script, redirect output globals, run its full pipeline,
        compare records and serialized CSV, then remove temporary artifacts.
    Raises:
        ValueError: If regenerated data differs; import/runtime errors propagate.
        Does not overwrite historical reports, figures or processed data.
    """
    spec = importlib.util.spec_from_file_location('milestone1', PROJECT/'docs/milestone1/milestone1_analysis.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        module.PROCESSED_DIR, module.REPORT_DIR, module.VISUALIZATION_DIR = root/'data', root/'reports', root/'figures'
        generated = module.run_pipeline()
        original = PROJECT/'docs/milestone1/processed_data/marketing_campaign_processed.csv'
        pd.testing.assert_frame_equal(generated.reset_index(drop=True), pd.read_csv(original), check_dtype=False)
        same_hash = digest(original) == digest(module.PROCESSED_DIR/'marketing_campaign_processed.csv')
        require(same_hash, 'Milestone 1 serialized data differs from historical hash')
        return dict(rows=len(generated), processed_hash_matches=same_hash, generated_eda_figures=8)


def validate_interpretation(manifest):
    """Recompute saved permutation importance and selected clustering silhouette.

    Args:
        manifest (dict): Pinned repository-relative experiment source runs.
    Returns:
        dict: Four permutation matches and selected K-Means silhouette.
    Processing:
        Refit the first training fold only, repeat original permutation settings,
        compare score decreases/SD, and recompute silhouette on transformed data.
    Raises:
        ValueError: On mismatched importance or silhouette. Fits temporary
            validation models in memory; never tunes or modifies held-out models.
    """
    c = load_config(PROJECT/manifest['runs'][MODELS[0]]/'config.json')
    data, _ = load_data(c); x = features(data, c)
    train, _, folds, _ = shared_split(data, c); a, b = folds[0]
    results = {}
    for name in MODELS:
        run = PROJECT/manifest['runs'][name]
        pipeline = clone(joblib.load(run/'model.joblib')['pipeline'])
        pipeline.fit(x.iloc[train[a]], data.Response.iloc[train[a]])
        perm = permutation_importance(pipeline, x.iloc[train[b]], data.Response.iloc[train[b]],
                                      scoring='f1', n_repeats=PERMUTATION_REPEATS, random_state=SEED, n_jobs=1)
        saved = pd.read_csv(run/'permutation_importance.csv')
        require(np.allclose(perm.importances_mean, saved.importance), f'{name}: permutation mean mismatch')
        require(np.allclose(perm.importances_std, saved['std']), f'{name}: permutation SD mismatch')
        results[name] = 'mean and SD reproduced on first training validation fold'
    run = PROJECT/manifest['runs']['kmeans']; c = read_json(run/'config.json')
    artifact = joblib.load(run/'model.joblib')['pipeline']
    z = artifact.named_steps['preprocessor'].transform(features(data, c, clustering=True))
    with threadpool_limits(limits=1):
        silhouette = float(silhouette_score(z, artifact.named_steps['model'].labels_))
    saved = pd.read_csv(run/'cluster_metrics.csv').set_index('k')
    require(np.isclose(silhouette, saved.loc[artifact.named_steps['model'].n_clusters, 'silhouette']), 'Silhouette mismatch')
    return dict(permutation=results, silhouette=silhouette)


def replay_training(manifest):
    """Execute each existing training entry point and compare fresh output to history.

    Args:
        manifest (dict): Pinned original runs for comparison.
    Returns:
        dict: Fresh run paths and probability/assignment reproducibility checks.
    Processing:
        Run four classifiers and K-Means as subprocesses; compare model outputs
        to pinned originals, run comparison, artifact verification and explanation.
    Raises:
        CalledProcessError: If an entry point fails.
        ValueError: If fresh thresholds/probabilities/labels differ. Creates new
            timestamped runs and updates the explicit Milestone 2 summary only.
    """
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT/'.experiment_dependencies') + os.pathsep + str(PROJECT)
    results = {}
    for name in (*MODELS, 'kmeans'):
        print('Replaying '+name, flush=True)
        subprocess.run([sys.executable, '-m', f'experiments.{name}.train'], cwd=PROJECT, env=env, check=True)
        run, original = latest(name), PROJECT/manifest['runs'][name]
        if name != 'kmeans':
            new, old = [pd.read_csv(p/'test_predictions.csv') for p in (run, original)]
            require(new.ID.tolist() == old.ID.tolist(), f'{name}: replay IDs mismatch')
            require(np.allclose(new.predicted_probability, old.predicted_probability, atol=1e-12, rtol=0), f'{name}: replay probability mismatch')
            require(np.array_equal(new.predicted_class, old.predicted_class), f'{name}: replay classes mismatch')
            require(np.array_equal(new.selected_threshold, old.selected_threshold), f'{name}: replay threshold mismatch')
            results[name] = dict(run=run.relative_to(PROJECT).as_posix(), max_probability_difference=float(np.max(np.abs(new.predicted_probability-old.predicted_probability))))
        else:
            new, old = [pd.read_csv(p/'cluster_assignments.csv') for p in (run, original)]
            require(new.equals(old), 'K-Means replay assignments mismatch')
            results[name] = dict(run=run.relative_to(PROJECT).as_posix(), assignments_match=True)
    for module in ('experiments.comparison.compare_classifiers', 'experiments.verify_outputs'):
        subprocess.run([sys.executable, '-m', module], cwd=PROJECT, env=env, check=True)
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, '-m', 'experiments.kmeans.explain_results', '--output', str(Path(folder)/'explanation.md')], cwd=PROJECT, env=env, check=True)
    return results


def main():
    """Run full workflow reproducibility QA and save its measured results.

    Args:
        None: CLI --skip-training skips new full training runs when requested.
    Returns:
        None: Writes docs/milestone3/workflow_validation.json after all checks pass.
    Processing:
        Reproduce Milestone 1, interpretation and optionally all training entry points;
        hash pinned artifacts before/after to verify preservation.
    Raises:
        Exception: Any validation/training failure propagates without reporting success.
        Writes new timestamped runs when training is enabled.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-training', action='store_true')
    args = parser.parse_args()
    manifest = read_json(Path(__file__).with_name('sources.json'))
    before = {p: digest(p) for path in manifest['runs'].values() for p in (PROJECT/path).iterdir() if p.is_file()}
    result = dict(milestone1=validate_milestone1(), interpretation=validate_interpretation(manifest))
    result['training'] = 'skipped by request' if args.skip_training else replay_training(manifest)
    require(all(digest(p) == value for p, value in before.items()), 'Historical run artifacts changed')
    result['historical_artifacts_unchanged'] = True
    save_json(PROJECT/'docs/milestone3/workflow_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
