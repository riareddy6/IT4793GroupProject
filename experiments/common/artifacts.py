"""Completed-run discovery without loading or fitting estimators."""
import json
from .result_saver import ROOT


def latest(name):
    """Find the newest completed run for an experiment.

    Args:
        name (str): Experiment directory under experiments/outputs.

    Returns:
        Path: Newest lexically ordered run with complete metadata.

    Processing:
        Inspect timestamped run directories in descending order, skipping failures.

    Raises:
        ValueError: If no completed run exists.
        JSONDecodeError: If a candidate metadata file is malformed.
    """
    for path in sorted((ROOT / 'outputs' / name).glob('run_*'), reverse=True):
        meta = path / 'run_metadata.json'
        if meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('status') == 'complete':
            return path
    raise ValueError(f'{name} has no successful run')
