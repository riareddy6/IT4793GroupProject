import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sklearn

ROOT = Path(__file__).resolve().parents[1]

def save_json(path, value):
    """Serialize a JSON-compatible value while rejecting nonfinite numbers.

    Args:
        path (Path): Destination file; parent directory must exist.
        value (JSON-compatible): Object containing only serializable finite values.

    Returns:
        None: Writes indented UTF-8 JSON to path.

    Processing:
        Serialize with allow_nan=False and write the result to the destination.

    Raises:
        ValueError: For NaN or infinity. TypeError: For unsupported values. OSError: For unwritable paths. Overwrites the named file.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def digest(path):
    """Compute a file content hash for experiment provenance.

    Args:
        path (Path): Existing readable file.

    Returns:
        str: Hexadecimal SHA-256 digest.

    Processing:
        Read bytes and calculate SHA-256 without changing the file.

    Raises:
        OSError: If the file cannot be read. Reads the complete file into memory.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()

def new_run(name, config):
    """Create a unique timestamped experiment directory and save its configuration.

    Args:
        name (str): Experiment identifier under experiments/outputs.
        config (dict): JSON-compatible resolved experiment configuration.

    Returns:
        Path: New run directory containing config.json.

    Processing:
        Hash sorted configuration, combine UTC microsecond timestamp with short hash, create directory exclusively and serialize configuration.

    Raises:
        FileExistsError: On a directory collision. OSError/TypeError/ValueError: On creation or serialization failure. Creates directory and config file.
    """
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    path = ROOT / 'outputs' / name / (datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f_') + tag)
    path.mkdir(parents=True, exist_ok=False)
    save_json(path / 'config.json', config)
    return path

def metadata(config, data, split=None):
    """Build reproducibility metadata for a running experiment.

    Args:
        config (dict): Resolved dataset/raw_dataset paths.
        data (pd.DataFrame): Loaded customer records.
        split (str | None): Shared split identifier or None for clustering; default None.

    Returns:
        dict: UTC time, Python/sklearn versions, seed, rows, file hashes, split ID and running status.

    Processing:
        Inspect environment and dataset lengths; hash both source files and attach optional split identity.

    Raises:
        OSError: If source files cannot be hashed. Does not save files; caller updates status and persists metadata.
    """
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                sklearn_version=sklearn.__version__, random_seed=42, rows=len(data),
                dataset_sha256=digest(Path(config['dataset'])), raw_sha256=digest(Path(config['raw_dataset'])),
                split_id=split, status='running')
