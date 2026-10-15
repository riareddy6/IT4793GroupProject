import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sklearn

ROOT = Path(__file__).resolve().parents[1]

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def new_run(name, config):
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    path = ROOT / 'outputs' / name / (datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f_') + tag)
    path.mkdir(parents=True, exist_ok=False)
    save_json(path / 'config.json', config)
    return path

def metadata(config, data, split=None):
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                sklearn_version=sklearn.__version__, random_seed=42, rows=len(data),
                dataset_sha256=digest(Path(config['dataset'])), raw_sha256=digest(Path(config['raw_dataset'])),
                split_id=split, status='running')
