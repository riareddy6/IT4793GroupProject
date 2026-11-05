import json
from pathlib import Path
import pandas as pd
from .data_validation import validate

PROJECT = Path(__file__).resolve().parents[2]

def load_config(path):
    """Read and validate a classifier or clustering JSON configuration.

    Args:
        path (str | Path): Configuration file; relative dataset paths resolve from repository root.

    Returns:
        dict: Configuration with absolute dataset and raw_dataset paths.

    Processing:
        Parse JSON; enforce established target, seed, split, CV, metric and threshold settings; resolve data paths.

    Raises:
        ValueError: If established experiment settings differ. OSError/JSONDecodeError: For unreadable or invalid JSON. No files are modified.
    """
    config = json.loads(Path(path).read_text())
    required = dict(target='Response', random_state=42, test_size=.2, cv_folds=5,
                    primary_metric='f1', positive_class=1, default_threshold=.5)
    if any(config.get(k) != v for k,v in required.items()):
        raise ValueError('Initial experiments require the documented shared dataset settings')
    for key in ('dataset', 'raw_dataset'):
        config[key] = str(PROJECT / config[key])
    return config

def load_data(config):
    """Load processed records and restore original missing Income markers before fold-local imputation.

    Args:
        config (dict): Validated configuration with dataset and raw_dataset paths.

    Returns:
        tuple[pd.DataFrame, dict]: Records with missing/sentinel incomes restored to NaN and validation audit.

    Processing:
        Read processed CSV, validate records, align raw income by unique ID, replace raw missing or 666666 incomes with NaN.

    Raises:
        ValueError: If IDs cannot be aligned or records fail validation. OSError: For missing inputs. Changes only the in-memory DataFrame.
    """
    data = pd.read_csv(config['dataset'])
    audit = validate(data)
    raw = pd.read_csv(config['raw_dataset'], sep='\t').set_index('ID')
    if not raw.index.is_unique or not data.ID.isin(raw.index).all():
        raise ValueError('Cannot safely restore original missing incomes')
    income = data.ID.map(raw.Income)
    mask = income.isna() | income.eq(666666)
    data.loc[mask, 'Income'] = float('nan')
    audit['restored_missing_income'] = int(mask.sum())
    return data, audit
