import json
from pathlib import Path
import pandas as pd
from .data_validation import validate

PROJECT = Path(__file__).resolve().parents[2]

def load_config(path):
    config = json.loads(Path(path).read_text())
    required = dict(target='Response', random_state=42, test_size=.2, cv_folds=5,
                    primary_metric='f1', positive_class=1, default_threshold=.5)
    if any(config.get(k) != v for k,v in required.items()):
        raise ValueError('Initial experiments require the documented shared dataset settings')
    for key in ('dataset', 'raw_dataset'):
        config[key] = str(PROJECT / config[key])
    return config

def load_data(config):
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
