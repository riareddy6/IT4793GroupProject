import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from .result_saver import ROOT, digest, save_json
from .settings import SEED, TEST_SIZE, CV_FOLDS

def shared_split(data, config):
    """Reuse or create identical stratified train/test indices and CV folds for all classifiers.

    Args:
        data (pd.DataFrame): Customer records with ID and binary Response, in persisted row order.
        config (dict): Resolved dataset and raw_dataset paths.

    Returns:
        tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray,np.ndarray]], str]: Global train/test positions, folds relative to train, and split hash.

    Processing:
        Hash data sources plus established seed/split/fold settings; load persisted split or create stratified 80/20 and shuffled five-fold indices; check partition invariants.

    Raises:
        AssertionError: If persisted identity or partition coverage is inconsistent. OSError/JSONDecodeError: For invalid files. Creates shared directory and, only if absent, split.json and assignments.csv.
    """
    identity = dict(dataset=digest(Path(config['dataset'])), raw=digest(Path(config['raw_dataset'])),
                    seed=SEED, test_size=TEST_SIZE, folds=CV_FOLDS)
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = ROOT / 'outputs' / 'shared' / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'split.json'
    if path.exists():
        s = json.loads(path.read_text())
    else:
        train, test = train_test_split(np.arange(len(data)), test_size=TEST_SIZE, random_state=SEED, stratify=data.Response)
        folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(train, data.Response.iloc[train]))
        s = dict(identity=identity, train=train.tolist(), test=test.tolist(), folds=[dict(train=a.tolist(), validation=b.tolist()) for a,b in folds])
        save_json(path, s)
        assignments = pd.DataFrame({'row_index':np.arange(len(data)), 'ID':data.ID, 'partition':'test', 'fold':-1})
        assignments.loc[train, 'partition'] = 'train'
        for i, (_, val) in enumerate(folds):
            assignments.loc[train[val], 'fold'] = i
        assignments.to_csv(folder / 'assignments.csv', index=False)
    assert s['identity'] == identity
    assert len(set(s['train']) & set(s['test'])) == 0
    assert sorted(s['train'] + s['test']) == list(range(len(data)))
    return np.array(s['train']), np.array(s['test']), [(np.array(f['train']),np.array(f['validation'])) for f in s['folds']], tag

if __name__ == '__main__':
    from .data_loader import load_config, load_data
    c = load_config(ROOT / 'logistic_regression' / 'config.json')
    d, a = load_data(c)
    tr, te, folds, tag = shared_split(d,c)
    save_json(ROOT / 'outputs' / 'shared' / tag / 'data_audit.json', a)
    print(f'Shared split {tag}: {len(tr)} train, {len(te)} test')
