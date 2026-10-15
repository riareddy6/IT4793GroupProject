import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from .result_saver import ROOT, digest, save_json

def shared_split(data, config):
    identity = dict(dataset=digest(__import__('pathlib').Path(config['dataset'])), raw=digest(__import__('pathlib').Path(config['raw_dataset'])),
                    seed=42, test_size=.2, folds=5)
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = ROOT / 'outputs' / 'shared' / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'split.json'
    if path.exists():
        s = json.loads(path.read_text())
    else:
        train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.Response)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, data.Response.iloc[train]))
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
