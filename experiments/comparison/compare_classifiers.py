import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.common.result_saver import ROOT, new_run, save_json
from experiments.common.evaluation import plt, figure

MODELS=['logistic_regression','decision_tree','random_forest','gradient_boosting']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs=4,help='Explicit run directories in LR, DT, RF, GB order; otherwise latest completed per model')
    args=parser.parse_args()
    runs=[]
    for i,name in enumerate(MODELS):
        candidates=[Path(args.runs[i])] if args.runs else sorted((ROOT/'outputs'/name).glob('run_*'),reverse=True)
        complete=[p for p in candidates if (p/'run_metadata.json').exists() and json.loads((p/'run_metadata.json').read_text()).get('status')=='complete']
        if not complete:
            raise ValueError(f'No complete run for {name}')
        runs.append(complete[0])
    metadata=[json.loads((p/'run_metadata.json').read_text()) for p in runs]
    for key in ['split_id','dataset_sha256','raw_sha256','features','sklearn_version']:
        if any(m[key]!=metadata[0][key] for m in metadata):
            raise ValueError(f'Incompatible runs: {key}')
    rows=[]
    for name,p in zip(MODELS,runs):
        cv=json.loads((p/'cv_metrics.json').read_text())
        test=json.loads((p/'test_metrics.json').read_text())
        rows.append(dict(model=name,run=str(p),cv_f1=cv['mean']['f1'],cv_f1_std=cv['std']['f1'],
                         threshold=test['selected_threshold'],
                         **{'test_default_'+k:v for k,v in test['default_threshold'].items() if k not in ('confusion_matrix','roc_auc')},
                         **{'test_selected_'+k:v for k,v in test['selected'].items() if k not in ('confusion_matrix','roc_auc')}))
    table=pd.DataFrame(rows).sort_values('cv_f1',ascending=False)
    out=new_run('comparison',dict(runs=[str(p) for p in runs]))
    table.to_csv(out/'model_comparison.csv',index=False)
    table.set_index('model')[['cv_f1','test_default_f1','test_selected_f1']].plot.bar(figsize=(10,6),ylim=(0,1),title='Positive-class F1: CV and independent test');figure(out/'model_comparison.png')
    winner=table.iloc[0]['model']
    summary=f'''# Milestone 2 classifier comparison

Provisional recommendation: **{winner}**, selected by training CV F1 at threshold 0.50, with stability and interpretability considered below. Test scores are final descriptive evidence, not a model-selection search.

{table.to_string(index=False)}

Higher recall reaches more responders; higher precision reduces wasted contacts. F1 balances these without assigning monetary costs. Tune campaign policy using development data and explicit contact costs, revenue and capacity. Logistic Regression offers signed conditional coefficients; the depth-limited tree offers readable rules; ensembles capture nonlinear interactions with less direct interpretability. CV standard deviations indicate fold variability, not confidence intervals. Threshold-development F1 is optimistic because the same OOF labels choose the threshold; compare final test results separately. Accuracy alone favors the non-responder majority.

The fixed configurations were not hyperparameter-searched. Correlated spend totals and component predictors redistribute feature importance. Prior acceptance predictors require historical availability before the current campaign. All customer snapshots must precede the offer.

## Segmentation

K Means is excluded from this leaderboard. See its independent cluster profiles, silhouette scores, size distribution and descriptive Response rates. Response was excluded from fitting and selection. Profile interpretation should be reviewed by the marketing team before using segments.
'''
    (out/'model_selection_summary.md').write_text(summary,encoding='utf-8')
    save_json(out/'run_metadata.json',dict(status='complete',source_runs=[str(p) for p in runs],split_id=metadata[0]['split_id']))
    print(out)

if __name__=='__main__':
    main()
