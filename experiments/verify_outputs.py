"""Verify saved runs and produce an execution report without retraining."""
import json
from pathlib import Path
import joblib
import pandas as pd
from PIL import Image
from experiments.common.result_saver import ROOT

CLASSIFIERS=['logistic_regression','decision_tree','random_forest','gradient_boosting']
FILES=['model.joblib','config.json','run_metadata.json','cv_metrics.json','test_metrics.json',
       'threshold_analysis.csv','test_predictions.csv','confusion_matrix.png',
       'precision_recall_curve.png','feature_importance.csv','feature_importance.png','classification_report.txt']
KFILES=['model.joblib','config.json','run_metadata.json','cluster_metrics.csv','cluster_assignments.csv',
        'cluster_profiles.csv','silhouette_scores.png','cluster_size_distribution.png','cluster_response_rates.png']

def latest(name):
    for p in sorted((ROOT/'outputs'/name).glob('run_*'),reverse=True):
        if (p/'run_metadata.json').exists() and json.loads((p/'run_metadata.json').read_text())['status']=='complete':
            return p
    raise ValueError(name+' has no successful run')

def main():
    report=['# Verified Milestone 2 execution','', 'Original project files were preserved. Created `experiments/` with six shared utility modules, five independent training modules and JSON configurations, result comparison, pinned requirements, documentation, verification and separate timestamped outputs. Local dependencies are in `.experiment_dependencies/`.','',
            '## Exact execution commands','', 'From repository root in PowerShell:','', '```powershell', r'$env:PYTHONPATH="$PWD\.experiment_dependencies"',
            'python -m experiments.common.split_manager']
    report += ['python -m experiments.'+name+'.train' for name in CLASSIFIERS+['kmeans']]
    report += ['python -m experiments.comparison.compare_classifiers','python -m experiments.verify_outputs','```','',
               '## Classification results','', '| Model | CV F1 mean ± SD (.50) | Test F1 (.50) | Threshold | Test precision | Test recall | Test F1 | AP |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    paths=[];split_ids=[]
    for name in CLASSIFIERS+['kmeans']:
        p=latest(name)
        for file in FILES if name in CLASSIFIERS else KFILES:
            f=p/file
            assert f.is_file() and f.stat().st_size>0, f
            if f.suffix=='.json':json.loads(f.read_text())
            elif f.suffix=='.png':
                with Image.open(f) as im:im.verify()
            elif f.suffix=='.csv':pd.read_csv(f)
        joblib.load(p/'model.joblib')
        m=json.loads((p/'run_metadata.json').read_text())
        paths.append(f'- `{name}`: [{p.name}]({p.relative_to(ROOT).as_posix()}/run_metadata.json), {m["rows"]} rows, {m["feature_count"]} input features.')
        if name in CLASSIFIERS:
            split_ids.append(m['split_id'])
            test=json.loads((p/'test_metrics.json').read_text());cv=json.loads((p/'cv_metrics.json').read_text());s=test['selected']
            report.append(f'| {name} | {cv["mean"]["f1"]:.4f} ± {cv["std"]["f1"]:.4f} | {test["default_threshold"]["f1"]:.4f} | {test["selected_threshold"]:.2f} | {s["precision"]:.4f} | {s["recall"]:.4f} | {s["f1"]:.4f}  | {s["average_precision"]:.4f} |')
            preds=pd.read_csv(p/'test_predictions.csv')
            assert len(preds)==448 and preds.ID.is_unique
            recomputed=(preds.predicted_probability>=test['selected_threshold']).astype(int)
            assert (recomputed==preds.predicted_class).all()
            paths.append(f'  Training/test: {m["train_rows"]}/{m["test_rows"]}; captured warnings: {len(json.loads((p/"warnings.json").read_text()))}.')
        else:
            profiles=pd.read_csv(p/'cluster_profiles.csv')
            assert int(profiles['size'].sum())==2237
            segmentation=['', '## Customer segmentation','', 'Standalone interpretation: [K Means results](kmeans/RESULTS_EXPLANATION.md). The classifiers use the processed data directly; cluster labels are not classification inputs.', '', f'Selected k={m["selected_k"]}; 2,237 customers; 16 standardized numerical inputs. All candidate k values, silhouettes, inertia and size constraints are saved. Response was excluded from fitting and selection.', '',profiles.to_string(index=False)]
            sizes=profiles[['cluster','size','response_rate','Income','MntWines','MntMeatProducts','NumWebPurchases','Kidhome','Teenhome']]
            segmentation += ['', 'Interpretation from original-unit profiles:', '',sizes.to_string(index=False)]
            low=profiles.sort_values('Income').iloc[0];high=profiles.sort_values('Income').iloc[-1]
            segmentation += ['', f'Cluster {int(low.cluster)} contains {int(low["size"]):,} customers with lower mean income ({low.Income:,.0f}), lower wine/meat spending and fewer web purchases; it has more young children on average. Cluster {int(high.cluster)} contains {int(high["size"]):,} customers with higher mean income ({high.Income:,.0f}), substantially higher spending and more purchases. Their observed response rates are {low.response_rate:.1%} and {high.response_rate:.1%}. This two-group solution is interpretable as lower- and higher-spending customer segments; these are descriptive labels, not validated targeting rules.']
    assert len(set(split_ids))==1
    p=latest('comparison')
    for f in ['model_comparison.csv','model_comparison.png','model_selection_summary.md']:
        assert (p/f).stat().st_size>0
    with Image.open(p/'model_comparison.png') as im:im.verify()
    comparison=pd.read_csv(p/'model_comparison.csv')
    report += ['',f'Provisional recommendation by training CV F1: **{comparison.iloc[0].model}**. See the [selection summary]({p.relative_to(ROOT).as_posix()}/model_selection_summary.md) for marketing and interpretability tradeoffs.',
               '', 'All four classifiers use 1,789 training observations, 448 independent test observations, the same persisted folds, and identical underlying input features (counts listed below). One-hot encoding changes the transformed feature count. Numerical scaling is enabled for Logistic Regression and K Means. Exact estimator parameters are in each run config.','',
               '## Saved runs',''] + paths + [f'- Comparison: [{p.name}]({p.relative_to(ROOT).as_posix()}/model_comparison.csv).'] + segmentation
    failed=list((ROOT.parent/'outputs').glob('*/run_*/error.txt'))+list((ROOT/'outputs').glob('*/run_*/error.txt'))
    report += ['', '## Warnings, limitations and next steps','',
               'The original virtual environment was inaccessible; dependencies were installed locally. Pip reported conflicts with globally installed streamlit/sklearn-compat; these experiments do not import those packages and use the isolated local dependency path. An initial path-resolution error was corrected and its failed run retained. Initial sandbox dependency-read failures occurred before model training. Failed records: '+', '.join(str(f.relative_to(ROOT.parent)) for f in failed),
               '', 'Threshold OOF results are development estimates; the chosen threshold is evaluated once on frozen test probabilities. Models are provisionally ranked by training CV F1, not test accuracy. Historical feature timing and the fixed 2014 date convention need verification. Existing globally imputed incomes were restored in memory; source files were untouched. Correlated aggregates complicate importance. Segments and their response rates are descriptive; cluster stability and operational interpretability remain to be reviewed.',
               '', 'Next: validate pre-offer data availability, review segment profiles with marketing stakeholders, define cost/capacity objectives, and use nested CV or fresh holdout data for subsequent tuning. See [README](README.md) for preprocessing, configurations, reproduction and limitations.', '',
               'Historical run artifacts are preserved. Current summaries and new comparisons omit the removed ranking metric and its curve; classifiers were not retrained for this reporting change.', '',
               'Verification: all required artifacts exist and are nonempty; JSON/CSV parsing, image decoding, joblib loading, prediction-threshold consistency, identical split IDs, and cluster totals passed.']
    (ROOT/'EXECUTION_SUMMARY.md').write_text('\n'.join(report),encoding='utf-8')
    print('\n'.join(report[:35]))

if __name__=='__main__':main()
