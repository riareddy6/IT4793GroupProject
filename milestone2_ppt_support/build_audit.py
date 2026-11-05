"""Read-only source audit; writes only the new audit report and support directory. No fitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.experiment_dependencies'))
import json, hashlib, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

OUT=ROOT/'milestone2_ppt_support'
def read(p):
    """Decode a historical audit JSON source.

    Args:
        p (str | Path): Existing JSON path.

    Returns:
        dict | list: Parsed value.

    Processing:
        Read UTF-8 and call json.loads.

    Raises:
        OSError/JSONDecodeError: For unreadable or invalid input. No output.
    """
    return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):
    """Format a repository-relative audit source path.

    Args:
        p (str | Path): Path within ROOT.

    Returns:
        str: Relative POSIX path.

    Processing:
        Resolve relative representation with Path.relative_to and as_posix.

    Raises:
        ValueError: If p is outside ROOT. No files changed.
    """
    return Path(p).relative_to(ROOT).as_posix()
def link(p):
    """Format a Markdown link to a repository audit artifact.

    Args:
        p (str | Path): Artifact path within ROOT.

    Returns:
        str: Markdown label and target using repository-relative path.

    Processing:
        Call rel and interpolate the resulting path.

    Raises:
        ValueError: If p is outside ROOT. No files changed.
    """
    return f'[{rel(p)}]({rel(p)})'
def table(df):
    """Format a DataFrame as a Markdown audit table.

    Args:
        df (pd.DataFrame): Data with printable scalar cells.

    Returns:
        str: Header, separator and formatted data rows.

    Processing:
        Format floats to eight decimals and escape pipe/newline content via nested fmt.

    Raises:
        TypeError: For unsupported formatting values. No files changed.
    """
    def fmt(v):
        """Format a scalar cell safely for the historical Markdown table.

        Args:
            v (object): Printable scalar; floats receive eight decimal places.

        Returns:
            str: Sanitized cell text.

        Processing:
            Format float values; otherwise replace pipes and newlines.

        Raises:
            Exception: Custom string conversion errors propagate. No file output.
        """
        if isinstance(v,(float,np.floating)): return f'{v:.8f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def csv(df,name):
    """Export a historical audit support table.

    Args:
        df (pd.DataFrame): Audit table.
        name (str): Destination filename within OUT.

    Returns:
        None: Writes a CSV with no row index and 15-significant-digit floats.

    Processing:
        Call DataFrame.to_csv in the support output directory.

    Raises:
        OSError: On failed export. Overwrites the named support file only when the historical audit script runs.
    """
    df.to_csv(OUT/name,index=False,float_format='%.15g')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
def save(name):
    """Export and close a historical audit plot.

    Args:
        name (str): PNG filename within OUT.

    Returns:
        None: Writes a 300-DPI PNG.

    Processing:
        Apply tight layout, save with tight bounds and close the active plot.

    Raises:
        OSError: On image writing failure. Changes matplotlib active figure state.
    """
    plt.tight_layout();plt.savefig(OUT/name,dpi=300,bbox_inches='tight');plt.close()
names=['logistic_regression','decision_tree','random_forest','gradient_boosting']
pretty=['Logistic Regression','Decision Tree','Random Forest','Gradient Boosting']
runs={n:next(p for p in sorted((ROOT/'experiments/outputs'/n).glob('run_*'),reverse=True) if read(p/'run_metadata.json').get('status')=='complete') for n in names+['kmeans']}
raw=pd.read_csv(ROOT/'data/marketing_campaign.csv',sep='\t')
processed=pd.read_csv(ROOT/'docs/milestone1/processed_data/marketing_campaign_processed.csv')
meta={n:read(p/'run_metadata.json') for n,p in runs.items()}
splitpath=ROOT/'experiments/outputs/shared'/meta[names[0]]['split_id']/'split.json'
split=read(splitpath); train=split['train']; test=split['test']
restored=processed.copy(); original=processed.ID.map(raw.set_index('ID').Income)
restored.loc[original.isna()|original.eq(666666),'Income']=np.nan
quality=read(ROOT/'docs/milestone1/reports/data_quality_summary.json')
checks=[]; perf=[]; configs=[]; artifacts={}; model_details=[]
for n,label in zip(names,pretty):
    p=runs[n]; m=meta[n]; cfg=read(p/'config.json'); cv=read(p/'cv_metrics.json'); tm=read(p/'test_metrics.json'); pred=pd.read_csv(p/'test_predictions.csv'); oof=pd.read_csv(p/'oof_predictions.csv'); th=pd.read_csv(p/'threshold_analysis.csv')
    with warnings.catch_warnings(record=True) as caught:
        obj=joblib.load(p/'model.joblib')
    artifacts[n]=obj
    pipe=obj['pipeline']; prep=pipe.named_steps['preprocessor']; estimator=pipe.named_steps['model']
    fi=pd.read_csv(p/'feature_importance.csv')
    selected=tm['selected']; cm=np.array(selected['confusion_matrix']).ravel()
    perf.append(dict(model=label,cv_f1=cv['mean']['f1'],cv_f1_sample_sd=cv['std']['f1'],test_f1=selected['f1'],precision=selected['precision'],recall=selected['recall'],accuracy=selected['accuracy'],macro_f1=selected['macro_f1'],roc_auc=selected.get('roc_auc',np.nan),average_precision=selected['average_precision'],threshold=tm['selected_threshold'],default_test_f1=tm['default_threshold']['f1'],TN=cm[0],FP=cm[1],FN=cm[2],TP=cm[3]))
    configs.append(dict(model=label,estimator=f'{type(estimator).__module__}.{type(estimator).__name__}',explicit_parameters=json.dumps(cfg['parameters'],sort_keys=True),numeric_scaling=cfg['scale_numeric'],class_weight=cfg['parameters'].get('class_weight','None'),seed=42,train_rows=len(train),test_rows=len(test),cv='shared shuffled 5-fold StratifiedKFold',features_before_encoding=m['feature_count'],features_after_encoding=len(prep.get_feature_names_out())))
    assert list(pred.row_index)==test and list(oof.row_index)==train
    assert pred.ID.tolist()==processed.iloc[test].ID.tolist()
    assert pred.true_label.tolist()==processed.iloc[test].Response.tolist()
    assert oof.true_label.tolist()==processed.iloc[train].Response.tolist()
    prediction=(pred.predicted_probability>=tm['selected_threshold']).astype(int)
    assert np.array_equal(prediction,pred.predicted_class)
    assert np.array_equal(confusion_matrix(pred.true_label,prediction),selected['confusion_matrix'])
    for key,func in [('f1',f1_score),('precision',precision_score),('recall',recall_score),('accuracy',accuracy_score)]:
        assert np.isclose(func(pred.true_label,prediction),selected[key],atol=1e-12)
    assert np.allclose(th.threshold,np.round(np.arange(.1,.901,.05),2))
    for row in th.itertuples(): assert np.isclose(f1_score(oof.true_label,oof.oof_probability>=row.threshold),row.f1)
    chosen=th.assign(distance=(th.threshold-.5).abs()).sort_values(['f1','precision','distance'],ascending=[False,False,True]).iloc[0]
    assert chosen.threshold==tm['selected_threshold']==obj['threshold']
    for idx,fold in enumerate(split['folds']):
        a=set(fold['train']);b=set(fold['validation'])
        assert not a.intersection(b) and a.union(b)==set(range(len(train)))
        saved=cv['folds'][idx]
        fold_oof=oof.iloc[fold['validation']]
        assert np.isclose(f1_score(fold_oof.true_label,fold_oof.oof_probability>=.5),saved['f1'])
    assert np.isclose(np.mean([f['f1'] for f in cv['folds']]),cv['mean']['f1'])
    assert np.isclose(np.std([f['f1'] for f in cv['folds']],ddof=1),cv['std']['f1'])
    assert m['features']==meta[names[0]]['features'] and m['split_id']==meta[names[0]]['split_id']
    assert len(fi)==len(prep.get_feature_names_out()) and fi.feature.tolist()==prep.get_feature_names_out().tolist()
    fitted_values=estimator.coef_[0] if hasattr(estimator,'coef_') else estimator.feature_importances_
    assert np.allclose(fi.importance,fitted_values)
    checks.append(f'{label}: saved pipeline loaded; selected-threshold classes, TN/FP/FN/TP and four primary test metrics match complete saved predictions; all 17 threshold F1 scores match saved OOF predictions; selected threshold and feature importance match artifacts. Load warnings: {len(caught)}.')
    model_details.append({'model':n,'all_estimator_parameters':estimator.get_params(),'encoded_feature_names':prep.get_feature_names_out().tolist(),'numeric_imputer_statistics':prep.named_transformers_['numeric'].named_steps['imputer'].statistics_.tolist(),'categorical_categories':[a.tolist() for a in prep.named_transformers_['categorical'].named_steps['encoder'].categories_]})
    plt.figure(figsize=(7,5));plt.imshow(selected['confusion_matrix'],cmap='Blues');plt.xticks([0,1],['Predicted No','Predicted Yes']);plt.yticks([0,1],['Actual No','Actual Yes']);plt.title(f'{label}: test confusion matrix\nThreshold {tm["selected_threshold"]:.2f}')
    for i in range(2):
        for j in range(2):plt.text(j,i,str(selected['confusion_matrix'][i][j]),ha='center',va='center',fontsize=22,color='white' if selected['confusion_matrix'][i][j]>200 else 'black')
    save(f'confusion_matrix_{n}.png')
    top=fi.assign(magnitude=fi.importance.abs()).nlargest(10,'magnitude').sort_values('importance')
    plt.figure(figsize=(10,6));plt.barh(top.feature,top.importance,color=['#b65b52' if x<0 else '#286a92' for x in top.importance]);plt.title(label+(': signed coefficients' if n==names[0] else ': impurity importance'));plt.xlabel('Coefficient (scaled numeric / indicator)' if n==names[0] else 'Impurity importance');save(f'feature_importance_{n}.png')
    csv(fi,f'feature_importance_{n}.csv')
performance=pd.DataFrame(perf);configuration=pd.DataFrame(configs)
csv(performance,'classification_performance_comparison.csv');csv(configuration,'model_configuration_comparison.csv')
(OUT/'saved_model_inspection.json').write_text(json.dumps(model_details,indent=2,default=str),encoding='utf-8')
for field,title,filename in [('test_f1','Test positive-class F1 at OOF-selected thresholds','test_f1_comparison.png')]:
    plt.figure(figsize=(10,5));plt.bar(pretty,performance[field],color='#286a92');plt.ylim(0,1);plt.ylabel('F1 score');plt.title(title)
    for i,v in enumerate(performance[field]):plt.text(i,v+.02,f'{v:.4f}',ha='center')
    save(filename)
ax=performance.set_index('model')[['precision','recall']].plot.bar(figsize=(10,6),color=['#286a92','#d79642'],rot=10);ax.set_ylim(0,1);ax.set_ylabel('Score');ax.set_xlabel('');ax.set_title('Test precision and recall at OOF-selected thresholds');save('precision_recall_comparison.png')
summary=pd.DataFrame([{'stage':'Raw source','rows':len(raw),'columns':raw.shape[1],'meaning':'29 total columns = 28 non-target columns, including ID and constants'}, {'stage':'Milestone 1 processed CSV','rows':len(processed),'columns':processed.shape[1],'meaning':'33 total columns, not 33 model predictors'}, {'stage':'Classification before encoding','rows':len(processed),'columns':28,'meaning':'26 numeric + 2 categorical predictors'}, {'stage':'Training before encoding','rows':len(train),'columns':28,'meaning':'Response stored separately'}, {'stage':'Testing before encoding','rows':len(test),'columns':28,'meaning':'Response stored separately'}, {'stage':'Final classification after encoding','rows':len(processed),'columns':int(configuration.features_after_encoding.iloc[0]),'meaning':'26 numeric + all education/marital indicator columns'}, {'stage':'K Means input','rows':len(processed),'columns':16,'meaning':'Numerical features; full dataset'}])
csv(summary,'dataset_summary.csv')
distribution=pd.DataFrame([dict(partition=label,rows=len(d),nonresponders=int((d.Response==0).sum()),responders=int(d.Response.sum()),response_rate=float(d.Response.mean())) for label,d in [('Full',processed),('Train',processed.iloc[train]),('Test',processed.iloc[test])]]);csv(distribution,'target_distribution.csv')
kp=runs['kmeans']; km=pd.read_csv(kp/'cluster_metrics.csv');profiles=pd.read_csv(kp/'cluster_profiles.csv');assign=pd.read_csv(kp/'cluster_assignments.csv'); kobj=joblib.load(kp/'model.joblib');kmodel=kobj['pipeline'].named_steps['model']
assert np.array_equal(kmodel.labels_,assign.cluster)
assert assign.ID.tolist()==processed.ID.tolist() and assign.Response.tolist()==processed.Response.tolist()
spend=['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
enriched=restored.assign(cluster=assign.cluster)
profiles['share']=profiles['size']/len(processed)
profiles['median_income_observed']=enriched.groupby('cluster').Income.median().values
profiles['mean_total_spending']=enriched.groupby('cluster')[spend].sum().sum(axis=1).values/profiles['size']
profiles['accepted']=assign.groupby('cluster').Response.sum().values
profiles['not_accepted']=profiles['size']-profiles['accepted']
profiles['income_missing_excluded']=enriched.groupby('cluster').Income.apply(lambda s:s.isna().sum()).values
assert np.allclose(profiles.response_rate,profiles.accepted/profiles['size'])
for f in meta['kmeans']['features']:
    values=enriched.groupby('cluster')[f].mean()
    if f=='Customer_Age': values=enriched.assign(Customer_Age=2014-enriched.Year_Birth).groupby('cluster').Customer_Age.mean()
    assert np.allclose(values,profiles[f])
csv(profiles,'cluster_profile_comparison.csv');csv(km,'k_selection_comparison.csv')
for field,title,filename in [('silhouette','K Means: silhouette by candidate K','silhouette_comparison.png'),('inertia','K Means: inertia by candidate K','inertia_comparison.png')]:
    plt.figure(figsize=(9,5));plt.plot(km.k,km[field],marker='o',color='#286a92');plt.xticks(km.k);plt.xlabel('Number of clusters (K)');plt.ylabel(field.capitalize());plt.title(title);save(filename)
for field,title,filename in [('size','Customer counts by cluster','cluster_size_comparison.png'),('response_rate','Observed offer response rate by cluster','cluster_response_rate_comparison.png')]:
    plt.figure(figsize=(8,5));plt.bar(['Cluster 0','Cluster 1'],profiles[field],color=['#286a92','#d79642']);plt.title(title);plt.ylabel('Customers' if field=='size' else 'Response rate')
    for i,v in enumerate(profiles[field]):plt.text(i,v*1.02,f'{int(v):,}' if field=='size' else f'{v:.4%}',ha='center')
    plt.ylim(0,max(profiles[field])*1.15);save(filename)
hashrows=[]
for name in ['dataset','raw']:
    path=ROOT/('docs/milestone1/processed_data/marketing_campaign_processed.csv' if name=='dataset' else 'data/marketing_campaign.csv')
    actual=hashlib.sha256(path.read_bytes()).hexdigest();expected=meta[names[0]][name+'_sha256'];assert actual==expected
    hashrows.append(dict(file=rel(path),sha256=actual,matches_saved_metadata=True))
csv(pd.DataFrame(hashrows),'dataset_hash_verification.csv')
image_count=0
for p in (ROOT/'experiments/outputs').rglob('*.png'):
    with Image.open(p) as im:im.verify()
    image_count+=1
for p in OUT.glob('*.png'):
    with Image.open(p) as im:im.verify()
checks += [f'Current raw/processed SHA256 match saved metadata; {image_count} existing experiment PNGs decode successfully.', 'Saved K Means labels, IDs, Response values, cluster counts and response rates match full assignments. No models fitted or retrained.', 'Existing verify_outputs.py was inspected but not executed because it overwrites experiments/EXECUTION_SUMMARY.md. The independent checks above ran during this audit.']
for cp in sorted((ROOT/'experiments/outputs/comparison').glob('run_*')):
    ct=pd.read_csv(cp/'model_comparison.csv')
    for n,label in zip(names,pretty):
        row=ct[ct.model==n].iloc[0];verified=performance[performance.model==label].iloc[0]
        assert np.isclose(row.cv_f1,verified.cv_f1) and np.isclose(row.test_selected_f1,verified.test_f1)
checks.append('Both saved comparison CSVs agree with source CV/test F1. All five saved fold F1 scores and their mean/sample SD match OOF records; fold train/validation indices are disjoint and exhaustive. All 16 saved cluster feature means match complete source records under the documented 2014 age convention.')
(OUT/'audit_verification.txt').write_text('\n'.join(checks),encoding='utf-8')

report=[]
def add(s):
    """Append one section fragment to the historical audit report.

    Args:
        s (str): Markdown fragment.

    Returns:
        None: Adds s to the global report list.

    Processing:
        Append the provided text without writing the final report yet.

    Raises:
        No expected exceptions for valid input. Mutates global report list.
    """
    report.append(s)
add('# Milestone 2 PowerPoint Evidence Audit\n\nAudit date: October 20, 2026. Scope: repository files and retained successful October 13 training runs, plus October 13/14 comparison outputs. This audit did not fit, tune, or retrain any model, modify source data/code/configuration, or overwrite experiment artifacts. Newly derived descriptive statistics use complete saved customer assignments and source records; they are explicitly identified. All charts are new 300-DPI PNG renderings of verified saved values.\n\n**Evidence labels:** Verified = directly supported by source/artifacts or an audit consistency check; Derived = arithmetic from complete existing records; Inferred = interpretation, not an experimental measurement; Missing = not retained or unavailable. No presentation file was found, so slide corrections address only the claims supplied in the request; unseen slide content is unverifiable.')
add('## Evidence register\n\n'+table(pd.DataFrame([dict(experiment=n,source=link(p/'run_metadata.json'),status=meta[n]['status']) for n,p in runs.items()])))
add('## 1. Dataset dimensions and preparation\n\n'+table(summary)+'\n\n'+table(distribution))
add('### Resolving 28 versus 33\n\nVerified: the raw file has **29 total columns**: 28 non-target columns including ID and two constants. Milestone 1 adds four columns (`Customer_Age`, `Total_Spending`, `Number_of_Children`, `Past_Campaign_Responses`), producing **33 total columns**. These are not 33 usable predictors. Modeling drops ID and Response (33→31), removes two constants (31→29), replaces `Dt_Customer` with tenure (no net change), and replaces `Year_Birth` with the already-existing `Customer_Age` (29→28). Final one-hot encoding expands the 2 categorical variables into '+str(configuration.features_after_encoding.iloc[0]-26)+' indicators, giving **'+str(configuration.features_after_encoding.iloc[0])+' transformed features**.\n\nSources: '+link(ROOT/'docs/milestone1/milestone1_analysis.py')+' (`clean_data`, `engineer_features`), '+link(ROOT/'experiments/common/preprocessing.py')+' (`features`, `preprocessor`); saved model inspection: [support JSON](milestone2_ppt_support/saved_model_inspection.json).')
add('### Complete classification inputs\n\n'+', '.join('`'+x+'`' for x in meta[names[0]]['features'])+'.\n\n`Education` and `Marital_Status` are categorical; the other 26 are numeric. No K Means labels are included. ID is excluded as identifier; Response is the target; `Z_CostContact` and `Z_Revenue` are excluded because constant. Raw birth year and registration date are replaced by age and tenure. No other original predictive columns are silently dropped.')
add('### Cleaning, missingness and timing\n\nVerified raw missing values: '+str(raw.isna().sum()[raw.isna().sum()>0].to_dict())+'. All other raw columns have zero missing entries. Raw Income also contains one `666666` value treated by project code as a sentinel (the data-entry interpretation itself is an assumption). Milestone 1 removes three records with Year_Birth < 1926; no duplicate records are removed. It retains the income-affected records, replaces the sentinel with missing, and fills 25 incomes with global cleaned-data median **51,371**. The saved processed CSV has **zero missing values in all 33 columns**.\n\nMilestone 2 restores those **25 Income cells to NaN in memory**, aligned by ID to the raw file; it does not restore deleted customer records. Numeric `SimpleImputer(strategy="median")` then learns from each fold training subset, and final training learns from all 1,789 training records. Categoricals use most-frequent imputation and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` with all categories retained (no reference category dropped). Logistic Regression additionally uses numeric `StandardScaler` (centering and unit variance); trees do not scale. Numeric/categorical imputation, encoding and LR scaling all live inside the cloned pipeline fitted within each fold. K Means uses full-data median imputation/scaling independently.\n\nThe saved `data_audit.missing` values describe the CSV **before restoration**, so Income=0 there and `restored_missing_income=25` are compatible. After restoration there are 25 missing income cells; saved fitted imputers provide finite replacements. Final saved models have '+str(configuration.features_after_encoding.iloc[0])+' features; fold encoders can omit categories absent from that training fold, and per-fold transformed dimensions are not saved.\n\nAge in Milestone 1 is `2026-Year_Birth`; classification and clustering overwrite it as `2014-Year_Birth`. Tenure is `2014-12-31-Dt_Customer`. This fixed date is not a verified offer date. Three birth-year exclusions use a 2026 rule despite the later 2014 modeling reference.\n\nSources: '+link(ROOT/'experiments/common/data_loader.py')+' (`load_data`), '+link(ROOT/'experiments/common/data_validation.py')+' (`validate`), '+link(ROOT/'experiments/common/evaluation.py')+' (`run_classifier`), '+link(ROOT/'docs/milestone1/reports/data_quality_summary.json')+'.\n\nCleaned dataset dimensions are 2,237×33; model matrix dimensions are 1,789×28 train and 448×28 test before encoding, and 1,789×'+str(configuration.features_after_encoding.iloc[0])+' and 448×'+str(configuration.features_after_encoding.iloc[0])+' under the final fitted encoders. Labels have 1,789 and 448 entries respectively.')
add('## 2. Classification experiment configuration\n\n'+table(configuration)+'\n\nAll four estimators are scikit-learn classifiers. Explicit parameters above come from each saved run config, and fitted artifacts were loaded for inspection. Full effective estimator parameters, including defaults, are in [saved_model_inspection.json](milestone2_ppt_support/saved_model_inspection.json). No resampling/SMOTE or sample weights appear in the fitting code; class weights are balanced for LR, DT and RF; Gradient Boosting has no class balancing configured.\n\nVerified: stratified 80/20 split, seed 42, same persisted train/test row indices and identical five shuffled stratified training folds for every classifier. The saved split identity is `'+meta[names[0]]['split_id']+'`. CV validation sizes: '+str([len(f['validation']) for f in split['folds']])+'. Config settings are fixed; no GridSearchCV, RandomizedSearchCV or parameter-search loop exists in these entry points. Manual fixed configuration is supported; how people originally chose those values is not recorded. All four use identical 28 underlying features.\n\nSource: '+link(splitpath)+', `shared_split` in '+link(ROOT/'experiments/common/split_manager.py')+', each `experiments/<model>/train.py` and saved `config.json`.')
add('## 3. Classification performance\n\nAll final test columns below use each model’s OOF-selected threshold; CV F1 uses **0.50**. SD is the sample SD across five folds (`ddof=1`), not a confidence interval. ROC AUC is reported because it already exists in the historical JSON files.\n\n'+table(performance))
add('### Threshold verification\n\nVerified: 17 thresholds from 0.10 through 0.90 inclusive in 0.05 steps are evaluated against training **out-of-fold probabilities**. Selection maximizes pooled positive-class OOF F1, breaking ties by higher precision, then closeness to 0.50. Each classifier chooses separately. Selected thresholds are frozen before predicting the 448 test customers. The test set used for final evaluation is the saved common holdout, not the threshold-development set. Audit recomputation from complete saved predictions confirmed selected precision/recall/F1/accuracy, confusion matrices, threshold decisions, and all threshold-grid F1 values.\n\nThreshold-development performance is optimistic for the threshold it selects because the same OOF labels choose and score it; it is not an independent estimate. Test threshold-selection leakage is **not evident** in the implemented workflow. Subsequent selection or tuning using the already-viewed test leaderboard could bias future claims. CV-at-0.50 and test-at-selected-threshold are different evaluation policies and should be labeled separately.\n\nExisting precision–recall evidence: each successful run has `precision_recall_curve.png`, `test_metrics.json` average precision, `threshold_analysis.csv`, `oof_predictions.csv`, and `test_predictions.csv`. Average precision and ROC AUC use probabilities, not the selected binary threshold. Historical ROC AUC values/plots persist, although current `evaluation.metrics` and `compare_classifiers.py` omit ROC AUC; an exact rerun of current reporting will not regenerate those historical fields.\n\nSources: `metrics`, `run_classifier` in '+link(ROOT/'experiments/common/evaluation.py')+'; each run’s `cv_metrics.json`, `test_metrics.json`, `threshold_development.json`, and CSVs. [Audit verification](milestone2_ppt_support/audit_verification.txt).')
add('### Model recommendation\n\n'+ '; '.join(f'Highest {col}: **{performance.loc[performance[col].idxmax(),"model"]}** ({performance[col].max():.8f})' for col in ['cv_f1','test_f1','precision','recall'])+'.\n\nLR is the provisional CV-based recommendation; RF is the strongest observed test-F1/precision alternative. The small CV F1 gap is '+f'{performance.cv_f1.iloc[0]-performance.cv_f1.iloc[2]:.8f}'+', and RF’s test F1 advantage over LR is '+f'{performance.test_f1.iloc[2]-performance.test_f1.iloc[0]:.8f}'+'. No repeated splits, confidence intervals or paired significance test establish a reliable winner. Only 67 positives are in the holdout. The saved majority baseline has accuracy 0.85044643 but positive-class F1=0; accuracy alone is misleading. Operational choice requires contact cost, benefit and capacity, none measured here.')
add('## 4. Feature importance and interpretation')
lrfi=pd.read_csv(runs[names[0]]/'feature_importance.csv')
add('### Logistic Regression: top signed coefficients\n\nLargest positive coefficients:\n\n'+table(lrfi[lrfi.importance>0].nlargest(10,'importance')[['feature','importance']])+'\n\nMost negative coefficients:\n\n'+table(lrfi[lrfi.importance<0].nsmallest(10,'importance')[['feature','importance']])+'\n\nNumeric coefficients refer to a one-training-standard-deviation change after median imputation and StandardScaler; one-hot coefficients refer to indicator columns, which are not standardized. All categories are retained, and regularization/correlated variables complicate reference-category comparisons. Coefficients are conditional log-odds associations, not causal effects or raw-unit changes.')
for n,label in zip(names[2:],pretty[2:]):
    fi=pd.read_csv(runs[n]/'feature_importance.csv')
    add('### '+label+' top 10\n\n'+table(fi.nlargest(10,'importance')[['feature','importance']])+'\n\nSource: '+link(runs[n]/'feature_importance.csv')+'. Method: fitted estimator `feature_importances_` (normalized impurity-based importance), verified against loaded artifact.')
add('Each classifier also has `permutation_importance.csv` (five repeats, seed 42, scoring F1 at default 0.50), evaluated on the first saved training validation fold after fitting its training portion; it does not use the final test set. Existing `model_behavior.md`, coefficient/importance PNGs, classification reports and a Decision Tree visualization provide additional interpretation. Totals and their components overlap: spending totals, child totals and historical campaign totals can redistribute importance. Neither coefficients, impurity nor permutation values prove causality. Sources: `run_classifier` interpretation block and each successful run’s outputs.')
add('## 5. K Means clustering configuration\n\nVerified **16 numeric inputs**: '+', '.join('`'+f+'`' for f in meta['kmeans']['features'])+'.\n\nSix product-spending features represent spending; four purchase counts represent transactions/deals/channels; website visits represent engagement; Income, age and household counts represent demographics; Recency represents purchase timing. Income and Recency are included. ID, Response, AcceptedCmp1–5 and Past_Campaign_Responses are excluded; aggregate Total_Spending and Number_of_Children are also not clustering inputs. Features are manually listed, not automatically selected.\n\nK Means is scikit-learn `KMeans`, Euclidean-distance clustering, numeric median imputation and `StandardScaler` on **all 2,237 customers**. Explicit code settings: n_clusters per candidate, n_init=20, random_state=42. Saved artifact effective settings: `'+json.dumps(kmodel.get_params(),sort_keys=True)+'`. Fitted selected model iterations: **'+str(kmodel.n_iter_)+'**. Initialization is k-means++; default max_iter=300 and tol=0.0001 are effective artifact parameters, not explicitly configured in JSON. Twenty initializations choose a low-inertia solution for a given K; this does not guarantee global optimality.\n\nClustering is independent from supervised classification. It never calls `shared_split`; Response is attached only after fitting/selection. Classifiers do not consume cluster assignments, and no cluster-specific classifiers appear. Sources: '+link(ROOT/'experiments/kmeans/train.py')+', '+link(kp/'config.json')+', '+link(kp/'model.joblib')+'.')
kt=km.copy();kt['cluster_sizes']='Not saved in full; see min/max'
kt.loc[kt.k==2,'cluster_sizes']='1304, 933 (cluster IDs 0, 1)'
add('## 6. Selecting the number of clusters\n\n'+table(kt)+'\n\nVerified K=2 has the highest silhouette **0.3037603760365297**, correctly rounded to **0.3038**. K=3 is next at 0.23406930625174138 (gap '+f'{km.silhouette.iloc[0]-km.silhouette.iloc[1]:.8f}'+'); none matches K=2 under this rule. All candidates pass the minimum-cluster-size constraint (5%, requiring at least 112 of 2,237 customers). The selector maximizes silhouette among eligible candidates; exact ties favor lower K. Interpretability is discussed afterward and is **not a numerical selection criterion**.\n\nInertia values were saved, but no elbow-selection algorithm or original elbow plot is evident. This audit creates an inertia chart from those saved values; it does not imply that elbow analysis originally selected K. Full per-cluster sizes/labels for K=3–8 were not retained, only min/max sizes; they cannot be reconstructed without fitting and are deliberately not invented.\n\nK=2 is supported as the best candidate **among 2–8 for these features, scaling and silhouette constraint**. It is not proven globally or commercially optimal. No stability/resampling study is saved. Sources: '+link(kp/'cluster_metrics.csv')+', '+link(kp/'selection.json')+' and `main` in K Means training.')
add('## 7. Customer cluster profiles\n\nThe complete CSV includes all 16 feature means, sizes and response rates plus new explicitly derived summaries. Income means in the original profiles omit restored missing/sentinel incomes; K Means fitting instead imputes them. Median income below is newly derived from the complete observed-income records after the same restoration, not a previously saved statistic. Total spending is newly derived by summing all six spending columns; counts and shares are derived from complete saved assignments.\n\n'+table(profiles[['cluster','size','share','Income','median_income_observed','income_missing_excluded','mean_total_spending','Recency','Kidhome','Teenhome','accepted','not_accepted','response_rate']])+'\n\nChannel/engagement comparisons:\n\n'+table(profiles[['cluster','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','Customer_Age']]))
low=profiles.iloc[0];high=profiles.iloc[1]
add('The supplied counts **1,304** and **933**, rounded incomes **38,257** and **70,988**, and response rates **10.0%** and **21.8%** are verified. A dollar sign is not established by the repository’s data dictionary; use “income (source currency units)” unless currency is independently confirmed.\n\nDerived from unrounded counts: absolute response-rate difference = '+f'{high.response_rate-low.response_rate:.12f}'+', or **'+f'{(high.response_rate-low.response_rate)*100:.8f} percentage points**. Response-rate ratio = **'+f'{(high.accepted/high["size"])/(low.accepted/low["size"]):.10f}'+ '**.\n\n“Lower Spending Customers” and “Higher Spending Customers” are empirically supported by all six product-spending means and total spending, with higher web/catalog/store purchase frequencies in Cluster 1. Cluster 0 has more young children and more website visits despite fewer purchases; that is a descriptive pattern, not proof of why purchasing differs. The response comparison is retrospective and in-sample for segmentation; it does not establish incremental campaign lift, targeting profitability or a causal income effect. Sources: '+link(kp/'cluster_profiles.csv')+', '+link(kp/'cluster_assignments.csv')+', [derived complete profiles](milestone2_ppt_support/cluster_profile_comparison.csv).')
add('## 8. Reproducibility and project architecture\n\n| Component | Verified implementation |\n|---|---|\n| Data loading/validation | `common/data_loader.py::load_data`, `data_validation.py::validate` |\n| Feature preparation | `common/preprocessing.py::features`, `preprocessor` |\n| Persisted split/CV | `common/split_manager.py::shared_split`, shared split.json and assignments.csv |\n| Classifier evaluation | `common/evaluation.py::run_classifier`, `metrics` |\n| Saving/metadata | `common/result_saver.py::new_run`, `metadata`, `save_json` |\n| Independent entry points | each model’s train.py builds its estimator; K Means separate |\n| Saved pipelines | all five successful run model.joblib files loaded in this audit |\n| Saved-result comparison | `comparison/compare_classifiers.py`, no fitting |\n| Existing verification | `verify_outputs.py` checks artifacts and writes EXECUTION_SUMMARY.md |\n| Cluster explanation | `kmeans/explain_results.py` reads saved results, no fitting |\n\nSaved metadata records Python '+meta[names[0]]['python_version']+', scikit-learn '+meta[names[0]]['sklearn_version']+', seed 42, dataset hashes, configs and split IDs. requirements.txt pins six numerical/model libraries, but is not a full environment lock (e.g. Pillow used by verification is not pinned). Current raw and processed hashes match retained successful metadata. No source-code commit/hash is captured in run metadata, so exact historical code equivalence is not proven. Current evaluation no longer emits historical ROC AUC and K Means current code calls a per-run explanation absent from this older run; complete artifact-for-artifact reproduction is therefore not established. Absolute Windows paths in saved configurations require adjustment on another machine.\n\nThe root `outputs/logistic_regression/run_20261013_185401_860960_f3ed22a6/` is a retained **failed** attempt: its error.txt records a FileNotFoundError from a wrongly resolved dataset path. It is excluded from performance tables. The successful model runs are under `experiments/outputs/`. Both saved comparison CSVs agree on their shared numeric results; the older comparison retains ROC AUC columns while the later one omits them.\n\nExisting EXECUTION_SUMMARY.md and the handoff document state verification passed. This is retained documentary evidence, not a timestamped standalone verification log. The audit did not run verify_outputs.py because it overwrites the existing execution summary. Instead these read-only checks **actually ran successfully**:\n\n'+ '\n'.join('- '+x for x in checks)+'\n\nNo predictions were regenerated from models, and no training reproducibility run was performed. Artifact loadability and consistency are verified; deterministic numerical reproduction across environments is not claimed.')
risks=pd.DataFrame([
('Potential Issue','Historical feature timing','AcceptedCmp1–5 and their sum are used; campaign chronology is not independently verified. Spending/recency/complaints must also precede the offer.','preprocessing.features; run feature lists'),
('Confirmed Issue','Timing convention inconsistency','Milestone 1 age uses 2026; model age uses 2014; cleaning excludes birth years using a 2026 age cutoff. Fixed 2014-12-31 is not a verified prediction timestamp.','milestone1_analysis.clean_data/engineer_features; preprocessing.features'),
('Not Evident From Available Evidence','Fold preprocessing leakage','Imputation/encoding/scaling fit inside cloned fold pipelines; previously global income imputations are restored before fitting. Unsupervised full-data clustering is independent.','data_loader.load_data; evaluation.run_classifier'),
('Not Evident From Available Evidence','Test-based threshold optimization','Thresholds use training OOF predictions; test threshold is already frozen.','evaluation.run_classifier; threshold_analysis/OOF CSVs'),
('Confirmed Issue','Threshold-development optimism','Same pooled OOF outcomes choose and score threshold; development performance is not independent validation.','threshold_development.json; evaluation.run_classifier'),
('Confirmed Issue','Class imbalance','334/2237 positives; only 67 positive test cases; accuracy alone favors the majority.','target counts; test_metrics baselines'),
('Potential Issue','Model ranking uncertainty','Small observed CV differences, one holdout, no repeated splits or paired significance evidence; SD is not a confidence interval.','cv_metrics.json; split.json'),
('Potential Issue','Feature importance interpretation','Correlated totals/components and impurity cardinality bias can redistribute apparent importance; causal claims unsupported.','feature lists; feature_importance.csv; permutation_importance.csv'),
('Potential Issue','Segmentation overinterpretation','Moderate silhouette, full-data descriptive responses, correlated spending dimensions and untested stability limit generalization/business claims.','cluster_metrics.csv; cluster_profiles.csv'),
('Confirmed Issue','Reporting/code drift','Historical ROC AUC persists but current code omits it; root README says planning phase; handoff lists TEAM_HANDOFF_SUMMARY.md but actual filename is Oct14_HANDOFF_SUMMARY.md.','test_metrics.json; evaluation.py; README.md; Oct14_HANDOFF_SUMMARY.md'),
('Potential Issue','Repeated future use of the holdout','Comparing reported models is descriptive; future tuning/selection using these viewed test scores would compromise fresh evaluation.','comparison summary; common test split'),
('Not Evident From Available Evidence','Cluster labels in classifiers','No cluster inputs or cluster-specific training exist in inspected successful workflows.','features metadata; train.py; evaluation.py')],columns=['Category','Issue','Evidence-based explanation','Source'])
add('## 9. Methodological risks and limitations\n\n'+table(risks)+'\n\nAdditional discrepancies/limits: 28 “original features” counts non-target raw columns including nonpredictive ID/constants; 33 describes total processed columns. Income=0 missing in saved audit precedes the restoration of 25 cells. Stored classification training metrics use threshold 0.50, not necessarily each selected threshold. Old comparison includes ROC AUC while current comparison omits it. Missing full K=3–8 cluster sizes are not evidence of equal-sized clusters. The current K Means source would produce per-run results_explanation.md, absent from the retained run; the separate kmeans/RESULTS_EXPLANATION.md exists. These differences are reported rather than silently normalized.')
add('## 10. Presentation-ready tables and visualizations\n\nAll support is in [milestone2_ppt_support](milestone2_ppt_support/). CSVs preserve substantially more precision than slide labels. Generated figures use consistent English labels and 300 DPI.\n\n'+table(pd.DataFrame([dict(file=p.name,purpose=('Table/data or verification' if p.suffix!='.png' else 'Presentation figure')) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='build_audit.py']))+'\n\nExisting PR curves remain in each original classifier run. The new inertia plot uses existing inertia; no new elbow decision was made. Unavailable: full candidate cluster sizes for K=3–8, historical per-fold fitted pipelines/dimensions, a verified source-currency definition, confidence intervals, stability tests and the actual presentation deck. No figures requiring these missing data were fabricated. The support builder is retained for transparency; it only reads repository evidence and writes the audit/support outputs.')
slides=[
(6,'2,237 records, 33 features, 25 missing income records','Needs Clarification','2,237 customers; 33 processed columns; 28 classification inputs before encoding, '+str(configuration.features_after_encoding.iloc[0])+' after encoding. 24 originally missing incomes + 1 sentinel restored and fold-imputed.','Section 1; raw/processed CSV; preprocessing; data_loader'),
(8,'Threshold optimization procedure (exact slide wording unavailable)','Unverifiable','Choose a separate threshold per model using training OOF F1 over 0.10–0.90 (step 0.05); freeze before final common-holdout evaluation.','evaluation.run_classifier; threshold_analysis.csv; threshold_development.json'),
(9,'Classification hyperparameters (exact slide wording unavailable)','Unverifiable','LR: balanced, lbfgs, max_iter 2000; DT: depth 5, leaf 10, balanced; RF: 300 trees, leaf 3, balanced; GB: 200 trees, rate .05, depth 3. Seed 42 throughout.','Section 2 table; saved config.json'),
(10,'CV F1, Test F1, Precision, Recall (slide numbers unavailable)','Unverifiable','Use Section 3 table: CV F1 at .50; Test F1/precision/recall at model-specific OOF-selected thresholds.','cv_metrics.json; test_metrics.json; validated prediction CSVs'),
(11,'Logistic Regression versus Random Forest recommendation','Needs Clarification','LR has highest mean training CV F1 (0.5756); RF has highest observed test F1 (0.5874). Evidence does not establish a statistically reliable winner.','Section 3; current comparison summary'),
(12,'16 K Means features; K=2–8; best silhouette .3038','Correct','16 standardized numerical inputs, all 2,237 customers; K=2 wins silhouette among K=2–8 subject to every cluster >=5%; silhouette 0.3038.','K Means config/train.py; cluster_metrics.csv; selection.json'),
(13,'Cluster 0:1304/income 38257/10.0%; Cluster 1:933/income 70988/21.8%','Correct','Lower-/higher-spending groups: 1,304 vs 933 customers; mean income 38,257 vs 70,988 source units; observed responses 10.0460% vs 21.7578%. Associations, not causal lift.','cluster_profiles.csv; cluster_assignments.csv; Section 7'),
(14,'Reproducibility and output verification','Needs Clarification','Saved configs, seed, hashes, common split/folds, predictions and pipelines support inspection. Audit consistency checks passed; no retraining reproduction performed; current code differs from historical output schema.','Section 8; audit_verification.txt; metadata'),
(16,'Outstanding issues and future work (exact slide wording unavailable)','Unverifiable','Confirm pre-offer timing/date assumptions; acknowledge imbalance, small holdout and ranking uncertainty; assess cluster stability; use fresh/nested validation for future tuning and business-cost policy.','Section 9; repository limitations')]
add('## 11. Slide-specific corrections\n\nActual slide text/figures were not supplied or found. “Correct” applies only to the numeric claim given in the request; topic-only claims remain unverifiable. Suggested replacements are fully evidence based.\n\n'+table(pd.DataFrame(slides,columns=['Slide','Current supplied claim','Status','Recommended PowerPoint wording','Evidence']))+'\n\nFor Slide 10 insert the consolidated performance table from Section 3 or its CSV; do not mix default and selected-threshold precision/recall. For Slide 13 avoid an unsupported dollar currency label and include the 11.7118 percentage-point descriptive difference.')
add('## 12. Final summary for presentation revision\n\n**Confirmed results.** Four independently trained classifiers share 1,789 training and 448 test customers, 28 original model inputs and '+str(configuration.features_after_encoding.iloc[0])+' encoded inputs under final encoders. LR leads default-threshold training CV F1; RF leads observed selected-threshold test F1 and precision; LR leads recall. Independent K Means uses 16 standardized inputs across 2,237 customers; K=2 has the best tested silhouette, 0.30376038.\n\n**Technical corrections.** Distinguish total columns from predictors; 24 missing raw incomes plus one sentinel from the 25 restored cells; CV threshold .50 from selected-threshold test metrics; training OOF threshold selection from test tuning; code/artifact consistency from retraining reproducibility. Historical ROC AUC is real saved evidence despite its removal from current reporting.\n\n**Business insights.** Higher-spending customers have substantially higher purchase counts and a 21.7578% observed acceptance rate versus 10.0460% in the lower-spending group. RF offers the highest observed precision; LR recovers the most actual responders. These facts motivate controlled targeting tests, not claims of incremental profit.\n\n**Priority figures.** Test F1, precision/recall, four confusion matrices, silhouette by K, and cluster counts/response rates. Use coefficient and RF/GB importance charts when discussing associations. Inertia is optional context, not the original selection rule.\n\n**Prioritized edits before presenting:**\n\n1. Correct Slides 6, 8 and 10 with verified feature counts, restoration/imputation and threshold labels.\n2. Populate Slide 9 from saved configurations and retain exact numeric CSVs as backup.\n3. Qualify Slide 11: LR provisional CV choice; RF strongest observed test-F1 alternative; no established winner.\n4. Keep Slides 12–13 numeric results, add candidate-range/minimum-size and descriptive-association qualifications.\n5. Revise Slide 14 to state the checks that actually ran, and disclose historical/current-code schema drift.\n6. Use Slide 16 for historical timing validation, date-reference consistency, fresh evaluation for future tuning, cluster stability, and cost/capacity-driven campaign decisions.\n\nNo model changes or new experiments are required to make these evidence-based corrections.')
(ROOT/'milestone2_ppt_audit.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
print('Created milestone2_ppt_audit.md and',len(list(OUT.glob('*.png'))),'verified 300-DPI figures.')
print(configuration[['model','features_after_encoding']].to_string(index=False))
print(profiles[['cluster','accepted','not_accepted','mean_total_spending','median_income_observed']].to_string(index=False))
