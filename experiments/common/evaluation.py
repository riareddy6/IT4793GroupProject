"""Model-independent validation; estimators are owned by individual modules."""
import traceback
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            average_precision_score, confusion_matrix, classification_report,
                            ConfusionMatrixDisplay, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay)
from sklearn.tree import plot_tree
from .data_loader import load_config, load_data
from .preprocessing import features, preprocessor
from .split_manager import shared_split
from .result_saver import new_run, metadata, save_json
from .settings import THRESHOLDS, SEED, PERMUTATION_REPEATS
from .visualization import plt, figure

def metrics(y, p, threshold=.5):
    """Calculate standardized binary classification and probability-ranking metrics.

    Args:
        y (array-like): Nonempty 1D binary true labels aligned to p.
        p (array-like): Finite positive-class probabilities in [0,1], same length as y.
        threshold (float): Inclusive probability cutoff in [0,1]; default 0.5.

    Returns:
        dict: Accuracy, precision, recall, positive F1, macro F1, AP, ROC AUC (None for one-class y) and [[TN,FP],[FN,TP]].

    Processing:
        Validate vectors; predict positive when p >= threshold; call sklearn metrics with zero_division=0 and fixed confusion labels [0,1]. AP uses recall-weighted precision, not trapezoidal integration.

    Raises:
        ValueError: For invalid labels, probabilities, alignment or threshold. sklearn may warn on one-class AP. No fitting or file output.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError('Labels and probabilities must be nonempty aligned vectors')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Binary labels and finite probabilities in [0, 1] are required')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0, 1]')
    pred = (p >= threshold).astype(int)
    return dict(accuracy=float(accuracy_score(y,pred)), precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)), f1=float(f1_score(y,pred,zero_division=0)),
                macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
                average_precision=float(average_precision_score(y,p)),
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def select_threshold(table):
    """Select a threshold from training-OOF metric rows.

    Args:
        table (pd.DataFrame): Nonempty threshold, f1 and precision columns.

    Returns:
        float: Threshold maximizing F1, then precision, then proximity to 0.50.

    Processing:
        Stable-sort development scores using the established tie-breaking policy.

    Raises:
        ValueError: If the development table is empty or contains invalid scores.
        KeyError: If required columns are absent. Does not read test data.
    """
    values = table[['threshold', 'f1', 'precision']].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Threshold development scores must be finite and in [0, 1]')
    chosen = table.assign(distance=(table.threshold-.5).abs()).sort_values(
        ['f1', 'precision', 'distance'], ascending=[False, False, True], kind='stable').iloc[0]
    return float(chosen.threshold)

def run_classifier(name, config_path, estimator, scale=False):
    """Train and evaluate one estimator using the persisted split and fold-local preprocessing.

    Args:
        name (str): Experiment identifier used for the new output subdirectory.
        config_path (str | Path): Validated model JSON configuration path.
        estimator (sklearn classifier): Unfitted binary estimator with predict_proba and coef_ or feature_importances_.
        scale (bool): Standardize numeric predictors when True; default False.

    Returns:
        Path: Newly created timestamped run containing pipeline, predictions, CV/test metrics, threshold evidence, plots and metadata.

    Processing:
        Load data and shared folds; clone preprocessing/model within five folds; choose threshold from pooled training OOF probabilities; fit full training set; freeze threshold before holdout scoring; save baselines, importance and first-fold permutation evidence.

    Raises:
        Exception: Training, data or serialization failures propagate after failed metadata/error.txt are saved. Writes a new run; does not overwrite existing runs. Captures CV/full-fit warnings.
    """
    config = load_config(config_path)
    out = new_run(name,config)
    meta = {}
    try:
        data, audit = load_data(config)
        x, y = features(data,config), data.Response
        train, test, folds, split_id = shared_split(data,config)
        meta = metadata(config,data,split_id)
        meta.update(features=x.columns.tolist(), feature_count=len(x.columns), train_rows=len(train), test_rows=len(test), data_audit=audit)
        xt, yt = x.iloc[train], y.iloc[train]
        pipe = Pipeline([('preprocessor',preprocessor(x,scale)), ('model',estimator)])
        oof = np.empty(len(train))
        fold_metrics, recorded = [], []
        for a,b in folds:
            fitted = clone(pipe)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fitted.fit(xt.iloc[a],yt.iloc[a])
            recorded.extend(str(w.message) for w in caught)
            oof[b] = fitted.predict_proba(xt.iloc[b])[:,1]
            fold_metrics.append(metrics(yt.iloc[b],oof[b]))
        keys = [k for k in fold_metrics[0] if k != 'confusion_matrix']
        save_json(out / 'cv_metrics.json', dict(threshold=.5, folds=fold_metrics,
                  mean={k:float(np.mean([f[k] for f in fold_metrics])) for k in keys},
                  std={k:float(np.std([f[k] for f in fold_metrics],ddof=1)) for k in keys}))
        thresholds = pd.DataFrame([dict(threshold=float(t), **{k:v for k,v in metrics(yt,oof,t).items() if k!='confusion_matrix'}) for t in THRESHOLDS])
        thresholds.to_csv(out / 'threshold_analysis.csv',index=False)
        # Stable tie-break: higher precision, then closest to default.
        threshold = select_threshold(thresholds)
        save_json(out / 'threshold_development.json', dict(label='development performance, not independent validation',
                  selection='maximum training OOF positive F1; ties: precision then proximity to 0.50', threshold=threshold,
                  metrics=metrics(yt,oof,threshold)))
        pd.DataFrame({'row_index':train,'ID':data.ID.iloc[train].values,'true_label':yt.values,'oof_probability':oof}).to_csv(out / 'oof_predictions.csv',index=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipe.fit(xt,yt)
        recorded.extend(str(w.message) for w in caught)
        save_json(out / 'warnings.json',recorded)
        # Freeze configuration and threshold before the single test probability call.
        joblib.dump(dict(pipeline=pipe, threshold=threshold, feature_columns=x.columns.tolist(), reference_date=config['reference_date']), out / 'model.joblib')
        prob = pipe.predict_proba(x.iloc[test])[:,1]
        final = dict(default_threshold=metrics(y.iloc[test],prob), selected_threshold=threshold,
                     selected=metrics(y.iloc[test],prob,threshold), training=metrics(yt,pipe.predict_proba(xt)[:,1]))
        baselines = {}
        for strategy in ('most_frequent','stratified'):
            dummy = DummyClassifier(strategy=strategy,random_state=42).fit(np.zeros((len(train),1)),yt)
            baselines[strategy] = metrics(y.iloc[test],dummy.predict_proba(np.zeros((len(test),1)))[:,1])
        final['baselines'] = baselines
        save_json(out / 'test_metrics.json',final)
        pred = (prob>=threshold).astype(int)
        pd.DataFrame({'row_index':test,'ID':data.ID.iloc[test].values,'true_label':y.iloc[test].values,
                      'predicted_probability':prob,'predicted_class':pred,'selected_threshold':threshold,
                      'default_predicted_class':(prob>=.5).astype(int)}).to_csv(out / 'test_predictions.csv',index=False)
        report = classification_report(y.iloc[test],pred,zero_division=0)
        (out / 'classification_report.txt').write_text(report)
        ConfusionMatrixDisplay.from_predictions(y.iloc[test],pred); figure(out / 'confusion_matrix.png')
        PrecisionRecallDisplay.from_predictions(y.iloc[test],prob); figure(out / 'precision_recall_curve.png')
        RocCurveDisplay.from_predictions(y.iloc[test],prob,plot_chance_level=True); figure(out / 'roc_curve.png')
        names = pipe.named_steps['preprocessor'].get_feature_names_out()
        model = pipe.named_steps['model']
        coefficient = hasattr(model,'coef_')
        values = model.coef_[0] if coefficient else model.feature_importances_
        importance = pd.DataFrame({'feature':names,'importance':values,'method':'coefficient' if coefficient else 'impurity', 'feature_space':'transformed'})
        importance.to_csv(out / 'feature_importance.csv',index=False)
        top = importance.assign(magnitude=importance.importance.abs()).nlargest(20,'magnitude').sort_values('importance')
        plt.figure(figsize=(9,7)); plt.barh(top.feature,top.importance); plt.title(name + ': ' + ('signed coefficients' if coefficient else 'impurity importance')); figure(out / 'feature_importance.png')
        # Permutation on a saved validation fold; no extra use of the independent test set.
        a,b = folds[0]
        val_model = clone(pipe).fit(xt.iloc[a],yt.iloc[a])
        perm = permutation_importance(val_model,xt.iloc[b],yt.iloc[b],scoring='f1',n_repeats=PERMUTATION_REPEATS,random_state=SEED,n_jobs=1)
        pd.DataFrame({'feature':x.columns,'importance':perm.importances_mean,'std':perm.importances_std,'method':'validation_permutation_f1','feature_space':'original'}).to_csv(out / 'permutation_importance.csv',index=False)
        if name=='decision_tree':
            plt.figure(figsize=(32,16)); plot_tree(model,feature_names=names,class_names=['No','Yes'],filled=True,rounded=True,fontsize=7); figure(out / 'decision_tree.png')
        (out / 'model_behavior.md').write_text('Top features: '+', '.join(top.sort_values('magnitude',ascending=False).feature.head(8))+'.\nCoefficients describe conditional log odds per scaled numeric unit; correlated predictors can redistribute effects. Impurity importance favors high-cardinality predictors. Validation permutation measures F1 loss at threshold 0.50 and is descriptive. None establish causation.\n')
        meta['status']='complete'
        save_json(out / 'run_metadata.json',meta)
        print(out)
        return out
    except Exception:
        meta['status']='failed'
        save_json(out / 'run_metadata.json',meta)
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
