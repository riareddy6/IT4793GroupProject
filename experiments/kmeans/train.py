"""Independent behavioral segmentation; Response is used only after selection."""
import argparse
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
from experiments.common.data_loader import load_config, load_data
from experiments.common.preprocessing import features, preprocessor
from experiments.common.result_saver import new_run, metadata, save_json
from experiments.common.visualization import figure, plt
from experiments.kmeans.explain_results import write_explanation

def main():
    """Fit and select an independent behavioral K-Means segmentation.

    Args:
        None: CLI --config defaults to config.json beside this module.

    Returns:
        None: Prints the generated artifact directory or report; command-line side effects described below.

    Processing:
        Load/standardize all customer inputs; fit configured K range with fixed seeds; maximize eligible silhouette; save model, profiles, assignments, charts and explanation.

    Raises:
        Exception: Errors propagate after failed metadata and traceback are saved. Writes a new run. Response is attached after fitting and K selection.
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    args=parser.parse_args()
    c=load_config(args.config)
    out=new_run('kmeans',c)
    meta={}
    try:
        data,audit=load_data(c)
        x=features(data,c,clustering=True)
        meta=metadata(c,data)
        meta.update(features=x.columns.tolist(),feature_count=len(x.columns),data_audit=audit)
        prep=preprocessor(x,True)
        z=prep.fit_transform(x)
        rows=[]; models={}
        with threadpool_limits(limits=1):
            for k in range(c['k_min'],c['k_max']+1):
                model=KMeans(n_clusters=k,n_init=c['n_init'],random_state=c['random_state']).fit(z)
                sizes=np.bincount(model.labels_,minlength=k)
                rows.append(dict(k=k,silhouette=float(silhouette_score(z,model.labels_)),inertia=float(model.inertia_),
                                 min_cluster_size=int(sizes.min()),max_cluster_size=int(sizes.max()),
                                 eligible=bool(sizes.min()/len(data)>=c['min_cluster_fraction'])))
                models[k]=model
        table=pd.DataFrame(rows)
        eligible=table[table.eligible]
        if eligible.empty:
            raise ValueError('No cluster count satisfies minimum size constraint')
        selected=int(eligible.sort_values(['silhouette','k'],ascending=[False,True]).iloc[0].k)
        model=models[selected]
        labels=model.labels_
        table.to_csv(out/'cluster_metrics.csv',index=False)
        joblib.dump(dict(pipeline=Pipeline([('preprocessor',prep),('model',model)]),feature_columns=x.columns.tolist()),out/'model.joblib')
        pd.DataFrame({'row_index':data.index,'ID':data.ID,'cluster':labels,'Response':data.Response}).to_csv(out/'cluster_assignments.csv',index=False)
        profiles=x.assign(cluster=labels).groupby('cluster').mean()
        profiles['size']=pd.Series(labels).value_counts().sort_index()
        profiles['response_rate']=data.assign(cluster=labels).groupby('cluster').Response.mean()
        profiles.to_csv(out/'cluster_profiles.csv')
        plt.figure();plt.plot(table.k,table.silhouette,marker='o');plt.xlabel('k');plt.ylabel('Silhouette');figure(out/'silhouette_scores.png')
        profiles['size'].plot.bar(title='Cluster sizes');figure(out/'cluster_size_distribution.png')
        profiles.response_rate.plot.bar(title='Descriptive observed Response rates');figure(out/'cluster_response_rates.png')
        save_json(out/'selection.json',dict(k=selected,method='highest silhouette subject to every cluster >= 5% of customers; ties favor lower k',
                  interpretability='Inspect original-unit profiles for spending, purchasing, household and income patterns; clusters are descriptive, not causal.',response_used_for_selection=False))
        meta.update(status='complete',selected_k=selected)
        save_json(out/'run_metadata.json',meta)
        write_explanation(out,out/'results_explanation.md')
        print(out)
    except Exception:
        meta['status']='failed';save_json(out/'run_metadata.json',meta)
        (out/'error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':
    main()
