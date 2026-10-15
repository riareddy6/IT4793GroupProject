"""Write a standalone interpretation from saved K Means results; no fitting."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def write_explanation(run, output):
    run = Path(run)
    meta = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'complete':
        raise ValueError('A completed K Means run is required')
    with (run / 'cluster_profiles.csv').open(encoding='utf-8', newline='') as f:
        profiles = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    with (run / 'cluster_metrics.csv').open(encoding='utf-8', newline='') as f:
        candidates = list(csv.DictReader(f))
    selected = next(row for row in candidates if int(row['k']) == meta['selected_k'])
    lines = ['# K Means Customer Segmentation: Results and Interpretation', '',
             f'Source run: `{run.name}`. Dataset SHA256: `{meta["dataset_sha256"]}`.', '',
             '## Purpose and inputs', '',
             f'This independent experiment groups {meta["rows"]:,} customers using {meta["feature_count"]} numerical behavioral and demographic features. It does not predict Response.', '',
             'Inputs: ' + ', '.join(f'`{c}`' for c in meta['features']) + '.', '',
             'Missing numerical values are median-imputed and all inputs are standardized before Euclidean-distance clustering. ID, Response, campaign acceptance indicators and Past_Campaign_Responses are excluded.', '',
             '## Cluster-count selection', '',
             f'Candidate counts 2 through 8 were fitted with n_init=20 and random_state=42. The selected count is **k={meta["selected_k"]}**, with silhouette **{float(selected["silhouette"]):.4f}**. Selection maximizes silhouette among solutions in which every cluster contains at least 5% of customers; ties favor smaller k. Response is not used in this process.', '',
             '| k | Silhouette | Smallest cluster | Largest cluster | Eligible |',
             '|---|---:|---:|---:|---|']
    for row in candidates:
        lines.append(f'| {row["k"]} | {float(row["silhouette"]):.4f} | {row["min_cluster_size"]} | {row["max_cluster_size"]} | {row["eligible"]} |')
    lines += ['', '## Profiles in original units', '',
              'Income means below omit originally missing/sentinel values restored before imputation; clustering itself uses imputed values. Other means are from the original feature values. Currency units are those of the source data.', '',
              '| Cluster | Customers | Share | Mean income | Wine spending | Meat spending | Web purchases | Store purchases | Kidhome | Response rate |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in profiles:
        lines.append(f'| {int(r["cluster"])} | {int(r["size"]):,} | {r["size"]/meta["rows"]:.1%} | {r["Income"]:,.0f} | {r["MntWines"]:.1f} | {r["MntMeatProducts"]:.1f} | {r["NumWebPurchases"]:.2f} | {r["NumStorePurchases"]:.2f} | {r["Kidhome"]:.2f} | {r["response_rate"]:.1%} |')
    if len(profiles)==2:
        low,high=sorted(profiles,key=lambda r:r['Income'])
        lines += ['', f'**Cluster {int(low["cluster"])}: lower-income, lower-spending customers.** This group contains {int(low["size"]):,} customers, has fewer web and store purchases, and more young children per household on average. Its higher mean website visit count despite fewer web purchases suggests a browsing-to-purchase pattern worth investigating; it does not establish why customers buy less.', '',
                  f'**Cluster {int(high["cluster"])}: higher-income, higher-spending customers.** This group contains {int(high["size"]):,} customers and has greater product spending and purchase frequency. Premium or cross-sell offers are hypotheses to test with controlled campaigns.', '',
                  f'Observed response rates are {low["response_rate"]:.1%} and {high["response_rate"]:.1%}, a difference of {(high["response_rate"]-low["response_rate"])*100:.1f} percentage points. These rates were calculated after fitting and cluster-count selection. They are descriptive associations from this dataset, not independent estimates of future campaign performance.']
    lines += ['', '## Relationship to the four classifiers', '',
              'Logistic Regression, Decision Tree, Random Forest and Gradient Boosting read the Milestone 1 processed dataset directly and apply their own shared classification preprocessing. They do not read cluster_assignments.csv, use cluster labels, or train separately within clusters. K Means runs independently on the same source customers; its outputs are not an upstream transformation for classification.', '',
              '## Limitations and next steps', '',
              'Cluster labels are arbitrary. Silhouette compares compactness and separation under the chosen scaling and features; it does not prove business usefulness. Related spending and purchase features can give these behaviors greater weight. Results may change with outliers, feature choices and resampling. The age reference is the fixed year 2014, not a verified offer date.', '',
              'Review the profiles with marketing stakeholders, examine stability across seeds/resamples and test segment-specific offers. If cluster labels are later added to classification, fit the clustering pipeline only on each applicable training fold, then assign validation/test customers using that fitted pipeline to avoid leakage.', '',
              'See cluster_profiles.csv, cluster_metrics.csv and the three cluster charts in the source run for supporting results.']
    output=Path(output)
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'kmeans'/'RESULTS_EXPLANATION.md')
    args=parser.parse_args()
    run=args.run
    if run is None:
        run=next((p for p in sorted((ROOT/'outputs'/'kmeans').glob('run_*'),reverse=True)
                  if (p/'run_metadata.json').exists() and json.loads((p/'run_metadata.json').read_text())['status']=='complete'),None)
    if run is None:
        raise ValueError('No completed K Means run')
    print(write_explanation(run,args.output))

if __name__=='__main__':
    main()
