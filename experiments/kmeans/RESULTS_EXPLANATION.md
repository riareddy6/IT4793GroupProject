# K Means Customer Segmentation: Results and Interpretation

Source run: `run_20261013_185820_218800_afad3148`. Dataset SHA256: `be37fcb6f617302588e2910eccb8ca23a5b45450713c2d2a2d3898eee54207d5`.

## Purpose and inputs

This independent experiment groups 2,237 customers using 16 numerical behavioral and demographic features. It does not predict Response.

Inputs: `Income`, `Kidhome`, `Teenhome`, `Customer_Age`, `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`, `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`.

Missing numerical values are median-imputed and all inputs are standardized before Euclidean-distance clustering. ID, Response, campaign acceptance indicators and Past_Campaign_Responses are excluded.

## Cluster-count selection

Candidate counts 2 through 8 were fitted with n_init=20 and random_state=42. The selected count is **k=2**, with silhouette **0.3038**. Selection maximizes silhouette among solutions in which every cluster contains at least 5% of customers; ties favor smaller k. Response is not used in this process.

| k | Silhouette | Smallest cluster | Largest cluster | Eligible |
|---|---:|---:|---:|---|
| 2 | 0.3038 | 933 | 1304 | True |
| 3 | 0.2341 | 596 | 1008 | True |
| 4 | 0.1651 | 537 | 579 | True |
| 5 | 0.1566 | 261 | 571 | True |
| 6 | 0.1594 | 176 | 540 | True |
| 7 | 0.1520 | 146 | 537 | True |
| 8 | 0.1538 | 143 | 536 | True |

## Profiles in original units

Income means below omit originally missing/sentinel values restored before imputation; clustering itself uses imputed values. Other means are from the original feature values. Currency units are those of the source data.

| Cluster | Customers | Share | Mean income | Wine spending | Meat spending | Web purchases | Store purchases | Kidhome | Response rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,304 | 58.3% | 38,257 | 97.0 | 35.4 | 2.90 | 3.83 | 0.71 | 10.0% |
| 1 | 933 | 41.7% | 70,988 | 593.3 | 350.7 | 5.75 | 8.54 | 0.07 | 21.8% |

**Cluster 0: lower-income, lower-spending customers.** This group contains 1,304 customers, has fewer web and store purchases, and more young children per household on average. Its higher mean website visit count despite fewer web purchases suggests a browsing-to-purchase pattern worth investigating; it does not establish why customers buy less.

**Cluster 1: higher-income, higher-spending customers.** This group contains 933 customers and has greater product spending and purchase frequency. Premium or cross-sell offers are hypotheses to test with controlled campaigns.

Observed response rates are 10.0% and 21.8%, a difference of 11.7 percentage points. These rates were calculated after fitting and cluster-count selection. They are descriptive associations from this dataset, not independent estimates of future campaign performance.

## Relationship to the four classifiers

Logistic Regression, Decision Tree, Random Forest and Gradient Boosting read the Milestone 1 processed dataset directly and apply their own shared classification preprocessing. They do not read cluster_assignments.csv, use cluster labels, or train separately within clusters. K Means runs independently on the same source customers; its outputs are not an upstream transformation for classification.

## Limitations and next steps

Cluster labels are arbitrary. Silhouette compares compactness and separation under the chosen scaling and features; it does not prove business usefulness. Related spending and purchase features can give these behaviors greater weight. Results may change with outliers, feature choices and resampling. The age reference is the fixed year 2014, not a verified offer date.

Review the profiles with marketing stakeholders, examine stability across seeds/resamples and test segment-specific offers. If cluster labels are later added to classification, fit the clustering pipeline only on each applicable training fold, then assign validation/test customers using that fitted pipeline to avoid leakage.

See cluster_profiles.csv, cluster_metrics.csv and the three cluster charts in the source run for supporting results.
