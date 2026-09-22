# Milestone 1: Data Ready for Analysis

## Dataset Overview

- Source: Kaggle Customer Personality Analysis (`data/marketing_campaign.csv`)
- Raw shape: **2,240 rows x 29 columns**
- Processed shape: **2,237 rows x 33 columns**
- Target variable: **`Response`**, where `1` means the customer accepted the current campaign and `0` means they did not.
- The raw file was read with tab separation and was not modified.

### Column Data Types

- `ID`: `int64`
- `Year_Birth`: `int64`
- `Education`: `str`
- `Marital_Status`: `str`
- `Income`: `float64`
- `Kidhome`: `int64`
- `Teenhome`: `int64`
- `Dt_Customer`: `str`
- `Recency`: `int64`
- `MntWines`: `int64`
- `MntFruits`: `int64`
- `MntMeatProducts`: `int64`
- `MntFishProducts`: `int64`
- `MntSweetProducts`: `int64`
- `MntGoldProds`: `int64`
- `NumDealsPurchases`: `int64`
- `NumWebPurchases`: `int64`
- `NumCatalogPurchases`: `int64`
- `NumStorePurchases`: `int64`
- `NumWebVisitsMonth`: `int64`
- `AcceptedCmp3`: `int64`
- `AcceptedCmp4`: `int64`
- `AcceptedCmp5`: `int64`
- `AcceptedCmp1`: `int64`
- `AcceptedCmp2`: `int64`
- `Complain`: `int64`
- `Z_CostContact`: `int64`
- `Z_Revenue`: `int64`
- `Response`: `int64`

## Data Quality Issues

### Missing Values

- `Income`: 24

- Raw missing `Income` values: **24**
- Raw duplicate rows: **0**
- Date parsing failures for `Dt_Customer`: **0**
- Negative numeric values: **0**
- The target contains values `[0, 1]` and has **0** missing values.

### Potential Format and Outlier Issues

- Three `Year_Birth` values were before 1926, implying an age over 100 in 2026; these records were removed as clear year-entry errors.
- `Income = 666666` occurred once and was treated as an isolated data-entry sentinel. Other high incomes were retained because they may represent genuine customers.
- `Education` and `Marital_Status` are categorical strings. Date values match the expected `DD-MM-YYYY` format.

## Data Cleaning Performed

- Removed **3** rows with `Year_Birth < 1926`; reason: implied age over 100 in 2026.
- Replaced **1** sentinel income value(s) with missing, then filled **25** missing income values using the cleaned-data median of **51,371.00**. Median was used because Income is right-skewed and this preserves valid customer records without letting extreme values dominate.
- Removed **0** duplicate rows after cleaning.
- Final missing-value count: **0**; final duplicate count: **0**.

## Feature Engineering

- `Customer_Age = 2026 - Year_Birth`
- `Total_Spending` is the sum of the six product spending columns.
- `Number_of_Children = Kidhome + Teenhome`
- `Past_Campaign_Responses` is the sum of `AcceptedCmp1` through `AcceptedCmp5` only. The current `Response` target was excluded to prevent target leakage.

## EDA Completed

- Descriptive statistics: `reports/descriptive_statistics.csv`
- Correlation matrix: `reports/correlation_matrix.csv`
- Supporting relationship chart: `acceptance_rate_by_past_campaign_responses.png`

### Milestone 1 Presentation Visualizations

- `campaign_acceptance_distribution.png`
- `customer_age_distribution.png`
- `total_spending_by_campaign_acceptance.png`
- `income_by_campaign_acceptance.png`
- `acceptance_rate_by_number_of_children.png`
- `correlation_matrix.png`

### Supporting Relationship Visualizations

- `customer_age_by_campaign_acceptance.png`
- `acceptance_rate_by_past_campaign_responses.png`

All charts are PNG files saved at 300 DPI under `docs/visualizations`.

## Preliminary Findings

- The current campaign acceptance rate is 14.9% (334 of 2,237 customers), so the target is meaningfully imbalanced toward non-acceptance.
- Accepted customers have a median Total_Spending of 1,058, compared with 315 for non-accepted customers (a difference of 742).
- The median Income difference between accepted and non-accepted customers is 13,521 (64,044 versus 50,523).
- Accepted customers have a median age of 55, compared with 56 for non-accepted customers (a difference of -1 years); children-group acceptance rates range from 3.8% to 26.5%.
- Past campaign responses show a 82.7% spread between the lowest and highest observed current acceptance rates, making this a useful candidate predictor for later modeling.

## Data Readiness Status

**READY FOR ANALYSIS.** The processed dataset was exported to `processed_data/marketing_campaign_processed.csv`. The raw data was successfully read, documented cleaning was completed, required features were created, `Response` is complete, EDA ran successfully, and presentation-ready visualizations were generated. No classification, clustering, or other Milestone 2 modeling was performed.
