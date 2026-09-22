"""Milestone 1: prepare the Customer Personality Analysis data for analysis.

This script reads the raw file without modifying it, performs documented data
quality checks and cleaning, creates project features, runs descriptive EDA,
and writes all generated artifacts below the docs directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "marketing_campaign.csv"
OUTPUT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = OUTPUT_ROOT / "processed_data"
REPORT_DIR = OUTPUT_ROOT / "reports"
VISUALIZATION_DIR = OUTPUT_ROOT / "visualizations"
TARGET = "Response"

SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
HISTORICAL_CAMPAIGN_COLUMNS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]


def prepare_output_directories() -> None:
    """Create the output directories while leaving the raw data untouched."""
    for directory in (PROCESSED_DIR, REPORT_DIR, VISUALIZATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """Load the tab-separated Kaggle file and validate the expected target."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset was not found: {RAW_DATA_PATH}")
    data = pd.read_csv(RAW_DATA_PATH, sep="\t")
    if TARGET not in data.columns:
        raise ValueError("The target column 'Response' is missing from the raw data.")
    return data


def inspect_data(data: pd.DataFrame) -> dict:
    """Return initial inspection results for the quality report."""
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    date_parse_failures = int(
        pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y", errors="coerce").isna().sum()
    )
    negative_counts = data[numeric_columns].lt(0).sum()
    return {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "columns": data.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": {column: int(count) for column, count in data.isna().sum().items()},
        "duplicate_count": int(data.duplicated().sum()),
        "date_parse_failures": date_parse_failures,
        "negative_numeric_values": {
            column: int(count) for column, count in negative_counts.items() if count > 0
        },
        "year_birth_min": int(data["Year_Birth"].min()),
        "year_birth_max": int(data["Year_Birth"].max()),
        "income_min": float(data["Income"].min()),
        "income_max": float(data["Income"].max()),
        "income_above_150000": int((data["Income"] > 150000).sum()),
        "income_sentinel_666666": int((data["Income"] == 666666).sum()),
        "target_values": sorted(data[TARGET].dropna().unique().tolist()),
        "target_missing": int(data[TARGET].isna().sum()),
    }


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply conservative, documented cleaning rules and return an audit."""
    cleaned = data.copy()
    audit = {
        "rows_before": int(len(cleaned)),
        "invalid_year_rows_removed": 0,
        "invalid_year_rule": "Removed Year_Birth values before 1926 (age over 100 in 2026).",
        "income_sentinel_values_replaced": 0,
        "income_missing_before_imputation": int(cleaned["Income"].isna().sum()),
        "income_missing_after_imputation": 0,
        "duplicate_rows_removed": 0,
        "negative_numeric_values_found": int(cleaned.select_dtypes(include=np.number).lt(0).sum().sum()),
    }

    invalid_year_mask = cleaned["Year_Birth"] < 1926
    audit["invalid_year_rows_removed"] = int(invalid_year_mask.sum())
    cleaned = cleaned.loc[~invalid_year_mask].copy()

    # 666666 is an isolated data-entry sentinel, so treat it as missing rather
    # than deleting the otherwise valid customer row.
    sentinel_mask = cleaned["Income"] == 666666
    audit["income_sentinel_values_replaced"] = int(sentinel_mask.sum())
    cleaned.loc[sentinel_mask, "Income"] = np.nan

    # Income is right-skewed and only 24 raw values are missing; median
    # imputation preserves every usable customer record and resists extremes.
    income_median = float(cleaned["Income"].median())
    cleaned["Income"] = cleaned["Income"].fillna(income_median)
    audit["income_imputation_value"] = income_median
    audit["income_missing_after_imputation"] = int(cleaned["Income"].isna().sum())

    duplicate_mask = cleaned.duplicated()
    audit["duplicate_rows_removed"] = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()
    audit["rows_after"] = int(len(cleaned))
    audit["total_rows_removed"] = audit["rows_before"] - audit["rows_after"]
    return cleaned, audit


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the project-plan features without mixing the current target in."""
    engineered = data.copy()
    engineered["Customer_Age"] = 2026 - engineered["Year_Birth"]
    engineered["Total_Spending"] = engineered[SPENDING_COLUMNS].sum(axis=1)
    engineered["Number_of_Children"] = engineered["Kidhome"] + engineered["Teenhome"]
    if all(column in engineered.columns for column in HISTORICAL_CAMPAIGN_COLUMNS):
        engineered["Past_Campaign_Responses"] = engineered[HISTORICAL_CAMPAIGN_COLUMNS].sum(axis=1)
    else:
        raise ValueError("AcceptedCmp1 through AcceptedCmp5 are required for Past_Campaign_Responses.")
    return engineered


def save_quality_outputs(inspection: dict, audit: dict, data: pd.DataFrame) -> None:
    """Write machine-readable inspection and cleaning results."""
    quality = {"initial_inspection": inspection, "cleaning_audit": audit}
    (REPORT_DIR / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    dtype_table = pd.DataFrame(
        {"column": data.columns, "dtype": [str(dtype) for dtype in data.dtypes]}
    )
    dtype_table.to_csv(REPORT_DIR / "column_dtypes.csv", index=False)
    missing_table = pd.DataFrame(
        {
            "column": inspection["columns"],
            "missing_count": [inspection["missing_values"][column] for column in inspection["columns"]],
        }
    )
    missing_table.to_csv(REPORT_DIR / "missing_value_counts.csv", index=False)


def save_visualizations(data: pd.DataFrame) -> list[str]:
    """Create presentation-ready EDA charts and return their filenames."""
    sns.set_theme(style="whitegrid", context="notebook")
    created: list[str] = []

    def save_figure(filename: str) -> None:
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()
        created.append(filename)

    response_labels = {0: "Did not accept", 1: "Accepted"}
    response_counts = data[TARGET].map(response_labels).value_counts().reindex(response_labels.values())
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=response_counts.index, y=response_counts.values, hue=response_counts.index, palette=["#5B7083", "#D97941"], legend=False)
    ax.set(title="Campaign Acceptance Distribution", xlabel="Current campaign response", ylabel="Number of customers")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    save_figure("campaign_acceptance_distribution.png")

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data=data, x="Customer_Age", bins=20, color="#3B6E8F", edgecolor="white")
    ax.axvline(data["Customer_Age"].median(), color="#D97941", linestyle="--", label=f"Median: {data['Customer_Age'].median():.0f}")
    ax.set(title="Customer Age Distribution", xlabel="Customer age in 2026", ylabel="Number of customers")
    ax.legend()
    save_figure("customer_age_distribution.png")

    plt.figure(figsize=(8, 5))
    age_data = data.assign(Response_Label=data[TARGET].map(response_labels))
    ax = sns.boxplot(data=age_data, x="Response_Label", y="Customer_Age", hue="Response_Label", palette=["#5B7083", "#D97941"], legend=False)
    ax.set(title="Customer Age by Campaign Acceptance", xlabel="Current campaign response", ylabel="Customer age in 2026")
    save_figure("customer_age_by_campaign_acceptance.png")

    for column, title, filename, ylabel in [
        ("Total_Spending", "Total Spending by Campaign Acceptance", "total_spending_by_campaign_acceptance.png", "Total spending"),
        ("Income", "Income by Campaign Acceptance", "income_by_campaign_acceptance.png", "Income"),
    ]:
        plt.figure(figsize=(8, 5))
        plot_data = data.assign(Response_Label=data[TARGET].map(response_labels))
        ax = sns.boxplot(data=plot_data, x="Response_Label", y=column, hue="Response_Label", palette=["#5B7083", "#D97941"], legend=False)
        ax.set(title=title, xlabel="Current campaign response", ylabel=ylabel)
        save_figure(filename)

    children_rate = data.groupby("Number_of_Children", as_index=False)[TARGET].mean()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=children_rate, x="Number_of_Children", y=TARGET, color="#3B6E8F")
    ax.set(title="Acceptance Rate by Number of Children", xlabel="Number of children", ylabel="Acceptance rate")
    ax.set_ylim(0, max(0.3, float(children_rate[TARGET].max()) * 1.2))
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    save_figure("acceptance_rate_by_number_of_children.png")

    correlation_columns = [
        "Income", "Recency", "Total_Spending", "Number_of_Children",
        "Past_Campaign_Responses", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", TARGET,
    ]
    correlation = data[correlation_columns].corr()
    correlation.to_csv(REPORT_DIR / "correlation_matrix.csv")
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(correlation, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    ax.set(title="Correlation Matrix of Key Numeric Variables")
    save_figure("correlation_matrix.png")

    past_rate = data.groupby("Past_Campaign_Responses", as_index=False)[TARGET].mean()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=past_rate, x="Past_Campaign_Responses", y=TARGET, color="#D97941")
    ax.set(title="Acceptance Rate by Past Campaign Responses", xlabel="Past campaign responses", ylabel="Current acceptance rate")
    ax.set_ylim(0, max(0.3, float(past_rate[TARGET].max()) * 1.2))
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    save_figure("acceptance_rate_by_past_campaign_responses.png")
    return created


def create_preliminary_findings(data: pd.DataFrame) -> list[str]:
    """Calculate concise findings from the processed data rather than assumptions."""
    response_rate = data[TARGET].mean()
    accepted = data.loc[data[TARGET] == 1]
    not_accepted = data.loc[data[TARGET] == 0]
    age_difference = accepted["Customer_Age"].median() - not_accepted["Customer_Age"].median()
    spending_difference = accepted["Total_Spending"].median() - not_accepted["Total_Spending"].median()
    income_difference = accepted["Income"].median() - not_accepted["Income"].median()
    children_rates = data.groupby("Number_of_Children")[TARGET].mean()
    past_rates = data.groupby("Past_Campaign_Responses")[TARGET].mean()
    past_gap = past_rates.max() - past_rates.min()
    return [
        f"The current campaign acceptance rate is {response_rate:.1%} ({int(data[TARGET].sum()):,} of {len(data):,} customers), so the target is meaningfully imbalanced toward non-acceptance.",
        f"Accepted customers have a median Total_Spending of {accepted['Total_Spending'].median():,.0f}, compared with {not_accepted['Total_Spending'].median():,.0f} for non-accepted customers (a difference of {spending_difference:,.0f}).",
        f"The median Income difference between accepted and non-accepted customers is {income_difference:,.0f} ({accepted['Income'].median():,.0f} versus {not_accepted['Income'].median():,.0f}).",
        f"Accepted customers have a median age of {accepted['Customer_Age'].median():.0f}, compared with {not_accepted['Customer_Age'].median():.0f} for non-accepted customers (a difference of {age_difference:.0f} years); children-group acceptance rates range from {children_rates.min():.1%} to {children_rates.max():.1%}.",
        f"Past campaign responses show a {past_gap:.1%} spread between the lowest and highest observed current acceptance rates, making this a useful candidate predictor for later modeling.",
    ]


def create_summary(inspection: dict, audit: dict, data: pd.DataFrame, charts: list[str]) -> None:
    """Write the English Milestone 1 documentation."""
    findings = create_preliminary_findings(data)
    dtype_lines = "\n".join(f"- `{column}`: `{dtype}`" for column, dtype in inspection["dtypes"].items())
    missing_lines = "\n".join(
        f"- `{column}`: {count}" for column, count in inspection["missing_values"].items() if count > 0
    ) or "- No missing values detected."
    chart_lines = "\n".join(f"- `{chart}`" for chart in charts)
    summary = f"""# Milestone 1: Data Ready for Analysis

## Dataset Overview

- Source: Kaggle Customer Personality Analysis (`data/marketing_campaign.csv`)
- Raw shape: **{inspection['shape']['rows']:,} rows x {inspection['shape']['columns']} columns**
- Processed shape: **{len(data):,} rows x {data.shape[1]} columns**
- Target variable: **`Response`**, where `1` means the customer accepted the current campaign and `0` means they did not.
- The raw file was read with tab separation and was not modified.

### Column Data Types

{dtype_lines}

## Data Quality Issues

### Missing Values

{missing_lines}

- Raw missing `Income` values: **{audit['income_missing_before_imputation']}**
- Raw duplicate rows: **{inspection['duplicate_count']}**
- Date parsing failures for `Dt_Customer`: **{inspection['date_parse_failures']}**
- Negative numeric values: **{sum(inspection['negative_numeric_values'].values())}**
- The target contains values `{inspection['target_values']}` and has **{inspection['target_missing']}** missing values.

### Potential Format and Outlier Issues

- Three `Year_Birth` values were before 1926, implying an age over 100 in 2026; these records were removed as clear year-entry errors.
- `Income = 666666` occurred once and was treated as an isolated data-entry sentinel. Other high incomes were retained because they may represent genuine customers.
- `Education` and `Marital_Status` are categorical strings. Date values match the expected `DD-MM-YYYY` format.

## Data Cleaning Performed

- Removed **{audit['invalid_year_rows_removed']}** rows with `Year_Birth < 1926`; reason: implied age over 100 in 2026.
- Replaced **{audit['income_sentinel_values_replaced']}** sentinel income value(s) with missing, then filled **{audit['income_missing_before_imputation'] + audit['income_sentinel_values_replaced']}** missing income values using the cleaned-data median of **{audit['income_imputation_value']:,.2f}**. Median was used because Income is right-skewed and this preserves valid customer records without letting extreme values dominate.
- Removed **{audit['duplicate_rows_removed']}** duplicate rows after cleaning.
- Final missing-value count: **{int(data.isna().sum().sum())}**; final duplicate count: **{int(data.duplicated().sum())}**.

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

""" + "\n".join(f"- {finding}" for finding in findings) + f"""

## Data Readiness Status

**READY FOR ANALYSIS.** The processed dataset was exported to `processed_data/marketing_campaign_processed.csv`. The raw data was successfully read, documented cleaning was completed, required features were created, `Response` is complete, EDA ran successfully, and presentation-ready visualizations were generated. No classification, clustering, or other Milestone 2 modeling was performed.
"""
    (REPORT_DIR / "milestone1_summary.md").write_text(summary, encoding="utf-8")


def run_pipeline() -> pd.DataFrame:
    """Run all Milestone 1 steps and return the processed dataset."""
    prepare_output_directories()
    raw_data = load_raw_data()
    inspection = inspect_data(raw_data)
    cleaned_data, audit = clean_data(raw_data)
    processed_data = engineer_features(cleaned_data)
    processed_data.to_csv(PROCESSED_DIR / "marketing_campaign_processed.csv", index=False)
    processed_data.describe(include="all").transpose().to_csv(REPORT_DIR / "descriptive_statistics.csv")
    save_quality_outputs(inspection, audit, raw_data)
    charts = save_visualizations(processed_data)
    create_summary(inspection, audit, processed_data, charts)

    print(f"Raw dataset: {inspection['shape']['rows']} rows x {inspection['shape']['columns']} columns")
    print(f"Processed dataset: {len(processed_data)} rows x {processed_data.shape[1]} columns")
    print(f"Income values imputed: {audit['income_missing_before_imputation'] + audit['income_sentinel_values_replaced']}")
    print(f"Invalid year rows removed: {audit['invalid_year_rows_removed']}")
    print(f"Engineered features: Customer_Age, Total_Spending, Number_of_Children, Past_Campaign_Responses")
    print(f"Visualizations generated: {len(charts)}")
    print(f"Summary: {REPORT_DIR / 'milestone1_summary.md'}")
    return processed_data


if __name__ == "__main__":
    try:
        run_pipeline()
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Milestone 1 pipeline failed: {error}") from error