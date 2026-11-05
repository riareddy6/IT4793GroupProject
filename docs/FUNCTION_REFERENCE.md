# Function reference

Generated with `python -m experiments.milestone3.documentation`. Includes every project Python function, nested helper and test; installed dependencies are excluded. Purpose is the opening docstring sentence. Call lists are syntactic dependencies, not a runtime call graph.

Total documented functions: **66**. No docstring coverage exceptions.
## docs/milestone1/milestone1_analysis.py

### `prepare_output_directories()`

[Source](../docs/milestone1/milestone1_analysis.py#L44)

Create Milestone 1 artifact directories without modifying raw data.

Args:
    None: Uses configured module paths.

Returns:
    None: Creates processed_data, reports and visualizations directories if needed.

Processing:
    Call mkdir with parents=True and exist_ok=True for the three configured directories.

Raises:
    OSError: If a directory cannot be created. Existing artifacts are not removed.

**Dependencies / calls:** `directory.mkdir`.

### `load_raw_data()`

[Source](../docs/milestone1/milestone1_analysis.py#L63)

Load the raw tab-separated customer dataset and require its target column.

Args:
    None: Uses configured module paths.

Returns:
    pd.DataFrame: Raw records parsed with tab separation.

Processing:
    Check RAW_DATA_PATH exists, read TSV and verify Response is present.

Raises:
    FileNotFoundError: If raw data is absent. ValueError: If Response is missing. Does not modify the source file.

**Dependencies / calls:** `FileNotFoundError`, `RAW_DATA_PATH.exists`, `ValueError`, `pd.read_csv`.

### `inspect_data(data: pd.DataFrame)`

[Source](../docs/milestone1/milestone1_analysis.py#L86)

Summarize raw dimensions, missingness, types, dates and potential data quality issues.

Args:
    data (pd.DataFrame): Raw records including Income, Year_Birth, Dt_Customer and Response.

Returns:
    dict: JSON-compatible quality statistics including target values and sentinel counts.

Processing:
    Inspect numeric negatives, parse DD-MM-YYYY dates with coercion and summarize missingness, duplicates and ranges.

Raises:
    KeyError: For absent required columns. ValueError: For invalid numeric range conversions. Does not modify data or write files.

**Dependencies / calls:** `(data['Income'] == 666666).sum`, `(data['Income'] > 150000).sum`, `data.columns.tolist`, `data.dtypes.items`, `data.duplicated`, `data.duplicated().sum`, `data.isna`, `data.isna().sum`, `data.isna().sum().items`, `data.select_dtypes`, `data.select_dtypes(include=np.number).columns.tolist`, `data['Income'].max`, `data['Income'].min`, `data['Year_Birth'].max`, `data['Year_Birth'].min`, `data[TARGET].dropna`, `data[TARGET].dropna().unique`, `data[TARGET].dropna().unique().tolist`, `data[TARGET].isna`, `data[TARGET].isna().sum`, `data[numeric_columns].lt`, `data[numeric_columns].lt(0).sum`, `float`, `int`, `negative_counts.items`, `pd.to_datetime`, `pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce').isna`, `pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce').isna().sum`, `sorted`, `str`.

### `clean_data(data: pd.DataFrame)`

[Source](../docs/milestone1/milestone1_analysis.py#L127)

Apply the historical conservative customer cleaning rules and record their effects.

Args:
    data (pd.DataFrame): Raw customer records with Year_Birth and Income.

Returns:
    tuple[pd.DataFrame, dict]: Cleaned copy and audit of row removals, income replacement/imputation and duplicates.

Processing:
    Exclude births before 1926, replace income 666666 with missing, fill income with global cleaned median, then remove exact duplicate rows. This EDA imputation is restored to missing before model fitting.

Raises:
    KeyError: For missing required columns. Does not mutate the input or write files. Retains the historical 2026 exclusion convention.

**Dependencies / calls:** `cleaned.duplicated`, `cleaned.loc[~duplicate_mask].copy`, `cleaned.loc[~invalid_year_mask].copy`, `cleaned.select_dtypes`, `cleaned.select_dtypes(include=np.number).lt`, `cleaned.select_dtypes(include=np.number).lt(0).sum`, `cleaned.select_dtypes(include=np.number).lt(0).sum().sum`, `cleaned['Income'].fillna`, `cleaned['Income'].isna`, `cleaned['Income'].isna().sum`, `cleaned['Income'].median`, `data.copy`, `duplicate_mask.sum`, `float`, `int`, `invalid_year_mask.sum`, `len`, `sentinel_mask.sum`.

### `engineer_features(data: pd.DataFrame)`

[Source](../docs/milestone1/milestone1_analysis.py#L179)

Add age, total spending, household child count and historical campaign response count.

Args:
    data (pd.DataFrame): Cleaned data containing spending components, Year_Birth, Kidhome/Teenhome and AcceptedCmp1–5.

Returns:
    pd.DataFrame: Copy with Customer_Age, Total_Spending, Number_of_Children and Past_Campaign_Responses.

Processing:
    Use age 2026-Year_Birth for Milestone 1; sum six spending columns, two child counts and only prior campaign indicators. Modeling later replaces age with the fixed 2014 convention.

Raises:
    ValueError: If a historical campaign column is missing. KeyError: For other missing inputs. No target inclusion, file output or input mutation.

**Dependencies / calls:** `ValueError`, `all`, `data.copy`, `engineered[HISTORICAL_CAMPAIGN_COLUMNS].sum`, `engineered[SPENDING_COLUMNS].sum`.

### `save_quality_outputs(inspection: dict, audit: dict, data: pd.DataFrame)`

[Source](../docs/milestone1/milestone1_analysis.py#L205)

Export Milestone 1 inspection, cleaning, type and raw missingness reports.

Args:
    inspection (dict): Statistics returned by inspect_data.
    audit (dict): Cleaning decisions/counts returned by clean_data.
    data (pd.DataFrame): Frame supplying column names/dtypes; current pipeline passes raw data.

Returns:
    None: Writes data_quality_summary.json, column_dtypes.csv and missing_value_counts.csv under REPORT_DIR.

Processing:
    Combine inspection and audit into JSON; export dtype and inspection missing-count tables.

Raises:
    KeyError: For incomplete inspection. OSError: If report files cannot be written. Overwrites named report files.

**Dependencies / calls:** `(REPORT_DIR / 'data_quality_summary.json').write_text`, `dtype_table.to_csv`, `json.dumps`, `missing_table.to_csv`, `pd.DataFrame`, `str`.

### `save_visualizations(data: pd.DataFrame)`

[Source](../docs/milestone1/milestone1_analysis.py#L239)

Create Milestone 1 presentation-ready EDA figures and a correlation table.

Args:
    data (pd.DataFrame): Processed customer data with Response and engineered analysis features.

Returns:
    list[str]: Eight generated PNG filenames; also writes correlation_matrix.csv.

Processing:
    Configure seaborn styling; plot target counts, age, spending/income by response, children and past response rates, and numeric correlations. Export PNGs at 300 DPI.

Raises:
    KeyError: If required columns are missing. OSError: For failed exports. Changes plotting theme and overwrites named figures under VISUALIZATION_DIR.

**Dependencies / calls:** `ax.axvline`, `ax.bar_label`, `ax.legend`, `ax.set`, `ax.set_ylim`, `ax.yaxis.set_major_formatter`, `children_rate[TARGET].max`, `correlation.to_csv`, `created.append`, `data.assign`, `data.groupby`, `data.groupby('Number_of_Children', as_index=False)[TARGET].mean`, `data.groupby('Past_Campaign_Responses', as_index=False)[TARGET].mean`, `data['Customer_Age'].median`, `data[TARGET].map`, `data[TARGET].map(response_labels).value_counts`, `data[TARGET].map(response_labels).value_counts().reindex`, `data[correlation_columns].corr`, `float`, `max`, `past_rate[TARGET].max`, `plt.close`, `plt.figure`, `plt.savefig`, `plt.tight_layout`, `response_labels.values`, `save_figure`, `sns.barplot`, `sns.boxplot`, `sns.heatmap`, `sns.histplot`, `sns.set_theme`.

### `save_figure(filename: str)`

[Source](../docs/milestone1/milestone1_analysis.py#L257)

Save a current EDA figure and record its filename in the enclosing list.

Args:
    filename (str): PNG basename under VISUALIZATION_DIR.

Returns:
    None: Writes the figure and appends filename to the enclosing created list.

Processing:
    Apply tight layout, save at 300 DPI with tight bounds and close the active plot.

Raises:
    OSError: For failed image writing. Mutates the enclosing created list and closes the active figure.

**Dependencies / calls:** `created.append`, `plt.close`, `plt.savefig`, `plt.tight_layout`.

### `create_preliminary_findings(data: pd.DataFrame)`

[Source](../docs/milestone1/milestone1_analysis.py#L339)

Calculate descriptive campaign response findings from processed observations.

Args:
    data (pd.DataFrame): Processed records containing target, spending, income, age, children and prior campaign totals.

Returns:
    list[str]: English observations on prevalence, group medians and response-rate ranges.

Processing:
    Split accepted/nonaccepted records, calculate median differences and children/history response-rate ranges, then format statements.

Raises:
    KeyError: For missing analysis columns. No file output or mutation. Findings are associations, not causal conclusions.

**Dependencies / calls:** `accepted['Customer_Age'].median`, `accepted['Income'].median`, `accepted['Total_Spending'].median`, `children_rates.max`, `children_rates.min`, `data.groupby`, `data.groupby('Number_of_Children')[TARGET].mean`, `data.groupby('Past_Campaign_Responses')[TARGET].mean`, `data[TARGET].mean`, `data[TARGET].sum`, `int`, `len`, `not_accepted['Customer_Age'].median`, `not_accepted['Income'].median`, `not_accepted['Total_Spending'].median`, `past_rates.max`, `past_rates.min`.

### `create_summary(inspection: dict, audit: dict, data: pd.DataFrame, charts: list[str])`

[Source](../docs/milestone1/milestone1_analysis.py#L372)

Write the original Milestone 1 dataset-readiness report from computed evidence.

Args:
    inspection (dict): Raw-data inspection statistics.
    audit (dict): Cleaning audit.
    data (pd.DataFrame): Processed records used for preliminary findings.
    charts (list[str]): Generated EDA image basenames included in the report.

Returns:
    None: Writes REPORT_DIR/milestone1_summary.md.

Processing:
    Format dimensions, types, quality checks, cleaning, feature definitions, chart list and computed preliminary findings.

Raises:
    KeyError: For incomplete inputs. OSError: If the summary cannot be written. Overwrites the historical summary when explicitly run.

**Dependencies / calls:** `'\n'.join`, `(REPORT_DIR / 'milestone1_summary.md').write_text`, `create_preliminary_findings`, `data.duplicated`, `data.duplicated().sum`, `data.isna`, `data.isna().sum`, `data.isna().sum().sum`, `inspection['dtypes'].items`, `inspection['missing_values'].items`, `inspection['negative_numeric_values'].values`, `int`, `len`, `sum`.

### `run_pipeline()`

[Source](../docs/milestone1/milestone1_analysis.py#L475)

Run the complete Milestone 1 cleaning, feature engineering, EDA and export workflow.

Args:
    None: Uses configured module paths.

Returns:
    pd.DataFrame: Processed customer dataset; exports processed CSV, quality/statistical reports, EDA figures and summary.

Processing:
    Create output directories; load/inspect raw data; clean and engineer features; serialize processed data; create reports and charts; print dimensions.

Raises:
    OSError/ValueError: Loading or writing failures propagate. Overwrites named Milestone 1 outputs but never the raw source. For validation, redirect output globals to a temporary directory.

**Dependencies / calls:** `clean_data`, `create_summary`, `engineer_features`, `inspect_data`, `len`, `load_raw_data`, `prepare_output_directories`, `print`, `processed_data.describe`, `processed_data.describe(include='all').transpose`, `processed_data.describe(include='all').transpose().to_csv`, `processed_data.to_csv`, `save_quality_outputs`, `save_visualizations`.

## experiments/common/artifacts.py

### `latest(name)`

[Source](../experiments/common/artifacts.py#L6)

Find the newest completed run for an experiment.

Args:
    name (str): Experiment directory under experiments/outputs.

Returns:
    Path: Newest lexically ordered run with complete metadata.

Processing:
    Inspect timestamped run directories in descending order, skipping failures.

Raises:
    ValueError: If no completed run exists.
    JSONDecodeError: If a candidate metadata file is malformed.

**Dependencies / calls:** `(ROOT / 'outputs' / name).glob`, `ValueError`, `json.loads`, `json.loads(meta.read_text(encoding='utf-8')).get`, `meta.exists`, `meta.read_text`, `sorted`.

## experiments/common/data_loader.py

### `load_config(path)`

[Source](../experiments/common/data_loader.py#L8)

Read and validate a classifier or clustering JSON configuration.

Args:
    path (str | Path): Configuration file; relative dataset paths resolve from repository root.

Returns:
    dict: Configuration with absolute dataset and raw_dataset paths.

Processing:
    Parse JSON; enforce established target, seed, split, CV, metric and threshold settings; resolve data paths.

Raises:
    ValueError: If established experiment settings differ. OSError/JSONDecodeError: For unreadable or invalid JSON. No files are modified.

**Dependencies / calls:** `Path`, `Path(path).read_text`, `ValueError`, `any`, `config.get`, `dict`, `json.loads`, `required.items`, `str`.

### `load_data(config)`

[Source](../experiments/common/data_loader.py#L32)

Load processed records and restore original missing Income markers before fold-local imputation.

Args:
    config (dict): Validated configuration with dataset and raw_dataset paths.

Returns:
    tuple[pd.DataFrame, dict]: Records with missing/sentinel incomes restored to NaN and validation audit.

Processing:
    Read processed CSV, validate records, align raw income by unique ID, replace raw missing or 666666 incomes with NaN.

Raises:
    ValueError: If IDs cannot be aligned or records fail validation. OSError: For missing inputs. Changes only the in-memory DataFrame.

**Dependencies / calls:** `ValueError`, `data.ID.isin`, `data.ID.isin(raw.index).all`, `data.ID.map`, `float`, `income.eq`, `income.isna`, `int`, `mask.sum`, `pd.read_csv`, `pd.read_csv(config['raw_dataset'], sep='\t').set_index`, `validate`.

## experiments/common/data_validation.py

### `validate(data)`

[Source](../experiments/common/data_validation.py#L3)

Validate processed customer records and summarize data quality.

Args:
    data (pd.DataFrame): Records containing ID, Response, Year_Birth and numeric/categorical predictors.

Returns:
    dict: Dimensions, dtypes, missing counts, target counts, constant columns and birth-year range.

Processing:
    Require both binary target classes, unique IDs and finite observed numeric values; calculate audit statistics.

Raises:
    ValueError: For missing/invalid target, repeated IDs or infinite numeric values. KeyError: For missing required columns. No mutation.

**Dependencies / calls:** `ValueError`, `data.Response.value_counts`, `data.Response.value_counts().astype`, `data.Response.value_counts().astype(int).to_dict`, `data.Year_Birth.max`, `data.Year_Birth.min`, `data.dtypes.items`, `data.isna`, `data.isna().sum`, `data.isna().sum().astype`, `data.isna().sum().astype(int).to_dict`, `data.select_dtypes`, `data['Response'].isna`, `data['Response'].isna().any`, `data['Response'].unique`, `data[c].nunique`, `dict`, `int`, `len`, `np.isinf`, `np.isinf(data.select_dtypes(include='number')).any`, `np.isinf(data.select_dtypes(include='number')).any().any`, `set`, `str`.

## experiments/common/evaluation.py

### `metrics(y, p, threshold=0.5)`

[Source](../experiments/common/evaluation.py#L22)

Calculate standardized binary classification and probability-ranking metrics.

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

**Dependencies / calls:** `((p < 0) | (p > 1)).any`, `(p >= threshold).astype`, `ValueError`, `accuracy_score`, `average_precision_score`, `confusion_matrix`, `confusion_matrix(y, pred, labels=[0, 1]).tolist`, `dict`, `f1_score`, `float`, `len`, `np.asarray`, `np.isfinite`, `np.isfinite(p).all`, `np.isin`, `np.isin(y, [0, 1]).all`, `np.unique`, `precision_score`, `recall_score`, `roc_auc_score`.

### `select_threshold(table)`

[Source](../experiments/common/evaluation.py#L54)

Select a threshold from training-OOF metric rows.

Args:
    table (pd.DataFrame): Nonempty threshold, f1 and precision columns.

Returns:
    float: Threshold maximizing F1, then precision, then proximity to 0.50.

Processing:
    Stable-sort development scores using the established tie-breaking policy.

Raises:
    ValueError: If the development table is empty or contains invalid scores.
    KeyError: If required columns are absent. Does not read test data.

**Dependencies / calls:** `((values < 0) | (values > 1)).any`, `(table.threshold - 0.5).abs`, `ValueError`, `float`, `len`, `np.isfinite`, `np.isfinite(values).all`, `table.assign`, `table.assign(distance=(table.threshold - 0.5).abs()).sort_values`, `table[['threshold', 'f1', 'precision']].to_numpy`.

### `run_classifier(name, config_path, estimator, scale=False)`

[Source](../experiments/common/evaluation.py#L77)

Train and evaluate one estimator using the persisted split and fold-local preprocessing.

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

**Dependencies / calls:** `', '.join`, `(out / 'classification_report.txt').write_text`, `(out / 'error.txt').write_text`, `(out / 'model_behavior.md').write_text`, `(prob >= 0.5).astype`, `(prob >= threshold).astype`, `ConfusionMatrixDisplay.from_predictions`, `DummyClassifier`, `DummyClassifier(strategy=strategy, random_state=42).fit`, `Pipeline`, `PrecisionRecallDisplay.from_predictions`, `RocCurveDisplay.from_predictions`, `classification_report`, `clone`, `clone(pipe).fit`, `dict`, `dummy.predict_proba`, `features`, `figure`, `fitted.fit`, `fitted.predict_proba`, `float`, `fold_metrics.append`, `hasattr`, `importance.assign`, `importance.assign(magnitude=importance.importance.abs()).nlargest`, `importance.assign(magnitude=importance.importance.abs()).nlargest(20, 'magnitude').sort_values`, `importance.importance.abs`, `importance.to_csv`, `joblib.dump`, `len`, `load_config`, `load_data`, `meta.update`, `metadata`, `metrics`, `metrics(yt, oof, t).items`, `new_run`, `np.empty`, `np.mean`, `np.std`, `np.zeros`, `pd.DataFrame`, `pd.DataFrame({'feature': x.columns, 'importance': perm.importances_mean, 'std': perm.importances_std, 'method': 'validation_permutation_f1', 'feature_space': 'original'}).to_csv`, `pd.DataFrame({'row_index': test, 'ID': data.ID.iloc[test].values, 'true_label': y.iloc[test].values, 'predicted_probability': prob, 'predicted_class': pred, 'selected_threshold': threshold, 'default_predicted_class': (prob >= 0.5).astype(int)}).to_csv`, `pd.DataFrame({'row_index': train, 'ID': data.ID.iloc[train].values, 'true_label': yt.values, 'oof_probability': oof}).to_csv`, `permutation_importance`, `pipe.fit`, `pipe.named_steps['preprocessor'].get_feature_names_out`, `pipe.predict_proba`, `plot_tree`, `plt.barh`, `plt.figure`, `plt.title`, `preprocessor`, `print`, `recorded.extend`, `save_json`, `select_threshold`, `shared_split`, `str`, `thresholds.to_csv`, `top.sort_values`, `top.sort_values('magnitude', ascending=False).feature.head`, `traceback.format_exc`, `warnings.catch_warnings`, `warnings.simplefilter`, `x.columns.tolist`.

## experiments/common/preprocessing.py

### `features(data, config, clustering=False)`

[Source](../experiments/common/preprocessing.py#L7)

Prepare the established classification or clustering predictors without fitting transforms.

Args:
    data (pd.DataFrame): Processed customer data with engineered totals and original dates/birth years.
    config (dict): reference_date in YYYY-MM-DD format; clustering additionally requires features.
    clustering (bool): If True, retain the configured numeric clustering columns; default False.

Returns:
    pd.DataFrame: Predictor copy with fixed-reference age/tenure, no ID/Response and no verified constants.

Processing:
    Drop identifiers and target, replace dates/birth year, remove constant cost/revenue fields, optionally select clustering inputs.

Raises:
    KeyError: For missing input/configuration columns. ValueError: For invalid reference date/year. Invalid customer dates become NaN. Does not modify input.

**Dependencies / calls:** `data.drop`, `data.drop(columns=['Response', 'ID']).copy`, `int`, `pd.Timestamp`, `pd.to_datetime`, `x.drop`, `x.pop`, `x[c].nunique`.

### `preprocessor(x, scale=False)`

[Source](../experiments/common/preprocessing.py#L36)

Construct an unfitted, fold-local sklearn preprocessing transformer.

Args:
    x (pd.DataFrame): Predictor frame used only to identify numeric and categorical column names.
    scale (bool): Add StandardScaler after numeric median imputation when True; default False.

Returns:
    ColumnTransformer: Numeric imputation/scaling and categorical mode imputation/full one-hot encoding.

Processing:
    Identify dtypes; assemble independent numeric and categorical pipelines with unknown categories ignored.

Raises:
    TypeError: If x is not DataFrame-like. No fitting or file output; fitting errors propagate later from sklearn.

**Dependencies / calls:** `ColumnTransformer`, `OneHotEncoder`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `num_steps.append`, `x.select_dtypes`, `x.select_dtypes(include='number').columns.tolist`.

## experiments/common/result_saver.py

### `save_json(path, value)`

[Source](../experiments/common/result_saver.py#L10)

Serialize a JSON-compatible value while rejecting nonfinite numbers.

Args:
    path (Path): Destination file; parent directory must exist.
    value (JSON-compatible): Object containing only serializable finite values.

Returns:
    None: Writes indented UTF-8 JSON to path.

Processing:
    Serialize with allow_nan=False and write the result to the destination.

Raises:
    ValueError: For NaN or infinity. TypeError: For unsupported values. OSError: For unwritable paths. Overwrites the named file.

**Dependencies / calls:** `json.dumps`, `path.write_text`.

### `digest(path)`

[Source](../experiments/common/result_saver.py#L28)

Compute a file content hash for experiment provenance.

Args:
    path (Path): Existing readable file.

Returns:
    str: Hexadecimal SHA-256 digest.

Processing:
    Read bytes and calculate SHA-256 without changing the file.

Raises:
    OSError: If the file cannot be read. Reads the complete file into memory.

**Dependencies / calls:** `hashlib.sha256`, `hashlib.sha256(path.read_bytes()).hexdigest`, `path.read_bytes`.

### `new_run(name, config)`

[Source](../experiments/common/result_saver.py#L45)

Create a unique timestamped experiment directory and save its configuration.

Args:
    name (str): Experiment identifier under experiments/outputs.
    config (dict): JSON-compatible resolved experiment configuration.

Returns:
    Path: New run directory containing config.json.

Processing:
    Hash sorted configuration, combine UTC microsecond timestamp with short hash, create directory exclusively and serialize configuration.

Raises:
    FileExistsError: On a directory collision. OSError/TypeError/ValueError: On creation or serialization failure. Creates directory and config file.

**Dependencies / calls:** `datetime.now`, `datetime.now(timezone.utc).strftime`, `hashlib.sha256`, `hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest`, `json.dumps`, `json.dumps(config, sort_keys=True).encode`, `path.mkdir`, `save_json`.

### `metadata(config, data, split=None)`

[Source](../experiments/common/result_saver.py#L67)

Build reproducibility metadata for a running experiment.

Args:
    config (dict): Resolved dataset/raw_dataset paths.
    data (pd.DataFrame): Loaded customer records.
    split (str | None): Shared split identifier or None for clustering; default None.

Returns:
    dict: UTC time, Python/sklearn versions, seed, rows, file hashes, split ID and running status.

Processing:
    Inspect environment and dataset lengths; hash both source files and attach optional split identity.

Raises:
    OSError: If source files cannot be hashed. Does not save files; caller updates status and persists metadata.

**Dependencies / calls:** `Path`, `datetime.now`, `datetime.now(timezone.utc).isoformat`, `dict`, `digest`, `len`, `platform.python_version`.

## experiments/common/split_manager.py

### `shared_split(data, config)`

[Source](../experiments/common/split_manager.py#L10)

Reuse or create identical stratified train/test indices and CV folds for all classifiers.

Args:
    data (pd.DataFrame): Customer records with ID and binary Response, in persisted row order.
    config (dict): Resolved dataset and raw_dataset paths.

Returns:
    tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray,np.ndarray]], str]: Global train/test positions, folds relative to train, and split hash.

Processing:
    Hash data sources plus established seed/split/fold settings; load persisted split or create stratified 80/20 and shuffled five-fold indices; check partition invariants.

Raises:
    AssertionError: If persisted identity or partition coverage is inconsistent. OSError/JSONDecodeError: For invalid files. Creates shared directory and, only if absent, split.json and assignments.csv.

**Dependencies / calls:** `Path`, `StratifiedKFold`, `StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split`, `a.tolist`, `assignments.to_csv`, `b.tolist`, `dict`, `digest`, `enumerate`, `folder.mkdir`, `hashlib.sha256`, `hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest`, `json.dumps`, `json.dumps(identity, sort_keys=True).encode`, `json.loads`, `len`, `list`, `np.arange`, `np.array`, `path.exists`, `path.read_text`, `pd.DataFrame`, `range`, `save_json`, `set`, `sorted`, `test.tolist`, `train.tolist`, `train_test_split`.

## experiments/common/visualization.py

### `figure(path, dpi=300)`

[Source](../experiments/common/visualization.py#L7)

Save and close the current matplotlib figure.

Args:
    path (str | Path): Destination image path; parent must exist.
    dpi (int): Export resolution, default 300 dots per inch.

Returns:
    None: Writes an image at path.

Processing:
    Apply tight layout, export with tight bounds, and close the figure.

Raises:
    OSError: If the destination cannot be written.

**Dependencies / calls:** `plt.close`, `plt.savefig`, `plt.tight_layout`.

## experiments/comparison/compare_classifiers.py

### `main()`

[Source](../experiments/comparison/compare_classifiers.py#L12)

Compare compatible completed classifier runs without training.

Args:
    None: CLI --runs optionally accepts four run directories in LR, DT, RF, GB order; default newest completed per model.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Validate common metadata; recompute test metrics from saved probabilities; rank by CV F1; export CSV/chart/interpretation in a new comparison run.

Raises:
    ValueError: If runs are absent or incompatible. OSError: For artifact failures. Creates a new comparison directory, never modifies training runs.

**Dependencies / calls:** `(out / 'model_selection_summary.md').write_text`, `(p / 'cv_metrics.json').read_text`, `(p / 'run_metadata.json').exists`, `(p / 'run_metadata.json').read_text`, `(p / 'test_metrics.json').read_text`, `Path`, `ValueError`, `any`, `argparse.ArgumentParser`, `dict`, `enumerate`, `figure`, `json.loads`, `json.loads((p / 'run_metadata.json').read_text()).get`, `latest`, `metrics`, `new_run`, `parser.add_argument`, `parser.parse_args`, `pd.DataFrame`, `pd.DataFrame(rows).sort_values`, `pd.read_csv`, `print`, `rows.append`, `runs.append`, `save_json`, `str`, `table.set_index`, `table.set_index('model')[['cv_f1', 'test_default_f1', 'test_selected_f1']].plot.bar`, `table.to_csv`, `table.to_string`, `test['default_threshold'].items`, `test['selected'].items`, `zip`.

## experiments/decision_tree/train.py

### `main()`

[Source](../experiments/decision_tree/train.py#L8)

Train the configured decision_tree classifier through the shared evaluation pipeline.

Args:
    None: CLI --config defaults to config.json beside this module.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Parse configuration, construct the decision_tree sklearn estimator with configured parameters and call run_classifier with its established scaling policy.

Raises:
    SystemExit: For invalid CLI arguments. Data/training errors propagate. Writes a new run through run_classifier.

**Dependencies / calls:** `DecisionTreeClassifier`, `Path`, `Path(__file__).with_name`, `argparse.ArgumentParser`, `load_config`, `parser.add_argument`, `parser.parse_args`, `run_classifier`, `str`.

## experiments/gradient_boosting/train.py

### `main()`

[Source](../experiments/gradient_boosting/train.py#L8)

Train the configured gradient_boosting classifier through the shared evaluation pipeline.

Args:
    None: CLI --config defaults to config.json beside this module.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Parse configuration, construct the gradient_boosting sklearn estimator with configured parameters and call run_classifier with its established scaling policy.

Raises:
    SystemExit: For invalid CLI arguments. Data/training errors propagate. Writes a new run through run_classifier.

**Dependencies / calls:** `GradientBoostingClassifier`, `Path`, `Path(__file__).with_name`, `argparse.ArgumentParser`, `load_config`, `parser.add_argument`, `parser.parse_args`, `run_classifier`, `str`.

## experiments/kmeans/explain_results.py

### `write_explanation(run, output)`

[Source](../experiments/kmeans/explain_results.py#L9)

Write a standalone interpretation from completed K-Means artifacts.

Args:
    run (str | Path): Completed run with metadata, cluster profiles and candidate scores.
    output (str | Path): Destination Markdown file; parent must exist.

Returns:
    Path: Written explanation containing inputs, candidate scores, profiles and limitations.

Processing:
    Read saved CSV/JSON evidence, format original-unit profiles, describe lower/higher spending groups for two clusters and explain independence from classification.

Raises:
    ValueError: If run is incomplete. StopIteration: If selected K is absent. OSError: For missing inputs or unwritable output. Overwrites only the requested explanation.

**Dependencies / calls:** `', '.join`, `'\n'.join`, `(run / 'cluster_metrics.csv').open`, `(run / 'cluster_profiles.csv').open`, `(run / 'run_metadata.json').read_text`, `Path`, `ValueError`, `csv.DictReader`, `float`, `int`, `json.loads`, `len`, `lines.append`, `list`, `meta.get`, `next`, `output.write_text`, `row.items`, `sorted`.

### `main()`

[Source](../experiments/kmeans/explain_results.py#L67)

Regenerate a K-Means explanation from a completed run.

Args:
    None: CLI --run Path defaults to newest completed K-Means run; --output Path defaults to experiments/kmeans/RESULTS_EXPLANATION.md.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Parse arguments, select completed source and call write_explanation; print output path.

Raises:
    ValueError: If no completed run exists. OSError: For missing artifacts or unwritable output. Overwrites the requested explanation.

**Dependencies / calls:** `(ROOT / 'outputs' / 'kmeans').glob`, `(p / 'run_metadata.json').exists`, `(p / 'run_metadata.json').read_text`, `ValueError`, `argparse.ArgumentParser`, `json.loads`, `next`, `parser.add_argument`, `parser.parse_args`, `print`, `sorted`, `write_explanation`.

## experiments/kmeans/train.py

### `main()`

[Source](../experiments/kmeans/train.py#L18)

Fit and select an independent behavioral K-Means segmentation.

Args:
    None: CLI --config defaults to config.json beside this module.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Load/standardize all customer inputs; fit configured K range with fixed seeds; maximize eligible silhouette; save model, profiles, assignments, charts and explanation.

Raises:
    Exception: Errors propagate after failed metadata and traceback are saved. Writes a new run. Response is attached after fitting and K selection.

**Dependencies / calls:** `(out / 'error.txt').write_text`, `KMeans`, `KMeans(n_clusters=k, n_init=c['n_init'], random_state=c['random_state']).fit`, `Path`, `Path(__file__).with_name`, `Pipeline`, `ValueError`, `argparse.ArgumentParser`, `bool`, `data.assign`, `data.assign(cluster=labels).groupby`, `data.assign(cluster=labels).groupby('cluster').Response.mean`, `dict`, `eligible.sort_values`, `features`, `figure`, `float`, `int`, `joblib.dump`, `len`, `load_config`, `load_data`, `meta.update`, `metadata`, `new_run`, `np.bincount`, `parser.add_argument`, `parser.parse_args`, `pd.DataFrame`, `pd.DataFrame({'row_index': data.index, 'ID': data.ID, 'cluster': labels, 'Response': data.Response}).to_csv`, `pd.Series`, `pd.Series(labels).value_counts`, `pd.Series(labels).value_counts().sort_index`, `plt.figure`, `plt.plot`, `plt.xlabel`, `plt.ylabel`, `prep.fit_transform`, `preprocessor`, `print`, `profiles.response_rate.plot.bar`, `profiles.to_csv`, `profiles['size'].plot.bar`, `range`, `rows.append`, `save_json`, `silhouette_score`, `sizes.max`, `sizes.min`, `str`, `table.to_csv`, `threadpool_limits`, `traceback.format_exc`, `write_explanation`, `x.assign`, `x.assign(cluster=labels).groupby`, `x.assign(cluster=labels).groupby('cluster').mean`, `x.columns.tolist`.

## experiments/logistic_regression/train.py

### `main()`

[Source](../experiments/logistic_regression/train.py#L8)

Train the configured logistic_regression classifier through the shared evaluation pipeline.

Args:
    None: CLI --config defaults to config.json beside this module.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Parse configuration, construct the logistic_regression sklearn estimator with configured parameters and call run_classifier with its established scaling policy.

Raises:
    SystemExit: For invalid CLI arguments. Data/training errors propagate. Writes a new run through run_classifier.

**Dependencies / calls:** `LogisticRegression`, `Path`, `Path(__file__).with_name`, `argparse.ArgumentParser`, `load_config`, `parser.add_argument`, `parser.parse_args`, `run_classifier`, `str`.

## experiments/milestone3/build.py

### `read_json(path)`

[Source](../experiments/milestone3/build.py#L17)

Read UTF-8 JSON from a project artifact.

Args:
    path (Path): Existing JSON file.

Returns:
    dict | list: Decoded JSON value.

Processing:
    Read text and parse with the standard JSON decoder.

Raises:
    OSError: If the file cannot be read.
    JSONDecodeError: If the content is invalid.

**Dependencies / calls:** `Path`, `Path(path).read_text`, `json.loads`.

### `require(condition, message)`

[Source](../experiments/milestone3/build.py#L36)

Reject inconsistent artifacts even when Python assertions are disabled.

Args:
    condition (bool): Required invariant.
    message (str): Diagnostic describing the failed invariant.

Returns:
    None: Continues on a true condition.

Processing:
    Raise an explicit validation error on a false condition.

Raises:
    ValueError: If condition is false. No file side effects.

**Dependencies / calls:** `ValueError`.

### `verify_classifier(name, run, data, config, split, split_id)`

[Source](../experiments/milestone3/build.py#L56)

Verify one historical classifier against records, folds and fitted state.

Args:
    name (str): Classifier identifier in MODELS.
    run (Path): Completed run directory containing predictions and model.
    data (pd.DataFrame): Processed records with restored missing Income.
    config (dict): Shared resolved data configuration.
    split (dict): Persisted train/test indices and relative CV fold indices.
    split_id (str): Expected shared split directory identifier.

Returns:
    tuple[dict, dict]: Comparison row and dashboard model evidence.

Processing:
    Check hashes, ID alignment, saved predictions, fold metrics, OOF grid,
    threshold selection, fitted probabilities and estimator importance.

Raises:
    ValueError: If any saved artifact disagrees with recomputed evidence.
    OSError: If artifacts are missing. Loads trusted local joblib only;
        does not fit models or modify the source run.

**Dependencies / calls:** `Path`, `cv['folds'][i].items`, `data.iloc[indices].ID.tolist`, `data.iloc[indices].Response.tolist`, `dict`, `digest`, `enumerate`, `features`, `fpr.tolist`, `frame.ID.tolist`, `frame.row_index.tolist`, `frame.true_label.tolist`, `grid.to_dict`, `hasattr`, `importance.feature.tolist`, `importance.to_dict`, `joblib.load`, `len`, `list`, `metrics`, `metrics(oof.true_label, oof.oof_probability, t).items`, `np.allclose`, `np.array_equal`, `np.isclose`, `np.mean`, `np.std`, `pd.DataFrame`, `pd.read_csv`, `permutation.feature.tolist`, `permutation.to_dict`, `pipeline.named_steps['preprocessor'].get_feature_names_out`, `pipeline.named_steps['preprocessor'].get_feature_names_out().tolist`, `pipeline.predict_proba`, `precision.tolist`, `precision_recall_curve`, `range`, `read_json`, `recall.tolist`, `require`, `roc_curve`, `select_threshold`, `selected.items`, `set`, `sorted`, `test['selected'].items`, `tpr.tolist`, `validation.extend`, `x.columns.tolist`.

### `verify_clusters(run, data, config)`

[Source](../experiments/milestone3/build.py#L141)

Reconcile K-Means assignments, profiles, selected K and fitted pipeline.

Args:
    run (Path): Completed K-Means run directory.
    data (pd.DataFrame): Full restored-income records in original order.
    config (dict): Shared configuration used for data hash verification.

Returns:
    dict: Selected K, silhouette, candidate scores and enriched profiles.

Processing:
    Verify hashes, recompute original-unit means and response rates,
    compare fitted assignments and check silhouette-based K selection.

Raises:
    ValueError: If artifacts disagree. OSError if files cannot be read.
    Loads a trusted local model without refitting or changing labels.

**Dependencies / calls:** `Path`, `assignments.ID.tolist`, `assignments.Response.tolist`, `assignments.cluster.to_numpy`, `candidates.to_dict`, `candidates[candidates.eligible].sort_values`, `data.ID.tolist`, `data.Response.tolist`, `data.assign`, `data.assign(cluster=assignments.cluster.to_numpy()).groupby`, `data.assign(cluster=assignments.cluster.to_numpy()).groupby('cluster').Response.mean`, `data.assign(cluster=assignments.cluster.to_numpy()).groupby('cluster').Response.sum`, `dict`, `digest`, `features`, `float`, `grouped.mean`, `grouped.size`, `int`, `joblib.load`, `len`, `np.allclose`, `np.array_equal`, `pd.read_csv`, `pipeline.predict`, `profiles.set_index`, `profiles.to_dict`, `profiles[spending].sum`, `read_json`, `require`, `x.assign`, `x.assign(cluster=assignments.cluster.to_numpy()).groupby`.

### `create_plots(table, details, clusters, output)`

[Source](../experiments/milestone3/build.py#L191)

Export combined evaluation and customer-segmentation figures.

Args:
    table (pd.DataFrame): Verified model comparison rows.
    details (dict): Curves, confusion matrices, OOF grids and importance.
    clusters (dict): Verified cluster profiles and candidate scores.
    output (Path): Existing output directory for 300-DPI PNGs.

Returns:
    None: Writes ROC, PR, F1, confusion, threshold, importance and cluster PNGs.

Processing:
    Plot probability curves separately from selected-threshold outcomes;
    label confusion cells TN/FP/FN/TP and OOF curves as development scores.

Raises:
    OSError: If image export fails. Closes each generated figure.

**Dependencies / calls:** `ax.bar`, `ax.bar_label`, `ax.set_title`, `ax.set_ylim`, `figure`, `int`, `key.title`, `pd.DataFrame`, `pd.DataFrame(d['importance']).assign`, `pd.DataFrame(d['importance']).assign(magnitude=lambda x: x.importance.abs()).nlargest`, `pd.DataFrame(d['importance']).assign(magnitude=lambda x: x.importance.abs()).nlargest(12, 'magnitude').sort_values`, `plt.axhline`, `plt.axvline`, `plt.bar`, `plt.bar_label`, `plt.barh`, `plt.figure`, `plt.imshow`, `plt.legend`, `plt.plot`, `plt.step`, `plt.subplots`, `plt.text`, `plt.title`, `plt.xlabel`, `plt.xlim`, `plt.xticks`, `plt.ylabel`, `plt.ylim`, `plt.yticks`, `profiles[field].max`, `str`, `table.test_f1.idxmax`, `table.to_dict`, `x.importance.abs`, `zip`.

### `build(manifest_path)`

[Source](../experiments/milestone3/build.py#L264)

Build the entire verified report bundle from pinned experiment sources.

Args:
    manifest_path (Path): JSON with repository-relative runs and output path.

Returns:
    Path: Generated output directory.

Processing:
    Validate source artifacts, export comparison/profile CSVs, dashboard JSON,
    source hashes, figures and generated evaluation interpretation.

Raises:
    ValueError: If source evidence is inconsistent.
    OSError: If inputs are missing or outputs cannot be written. Overwrites
        only the manifest's generated report directory, never training runs.

**Dependencies / calls:** `(data.Response == 0).sum`, `Path`, `Path(manifest_path).relative_to`, `create_plots`, `data.Response.mean`, `data.Response.sum`, `data.iloc[split['test']].Response.mean`, `dict`, `digest`, `float`, `int`, `len`, `load_config`, `load_data`, `manifest['runs'].items`, `output.mkdir`, `p.is_file`, `p.relative_to`, `p.relative_to(PROJECT).as_posix`, `pd.DataFrame`, `pd.DataFrame(clusters['profiles']).to_csv`, `print`, `read_json`, `rows.append`, `run.iterdir`, `runs.values`, `save_json`, `str`, `table.cv_f1.idxmax`, `table.test_f1.idxmax`, `table.to_csv`, `verify_classifier`, `verify_clusters`, `write_evaluation`.

### `main()`

[Source](../experiments/milestone3/build.py#L317)

Run the verified report builder from the command line.

Args:
    None: Reads optional --manifest Path, default sources.json beside module.

Returns:
    None: Writes report assets through build and prints their directory.

Processing:
    Parse arguments and invoke the pinned-source report workflow.

Raises:
    SystemExit: For invalid CLI arguments; build errors propagate.

**Dependencies / calls:** `Path`, `Path(__file__).with_name`, `argparse.ArgumentParser`, `build`, `parser.add_argument`, `parser.parse_args`.

## experiments/milestone3/documentation.py

### `source_files()`

[Source](../experiments/milestone3/documentation.py#L8)

List all project Python sources while excluding installed dependencies.

Args:
    None: Uses the repository root derived from this module.

Returns:
    list[Path]: Sorted experiment, report, test and tooling Python files.

Processing:
    Search owned source directories and the historical audit support script.

Raises:
    OSError: If source directories are inaccessible. No files are changed.

**Dependencies / calls:** `(PROJECT / directory).rglob`, `files.extend`, `sorted`.

### `write_reference()`

[Source](../experiments/milestone3/documentation.py#L29)

Generate the function catalog and reject incomplete function docstrings.

Args:
    None: Reads Python sources returned by source_files.

Returns:
    int: Total documented functions, including nested and test functions.

Processing:
    Parse ASTs without importing executable scripts; include signatures,
    docstrings and statically observed call dependencies for each function.

Raises:
    ValueError: If any function lacks Args, Returns, Processing or Raises.
    OSError: If source reads or docs/FUNCTION_REFERENCE.md writing fails.

**Dependencies / calls:** `', '.join`, `'\n'.join`, `(PROJECT / 'docs/FUNCTION_REFERENCE.md').write_text`, `ValueError`, `all`, `ast.get_docstring`, `ast.parse`, `ast.unparse`, `ast.walk`, `inspect.cleandoc`, `isinstance`, `lines.extend`, `lines.insert`, `path.read_text`, `path.relative_to`, `path.relative_to(PROJECT).as_posix`, `print`, `sorted`, `source_files`.

### `write_evaluation(bundle, path)`

[Source](../experiments/milestone3/documentation.py#L69)

Write model evaluation and marketing interpretation from verified values.

Args:
    bundle (dict): Verified overview, comparison, model and clustering evidence.
    path (Path): Destination MODEL_EVALUATION.md document.

Returns:
    None: Writes metric definitions, actual results, charts and limitations.

Processing:
    Derive metric winners, comparison rows and confusion-based decisions
    from the evidence bundle; label CV, holdout and OOF development policies.

Raises:
    KeyError: If required evidence is absent.
    OSError: If documentation cannot be written.

**Dependencies / calls:** `'\n'.join`, `Path`, `Path(path).write_text`, `int`, `lines.append`, `max`.

## experiments/milestone3/validate_workflow.py

### `validate_milestone1()`

[Source](../experiments/milestone3/validate_workflow.py#L26)

Execute Milestone 1 in a temporary directory and compare processed data.

Args:
    None: Uses the original raw source and historical processed CSV.
Returns:
    dict: Dimensions and exact processed-data hash agreement.
Processing:
    Import the existing script, redirect output globals, run its full pipeline,
    compare records and serialized CSV, then remove temporary artifacts.
Raises:
    ValueError: If regenerated data differs; import/runtime errors propagate.
    Does not overwrite historical reports, figures or processed data.

**Dependencies / calls:** `Path`, `dict`, `digest`, `generated.reset_index`, `importlib.util.module_from_spec`, `importlib.util.spec_from_file_location`, `len`, `module.run_pipeline`, `pd.read_csv`, `pd.testing.assert_frame_equal`, `require`, `spec.loader.exec_module`, `tempfile.TemporaryDirectory`.

### `validate_interpretation(manifest)`

[Source](../experiments/milestone3/validate_workflow.py#L53)

Recompute saved permutation importance and selected clustering silhouette.

Args:
    manifest (dict): Pinned repository-relative experiment source runs.
Returns:
    dict: Four permutation matches and selected K-Means silhouette.
Processing:
    Refit the first training fold only, repeat original permutation settings,
    compare score decreases/SD, and recompute silhouette on transformed data.
Raises:
    ValueError: On mismatched importance or silhouette. Fits temporary
        validation models in memory; never tunes or modifies held-out models.

**Dependencies / calls:** `artifact.named_steps['preprocessor'].transform`, `clone`, `dict`, `features`, `float`, `joblib.load`, `load_config`, `load_data`, `np.allclose`, `np.isclose`, `pd.read_csv`, `pd.read_csv(run / 'cluster_metrics.csv').set_index`, `permutation_importance`, `pipeline.fit`, `read_json`, `require`, `shared_split`, `silhouette_score`, `threadpool_limits`.

### `replay_training(manifest)`

[Source](../experiments/milestone3/validate_workflow.py#L91)

Execute each existing training entry point and compare fresh output to history.

Args:
    manifest (dict): Pinned original runs for comparison.
Returns:
    dict: Fresh run paths and probability/assignment reproducibility checks.
Processing:
    Run four classifiers and K-Means as subprocesses; compare model outputs
    to pinned originals, run comparison, artifact verification and explanation.
Raises:
    CalledProcessError: If an entry point fails.
    ValueError: If fresh thresholds/probabilities/labels differ. Creates new
        timestamped runs and updates the explicit Milestone 2 summary only.

**Dependencies / calls:** `Path`, `dict`, `float`, `latest`, `new.ID.tolist`, `new.equals`, `np.abs`, `np.allclose`, `np.array_equal`, `np.max`, `old.ID.tolist`, `os.environ.copy`, `pd.read_csv`, `print`, `require`, `run.relative_to`, `run.relative_to(PROJECT).as_posix`, `str`, `subprocess.run`, `tempfile.TemporaryDirectory`.

### `main()`

[Source](../experiments/milestone3/validate_workflow.py#L131)

Run full workflow reproducibility QA and save its measured results.

Args:
    None: CLI --skip-training skips new full training runs when requested.
Returns:
    None: Writes docs/milestone3/workflow_validation.json after all checks pass.
Processing:
    Reproduce Milestone 1, interpretation and optionally all training entry points;
    hash pinned artifacts before/after to verify preservation.
Raises:
    Exception: Any validation/training failure propagates without reporting success.
    Writes new timestamped runs when training is enabled.

**Dependencies / calls:** `(PROJECT / path).iterdir`, `Path`, `Path(__file__).with_name`, `all`, `argparse.ArgumentParser`, `before.items`, `dict`, `digest`, `json.dumps`, `manifest['runs'].values`, `p.is_file`, `parser.add_argument`, `parser.parse_args`, `print`, `read_json`, `replay_training`, `require`, `save_json`, `validate_interpretation`, `validate_milestone1`.

## experiments/random_forest/train.py

### `main()`

[Source](../experiments/random_forest/train.py#L8)

Train the configured random_forest classifier through the shared evaluation pipeline.

Args:
    None: CLI --config defaults to config.json beside this module.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Parse configuration, construct the random_forest sklearn estimator with configured parameters and call run_classifier with its established scaling policy.

Raises:
    SystemExit: For invalid CLI arguments. Data/training errors propagate. Writes a new run through run_classifier.

**Dependencies / calls:** `Path`, `Path(__file__).with_name`, `RandomForestClassifier`, `argparse.ArgumentParser`, `load_config`, `parser.add_argument`, `parser.parse_args`, `run_classifier`, `str`.

## experiments/verify_outputs.py

### `main()`

[Source](../experiments/verify_outputs.py#L17)

Verify latest completed experiment artifacts and regenerate the Milestone 2 execution summary.

Args:
    None: No parameters; reads latest completed runs for four classifiers, clustering and comparison.

Returns:
    None: Prints the generated artifact directory or report; command-line side effects described below.

Processing:
    Check file presence, JSON/CSV decoding, PNG integrity, joblib loading, threshold classes, shared split and cluster totals; write experiments/EXECUTION_SUMMARY.md.

Raises:
    AssertionError/OSError/ValueError: On invalid or missing artifacts. Loads trusted local joblib files and overwrites the execution summary.

**Dependencies / calls:** `', '.join`, `'\n'.join`, `(ROOT / 'EXECUTION_SUMMARY.md').write_text`, `(ROOT / 'outputs').glob`, `(ROOT.parent / 'outputs').glob`, `(p / 'cv_metrics.json').read_text`, `(p / 'run_metadata.json').read_text`, `(p / 'test_metrics.json').read_text`, `(p / 'warnings.json').read_text`, `(p / f).stat`, `(preds.predicted_probability >= test['selected_threshold']).astype`, `(recomputed == preds.predicted_class).all`, `Image.open`, `f.is_file`, `f.read_text`, `f.relative_to`, `f.stat`, `im.verify`, `int`, `joblib.load`, `json.loads`, `latest`, `len`, `list`, `p.relative_to`, `p.relative_to(ROOT).as_posix`, `paths.append`, `pd.read_csv`, `print`, `profiles.sort_values`, `profiles.to_string`, `profiles['size'].sum`, `report.append`, `set`, `sizes.to_string`, `split_ids.append`, `str`.

## milestone2_ppt_support/build_audit.py

### `read(p)`

[Source](../milestone2_ppt_support/build_audit.py#L17)

Decode a historical audit JSON source.

Args:
    p (str | Path): Existing JSON path.

Returns:
    dict | list: Parsed value.

Processing:
    Read UTF-8 and call json.loads.

Raises:
    OSError/JSONDecodeError: For unreadable or invalid input. No output.

**Dependencies / calls:** `Path`, `Path(p).read_text`, `json.loads`.

### `rel(p)`

[Source](../milestone2_ppt_support/build_audit.py#L33)

Format a repository-relative audit source path.

Args:
    p (str | Path): Path within ROOT.

Returns:
    str: Relative POSIX path.

Processing:
    Resolve relative representation with Path.relative_to and as_posix.

Raises:
    ValueError: If p is outside ROOT. No files changed.

**Dependencies / calls:** `Path`, `Path(p).relative_to`, `Path(p).relative_to(ROOT).as_posix`.

### `link(p)`

[Source](../milestone2_ppt_support/build_audit.py#L49)

Format a Markdown link to a repository audit artifact.

Args:
    p (str | Path): Artifact path within ROOT.

Returns:
    str: Markdown label and target using repository-relative path.

Processing:
    Call rel and interpolate the resulting path.

Raises:
    ValueError: If p is outside ROOT. No files changed.

**Dependencies / calls:** `rel`.

### `table(df)`

[Source](../milestone2_ppt_support/build_audit.py#L65)

Format a DataFrame as a Markdown audit table.

Args:
    df (pd.DataFrame): Data with printable scalar cells.

Returns:
    str: Header, separator and formatted data rows.

Processing:
    Format floats to eight decimals and escape pipe/newline content via nested fmt.

Raises:
    TypeError: For unsupported formatting values. No files changed.

**Dependencies / calls:** `' | '.join`, `'\n'.join`, `df.itertuples`, `fmt`, `isinstance`, `len`, `map`, `str`, `str(v).replace`, `str(v).replace('|', '/').replace`.

### `fmt(v)`

[Source](../milestone2_ppt_support/build_audit.py#L80)

Format a scalar cell safely for the historical Markdown table.

Args:
    v (object): Printable scalar; floats receive eight decimal places.

Returns:
    str: Sanitized cell text.

Processing:
    Format float values; otherwise replace pipes and newlines.

Raises:
    Exception: Custom string conversion errors propagate. No file output.

**Dependencies / calls:** `isinstance`, `str`, `str(v).replace`, `str(v).replace('|', '/').replace`.

### `csv(df, name)`

[Source](../milestone2_ppt_support/build_audit.py#L98)

Export a historical audit support table.

Args:
    df (pd.DataFrame): Audit table.
    name (str): Destination filename within OUT.

Returns:
    None: Writes a CSV with no row index and 15-significant-digit floats.

Processing:
    Call DataFrame.to_csv in the support output directory.

Raises:
    OSError: On failed export. Overwrites the named support file only when the historical audit script runs.

**Dependencies / calls:** `df.to_csv`.

### `save(name)`

[Source](../milestone2_ppt_support/build_audit.py#L116)

Export and close a historical audit plot.

Args:
    name (str): PNG filename within OUT.

Returns:
    None: Writes a 300-DPI PNG.

Processing:
    Apply tight layout, save with tight bounds and close the active plot.

Raises:
    OSError: On image writing failure. Changes matplotlib active figure state.

**Dependencies / calls:** `plt.close`, `plt.savefig`, `plt.tight_layout`.

### `add(s)`

[Source](../milestone2_ppt_support/build_audit.py#L244)

Append one section fragment to the historical audit report.

Args:
    s (str): Markdown fragment.

Returns:
    None: Adds s to the global report list.

Processing:
    Append the provided text without writing the final report yet.

Raises:
    No expected exceptions for valid input. Mutates global report list.

**Dependencies / calls:** `report.append`.

## tests/test_pipeline.py

### `test_metric_orientation_and_probability_scores(self)`

[Source](../tests/test_pipeline.py#L21)

Check inclusive thresholds and asymmetric TN/FP/FN/TP counts.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts metric and sklearn probability-score agreement.
Processing:
    Score hand-labeled examples including a probability equal to cutoff.
Raises:
    AssertionError: On incorrect counts or ranking metrics. No file writes.

**Dependencies / calls:** `average_precision_score`, `metrics`, `roc_auc_score`, `self.assertAlmostEqual`, `self.assertEqual`.

### `test_invalid_metrics_and_one_class(self)`

[Source](../tests/test_pipeline.py#L43)

Reject malformed metric inputs and retain fixed one-class matrix shape.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Invalid inputs raise; one-class ROC AUC is undefined (None).
Processing:
    Exercise empty, unaligned, nonfinite, out-of-range and nonbinary cases.
Raises:
    AssertionError: If validation or fixed-label matrix behavior regresses.

**Dependencies / calls:** `metrics`, `self.assertEqual`, `self.assertIsNone`, `self.assertRaises`.

### `test_threshold_ties_and_training_only_interface(self)`

[Source](../tests/test_pipeline.py#L63)

Verify threshold tie-breaking without accepting test predictions.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts F1, precision and proximity ordering.
Processing:
    Use conflicting development scores to test both tie-break priorities.
Raises:
    AssertionError: If selected thresholds differ or invalid rows pass.

**Dependencies / calls:** `pd.DataFrame`, `select_threshold`, `self.assertEqual`, `self.assertRaises`.

### `test_fold_local_imputation_and_unknown_categories(self)`

[Source](../tests/test_pipeline.py#L82)

Ensure held-out values do not alter fitted numeric imputation statistics.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts median, unseen-category handling and finite transforms.
Processing:
    Fit on small training data, transform extreme/unknown validation data.
Raises:
    AssertionError: On leaked statistics or broken unknown-category support.

**Dependencies / calls:** `np.all`, `np.isfinite`, `np.isfinite(transformed).all`, `pd.DataFrame`, `prep.transform`, `preprocessor`, `preprocessor(train, scale=True).fit`, `self.assertEqual`, `self.assertTrue`.

### `test_data_features_and_shared_folds(self)`

[Source](../tests/test_pipeline.py#L103)

Verify project dimensions, restored Income and disjoint persisted folds.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts established data counts, features and fold coverage.
Processing:
    Load real data/configuration and compare absolute versus relative indices.
Raises:
    AssertionError: On changed data/splits; loader failures propagate.

**Dependencies / calls:** `(x.Customer_Age == 2014 - data.Year_Birth).all`, `data.Income.isna`, `data.Income.isna().sum`, `data.Response.value_counts`, `data.Response.value_counts().to_dict`, `features`, `len`, `list`, `load_config`, `load_data`, `np.concatenate`, `range`, `self.assertEqual`, `self.assertFalse`, `self.assertTrue`, `set`, `shared_split`, `sorted`.

### `test_config_validation_and_path_resolution(self)`

[Source](../tests/test_pipeline.py#L133)

Require historical split settings and repository-relative data resolution.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts path resolution and rejection of incompatible settings.
Processing:
    Modify only a temporary configuration, leaving source configuration intact.
Raises:
    AssertionError: If an incompatible seed is accepted. Writes temporary JSON.

**Dependencies / calls:** `Path`, `Path(c['dataset']).is_absolute`, `load_config`, `read_json`, `save_json`, `self.assertRaises`, `self.assertTrue`, `tempfile.TemporaryDirectory`.

### `test_strict_json_serialization(self)`

[Source](../tests/test_pipeline.py#L155)

Check UTF-8 round trips and rejection of nonfinite JSON payloads.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts exact payload recovery and invalid-number failure.
Processing:
    Write/read a temporary JSON and attempt NaN serialization.
Raises:
    AssertionError: For serialization regressions. Temporary files are cleaned.

**Dependencies / calls:** `Path`, `float`, `read_json`, `save_json`, `self.assertEqual`, `self.assertRaises`, `tempfile.TemporaryDirectory`.

### `test_all_functions_documented_and_parseable(self)`

[Source](../tests/test_pipeline.py#L175)

Require complete documentation sections for every owned Python function.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts parsability and section coverage, including nested tests.
Processing:
    Walk ASTs without importing side-effectful historical scripts.
Raises:
    AssertionError: If any function lacks its required documentation.

**Dependencies / calls:** `ast.get_docstring`, `ast.parse`, `ast.walk`, `isinstance`, `path.read_text`, `self.assertIn`, `source_files`.

### `test_generated_bundle_matches_source_scores(self)`

[Source](../tests/test_pipeline.py#L195)

Check dashboard/export metrics and profiles against saved source evidence.

Args:
    self (PipelineTests): unittest instance.
Returns:
    None: Asserts all four rows, selected thresholds and response counts.
Processing:
    Read the generated bundle and recompute from source prediction files.
Raises:
    AssertionError: If generated data drifts from its pinned sources.

**Dependencies / calls:** `len`, `metrics`, `pd.read_csv`, `read_json`, `self.assertAlmostEqual`, `self.assertEqual`, `sum`.
