import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def features(data, config, clustering=False):
    """Prepare the established classification or clustering predictors without fitting transforms.

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
    """
    x = data.drop(columns=['Response', 'ID']).copy()
    # Fixed historical reference avoids using the dataset maximum or execution date.
    dates = pd.to_datetime(x.pop('Dt_Customer'), format='%d-%m-%Y', errors='coerce')
    x['Customer_Tenure_Days'] = (pd.Timestamp(config['reference_date']) - dates).dt.days
    x['Customer_Age'] = int(config['reference_date'][:4]) - x.pop('Year_Birth')
    for c in ('Z_CostContact', 'Z_Revenue'):
        if c in x and x[c].nunique(dropna=False) <= 1:
            x = x.drop(columns=c)
    if clustering:
        x = x[config['features']]
    return x

def preprocessor(x, scale=False):
    """Construct an unfitted, fold-local sklearn preprocessing transformer.

    Args:
        x (pd.DataFrame): Predictor frame used only to identify numeric and categorical column names.
        scale (bool): Add StandardScaler after numeric median imputation when True; default False.

    Returns:
        ColumnTransformer: Numeric imputation/scaling and categorical mode imputation/full one-hot encoding.

    Processing:
        Identify dtypes; assemble independent numeric and categorical pipelines with unknown categories ignored.

    Raises:
        TypeError: If x is not DataFrame-like. No fitting or file output; fitting errors propagate later from sklearn.
    """
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in x if c not in numeric]
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        num_steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(num_steps), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ], verbose_feature_names_out=False)
