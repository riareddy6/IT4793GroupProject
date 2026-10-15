import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def features(data, config, clustering=False):
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
