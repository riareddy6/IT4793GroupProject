import numpy as np

def validate(data):
    if data['Response'].isna().any() or set(data['Response'].unique()) != {0, 1}:
        raise ValueError('Response must contain both binary classes and no missing values')
    if not data['ID'].is_unique:
        raise ValueError('ID must be unique for raw-data alignment')
    if np.isinf(data.select_dtypes(include='number')).any().any():
        raise ValueError('Infinite numeric values')
    return dict(rows=len(data), columns=len(data.columns), dtypes={c:str(t) for c,t in data.dtypes.items()},
                missing=data.isna().sum().astype(int).to_dict(), target_counts=data.Response.value_counts().astype(int).to_dict(),
                constants=[c for c in data if data[c].nunique(dropna=False)==1],
                year_birth_range=[int(data.Year_Birth.min()), int(data.Year_Birth.max())])
