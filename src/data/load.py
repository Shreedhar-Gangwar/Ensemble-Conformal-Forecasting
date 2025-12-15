import pandas as pd
from typing import Tuple

def load_raw(path: str) -> pd.DataFrame:
    """Load CSV, handle YEAR/MO/DY/HR or datetime columns, return DataFrame indexed by datetime."""
    data = pd.read_csv(path)
    # rename if present
    rename_map = {}
    if 'YEAR' in data.columns: rename_map['YEAR'] = 'year'
    if 'MO' in data.columns: rename_map['MO'] = 'month'
    if 'DY' in data.columns: rename_map['DY'] = 'day'
    if 'HR' in data.columns: rename_map['HR'] = 'hour'
    if rename_map:
        data = data.rename(columns=rename_map)

    if {'year','month','day','hour'}.issubset(set(data.columns)):
        data['datetime'] = pd.to_datetime(data[['year','month','day','hour']])
        data = data.sort_values('datetime').set_index('datetime')
        data = data.drop(columns=['year','month','day','hour'], errors='ignore')
    else:
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime').set_index('datetime')
        else:
            raise ValueError("No suitable datetime columns found (need YEAR/MO/DY/HR or datetime).")

    # drop rows that are entirely NA
    data.dropna(how='all', inplace=True)

    # downcast floats
    float_cols = data.select_dtypes(include=['float64']).columns
    if len(float_cols):
        data[float_cols] = data[float_cols].astype('float32')

    return data
