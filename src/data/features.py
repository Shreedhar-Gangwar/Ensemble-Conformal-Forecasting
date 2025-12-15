import numpy as np
import pandas as pd
from typing import List

def create_features(df: pd.DataFrame, target_cols: List[str], lags=None, roll_windows=None) -> pd.DataFrame:
    lags = lags or [1,2,3,6,12,24]
    roll_windows = roll_windows or [6,12,24,48]
    df_feat = df.copy()

    for col in target_cols:
        if col not in df_feat.columns:
            raise KeyError(f"Target column '{col}' not in dataframe.")
        for lag in lags:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    for col in target_cols:
        for w in roll_windows:
            df_feat[f'{col}_roll_mean_{w}'] = df_feat[col].rolling(window=w, min_periods=1).mean()
            df_feat[f'{col}_roll_std_{w}'] = df_feat[col].rolling(window=w, min_periods=1).std().fillna(0)
            df_feat[f'{col}_roll_min_{w}'] = df_feat[col].rolling(window=w, min_periods=1).min()
            df_feat[f'{col}_roll_max_{w}'] = df_feat[col].rolling(window=w, min_periods=1).max()

    hour = df_feat.index.hour
    doy = df_feat.index.dayofyear
    month = df_feat.index.month
    df_feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_feat['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    df_feat['doy_cos'] = np.cos(2 * np.pi * doy / 365)
    df_feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * month / 12)

    if 'T2M' in df_feat.columns and 'RH2M' in df_feat.columns:
        df_feat['temp_humidity'] = df_feat['T2M'] * df_feat['RH2M']

    if 'CLRSKY_SFC_SW_DWN' in df_feat.columns and 'SZA' in df_feat.columns:
        df_feat['solar_clear'] = df_feat['CLRSKY_SFC_SW_DWN'] * np.cos(np.pi * df_feat['SZA'] / 180)
    else:
        if 'CLRSKY_SFC_SW_DWN' not in df_feat.columns:
            df_feat['CLRSKY_SFC_SW_DWN'] = 0.0
        if 'SZA' not in df_feat.columns:
            df_feat['SZA'] = 0.0
        df_feat['solar_clear'] = df_feat['CLRSKY_SFC_SW_DWN'] * np.cos(np.pi * df_feat['SZA'] / 180)

    return df_feat
