import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def time_series_train_test_split(X: pd.DataFrame, y: pd.DataFrame, split_ratio: float = 0.8):
    split = int(len(X) * split_ratio)
    X_full_train, X_test = X.iloc[:split], X.iloc[split:]
    y_full_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_full_train, X_test, y_full_train, y_test

def make_model_calib_split(X_full_train: pd.DataFrame, y_full_train: pd.DataFrame, calib_ratio=0.2):
    n = len(X_full_train)
    cut = int(n * (1 - calib_ratio))
    X_model_train = X_full_train.iloc[:cut]
    X_calib = X_full_train.iloc[cut:]
    y_model_train = y_full_train.iloc[:cut]
    y_calib = y_full_train.iloc[cut:]
    return X_model_train, X_calib, y_model_train, y_calib

def scale_datasets(X_train: pd.DataFrame, X_calib: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_calib_s = pd.DataFrame(scaler.transform(X_calib), columns=X_train.columns, index=X_calib.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)
    return X_train_s, X_calib_s, X_test_s, scaler
