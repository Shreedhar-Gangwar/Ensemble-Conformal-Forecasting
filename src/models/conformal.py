import numpy as np
from typing import Tuple

def compute_split_conformal_quantile(y_calib: np.ndarray, calib_pred: np.ndarray, alpha: float = 0.05, scaling: float = 1.0) -> float:
    nonconf = np.abs(y_calib - calib_pred)
    q = scaling * np.quantile(nonconf, 1 - alpha)
    return float(q)

def form_constant_intervals(mean_pred: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    lower = mean_pred - q
    upper = mean_pred + q
    return lower, upper
