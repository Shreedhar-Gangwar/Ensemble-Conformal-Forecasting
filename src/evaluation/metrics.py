import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_metrics(true, pred, lower, upper, alpha=0.05, label=""):
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mae = mean_absolute_error(true, pred)
    rmse = (mean_squared_error(true, pred)) ** 0.5
    r2 = r2_score(true, pred)
    coverage = np.mean((true >= lower) & (true <= upper))
    pinaw = np.mean(upper - lower) / (true.max() - true.min() + 1e-12)
    mean_width = np.mean(upper - lower)
    print(f"\n--- {label} ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Coverage: {coverage * 100:.2f}% (target {(1-alpha)*100:.1f}%)")
    print(f"PINAW: {pinaw:.4f}")
    print(f"Mean Width: {mean_width:.4f}")
