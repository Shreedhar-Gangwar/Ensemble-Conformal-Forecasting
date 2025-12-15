#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.utils.io import load_config
from src.data.load import load_raw
from src.data.features import create_features
from src.data.preprocess import time_series_train_test_split, make_model_calib_split, scale_datasets
from src.models.ensemble import train_ensemble, ensemble_predict
from src.models.conformal import compute_split_conformal_quantile, form_constant_intervals
from src.evaluation.metrics import print_metrics

def main(config_path: str):
    cfg = load_config(config_path)
    np.random.seed(cfg.get('seed', 42))

    # 1) load
    data = load_raw(cfg['data_path'])

    # targets
    targets = cfg['targets']
    for t in targets:
        if t not in data.columns:
            raise KeyError(f"Target column {t} missing from data.")

    # 2) features
    data = create_features(data, targets, lags=cfg.get('lags'), roll_windows=cfg.get('roll_windows'))
    data = data.dropna().copy()

    # 3) split
    y = data[targets].copy()
    X = data.drop(columns=targets).copy()
    X_full_train, X_test, y_full_train, y_test = time_series_train_test_split(X, y, cfg.get('split_ratio', 0.8))
    X_model_train, X_calib, y_model_train, y_calib = make_model_calib_split(X_full_train, y_full_train, cfg.get('calib_ratio_of_train', 0.2))

    # 4) scale
    X_model_train_s, X_calib_s, X_test_s, scaler = scale_datasets(X_model_train, X_calib, X_test)

    # 5) train ensembles
    ensemble_size = cfg.get('ensemble_size', 7)
    solar_models = train_ensemble(cfg['solar_params'], X_model_train_s, y_model_train['ALLSKY_SFC_SW_DWN'], ensemble_size, seed_start=cfg.get('seed', 42))
    wind_models = train_ensemble(cfg['wind_params'], X_model_train_s, y_model_train['WS50M'], ensemble_size, seed_start=cfg.get('seed', 42)+1000)

    # 6) predict
    solar_test_mean, _, _, _ = ensemble_predict(solar_models, X_test_s)
    wind_test_mean, _, _, _ = ensemble_predict(wind_models, X_test_s)

    solar_calib_mean, _, _, _ = ensemble_predict(solar_models, X_calib_s)
    wind_calib_mean, _, _, _ = ensemble_predict(wind_models, X_calib_s)

    # 7) conformal
    q_solar = compute_split_conformal_quantile(y_calib['ALLSKY_SFC_SW_DWN'].values, solar_calib_mean, alpha=cfg.get('alpha', 0.05), scaling=cfg.get('scaling_factor', 1.15))
    q_wind = compute_split_conformal_quantile(y_calib['WS50M'].values, wind_calib_mean, alpha=cfg.get('alpha', 0.05), scaling=cfg.get('scaling_factor', 1.15))

    solar_lower, solar_upper = form_constant_intervals(solar_test_mean, q_solar)
    wind_lower, wind_upper = form_constant_intervals(wind_test_mean, q_wind)

    # 8) evaluation
    print_metrics(y_test['ALLSKY_SFC_SW_DWN'].values, solar_test_mean, solar_lower, solar_upper, alpha=cfg.get('alpha', 0.05), label='ALLSKY_SFC_SW_DWN (Ensemble mean)')
    print_metrics(y_test['WS50M'].values, wind_test_mean, wind_lower, wind_upper, alpha=cfg.get('alpha', 0.05), label='WS50M (Ensemble mean)')

    # 9) plot small window
    N = min(cfg.get('plot_n_points', 200), len(y_test))
    colors = [{ 'true': '#8B4500', 'pred': '#FFA500', 'interval': '#FFB347' },{ 'true': '#00008B', 'pred': '#00CED1', 'interval': '#87CEFA' }]
    units = { 'ALLSKY_SFC_SW_DWN': ' (W/m²)', 'WS50M': ' (m/s)' }
    preds = [solar_test_mean, wind_test_mean]
    lowers = [solar_lower, wind_lower]
    uppers = [solar_upper, wind_upper]

    for i, col in enumerate(targets):
        plt.figure(figsize=(14, 4))
        plt.plot(y_test[col].values[:N], label='True', marker='o', linestyle='-', color=colors[i]['true'])
        plt.plot(preds[i][:N], label='Ensemble Pred (mean)', marker='s', linestyle='--', color=colors[i]['pred'])
        plt.fill_between(range(N), lowers[i][:N], uppers[i][:N], color=colors[i]['interval'], alpha=0.3, label=f"{int((1-cfg.get('alpha',0.05))*100)}% Interval")
        plt.title(f"{col}: Ensemble XGBoost + Split Conformal (Agra)")
        plt.xlabel("Time (index order)")
        plt.ylabel(col + units.get(col, ''))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
