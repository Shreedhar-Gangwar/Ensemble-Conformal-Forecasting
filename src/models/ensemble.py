import copy
import numpy as np
from typing import List, Dict
from xgboost import XGBRegressor

def make_param_variants(base_params: Dict, n_variants: int, seed_start=0):
    variants = []
    for i in range(n_variants):
        p = copy.deepcopy(base_params)
        p['random_state'] = int(seed_start + i)
        rng = np.random.RandomState(seed_start + i)
        jitter = (rng.rand() - 0.5) * 0.2
        if 'subsample' in p:
            p['subsample'] = float(max(0.3, min(1.0, p['subsample'] * (1 + jitter))))
        if 'colsample_bytree' in p:
            p['colsample_bytree'] = float(max(0.3, min(1.0, p['colsample_bytree'] * (1 + jitter/2))))
        if 'reg_lambda' in p:
            p['reg_lambda'] = float(max(1e-6, p['reg_lambda'] * (1 + jitter)))
        if 'reg_alpha' in p:
            p['reg_alpha'] = float(max(0.0, p['reg_alpha'] * (1 + jitter)))
        variants.append(p)
    return variants

def train_ensemble(base_params: Dict, X_train, y_train, ensemble_size: int, seed_start=0):
    variants = make_param_variants(base_params, ensemble_size, seed_start=seed_start)
    models = []
    for i, params in enumerate(variants):
        m = XGBRegressor(**params)
        m.fit(X_train.values, y_train.values, eval_metric='rmse', verbose=False)
        models.append(m)
    return models

def ensemble_predict(models: List[XGBRegressor], X_df):
    preds_all = np.column_stack([m.predict(X_df.values) for m in models])
    mean_pred = preds_all.mean(axis=1)
    med_pred = np.median(preds_all, axis=1)
    std_pred = preds_all.std(axis=1)
    return mean_pred, med_pred, std_pred, preds_all
