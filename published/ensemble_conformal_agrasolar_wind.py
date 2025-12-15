# ensemble_conformal_agrasolar_wind.py
import copy
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# -------------------------
# 1) Load & preprocess
# -------------------------
data = pd.read_csv('your_dataset.csv')

# rename if present
rename_map = {}
if 'YEAR' in data.columns: rename_map['YEAR'] = 'year'
if 'MO' in data.columns: rename_map['MO'] = 'month'
if 'DY' in data.columns: rename_map['DY'] = 'day'
if 'HR' in data.columns: rename_map['HR'] = 'hour'
if rename_map:
    data = data.rename(columns=rename_map)

# create datetime if components exist
if {'year','month','day','hour'}.issubset(set(data.columns)):
    data['datetime'] = pd.to_datetime(data[['year','month','day','hour']])
    data = data.sort_values('datetime').set_index('datetime')
    data.drop(columns=['year','month','day','hour'], inplace=True, errors='ignore')
else:
    # fallback: if there's a datetime column
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime').set_index('datetime')
    else:
        raise ValueError("No suitable datetime columns found (need YEAR/MO/DY/HR or datetime).")

# drop rows with NA in original target columns (keep after feature engineering we'll drop more)
data.dropna(how='all', inplace=True)

# Reduce float64 to float32 to save memory
float_cols = data.select_dtypes(include=['float64']).columns
if len(float_cols):
    data[float_cols] = data[float_cols].astype('float32')

# -------------------------
# 2) Feature engineering
# -------------------------
def create_features(df, target_cols):
    df_feat = df.copy()
    # Lags for target columns
    lags = [1, 2, 3, 6, 12, 24]
    for col in target_cols:
        if col not in df_feat.columns:
            raise KeyError(f"Target column '{col}' not in dataframe.")
        for lag in lags:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)

    # Rolling windows
    windows = [6, 12, 24, 48]
    for col in target_cols:
        for w in windows:
            df_feat[f'{col}_roll_mean_{w}'] = df_feat[col].rolling(window=w, min_periods=1).mean()
            df_feat[f'{col}_roll_std_{w}'] = df_feat[col].rolling(window=w, min_periods=1).std().fillna(0)
            df_feat[f'{col}_roll_min_{w}'] = df_feat[col].rolling(window=w, min_periods=1).min()
            df_feat[f'{col}_roll_max_{w}'] = df_feat[col].rolling(window=w, min_periods=1).max()

    # Cyclical time features
    hour = df_feat.index.hour
    doy = df_feat.index.dayofyear
    month = df_feat.index.month
    df_feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_feat['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    df_feat['doy_cos'] = np.cos(2 * np.pi * doy / 365)
    df_feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Cross features
    if 'T2M' in df_feat.columns and 'RH2M' in df_feat.columns:
        df_feat['temp_humidity'] = df_feat['T2M'] * df_feat['RH2M']
    # safe solar_clear if SZA exists
    if 'CLRSKY_SFC_SW_DWN' in df_feat.columns and 'SZA' in df_feat.columns:
        df_feat['solar_clear'] = df_feat['CLRSKY_SFC_SW_DWN'] * np.cos(np.pi * df_feat['SZA'] / 180)
    else:
        # create fallback columns to avoid key errors downstream
        if 'CLRSKY_SFC_SW_DWN' not in df_feat.columns:
            df_feat['CLRSKY_SFC_SW_DWN'] = 0.0
        if 'SZA' not in df_feat.columns:
            df_feat['SZA'] = 0.0
        df_feat['solar_clear'] = df_feat['CLRSKY_SFC_SW_DWN'] * np.cos(np.pi * df_feat['SZA'] / 180)

    return df_feat

# Targets
target_cols = ['ALLSKY_SFC_SW_DWN', 'WS50M']
for t in target_cols:
    if t not in data.columns:
        raise KeyError(f"Target column {t} missing from data.")

data = create_features(data, target_cols)
data = data.dropna().copy()   # drop rows that became NA after lags/rolling

# -------------------------
# 3) Train / test split
# -------------------------
y = data[target_cols].copy()
X = data.drop(columns=target_cols).copy()

# simple 80/20 time-series split (no shuffle)
split = int(len(X) * 0.8)
X_full_train, X_test = X.iloc[:split], X.iloc[split:]
y_full_train, y_test = y.iloc[:split], y.iloc[split:]

# further split full_train into model_train and calibration (no shuffle)
X_model_train, X_calib, y_model_train, y_calib = train_test_split(
    X_full_train, y_full_train, test_size=0.2, shuffle=False
)

# -------------------------
# 4) Scaling
# -------------------------
scaler = StandardScaler()
X_model_train = pd.DataFrame(scaler.fit_transform(X_model_train), columns=X.columns, index=X_model_train.index)
X_calib = pd.DataFrame(scaler.transform(X_calib), columns=X.columns, index=X_calib.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# -------------------------
# 5) Base XGBoost params (from your tuned values)
# -------------------------
solar_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 363,
    'max_depth': 7,
    'learning_rate': 0.026795561710426293,
    'subsample': 0.7491029794990068,
    'colsample_bytree': 0.9786660186576371,
    'reg_alpha': 0.2787219260896555,
    'reg_lambda': 0.06512891065430851,
    'tree_method': 'hist',
    'n_jobs': -1,
    'verbosity': 0
}

wind_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 204,
    'max_depth': 10,
    'learning_rate': 0.03981542127309022,
    'subsample': 0.725505267028803,
    'colsample_bytree': 0.9985654722085591,
    'reg_alpha': 0.166158932608926,
    'reg_lambda': 0.08765813395564175,
    'tree_method': 'hist',
    'n_jobs': -1,
    'verbosity': 0
}

# -------------------------
# 6) Ensemble utilities
# -------------------------
def make_param_variants(base_params, n_variants, seed_start=0):
    """Return list of param dicts with small jitter and different random_state for diversity."""
    variants = []
    for i in range(n_variants):
        p = copy.deepcopy(base_params)
        p['random_state'] = int(seed_start + i)
        # small jitter for some params
        rng = np.random.RandomState(seed_start + i)
        jitter = (rng.rand() - 0.5) * 0.2  # +/-10%
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

def ensemble_predict(models, X_df):
    preds_all = np.column_stack([m.predict(X_df.values) for m in models])  # (n_samples, n_models)
    mean_pred = preds_all.mean(axis=1)
    med_pred = np.median(preds_all, axis=1)
    std_pred = preds_all.std(axis=1)
    return mean_pred, med_pred, std_pred, preds_all

# -------------------------
# 7) Train ensembles
# -------------------------
ENSEMBLE_SIZE = 7
seed = 42

solar_variants = make_param_variants(solar_params, ENSEMBLE_SIZE, seed_start=seed)
wind_variants = make_param_variants(wind_params, ENSEMBLE_SIZE, seed_start=seed + 1000)

solar_models = []
wind_models = []

print("Training solar ensemble...")
for i, params in enumerate(solar_variants):
    m = XGBRegressor(**params)
    m.fit(X_model_train.values, y_model_train['ALLSKY_SFC_SW_DWN'].values,
          eval_metric='rmse', verbose=False)
    solar_models.append(m)
    print(f"  solar model {i+1}/{ENSEMBLE_SIZE} trained (seed={params['random_state']})")

print("Training wind ensemble...")
for i, params in enumerate(wind_variants):
    m = XGBRegressor(**params)
    m.fit(X_model_train.values, y_model_train['WS50M'].values,
          eval_metric='rmse', verbose=False)
    wind_models.append(m)
    print(f"  wind model {i+1}/{ENSEMBLE_SIZE} trained (seed={params['random_state']})")

# -------------------------
# 8) Predictions + Conformal calibration (split conformal)
# -------------------------
# Ensemble predictions
solar_test_mean, solar_test_med, solar_test_std, solar_test_all = ensemble_predict(solar_models, X_test)
wind_test_mean, wind_test_med, wind_test_std, wind_test_all = ensemble_predict(wind_models, X_test)

solar_calib_mean, _, solar_calib_std, _ = ensemble_predict(solar_models, X_calib)
wind_calib_mean, _, wind_calib_std, _ = ensemble_predict(wind_models, X_calib)

# Split conformal using absolute residuals on calibration set
alpha = 0.05
scaling_factor = 1.15  # tune this if you want more conservative intervals (previously you used 1.5)

solar_nonconf = np.abs(y_calib['ALLSKY_SFC_SW_DWN'].values - solar_calib_mean)
wind_nonconf = np.abs(y_calib['WS50M'].values - wind_calib_mean)

# Use (1 - alpha) quantile (split conformal)
q_solar = scaling_factor * np.quantile(solar_nonconf, 1 - alpha)
q_wind = scaling_factor * np.quantile(wind_nonconf, 1 - alpha)

# Constant-width intervals
solar_lower = solar_test_mean - q_solar
solar_upper = solar_test_mean + q_solar
wind_lower = wind_test_mean - q_wind
wind_upper = wind_test_mean + q_wind

# -------------------------
# 9) Evaluation
# -------------------------
def print_metrics(true, pred, lower, upper, label=""):
    true = np.asarray(true)
    pred = np.asarray(pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
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

print_metrics(y_test['ALLSKY_SFC_SW_DWN'].values, solar_test_mean, solar_lower, solar_upper, label="ALLSKY_SFC_SW_DWN (Ensemble mean)")
print_metrics(y_test['WS50M'].values, wind_test_mean, wind_lower, wind_upper, label="WS50M (Ensemble mean)")

# -------------------------
# 10) Plot a small window comparing true/pred + interval
# -------------------------
colors = [
    {
        'true': '#8B4500',     # Solar: Very dark orange
        'pred': '#FFA500',     # Bright orange
        'interval': '#FFB347'  # Medium orange
    },
    {
        'true': '#00008B',     # Wind: Very dark blue
        'pred': '#00CED1',     # Bright cyan
        'interval': '#87CEFA'  # Light sky blue
    }
]

units = {
    'ALLSKY_SFC_SW_DWN': ' (W/m²)',
    'WS50M': ' (m/s)'
}

N = min(200, len(y_test))  # plot up to 200 points

preds = [solar_test_mean, wind_test_mean]
lowers = [solar_lower, wind_lower]
uppers = [solar_upper, wind_upper]

for i, col in enumerate(target_cols):
    plt.figure(figsize=(14, 4))
    plt.plot(y_test[col].values[:N], label='True', marker='o', linestyle='-', color=colors[i]['true'])
    plt.plot(preds[i][:N], label='Ensemble Pred (mean)', marker='s', linestyle='--', color=colors[i]['pred'])
    plt.fill_between(np.arange(N), lowers[i][:N], uppers[i][:N], color=colors[i]['interval'], alpha=0.3,
                     label=f"{int((1-alpha)*100)}% Interval")
    plt.title(f"{col}: Ensemble XGBoost + Split Conformal (Agra)")
    plt.xlabel("Time (index order)")
    plt.ylabel(col + units.get(col, ''))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# 11) Optional: Save ensemble members (commented)
# -------------------------
# for idx, m in enumerate(solar_models):
#     m.save_model(f"solar_model_ens_{idx}.json")
# for idx, m in enumerate(wind_models):
#     m.save_model(f"wind_model_ens_{idx}.json")

print("Done.")
