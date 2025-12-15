from xgboost import XGBRegressor
from typing import Dict

def build_xgb(params: Dict) -> XGBRegressor:
    # shallow wrapper so other parts import from here
    return XGBRegressor(**params)
