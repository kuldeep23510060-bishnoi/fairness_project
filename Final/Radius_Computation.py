import pandas as pd
from typing import Tuple

def compute_radius(alpha: float, train_df: pd.DataFrame, protected_col: str) -> Tuple[float, float]:

    tv_if = 2.0 * alpha - alpha * alpha
    l1_radius_if = 2.0 * tv_if
    
    # Demographic parity radius
    pi = train_df[protected_col].value_counts(normalize=True).to_dict()
    
    # Compute per-group corruption bounds
    gamma_js = {}
    for j, pj in pi.items():
        if pj > 1e-10:  # Avoid division by zero
            gamma_js[j] = alpha / ((1.0 - alpha) * pj)
        else:
            gamma_js[j] = 0.0
    
    # TV distance bounds
    tv_bounds = [2.0 * gamma_js.get(j, 0.0) for j in [0, 1]]
    
    # Use minimum to be conservative
    l1_radius_dp = min(tv_bounds) if tv_bounds else 0.0
    
    return l1_radius_dp, l1_radius_if