from typing import Optional, Sequence

import numpy as np
import pandas as pd

def apply_adversarial_corruption(
                                    df: pd.DataFrame,
                                    alpha: float,
                                    feature_cols: Sequence[str],
                                    numeric_cols: Sequence[str],
                                    categorical_cols: Sequence[str],
                                    prot_col: str,
                                    seed: Optional[int],
                                
                                ) -> pd.DataFrame:
    

    if alpha <= 0.0:
        return df.copy()
    rng = np.random.RandomState(seed)
    df_cor = df.copy()

    numeric_cols = numeric_cols
    categorical_cols = categorical_cols
    mask = rng.rand(len(df_cor)) < alpha
    idxs = df_cor.index[mask]

    for col in numeric_cols:
        if col in df_cor.columns and len(idxs) > 0:
            df_cor[col] = df_cor[col].astype(np.float32)
            vals = df_cor[col]
            std = float(vals.std())
            if std == 0 or np.isnan(std):
                noise = np.zeros(len(idxs), dtype=np.float32)
            else:
                noise = rng.normal(0.0, 0.1 * std, size=len(idxs)).astype(np.float32)
            df_cor.loc[idxs, col] = vals.loc[idxs].to_numpy(dtype=np.float32) + noise

    if "Y" in df_cor.columns and len(idxs) > 0:
        df_cor.loc[idxs, "Y"] = 1 - df_cor.loc[idxs, "Y"]

    cat_values = {col: sorted(df_cor[col].dropna().unique().tolist()) for col in categorical_cols + [group_col] if col in df_cor.columns}
    for idx in idxs:
        for col in categorical_cols:
            if col not in df_cor.columns:
                continue
            cur = df_cor.at[idx, col]
            choices = [v for v in cat_values.get(col, []) if v != cur]
            if choices:
                df_cor.at[idx, col] = rng.choice(choices)
        if prot_col in df_cor.columns:
            cur = df_cor.at[idx, prot_col]
            choices = [v for v in cat_values.get(prot_col, []) if v != cur]
            if choices:
                df_cor.at[idx, prot_col] = rng.choice(choices)

    return df_cor