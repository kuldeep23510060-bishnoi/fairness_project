import numpy as np

def apply_adversarial_corruption(df, alpha, numeric_cols, categorical_cols, protected_col, label_col, seed):

    if alpha == 0.0:
        return df.copy()
    
    rng = np.random.RandomState(seed)
    
    df_cor = df.copy()

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

    if label_col in df_cor.columns and len(idxs) > 0:
        df_cor.loc[idxs, label_col] = 1 - df_cor.loc[idxs, label_col]

    cat_values = {col: sorted(df_cor[col].dropna().unique().tolist()) for col in categorical_cols + [protected_col] if col in df_cor.columns}
    
    for idx in idxs:
        for col in categorical_cols:
            if col not in df_cor.columns:
                continue
            cur = df_cor.at[idx, col]
            choices = [v for v in cat_values.get(col, []) if v != cur]
            if choices:
                df_cor.at[idx, col] = rng.choice(choices)
        if protected_col in df_cor.columns:
            cur = df_cor.at[idx, protected_col]
            choices = [v for v in cat_values.get(protected_col, []) if v != cur]
            if choices:
                df_cor.at[idx, protected_col] = rng.choice(choices)

    return df_cor


    