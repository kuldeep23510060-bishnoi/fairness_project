import numpy as np
import pandas as pd
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, df, features, prot_col, pairs, pair_distances, gamma):

    model.eval()

    with torch.no_grad():
        X = torch.tensor(df[features].values, device=DEVICE, dtype=torch.float32)
        logits = model(X)

    preds = torch.sigmoid(logits).cpu().squeeze().numpy()
    df_eval = df.copy()
    df_eval["pred_score"] = preds
    df_eval["pred_label"] = (preds >= 0.5).astype(int)

    acc = float((df_eval["pred_label"] == df_eval["Y"]).mean())

    if prot_col in df_eval.columns:
        group0 = df_eval[df_eval[prot_col] == 0]
        group1 = df_eval[df_eval[prot_col] == 1]

        if len(group0) > 0 and len(group1) > 0:
            rate0 = group0["pred_score"].mean()
            rate1 = group1["pred_score"].mean()
            parity_gap = abs(rate0 - rate1)

    if pairs is not None and len(pairs) > 0 and pair_distances is not None:

        i, j = pairs[:, 0], pairs[:, 1]
        pred_diff = np.abs(df_eval.iloc[i]["pred_label"].values - df_eval.iloc[j]["pred_label"].values)

        if isinstance(pair_distances, torch.Tensor):
            pd_np = pair_distances.cpu().numpy()
        else:
            pd_np = np.array(pair_distances)
            
        violations = (pred_diff > (pd_np + gamma)).mean()

        if_viol = float(violations)
    else:
        if_viol = 0.0

    return acc, parity_gap, if_viol

