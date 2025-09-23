import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def train_naive(model, train_df, features, label_col, prot_col, mb, epochs, lr, if_lambda, dp_lambda, lambda_lr, pairs, pair_distances, gamma, DEVICE, logger):
    
    model.to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)

    lambda_if_param = nn.Parameter(torch.tensor(float(if_lambda), device=DEVICE, dtype=torch.float32))
    lambda_dp_param = nn.Parameter(torch.tensor(float(dp_lambda), device=DEVICE, dtype=torch.float32))

    params_for_lambda_opt = [p for p in (lambda_if_param, lambda_dp_param) if p.requires_grad]

    opt_lambda = optim.Adam(params_for_lambda_opt, lr=lambda_lr) if params_for_lambda_opt else None

    X_all = torch.tensor(train_df[features].values, device=DEVICE, dtype=torch.float32)
    y_all = torch.tensor(train_df[label_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    prot_arr = torch.tensor(train_df[prot_col].values, device=DEVICE).float().view(-1)
    N = len(train_df)

    have_pairs = (pairs is not None and len(pairs) > 0)

    if have_pairs:
        pairs_np = np.asarray(pairs, dtype=int)
        pair_dists_np = np.asarray(pair_distances, dtype=float)

    for epoch in range(epochs):

        perm = torch.randperm(N, device=DEVICE)

        epoch_ce, epoch_if, epoch_dp = [], [], []

        for start in range(0, N, mb):

            idx_t = perm[start:start + mb]

            if idx_t.numel() == 0:
                continue

            idx = idx_t.cpu().numpy()
            Xb = X_all[idx_t]
            yb = y_all[idx_t]
            prot_b = prot_arr[idx_t]

            logits = model(Xb)
            ce_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if have_pairs:
                mask_pairs = np.isin(pairs_np[:, 0], idx) & np.isin(pairs_np[:, 1], idx)
                sel_pair_ids = np.nonzero(mask_pairs)[0]
            else:
                sel_pair_ids = np.array([], dtype=int)

            if sel_pair_ids.size == 0:
                loss_if = torch.tensor(0.0, device=DEVICE)
            else:
                sel_pairs = pairs_np[sel_pair_ids]
                sel_dists = pair_dists_np[sel_pair_ids]
                preds = torch.sigmoid(logits).view(-1)
                local_pos = {int(v): k for k, v in enumerate(idx)}
                i_idxs = [local_pos[int(i)] for i in sel_pairs[:, 0]]
                j_idxs = [local_pos[int(j)] for j in sel_pairs[:, 1]]
                preds_i = preds[i_idxs]
                preds_j = preds[j_idxs]
                dvals = torch.tensor(sel_dists, device=DEVICE, dtype=torch.float32)
                pair_diff = torch.abs(preds_i - preds_j)
                pair_viol = F.relu(pair_diff - (dvals + gamma))
                loss_if = pair_viol.mean()

            # DP loss
            preds_batch = torch.sigmoid(logits).view(-1)
            mask1 = (prot_b > 0.5)
            mask0 = ~mask1
            if mask0.sum() == 0 or mask1.sum() == 0:
                loss_dp = torch.tensor(0.0, device=DEVICE)
            else:
                r1 = preds_batch[mask1].mean()
                r0 = preds_batch[mask0].mean()
                loss_dp = torch.abs(r0 - r1)

            loss_total = ce_loss + (lambda_if_param * loss_if) + (lambda_dp_param * loss_dp)

            opt.zero_grad()
            loss_total.backward()
            opt.step()

            if opt_lambda is not None:
                opt_lambda.zero_grad()
                lambda_loss = - (lambda_if_param * loss_if.detach() + lambda_dp_param * loss_dp.detach())
                lambda_loss.backward()
                opt_lambda.step()
                with torch.no_grad():
                    lambda_if_param.data.clamp_(min=0.0)
                    lambda_dp_param.data.clamp_(min=0.0)

            epoch_ce.append(float(ce_loss.detach().cpu().item()))
            epoch_if.append(float(loss_if.detach().cpu().item()) if isinstance(loss_if, torch.Tensor) else float(loss_if))
            epoch_dp.append(float(loss_dp.detach().cpu().item()) if isinstance(loss_dp, torch.Tensor) else float(loss_dp))

        avg_ce = float(np.mean(epoch_ce)) if epoch_ce else 0.0
        avg_if = float(np.mean(epoch_if)) if epoch_if else 0.0
        avg_dp = float(np.mean(epoch_dp)) if epoch_dp else 0.0
        lambda_if_val = float(lambda_if_param.detach().cpu().item())
        lambda_dp_val = float(lambda_dp_param.detach().cpu().item())
        logger.info(f"[Naive-Lagrangian] epoch {epoch+1}/{epochs} CE={avg_ce:.4f} IF_loss={avg_if:.6f} DP_loss={avg_dp:.6f} lambda_if={lambda_if_val:.6f} lambda_dp={lambda_dp_val:.6f}")