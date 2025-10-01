import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def train_dro(model,
              train_df,
              val_df,
              features,
              label_col,
              prot_col,
              mb,
              epochs,
              p_hat_dp,
              p_hat_if,
              max_l1_radius_dp,
              max_l1_radius_if,
              lr_theta,
              lr_lambda,
              lr_p,
              inner_p_steps,
              beta,
              DEVICE,
              logger,
              pairs,
              val_pairs,
              pair_distances,
              val_pair_dists,
              gamma):

    model.to(DEVICE)

    theta_params = [p for n, p in model.named_parameters() if n not in ("p_tilde_dp", "p_tilde_if", "lambdas")]
    opt_theta = optim.Adam(theta_params, lr=lr_theta)
    opt_lambda = optim.Adam([model.lambdas], lr=lr_lambda)
    opt_p_dp = optim.Adam([model.p_tilde_dp], lr=lr_p)
    opt_p_if = optim.Adam([model.p_tilde_if], lr=lr_p)

    N = len(train_df)
    X_all = torch.tensor(train_df[features].values, device=DEVICE, dtype=torch.float32)
    y_all = torch.tensor(train_df[label_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    female = torch.tensor(1 - train_df[prot_col].values, device=DEVICE).float().view(-1)
    male = torch.tensor(train_df[prot_col].values, device=DEVICE).float().view(-1)

    p_hat_dp_t = torch.tensor(p_hat_dp, device=DEVICE, dtype=torch.float32)
    p_hat_if_t = torch.tensor(p_hat_if, device=DEVICE, dtype=torch.float32)

    pairs_t = torch.tensor(pairs, device=DEVICE, dtype=torch.long)  
    pair_distances_t = torch.tensor(pair_distances, device=DEVICE, dtype=torch.float32)  

    X_val = torch.tensor(val_df[features].values, device=DEVICE, dtype=torch.float32)
    y_val = torch.tensor(val_df[label_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    female_val = torch.tensor(1 - val_df[prot_col].values, device=DEVICE).float().view(-1)
    male_val = torch.tensor(val_df[prot_col].values, device=DEVICE).float().view(-1)
    pairs_val_t = torch.tensor(val_pairs, device=DEVICE, dtype=torch.long)
    pair_distances_val_t = torch.tensor(val_pair_dists, device=DEVICE, dtype=torch.float32)

    history = {"loss": [], "lambda_dp": [], "lambda_if": [], "avg_constraint_dp": [], "avg_constraint_if": [],
               "val_loss": [], "val_acc": [], "val_parity": [], "val_if": []}

    num_features = len(features)
    scaling_factor = np.sqrt(num_features)

    def compute_group_rate(probs, mask, pweights):
        mask_flat = mask.squeeze()
        group_size = mask_flat.sum()
        if group_size < 1e-6:
            return torch.tensor(0.0, device=DEVICE)
        num = (pweights * probs * mask_flat).sum()
        denom = (pweights * mask_flat).sum()
        return num / denom if denom > 1e-12 else torch.tensor(0.0, device=DEVICE)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        epoch_losses = []
        epoch_cons_dp = []
        epoch_cons_if = []
        epoch_num_both = []

        for start in range(0, N, mb):
            idx = perm[start:start + mb]
            batch_size = len(idx)
            if batch_size == 0:
                continue

            Xb = X_all[idx]
            yb = y_all[idx]

            logits = model(Xb)
            probs = torch.sigmoid(logits).view(-1)
            hard_probs = torch.sigmoid(30 * logits).view(-1)
            losses = F.binary_cross_entropy_with_logits(logits.view(-1), yb, reduction='none')

            p_weights = torch.softmax(losses / beta, dim=0)
            dro_loss = (p_weights * losses).sum()

            in_batch = torch.zeros(N, device=DEVICE, dtype=torch.bool)
            in_batch[idx] = True
            both_in = in_batch[pairs_t[:, 0]] & in_batch[pairs_t[:, 1]]
            num_both = both_in.sum().item()
            epoch_num_both.append(num_both)

            if num_both == 0:
                cons_if = torch.tensor(0.0, device=DEVICE)
            else:
                pairs_in = pairs_t[both_in] 
                dists_in = pair_distances_t[both_in] / scaling_factor
                global_to_local = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
                global_to_local[idx] = torch.arange(batch_size, device=DEVICE)
                i_locals = global_to_local[pairs_in[:, 0]]
                j_locals = global_to_local[pairs_in[:, 1]]

                preds_detached = hard_probs.detach()
                pair_diff_det = torch.abs(preds_detached[i_locals] - preds_detached[j_locals])

                violations = F.relu(pair_diff_det - (dists_in + gamma))

                pt_if_i = model.p_tilde_if[pairs_in[:, 0]]
                pt_if_j = model.p_tilde_if[pairs_in[:, 1]]
                pair_weights_if = 0.5 * (pt_if_i + pt_if_j)
                wsum_if = pair_weights_if.sum()
                cons_if = (pair_weights_if * violations).sum() / (wsum_if + 1e-12) if wsum_if > 1e-12 else torch.tensor(0.0, device=DEVICE)

            pt_dp_batch = model.p_tilde_dp[idx]
            mask_b_f = female[idx].view(-1, 1) 
            mask_b_m = male[idx].view(-1, 1)
            probs_col = hard_probs.view(-1, 1)

            rate_A1 = compute_group_rate(probs_col.squeeze(), mask_b_f, pt_dp_batch)
            rate_A0 = compute_group_rate(probs_col.squeeze(), mask_b_m, pt_dp_batch)
            cons_dp = (rate_A0 - rate_A1).abs()

            lambdas = model.lambdas
            lagr = dro_loss + lambdas[0] * cons_dp + lambdas[1] * cons_if

            opt_theta.zero_grad()
            lagr.backward()
            opt_theta.step()

            opt_lambda.zero_grad()
            lambda_loss = - (model.lambdas[0] * cons_dp.detach() + model.lambdas[1] * cons_if.detach())
            lambda_loss.backward()
            opt_lambda.step()
            model.clamp_lambdas()

            preds_detached = hard_probs.detach() 

            if num_both > 0:
                for _ in range(inner_p_steps):
                    pt_i = model.p_tilde_if[pairs_in[:, 0]]
                    pt_j = model.p_tilde_if[pairs_in[:, 1]]
                    pair_weights_p = 0.5 * (pt_i + pt_j)
                    wsum_p = pair_weights_p.sum()
                    if wsum_p > 1e-12:
                        cons_if_p = (pair_weights_p * violations).sum() / (wsum_p + 1e-12) 
                        p_loss_if = - (model.lambdas[1].detach() * cons_if_p)
                        opt_p_if.zero_grad()
                        if p_loss_if.requires_grad:
                            p_loss_if.backward()
                            opt_p_if.step()
                    model.project_p_tilde_if(p_hat_if_t, max_l1_radius_if)
            else:
                model.project_p_tilde_if(p_hat_if_t, max_l1_radius_if)

            for _ in range(inner_p_steps):
                pt_dp_i = model.p_tilde_dp[idx]

                mask_flat_f = mask_b_f.squeeze()
                mask_flat_m = mask_b_m.squeeze()
                group_size_f = mask_flat_f.sum()
                group_size_m = mask_flat_m.sum()

                if group_size_f < 1e-6 or group_size_m < 1e-6:
                    model.project_p_tilde_dp(p_hat_dp_t, max_l1_radius_dp)
                    continue

                num_f = (pt_dp_i * preds_detached * mask_flat_f).sum()
                denom_f = (pt_dp_i * mask_flat_f).sum()
                rate_A1_p = num_f / denom_f if denom_f > 1e-12 else torch.tensor(0.0, device=DEVICE)

                num_m = (pt_dp_i * preds_detached * mask_flat_m).sum()
                denom_m = (pt_dp_i * mask_flat_m).sum()
                rate_A0_p = num_m / denom_m if denom_m > 1e-12 else torch.tensor(0.0, device=DEVICE)

                cons_dp_p = (rate_A0_p - rate_A1_p).abs()
                p_loss_dp = - (model.lambdas[0].detach() * cons_dp_p)
                opt_p_dp.zero_grad()
                if p_loss_dp.requires_grad:
                    p_loss_dp.backward()
                    opt_p_dp.step()
                model.project_p_tilde_dp(p_hat_dp_t, max_l1_radius_dp)

            epoch_losses.append(dro_loss.detach().item())
            epoch_cons_dp.append(cons_dp.detach().item() if isinstance(cons_dp, torch.Tensor) else float(cons_dp))
            epoch_cons_if.append(cons_if.detach().item() if isinstance(cons_if, torch.Tensor) else float(cons_if))

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dp = np.mean(epoch_cons_dp) if epoch_cons_dp else 0.0
        avg_if = np.mean(epoch_cons_if) if epoch_cons_if else 0.0
        avg_num_both = np.mean(epoch_num_both) if epoch_num_both else 0.0

        history["loss"].append(avg_loss)
        history["avg_constraint_dp"].append(avg_dp)
        history["avg_constraint_if"].append(avg_if)
        history["lambda_dp"].append(model.lambdas[0].detach().item())
        history["lambda_if"].append(model.lambdas[1].detach().item())

        logger.info(f"[DRO-Joint] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} avg_dp={avg_dp:.6f} avg_if={avg_if:.6f} avg_num_both={avg_num_both:.2f} lambda_dp={model.lambdas[0].item():.4f} lambda_if={model.lambdas[1].item():.4f}")

        with torch.no_grad():
            logits_val = model(X_val)
            probs_val = torch.sigmoid(logits_val).view(-1)
            losses_val = F.binary_cross_entropy_with_logits(logits_val.view(-1), y_val, reduction='none')
            val_loss = losses_val.mean().item()

            val_acc = ((probs_val > 0.5).float() == y_val).float().mean().item()

            hard_val = (probs_val > 0.5).float()
            mask_val_f = female_val > 0.5
            mask_val_m = male_val > 0.5
            rate_A1_val = hard_val[mask_val_f].mean().item() if mask_val_f.sum() > 0 else 0.0
            rate_A0_val = hard_val[mask_val_m].mean().item() if mask_val_m.sum() > 0 else 0.0
            val_parity = abs(rate_A0_val - rate_A1_val)

            if len(pairs_val_t) > 0:
                i_val = pairs_val_t[:, 0]
                j_val = pairs_val_t[:, 1]
                pair_diff_val = torch.abs(hard_val[i_val] - hard_val[j_val])
                dists_val = pair_distances_val_t / scaling_factor
                violations_val = F.relu(pair_diff_val - (dists_val + gamma))
                val_if = violations_val.mean().item()
            else:
                val_if = 0.0

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_parity"].append(val_parity)
            history["val_if"].append(val_if)

            logger.info(f"[DRO-Joint Val] epoch {epoch+1}/{epochs} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_parity={val_parity:.6f} val_if={val_if:.6f}")

    return history