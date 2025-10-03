import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict

def train_dro(model, train_df, val_df, features, label_col, prot_col, mb, epochs, p_hat_dp, p_hat_if, max_l1_radius_dp, max_l1_radius_if, lr_theta, lr_lambda, lr_p, inner_p_steps, beta, DEVICE, logger, pairs, val_pairs, pair_distances, val_pair_dists, gamma):

    trainer = DROTrainer(model, DEVICE)

    model.to(DEVICE)
    
    theta_params = [
        p for n, p in model.named_parameters() 
        if n not in ("p_tilde_dp", "p_tilde_if", "lambdas")
    ]
    
    opt_theta = optim.Adam(theta_params, lr=lr_theta)
    opt_lambda = optim.Adam([model.lambdas], lr=lr_lambda)
    opt_p_dp = optim.Adam([model.p_tilde_dp], lr=lr_p)
    opt_p_if = optim.Adam([model.p_tilde_if], lr=lr_p)
    
    N = len(train_df)
    X_all = torch.tensor(train_df[features].values, device=DEVICE, dtype=torch.float32)
    y_all = torch.tensor(train_df[label_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    female = torch.tensor(1 - train_df[prot_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    male = torch.tensor(train_df[prot_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    
    p_hat_dp_t = torch.tensor(p_hat_dp, device=DEVICE, dtype=torch.float32)
    p_hat_if_t = torch.tensor(p_hat_if, device=DEVICE, dtype=torch.float32)
    
    pairs_t = torch.tensor(pairs, device=DEVICE, dtype=torch.long)
    pair_distances_t = torch.tensor(pair_distances, device=DEVICE, dtype=torch.float32)
    
    X_val = torch.tensor(val_df[features].values, device=DEVICE, dtype=torch.float32)
    y_val = torch.tensor(val_df[label_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    female_val = torch.tensor(1 - val_df[prot_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    male_val = torch.tensor(val_df[prot_col].values, device=DEVICE, dtype=torch.float32).view(-1)
    pairs_val_t = torch.tensor(val_pairs, device=DEVICE, dtype=torch.long)
    pair_distances_val_t = torch.tensor(val_pair_dists, device=DEVICE, dtype=torch.float32)
    
    history = {
        "loss": [], "lambda_dp": [], "lambda_if": [],
        "avg_constraint_dp": [], "avg_constraint_if": [],
        "val_loss": [], "val_acc": [], "val_parity": [], "val_if": []
    }
    
    num_features = len(features)
    scaling_factor = np.sqrt(num_features)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        
        epoch_metrics = {
            'losses': [], 'cons_dp': [], 'cons_if': [], 'num_both': []
        }
        
        for start in range(0, N, mb):
            idx = perm[start:start + mb]
            batch_size = len(idx)
            if batch_size == 0:
                continue
            
            Xb = X_all[idx]
            yb = y_all[idx]
            
            logits = model(Xb)
            
            hard_probs = torch.sigmoid(30 * logits).view(-1)
            
            losses = F.binary_cross_entropy_with_logits(logits.view(-1), yb, reduction='none')
            importance_weights = torch.softmax(losses / beta, dim=0)
            dro_loss = (importance_weights * losses).sum()
            
            in_batch = torch.zeros(N, device=DEVICE, dtype=torch.bool)
            in_batch[idx] = True
            both_in = in_batch[pairs_t[:, 0]] & in_batch[pairs_t[:, 1]]
            num_both = both_in.sum().item()
            epoch_metrics['num_both'].append(num_both)
            
            if num_both > 0:
                pairs_in = pairs_t[both_in]
                dists_in = pair_distances_t[both_in] / scaling_factor
                
                idx_to_local = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
                idx_to_local[idx] = torch.arange(batch_size, device=DEVICE)
                
                violations = trainer._compute_fairness_violations(
                    pairs_in, dists_in, hard_probs, idx_to_local, gamma
                )
                
                pt_if_i = model.p_tilde_if[pairs_in[:, 0]]
                pt_if_j = model.p_tilde_if[pairs_in[:, 1]]
                pair_weights = 0.5 * (pt_if_i + pt_if_j)
                weight_sum = pair_weights.sum()
                
                cons_if = (pair_weights * violations).sum() / (weight_sum + trainer.eps) \
                    if weight_sum > trainer.eps else torch.tensor(0.0, device=DEVICE)
            else:
                cons_if = torch.tensor(0.0, device=DEVICE)
            
            pt_dp_batch = model.p_tilde_dp[idx]
            mask_f = female[idx].view(-1, 1)
            mask_m = male[idx].view(-1, 1)
            
            rate_f = trainer._compute_group_rate(hard_probs, mask_f, pt_dp_batch)
            rate_m = trainer._compute_group_rate(hard_probs, mask_m, pt_dp_batch)
            cons_dp = (rate_m - rate_f).abs()
            
            lagrangian = dro_loss + model.lambdas[0] * cons_dp + model.lambdas[1] * cons_if
            
            opt_theta.zero_grad()
            lagrangian.backward()
            opt_theta.step()
            
            opt_lambda.zero_grad()
            lambda_loss = -(
                model.lambdas[0] * cons_dp.detach() + 
                model.lambdas[1] * cons_if.detach()
            )
            lambda_loss.backward()
            opt_lambda.step()
            model.clamp_lambdas()
            
            preds_detached = hard_probs.detach()
            
            if num_both > 0:
                trainer._update_if_weights(
                    pairs_in, preds_detached, dists_in, idx_to_local, gamma,
                    opt_p_if, model.lambdas[1],
                    p_hat_if_t, max_l1_radius_if, inner_p_steps
                )
            else:
                model.project_p_tilde_if(p_hat_if_t, max_l1_radius_if)
            
            trainer._update_dp_weights(
                idx, preds_detached, mask_f, mask_m, opt_p_dp,
                model.lambdas[0], p_hat_dp_t, max_l1_radius_dp, inner_p_steps
            )
            
            epoch_metrics['losses'].append(dro_loss.detach().item())
            epoch_metrics['cons_dp'].append(
                cons_dp.detach().item() if isinstance(cons_dp, torch.Tensor) else float(cons_dp)
            )
            epoch_metrics['cons_if'].append(
                cons_if.detach().item() if isinstance(cons_if, torch.Tensor) else float(cons_if)
            )
        
        avg_loss = np.mean(epoch_metrics['losses']) if epoch_metrics['losses'] else 0.0
        avg_dp = np.mean(epoch_metrics['cons_dp']) if epoch_metrics['cons_dp'] else 0.0
        avg_if = np.mean(epoch_metrics['cons_if']) if epoch_metrics['cons_if'] else 0.0
        avg_num_both = np.mean(epoch_metrics['num_both']) if epoch_metrics['num_both'] else 0.0
        
        history["loss"].append(avg_loss)
        history["avg_constraint_dp"].append(avg_dp)
        history["avg_constraint_if"].append(avg_if)
        history["lambda_dp"].append(model.lambdas[0].detach().item())
        history["lambda_if"].append(model.lambdas[1].detach().item())
        
        logger.info(
            f"[DRO-Joint] epoch {epoch+1}/{epochs} "
            f"loss={avg_loss:.4f} avg_dp={avg_dp:.6f} avg_if={avg_if:.6f} "
            f"avg_num_both={avg_num_both:.2f} "
            f"lambda_dp={model.lambdas[0].item():.4f} "
            f"lambda_if={model.lambdas[1].item():.4f}"
        )
        
        model.eval()
        val_metrics = trainer._validate(
            X_val, y_val, female_val, male_val,
            pairs_val_t, pair_distances_val_t, gamma, scaling_factor
        )
        
        for key, value in val_metrics.items():
            history[key].append(value)
        
        logger.info(
            f"[DRO-Joint Val] epoch {epoch+1}/{epochs} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_acc={val_metrics['val_acc']:.4f} "
            f"val_parity={val_metrics['val_parity']:.6f} "
            f"val_if={val_metrics['val_if']:.6f}"
        )
    
    return history