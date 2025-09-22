import logging
import torch
import torch.nn.functional as F
import torch.optim as optim


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dro ( model,
                train_df,
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
                gamma ):

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

    history = {"loss": [], "lambda_dp": [], "lambda_if": [], "avg_constraint_dp": [], "avg_constraint_if": []}


    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        epoch_losses = []
        epoch_cons_dp = []
        epoch_cons_if = []

        for start in range(0, N, mb):
            idx = perm[start:start + mb].cpu().numpy()
            if idx.size == 0:
                continue
            idx_t = torch.tensor(idx, device=DEVICE, dtype=torch.long)

            Xb = X_all[idx_t]
            yb = y_all[idx_t]

            logits = model(Xb)
            probs = torch.sigmoid(logits).view(-1)
            losses = F.binary_cross_entropy_with_logits(logits.view(-1), yb, reduction='none')

            p_weights = torch.softmax(losses / beta, dim=0)
            dro_loss = (p_weights * losses).sum()

            with torch.no_grad():
                if Xb.size(0) > 1:
                    dists = torch.cdist(Xb, Xb, p=2)
                    idx_diag = torch.arange(dists.size(0), device=DEVICE)
                    dists[idx_diag, idx_diag] = float('inf')
                    nn_idx = dists.argmin(dim=1)
                else:
                    nn_idx = torch.zeros((Xb.size(0),), dtype=torch.long, device=DEVICE)

            preds_detached = probs.detach()
            preds_nn_det = preds_detached[nn_idx]
            pair_diff_det = torch.abs(preds_detached - preds_nn_det)

            pt_if_i = model.p_tilde_if[idx_t]
            
            if Xb.size(0) > 1:
                nn_global_idx = idx[nn_idx.cpu().numpy()]
                pt_if_j = model.p_tilde_if[torch.tensor(nn_global_idx, device=DEVICE, dtype=torch.long)]
            else:
                pt_if_j = model.p_tilde_if[idx_t]
            pair_weights_if = 0.5 * (pt_if_i + pt_if_j)
            wsum_if = pair_weights_if.sum()
            if wsum_if.item() <= 1e-12:
                cons_if = torch.tensor(0.0, device=DEVICE)
            else:
                cons_if = (pair_weights_if * pair_diff_det).sum() / (wsum_if + 1e-12)
                

            pt_dp_batch = model.p_tilde_dp[idx_t]
            mask_b_f = (female[idx].reshape(-1, 1)).float()
            mask_b_m = (male[idx].reshape(-1, 1)).float()
            probs_col = probs.view(-1, 1)

            def group_rate(p_probs_col, mask_b, pweights_full):
                sel = mask_b.squeeze() > 0.5
                if sel.sum().item() == 0:
                    return torch.tensor(0.0, device=DEVICE)
                p_sel = pweights_full[sel]
                preds_sel = p_probs_col.squeeze()[sel]
                total_w = p_sel.sum()
                return (p_sel * preds_sel).sum() / (total_w + 1e-12)

            rate_A1 = group_rate(probs_col, mask_b_f, pt_dp_batch)
            rate_A0 = group_rate(probs_col, mask_b_m, pt_dp_batch)
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

            for _ in range(inner_p_steps):
                pt_i = model.p_tilde_if[idx_t]
                if Xb.size(0) > 1:
                    pt_j = model.p_tilde_if[torch.tensor(nn_global_idx, device=DEVICE, dtype=torch.long)]
                else:
                    pt_j = model.p_tilde_if[idx_t]
                pair_weights_p = 0.5 * (pt_i + pt_j)
                wsum_p = pair_weights_p.sum()
                if wsum_p.item() <= 1e-12:
                    model.project_p_tilde_if(p_hat_if_t, max_l1_radius_if)
                else:
                    cons_if_p = (pair_weights_p * pair_diff_det).sum() / (wsum_p + 1e-12)
                    p_loss_if = - (model.lambdas[1].detach() * cons_if_p)
                    opt_p_if.zero_grad()
                    if p_loss_if.requires_grad:
                        p_loss_if.backward()
                        opt_p_if.step()
                    model.project_p_tilde_if(p_hat_if_t, max_l1_radius_if)

            for _ in range(inner_p_steps):
                pt_dp_i = model.p_tilde_dp[idx_t]

                def group_rate_p(preds_col_det, mask_b, pweights_full):
                    sel = mask_b.squeeze() > 0.5
                    if sel.sum().item() == 0:
                        return None
                    p_sel = pweights_full[sel]
                    preds_sel = preds_col_det.squeeze()[sel]
                    total_w = p_sel.sum()
                    return (p_sel * preds_sel).sum() / (total_w + 1e-12)

                rate_A1_p = group_rate_p(preds_detached.view(-1, 1), mask_b_f, pt_dp_i)
                rate_A0_p = group_rate_p(preds_detached.view(-1, 1), mask_b_m, pt_dp_i)

                if (rate_A1_p is None) or (rate_A0_p is None):
                    model.project_p_tilde_dp(p_hat_dp_t, max_l1_radius_dp)
                else:
                    cons_dp_p = (rate_A0_p - rate_A1_p).abs()
                    p_loss_dp = - (model.lambdas[0].detach() * cons_dp_p)
                    opt_p_dp.zero_grad()
                    if p_loss_dp.requires_grad:
                        p_loss_dp.backward()
                        opt_p_dp.step()
                    model.project_p_tilde_dp(p_hat_dp_t, max_l1_radius_dp)

            epoch_losses.append(float(dro_loss.detach().cpu().item()))
            epoch_cons_dp.append(float(cons_dp.detach().cpu().item()) if isinstance(cons_dp, torch.Tensor) else float(cons_dp))
            epoch_cons_if.append(float(cons_if.detach().cpu().item()) if isinstance(cons_if, torch.Tensor) else float(cons_if))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        avg_dp = float(np.mean(epoch_cons_dp)) if epoch_cons_dp else 0.0
        avg_if = float(np.mean(epoch_cons_if)) if epoch_cons_if else 0.0\

        history["loss"].append(avg_loss)
        history["avg_constraint_dp"].append(avg_dp)
        history["avg_constraint_if"].append(avg_if)
        history["lambda_dp"].append(float(model.lambdas[0].detach().cpu().item()))
        history["lambda_if"].append(float(model.lambdas[1].detach().cpu().item()))

        logger.info(f"[DRO-Joint] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} avg_dp={avg_dp:.6f} avg_if={avg_if:.6f} lambda_dp={model.lambdas[0].item():.4f} lambda_if={model.lambdas[1].item():.4f}")


    return history