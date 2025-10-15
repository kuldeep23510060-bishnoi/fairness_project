import torch
import warnings
import torch
from math import sqrt
import torch.nn.functional as F
from evaluation import evaluate

class DROTrainer:
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.eps = config.eps
    

    def _compute_dro_loss(self, logits, targets, beta, eps):

        losses = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='none')
        scaled_losses = losses / (beta + eps)
        scaled_losses_stable = scaled_losses - scaled_losses.max()
        importance_weights = F.softmax(scaled_losses_stable, dim=0)
        dro_loss = (importance_weights * losses).sum()
        
        return dro_loss

    def _compute_group_rate(self, probs, mask, weights):

        mask_flat = mask.squeeze()
        group_size = mask_flat.sum()
        
        if group_size < self.eps:
            return torch.tensor(0.0, device=self.device)
        
        weighted_sum = (weights * probs * mask_flat).sum()
        weight_total = (weights * mask_flat).sum()
        
        rate = weighted_sum / (weight_total + self.eps)

        return rate
 
    def _compute_if_violations(self, pairs_in, dists_in, preds, idx_to_local): 

        i_local = idx_to_local[pairs_in[:, 0]]
        j_local = idx_to_local[pairs_in[:, 1]]

        pair_diff = torch.abs(preds[i_local] - preds[j_local])
        violations = F.relu(pair_diff - (dists_in + self.config.gamma))

        return violations
    

    def _compute_fairness_constraints(self, model, hard_probs, idx, female, male, pairs_in, dists_in, idx_to_local, num_both, device):

        pt_dp_batch = model.p_tilde_dp[idx]

        mask_f = female[idx].view(-1, 1)
        mask_m = male[idx].view(-1, 1)
        
        rate_f = self._compute_group_rate(hard_probs, mask_f, pt_dp_batch)
        rate_m = self._compute_group_rate(hard_probs, mask_m, pt_dp_batch)

        cons_dp = torch.abs(rate_m - rate_f)
        
        if num_both > 0:
            violations = self._compute_if_violations(pairs_in, dists_in, hard_probs, idx_to_local)
            pt_if_i = model.p_tilde_if[pairs_in[:, 0]]
            pt_if_j = model.p_tilde_if[pairs_in[:, 1]]
            pair_weights = 0.5 * (pt_if_i + pt_if_j)
            weight_sum = pair_weights.sum()
            cons_if = (pair_weights * violations).sum() / (weight_sum + self.eps)
        else:
            cons_if = torch.tensor(0.0, device=device)
    
        return cons_dp, cons_if
    

    def _get_batch_pair_info(self, idx, pairs_t, pair_distances_t, N, device, scaling_factor, eps):
        
        batch_size = len(idx)
        
        in_batch = torch.zeros(N, device=device, dtype=torch.bool)

        in_batch[idx] = True
        
        both_in = in_batch[pairs_t[:, 0]] & in_batch[pairs_t[:, 1]]
        num_both = both_in.sum().item()
        
        idx_to_local = torch.full((N,), -1, dtype=torch.long, device=device)
        idx_to_local[idx] = torch.arange(batch_size, device=device)
        
        if num_both > 0:
            pairs_in = pairs_t[both_in]
            dists_in = pair_distances_t[both_in] / (scaling_factor + eps)
        else:
            pairs_in = torch.empty((0, 2), dtype=torch.long, device=device)
            dists_in = torch.empty(0, dtype=torch.float32, device=device)
        
        return pairs_in, dists_in, idx_to_local, num_both
    

    
    def _update_if_weights(self, model, pairs_in, preds_detached, dists_in, idx_to_local, opt_p_if, p_hat_if, max_radius_if, inner_steps):

        violations = self._compute_if_violations(pairs_in, dists_in, preds_detached, idx_to_local)
        
        for step in range(inner_steps):

            pt_i = model.p_tilde_if[pairs_in[:, 0]]

            pt_j = model.p_tilde_if[pairs_in[:, 1]]

            pair_weights = 0.5 * (pt_i + pt_j)
            
            weight_sum = pair_weights.sum()
            
            constraint = (pair_weights * violations).sum() / (weight_sum + self.eps)
            
            loss = -(model.lambdas[1].detach() * constraint)
            
            opt_p_if.zero_grad()
            loss.backward()
            opt_p_if.step()
            
            model.project_p_tilde_if(p_hat_if, max_radius_if)


    def _update_dp_weights(self, model, idx, preds_detached, mask_f, mask_m, opt_p_dp, p_hat_dp, max_radius_dp, inner_steps):

        mask_flat_f = mask_f.squeeze()
        mask_flat_m = mask_m.squeeze()
        
        for step in range(inner_steps):

            pt_dp = model.p_tilde_dp[idx]
            
            num_f = (pt_dp * preds_detached * mask_flat_f).sum()
            denom_f = (pt_dp * mask_flat_f).sum()
            rate_f = num_f / (denom_f + self.eps)
            
            num_m = (pt_dp * preds_detached * mask_flat_m).sum()
            denom_m = (pt_dp * mask_flat_m).sum()
            rate_m = num_m / (denom_m + self.eps)

            constraint = torch.abs(rate_m - rate_f)

            loss = -(model.lambdas[0].detach() * constraint)
            
            opt_p_dp.zero_grad()
            loss.backward()
            opt_p_dp.step()
            
            model.project_p_tilde_dp(p_hat_dp, max_radius_dp)


    def _update_distribution_weights(self, model, hard_probs, idx, female, male, pairs_in, dists_in, idx_to_local, opt_p_dp, opt_p_if, p_hat_dp_t, p_hat_if_t, max_l1_radius_dp, max_l1_radius_if, inner_p_steps, num_both):

        preds_detached = hard_probs.detach()
        mask_f = female[idx].view(-1, 1)
        mask_m = male[idx].view(-1, 1)
        
        self._update_dp_weights(model, idx, preds_detached, mask_f, mask_m, opt_p_dp, p_hat_dp_t, max_l1_radius_dp, inner_p_steps)
        
        if num_both > 0:
            self._update_if_weights(model, pairs_in, preds_detached, dists_in, idx_to_local, opt_p_if, p_hat_if_t, max_l1_radius_if, inner_p_steps)
        else:
            model.project_p_tilde_if(p_hat_if_t, max_l1_radius_if)
    
    def _compute_gradient_norm(self, parameters):

        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return sqrt(total_norm) if total_norm > 0 else 0.0
    
    def _update_model_parameters(self, model, opt_theta, opt_lambda, dro_loss, cons_dp, cons_if, max_grad_norm):

        lagrangian = dro_loss + model.lambdas[0] * cons_dp + model.lambdas[1] * cons_if
        
        opt_theta.zero_grad()
        

        lagrangian.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(opt_theta.param_groups[0]['params'], max_norm=max_grad_norm)

        grad_norm = self._compute_gradient_norm(opt_theta.param_groups[0]['params'])
        
        opt_theta.step()

        opt_lambda.zero_grad()
        lambda_loss = -(model.lambdas[0] * cons_dp.detach() + model.lambdas[1] * cons_if.detach())
        lambda_loss.backward()
        opt_lambda.step()
        
        model.clamp_lambdas(max_val=10.0)
        
        return grad_norm
    

    def validate(self, X_val, y_val, pairs_val, pair_dists_val, scaling_factor, df_val, features, prot_col):

        self.model.eval()
        
        with torch.no_grad():
            logits_val = self.model(X_val)
            losses_val = F.binary_cross_entropy_with_logits(logits_val.view(-1), y_val, reduction='none')
            val_loss = losses_val.mean().item()
        
        if pairs_val is not None:
            if isinstance(pairs_val, torch.Tensor):
                pairs_numpy = pairs_val.cpu().numpy()
            else:
                pairs_numpy = pairs_val
        else:
            pairs_numpy = None
        
        if pair_dists_val is not None:
            scaled_pair_dists = pair_dists_val / (scaling_factor + self.eps)
            if isinstance(scaled_pair_dists, torch.Tensor):
                scaled_pair_dists = scaled_pair_dists.cpu()
        else:
            scaled_pair_dists = None
        
        val_acc, val_parity, val_if = evaluate(
            model=self.model,
            df=df_val,
            features=features,
            prot_col=prot_col,
            pairs=pairs_numpy,
            pair_distances=scaled_pair_dists,
            gamma=self.config.gamma,
            device=X_val.device
        )

        
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_parity': val_parity,
            'val_if': val_if
        }


