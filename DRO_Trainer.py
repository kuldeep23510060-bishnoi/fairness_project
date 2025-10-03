import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict


class DROTrainer:
    
    def __init__(self, model, device, eps=1e-12):
        self.model = model
        self.device = device
        self.eps = eps  
        
    def _compute_group_rate(self, probs: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

        mask_flat = mask.squeeze()
        group_size = mask_flat.sum()
        
        if group_size < self.eps:
            return torch.tensor(0.0, device=self.device)
        
        weighted_sum = (weights * probs * mask_flat).sum()
        weight_total = (weights * mask_flat).sum()
        
        return weighted_sum / (weight_total + self.eps)
    
    def _compute_fairness_violations(self, pairs_in: torch.Tensor, dists_in: torch.Tensor, preds: torch.Tensor, idx_to_local: torch.Tensor, gamma: float) -> torch.Tensor:

        i_local = idx_to_local[pairs_in[:, 0]]
        j_local = idx_to_local[pairs_in[:, 1]]
        
        pair_diff = torch.abs(preds[i_local] - preds[j_local])
        violations = F.relu(pair_diff - (dists_in + gamma))
        
        return violations
    
    def _update_if_weights(self, pairs_in: torch.Tensor, preds_detached: torch.Tensor, dists_in: torch.Tensor, idx_to_local: torch.Tensor, gamma: float, opt_p_if: optim.Optimizer, lambda_if: torch.Tensor, p_hat_if: torch.Tensor,  max_radius_if: float, inner_steps: int):


        violations = self._compute_fairness_violations(pairs_in, dists_in, preds_detached, idx_to_local, gamma)
        
        for _ in range(inner_steps):
            pt_i = self.model.p_tilde_if[pairs_in[:, 0]]
            pt_j = self.model.p_tilde_if[pairs_in[:, 1]]
            pair_weights = 0.5 * (pt_i + pt_j)
            weight_sum = pair_weights.sum()
            
            if weight_sum > self.eps:
                constraint = (pair_weights * violations).sum() / (weight_sum + self.eps)
                loss = -(lambda_if.detach() * constraint)  
                
                opt_p_if.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                    opt_p_if.step()
            
            self.model.project_p_tilde_if(p_hat_if, max_radius_if)
    
    def _update_dp_weights(self, idx: torch.Tensor, preds_detached: torch.Tensor, 
                        mask_f: torch.Tensor, mask_m: torch.Tensor, 
                        opt_p_dp: optim.Optimizer, lambda_dp: torch.Tensor, 
                        p_hat_dp: torch.Tensor, max_radius_dp: float, 
                        inner_steps: int):
        
        # Extract masks for current batch
        mask_flat_f = mask_f.squeeze()[idx] if mask_f.dim() > 1 else mask_f[idx]
        mask_flat_m = mask_m.squeeze()[idx] if mask_m.dim() > 1 else mask_m[idx]
        
        # Early exit if either group is empty in this batch
        if mask_flat_f.sum() < self.eps or mask_flat_m.sum() < self.eps:
            self.model.project_p_tilde_dp(p_hat_dp, max_radius_dp)
            return
        
        for _ in range(inner_steps):
            # Extract weights for batch samples
            pt_dp = self.model.p_tilde_dp[idx]
            
            # Female group rate computation
            num_f = (pt_dp * preds_detached * mask_flat_f).sum()
            denom_f = (pt_dp * mask_flat_f).sum()
            rate_f = num_f / (denom_f + self.eps)
            
            # Male group rate computation
            num_m = (pt_dp * preds_detached * mask_flat_m).sum()
            denom_m = (pt_dp * mask_flat_m).sum()
            rate_m = num_m / (denom_m + self.eps)
            
            # Squared constraint (smooth gradient)
            constraint = (rate_m - rate_f).pow(2)
            loss = -(lambda_dp.detach() * constraint)
            
            # Gradient step
            opt_p_dp.zero_grad()
            if loss.requires_grad:
                loss.backward()
                opt_p_dp.step()
            
            # Project back to feasible set
            self.model.project_p_tilde_dp(p_hat_dp, max_radius_dp)
    
    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor,
                 female_val: torch.Tensor, male_val: torch.Tensor,
                 pairs_val: torch.Tensor, pair_dists_val: torch.Tensor,
                 gamma: float, scaling_factor: float) -> Dict[str, float]:

        with torch.no_grad():
            logits_val = self.model(X_val)
            probs_val = torch.sigmoid(logits_val).view(-1)
            
            losses_val = F.binary_cross_entropy_with_logits(
                logits_val.view(-1), y_val, reduction='none'
            )
            val_loss = losses_val.mean().item()
            
            preds_val = (probs_val > 0.5).float()
            val_acc = (preds_val == y_val).float().mean().item()
            
            mask_f = female_val > 0.5
            mask_m = male_val > 0.5
            rate_f = preds_val[mask_f].mean().item() if mask_f.sum() > 0 else 0.0
            rate_m = preds_val[mask_m].mean().item() if mask_m.sum() > 0 else 0.0
            val_parity = abs(rate_m - rate_f)
            
            if len(pairs_val) > 0:
                pair_diff = torch.abs(
                    preds_val[pairs_val[:, 0]] - preds_val[pairs_val[:, 1]]
                )
                dists = pair_dists_val / scaling_factor
                violations = F.relu(pair_diff - (dists + gamma))
                val_if = violations.mean().item()
            else:
                val_if = 0.0
        
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_parity': val_parity,
            'val_if': val_if
        }
    


