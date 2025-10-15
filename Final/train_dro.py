import torch
import json
import numpy as np
import torch.optim as optim

from DROTrainer import DROTrainer
from TrainingVisualizer import TrainingVisualizer
from MetricsAccumulator import MetricsAccumulator
from utils import prepare_data_tensors

def train_dro(model, train_df, val_df, features, label_col, prot_col, config, p_hat_dp, p_hat_if, device, logger, pairs, val_pairs, pair_distances, val_pair_dists, save_dir):

    trainer = DROTrainer(model, device, config)

    model.to(device)
    
    visualizer = TrainingVisualizer(save_dir)
    
    theta_params = [p for n, p in model.named_parameters() if n not in ("p_tilde_dp", "p_tilde_if", "lambdas") ]
    
    opt_theta = optim.AdamW(theta_params, lr=config.lr_theta, weight_decay=config.weight_decay)
    opt_lambda = optim.Adam([model.lambdas], lr=config.lr_lambda)
    opt_p_dp = optim.Adam([model.p_tilde_dp], lr=config.lr_p)
    opt_p_if = optim.Adam([model.p_tilde_if], lr=config.lr_p)
    
    N = len(train_df)

    logger.info(f"Training set: {N} samples, {len(features)} features")
    logger.info(f"Validation set: {len(val_df)} samples")
    
    X_all, y_all, female, male = prepare_data_tensors(train_df, features, label_col, prot_col, device)
    
    p_hat_dp_t = torch.tensor(p_hat_dp, device=device, dtype=torch.float32)
    p_hat_if_t = torch.tensor(p_hat_if, device=device, dtype=torch.float32)
    pairs_t = torch.tensor(pairs, device=device, dtype=torch.long)
    pair_distances_t = torch.tensor(pair_distances, device=device, dtype=torch.float32)
    

    X_val, y_val, female_val, male_val = prepare_data_tensors(val_df, features, label_col, prot_col, device)

    pairs_val_t = torch.tensor(val_pairs, device=device, dtype=torch.long)
    pair_distances_val_t = torch.tensor(val_pair_dists, device=device, dtype=torch.float32)
    

    history = {
        "loss": [], "lambda_dp": [], "lambda_if": [],
        "avg_constraint_dp": [], "avg_constraint_if": [],
        "avg_grad_norm": [],
        "val_loss": [], "val_acc": [], 
        "val_parity": [], "val_if": []
    }
    
    scaling_factor = np.sqrt(len(features))
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for {config.epochs} epochs")
    logger.info(f"Batch size: {config.mb}, Learning rates: θ={config.lr_theta}, λ={config.lr_lambda}, p={config.lr_p}")

# -----------------------------------------------------------------------------------------------------    
    for epoch in range(config.epochs):

        model.train()
        perm = torch.randperm(N, device=device)
        metrics = MetricsAccumulator()
        
        for batch_idx, start in enumerate(range(0, N, config.mb)):

            idx = perm[start:start + config.mb]
            
            Xb, yb = X_all[idx], y_all[idx]
            
            logits = model(Xb)
            
            hard_probs = hard_probs = torch.sigmoid(config.temperature * logits).view(-1)

            dro_loss = trainer._compute_dro_loss(logits, yb, config.beta,  config.eps)

            pairs_in, dists_in, idx_to_local, num_both = trainer._get_batch_pair_info(idx, pairs_t, pair_distances_t, N, device, scaling_factor, config.eps)
            

            cons_dp, cons_if = trainer._compute_fairness_constraints(model, hard_probs, idx, female, male,pairs_in, dists_in, idx_to_local, num_both, device)
            
            grad_norm = trainer._update_model_parameters(model, opt_theta, opt_lambda, dro_loss, cons_dp, cons_if, config.max_grad_norm)

            trainer._update_distribution_weights(model, hard_probs, idx, female, male, pairs_in, dists_in, idx_to_local, opt_p_dp, opt_p_if, p_hat_dp_t, p_hat_if_t, config.max_l1_radius_dp, config.max_l1_radius_if, config.inner_p_steps, num_both)

            metrics.add(
                dro_loss.detach().item(),
                cons_dp.detach().item() if isinstance(cons_dp, torch.Tensor) else float(cons_dp),
                cons_if.detach().item() if isinstance(cons_if, torch.Tensor) else float(cons_if),
                num_both,
                grad_norm,
                model.lambdas[0].detach().item(),
                model.lambdas[1].detach().item()
            )
            
            if (batch_idx + 1) % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        

        epoch_avg = metrics.get_averages()
        
        history["loss"].append(epoch_avg['avg_loss'])
        history["avg_constraint_dp"].append(epoch_avg['avg_dp'])
        history["avg_constraint_if"].append(epoch_avg['avg_if'])
        history["avg_grad_norm"].append(epoch_avg['avg_grad_norm'])
        history["lambda_dp"].append(model.lambdas[0].detach().item())
        history["lambda_if"].append(model.lambdas[1].detach().item())
        
        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            logger.info(
                f"[Epoch {epoch+1:>4}/{config.epochs}] "
                f"Loss: {epoch_avg['avg_loss']:.4f} | "
                f"DP: {epoch_avg['avg_dp']:.6f} | "
                f"IF: {epoch_avg['avg_if']:.6f} | "
                f"λ_DP: {model.lambdas[0].item():.4f} | "
                f"λ_IF: {model.lambdas[1].item():.4f} | "
                f"GradNorm: {epoch_avg['avg_grad_norm']:.4f}"
            )
        
        val_metrics = trainer.validate(X_val, y_val, pairs_val_t, pair_distances_val_t, scaling_factor, val_df, features, prot_col)
        
        for key, value in val_metrics.items():
            if key in history:
                history[key].append(value)
        
        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            logger.info(
                f"[Val   {epoch+1:>4}/{config.epochs}] "
                f"Loss: {val_metrics['val_loss']:.4f} | "
                f"Acc: {val_metrics['val_acc']:.4f} | "
                f"Parity: {val_metrics['val_parity']:.6f} | "
                f"IF: {val_metrics['val_if']:.6f}"
            )
        
        if val_metrics['val_loss'] < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            if save_dir:
                checkpoint_path = save_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_theta_state_dict': opt_theta.state_dict(),
                    'optimizer_lambda_state_dict': opt_lambda.state_dict(),
                    'val_loss': val_metrics['val_loss'],
                    'val_acc': val_metrics['val_acc'],
                    'val_parity': val_metrics['val_parity'],
                    'val_if': val_metrics['val_if'],
                    'history': history,
                    'config': config.to_dict()
                }, checkpoint_path)
                logger.info(f"Best model saved (epoch {epoch+1}, val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (patience={config.early_stopping_patience})")
                break
        
        if save_dir and (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_theta_state_dict': opt_theta.state_dict(),
                'optimizer_lambda_state_dict': opt_lambda.state_dict(),
                'history': history,
                'config': config.to_dict()
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    logger.info("Training completed successfully")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    if visualizer and save_dir:
        logger.info("Generating visualizations and reports...")
        visualizer.plot_training_curves(history)
        visualizer.create_summary_report(history, config)
        
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")
        
        final_model_path = save_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'history': history
        }, final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info(f"All outputs saved to: {save_dir}")
    
    return history