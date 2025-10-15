import torch

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from math import sqrt
from Model import  DROModel
from Radius_Computation import compute_radius
from evaluation import evaluate
from Config import DROConfig
from utils import set_seed
from train_dro import train_dro

def run_experiments(dataset_name, outroot, logger, hyperparams, seeds, alphas, data_root, device, meta):

    outroot = Path(outroot) / dataset_name
    outroot.mkdir(parents=True, exist_ok=True)
    
    hyperparams_path = outroot / "hyperparameters.json"

    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    logger.info(f"Hyperparameters saved to: {hyperparams_path}")
    
    protected_col = meta["protected_col"]
    label_col = meta["label_col"]
    feature_cols = meta["feature_cols"]
    
    input_base = data_root / "Dataset" / f"{dataset_name.capitalize()}_Datasets"
    
    logger.info(f"{'='*80}")
    logger.info(f"EXPERIMENT CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Corruption levels (alpha): {alphas}")
    logger.info(f"Protected attribute: {protected_col}")
    logger.info(f"Label column: {label_col}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"{'='*80}\n")
    
    per_seed_results = []
    
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING SEED {seed} ({seed_idx+1}/{len(seeds)})")
        logger.info(f"{'='*80}")
        
        set_seed(seed)

        seed_dir = input_base / f"seed_{seed}"
        
        seed_output_dir = outroot / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        
        test_dir = seed_dir / "test"

        test_df = pd.read_csv(test_dir / "test.csv")
        test_pairs = np.load(test_dir / "test_pairs.npy")
        test_pair_dists = np.load(test_dir / "test_pair_dists.npy")
        logger.info(f"Test data loaded: {len(test_df)} samples, {len(test_pairs)} pairs")

        

        results = {"alpha": [], "dro_acc": [], "dro_parity": [], "dro_if": [], "train_time": [], "best_epoch": []}
        
        for alpha_idx, alpha in enumerate(alphas):
            logger.info(f"\n{'-'*80}")
            logger.info(f"ALPHA = {alpha} ({alpha_idx+1}/{len(alphas)})")
            logger.info(f"{'-'*80}")
            
            train_dir = seed_dir / "train"
            train_df = pd.read_csv(train_dir / f"train_alpha_{alpha}.csv")
            train_pairs = np.load(train_dir / f"train_alpha_{alpha}_pairs.npy")
            train_pair_dists = np.load(train_dir / f"train_alpha_{alpha}_pair_dists.npy")
            
            val_dir = seed_dir / "val"
            val_df = pd.read_csv(val_dir / f"val_alpha_{alpha}.csv")
            val_pairs = np.load(val_dir / f"val_alpha_{alpha}_pairs.npy")
            val_pair_dists = np.load(val_dir / f"val_alpha_{alpha}_pair_dists.npy")
            
            N = len(train_df)
            logger.info(f"Data loaded - Train: {N}, Val: {len(val_df)}, Test: {len(test_df)}")
            

            max_l1_radius_dp, max_l1_radius_if = compute_radius(alpha, train_df, protected_col)

            logger.info(f"DRO radii computed - DP: {max_l1_radius_dp:.6f}, IF: {max_l1_radius_if:.6f}")
            
            p_hat = np.ones(N, dtype=np.float32) / float(N)
            
            config = DROConfig(
                mb=hyperparams['MB_SIZE'],
                epochs=hyperparams['NUM_EPOCHS'],
                max_l1_radius_dp=max_l1_radius_dp,
                max_l1_radius_if=max_l1_radius_if,
                lr_theta=hyperparams['LR_THETA'],
                lr_lambda=hyperparams['LR_LAMBDA'],
                lr_p=hyperparams['LR_P'],
                inner_p_steps=hyperparams['INNER_P_STEPS'],
                beta=hyperparams['BETA_DRO'],
                gamma=hyperparams['GAMMA'],
                dropout=hyperparams.get('DROPOUT', 0.1),
            )
            
            run_output_dir = seed_output_dir / f"alpha_{alpha}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            config.save(run_output_dir / "config.json")
            
            dro_model = DROModel(in_dim=len(feature_cols), n_samples=N, hidden=hyperparams['HIDDEN'], init_lambda=0.1, device=device, dropout=config.dropout).to(device)
            
            with torch.no_grad():
                p_hat_tensor = torch.tensor(p_hat, device=device, dtype=torch.float32)
                dro_model.p_tilde_dp.copy_(p_hat_tensor)
                dro_model.p_tilde_if.copy_(p_hat_tensor)
            
            logger.info(f"Model initialized - Parameters: {sum(p.numel() for p in dro_model.parameters()):,}")
            
            logger.info("Starting training...")
            start_time = datetime.now()
            
            history = train_dro(
                model=dro_model,
                train_df=train_df,
                val_df=val_df,
                features=feature_cols,
                label_col=label_col,
                prot_col=protected_col,
                config=config,
                p_hat_dp=p_hat,
                p_hat_if=p_hat,
                device=device,
                logger=logger,
                pairs=train_pairs,
                val_pairs=val_pairs,
                pair_distances=train_pair_dists,
                val_pair_dists=val_pair_dists,
                save_dir=run_output_dir
            )
            
            train_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training completed in {train_time:.2f}s ({train_time/60:.2f} min)")
            
            best_epoch = int(np.argmin(history['val_loss'])) + 1 if history.get('val_loss') else len(history['loss'])
            
            logger.info("Evaluating on test set...")
            
            dro_acc, dro_par, dro_if = evaluate(dro_model, test_df, feature_cols, protected_col,test_pairs, test_pair_dists, hyperparams['GAMMA'], device)
            
            logger.info(f"Test Results - Acc: {dro_acc:.4f}, Parity: {dro_par:.6f}, IF: {dro_if:.6f}")
            
            results["alpha"].append(alpha)
            results["dro_acc"].append(dro_acc)
            results["dro_parity"].append(dro_par)
            results["dro_if"].append(dro_if)
            results["train_time"].append(train_time)
            results["best_epoch"].append(best_epoch)
            
            pd.DataFrame(results).to_csv(
                seed_output_dir / "results_partial.csv", index=False
            )
            
            del dro_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Alpha {alpha} completed successfully")
                

        if results["alpha"]:
            df_seed = pd.DataFrame(results)
            df_seed["seed"] = seed
            df_seed.to_csv(seed_output_dir / "results_complete.csv", index=False)
            per_seed_results.append(df_seed)
            logger.info(f"Seed {seed} completed - {len(results['alpha'])} alphas processed")
        else:
            logger.warning(f"No results for seed {seed}")
    
    if not per_seed_results:
        logger.error("No results to aggregate across seeds")
        return pd.DataFrame()
    
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATING RESULTS")
    logger.info(f"{'='*80}")
    
    all_df = pd.concat(per_seed_results, ignore_index=True)
    all_df.to_csv(outroot / "summary_all_seeds_raw.csv", index=False)
    logger.info(f"Raw results saved: {outroot / 'summary_all_seeds_raw.csv'}")
    
    grouped = all_df.groupby("alpha")
    summary_rows = []
    
    for alpha, df_alpha in grouped:
        row = {"alpha": alpha, "n_seeds": len(df_alpha)}
        
        for metric in ["dro_acc", "dro_parity", "dro_if", "train_time"]:
            if metric in df_alpha.columns:
                values = df_alpha[metric].values
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                sem_val = std_val / sqrt(len(values)) if len(values) > 1 else 0.0
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
                row[f"{metric}_sem"] = sem_val
                row[f"{metric}_min"] = min_val
                row[f"{metric}_max"] = max_val
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows).sort_values("alpha")
    summary_path = outroot / "summary_all_seeds_agg.csv"
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(summary_df.to_string(index=False))
    logger.info(f"{'='*80}")
    logger.info(f"Aggregated results saved: {summary_path}")
    logger.info(f"All outputs saved to: {outroot}")
    logger.info(f"{'='*80}\n")
    
    return summary_df