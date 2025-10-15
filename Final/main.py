import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
from pathlib import Path
import json
from datetime import datetime

from logger import setup_logger
from run_experiments import run_experiments



def load_dataset_metadata(data_root, dataset_name):

    meta_path = data_root / "Data" / f"{dataset_name}_meta.json"
    
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    
    return meta

def main():

    data_root = Path("/home/kuldeep/Fairness_Poject")
    output_root = Path("outputs")
    log_dir = Path("logs")
    
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    logger = setup_logger("dro_experiment", log_file, 1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    hyperparams = {
        'HIDDEN': 1024,
        'MB_SIZE': 4096,
        'NUM_EPOCHS': 200,
        'LR_THETA': 1e-3,
        'LR_LAMBDA': 5e-4,
        'LR_P': 1e-3,
        'INNER_P_STEPS': 10,
        'GAMMA': 0.02,
        'BETA_DRO': 5,
        'DROPOUT': 0.01,
    }
    
    dataset_name = "adult"
    seeds = [12]
    alphas = [0.1]
    
    logger.info("="*80)
    logger.info("DRO FAIRNESS EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Alphas: {alphas}")
    logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    logger.info("="*80 + "\n")
    
    meta = load_dataset_metadata(data_root, dataset_name)
    logger.info(f"Dataset metadata loaded successfully")
    logger.info(f"Protected attribute: {meta['protected_col']}")
    logger.info(f"Label: {meta['label_col']}")
    logger.info(f"Features: {len(meta['feature_cols'])} columns\n")
    
    summary_df = run_experiments(
        dataset_name=dataset_name,
        outroot=output_root,
        logger=logger,
        hyperparams=hyperparams,
        seeds=seeds,
        alphas=alphas,
        data_root=data_root,
        device=device,
        meta=meta
    )
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTS COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_root / dataset_name}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
        


if __name__ == "__main__":
    main()