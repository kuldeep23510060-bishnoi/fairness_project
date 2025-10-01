import os
import logging
import json
import random
import numpy as np
import pandas as pd
import torch
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
from compute_radius import compute_radius
from Model import DROModel, SimpleMLP
from train_dro import train_dro
from evaluation import evaluate
import argparse

OUTPUT_DIR = "./outputs" 

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("dro")


HIDDEN = 128
MB_SIZE = 5120
NUM_EPOCHS = 100
LR_THETA = 1e-3
LR_LAMBDA = 5e-4
LR_P = 1e-3
INNER_P_STEPS = 10
GAMMA = 0.02
BETA_DRO = 50
# seeds = [1, 12, 123, 1234, 12345, 54321, 5432, 543, 54, 5]
seeds = [123]
# alphas = [0, 0.1, 0.2, 0.3, 0.4]
alphas = [0.2]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiments(name, outroot, logger, HIDDEN, MB_SIZE, NUM_EPOCHS, LR_THETA, LR_LAMBDA, LR_P, INNER_P_STEPS, GAMMA, BETA_DRO, seeds, alphas, data_root):
    outroot = Path(outroot) / name
    outroot.mkdir(parents=True, exist_ok=True)

    meta_path = Path(data_root) / "Data" / f"{name}_meta.json"
    input_base = Path(data_root) / "Dataset" / f"{name.capitalize()}_Datasets"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {meta_path}. Please check the path.")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    protected_col = meta["protected_col"]
    label_col = meta["label_col"]
    feature_cols = meta.get("feature_cols", [])

    per_seed_results = []

    for seed in seeds:
        logger.info(f"[{name}] starting seed={seed}")
        set_seed(seed)

        seed_dir = input_base / f"seed_{seed}"

        test_dir = seed_dir / "test"
        test_df = pd.read_csv(test_dir / "test.csv")
        test_pairs = np.load(test_dir / "test_pairs.npy")
        test_pair_dists = np.load(test_dir / "test_pair_dists.npy")

        val_dir = seed_dir / "val"

        results = {"alpha": [], "dro_acc": [], "dro_parity": [], "dro_if": []}

        for alpha in alphas:
            logger.info(f"[{name} seed={seed}] alpha={alpha}")

            train_dir = seed_dir / "train"
            train_df = pd.read_csv(train_dir / f"train_alpha_{alpha}.csv")
            train_pairs = np.load(train_dir / f"train_alpha_{alpha}_pairs.npy")
            train_pair_dists = np.load(train_dir / f"train_alpha_{alpha}_pair_dists.npy")

            val_df = pd.read_csv(val_dir / f"val_alpha_{alpha}.csv")
            val_pairs = np.load(val_dir / f"val_alpha_{alpha}_pairs.npy")
            val_pair_dists = np.load(val_dir / f"val_alpha_{alpha}_pair_dists.npy")

            N = len(train_df)

            max_l1_radius_dp, max_l1_radius_if = compute_radius(alpha, train_df, protected_col)
            p_hat = np.ones(N, dtype=np.float32) / float(N)

            # Train DRO model
            dro_model = DROModel(len(feature_cols), n_samples=N, hidden=HIDDEN, init_lambda=0.1, DEVICE=DEVICE).to(DEVICE)

            with torch.no_grad():
                dro_model.p_tilde_dp.copy_(torch.tensor(p_hat, device=DEVICE, dtype=torch.float32))
                dro_model.p_tilde_if.copy_(torch.tensor(p_hat, device=DEVICE, dtype=torch.float32))

            _ = train_dro(dro_model, train_df, val_df, feature_cols, label_col, protected_col, mb=MB_SIZE, epochs=NUM_EPOCHS,
                          p_hat_dp=p_hat,
                          p_hat_if=p_hat,
                          max_l1_radius_dp=max_l1_radius_dp,
                          max_l1_radius_if=max_l1_radius_if,
                          lr_theta=LR_THETA,
                          lr_lambda=LR_LAMBDA,
                          lr_p=LR_P,
                          inner_p_steps=INNER_P_STEPS,
                          beta=BETA_DRO,
                          DEVICE=DEVICE, 
                          logger=logger,
                          pairs=train_pairs,
                          val_pairs=val_pairs,
                          pair_distances=train_pair_dists,
                          val_pair_dists=val_pair_dists,
                          gamma=GAMMA)

            dro_acc, dro_par, dro_if = evaluate(dro_model, test_df, feature_cols, protected_col, test_pairs, test_pair_dists, GAMMA, DEVICE)
            logger.info(f"[{name} seed={seed} DRO] acc={dro_acc:.4f} parity={dro_par:.6f} if={dro_if:.6f}")

            results["alpha"].append(alpha)
            results["dro_acc"].append(dro_acc)
            results["dro_parity"].append(dro_par)
            results["dro_if"].append(dro_if)

            pd.DataFrame(results).to_csv(outroot / f"summary_seed_{seed}.csv", index=False)

        df_seed = pd.DataFrame(results)
        df_seed["seed"] = seed
        per_seed_results.append(df_seed)

    all_df = pd.concat(per_seed_results, ignore_index=True)
    all_df.to_csv(outroot / "summary_all_seeds_raw.csv", index=False)

    grouped = all_df.groupby("alpha")
    summary_rows = []
    for alpha, df_alpha in grouped:
        row = {"alpha": alpha, "n_seeds": len(seeds)}
        for m in ["dro_acc", "dro_parity", "dro_if"]:
            mean_val = df_alpha[m].mean()
            std_val = df_alpha[m].std(ddof=1) if len(seeds) > 1 else 0.0
            sem_val = std_val / (sqrt(len(seeds)) if len(seeds) > 1 else 1.0)
            row[f"{m}_mean"] = mean_val
            row[f"{m}_sem"] = sem_val
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("alpha")
    summary_df.to_csv(outroot / "summary_all_seeds_agg.csv", index=False)

    metrics_to_plot = [
        ("acc", "Accuracy", "acc_plot"),
        ("if", "IFViolation", "if_plot"),
        ("parity", "ParityGap", "parity_plot"),
    ]
    for m, nice, fname in metrics_to_plot:
        plt.figure(figsize=(6, 4))
        x = summary_df["alpha"].values
        for model in ["dro"]:
            y = summary_df[f"{model}_{m}_mean"].values
            e = summary_df[f"{model}_{m}_sem"].values
            plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=f"{model.upper()}")
            plt.fill_between(x, y - e, y + e, alpha=0.12)
        plt.xlabel("alpha")
        if "acc" in m:
            plt.ylabel("Accuracy")
        elif "parity" in m:
            plt.ylabel("Parity gap")
        else:
            plt.ylabel("IF violation")
        plt.title(f"{name} — {nice} (mean ± SEM)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outroot / f"{fname}.png")
        plt.close()

    logger.info(f"[{name}] finished. Outputs saved to {outroot}")
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRO experiments on fairness datasets.")
    parser.add_argument("--datasets", nargs="+", default=["adult"], choices=["adult"],
                        help="Datasets to run experiments on.")
    parser.add_argument("--output_dir", default="./outputs", help="Directory to save outputs.")
    parser.add_argument("--data_root", default="/home/kuldeep/Fairness_Poject", help="Root directory for datasets and metadata.")
    
    args = parser.parse_args()
    
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name in args.datasets:
        run_experiments(name, OUTPUT_DIR, logger, HIDDEN, MB_SIZE, NUM_EPOCHS, LR_THETA, LR_LAMBDA, LR_P, INNER_P_STEPS, GAMMA, BETA_DRO, seeds, alphas, args.data_root)