import random
import numpy as np
import pandas as pd
import torch
from math import sqrt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from corruption import apply_adversarial_corruption
from sample_local_pairs import sample_local_pairs
from compute_radius import compute_radius
from Model import DROModel, SimpleMLP
from train_dro import train_dro
from train_naive import train_naive
from evaluation import evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s: int):
    
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_experiments(name, df_all, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols, alphas, seeds, outroot, logger,
                    HIDDEN, MB_SIZE, NUM_EPOCHS, LR_THETA,LR_LAMBDA, LR_P, INNER_P_STEPS, GAMMA, BETA_DRO):

    outroot = Path(outroot) / name
    outroot.mkdir(parents=True, exist_ok=True)

    per_seed_results = []

    for seed in seeds:

        logger.info(f"[{name}] starting seed={seed}")
        set_seed(seed)

        df = df_all.copy().reset_index(drop=True)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
        test_df[feature_cols] = scaler.transform(test_df[feature_cols].values).astype(np.float32)


        results = {"alpha": [], "dro_acc": [], "dro_parity": [], "dro_if": []}

        for alpha in alphas:

            logger.info(f"[{name} seed={seed}] alpha={alpha}")
            train_df = apply_adversarial_corruption(train_df, alpha, numeric_cols, categorical_cols, protected_col,  label_col, seed=seed)
            train_pairs, train_pair_dists = sample_local_pairs(train_df[feature_cols].values, k=10)
            test_pairs, test_pair_dists = sample_local_pairs(test_df[feature_cols].values, k=10)

            N = len(train_df)

            max_l1_radius_dp, max_l1_radius_if= compute_radius(alpha, train_df, protected_col)
            p_hat = np.ones(N, dtype=np.float32) / float(N)


            naive_model = SimpleMLP(len(feature_cols), hidden=HIDDEN).to(DEVICE)

            train_naive(naive_model,
                        train_df,
                        feature_cols,
                        label_col,
                        prot_col=protected_col,
                        mb=MB_SIZE,
                        epochs=NUM_EPOCHS,
                        lr=LR_THETA,
                        if_lambda=0.1,
                        dp_lambda=0.1,
                        lambda_lr=LR_LAMBDA,
                        pairs=train_pairs,
                        pair_distances=train_pair_dists,
                        gamma=GAMMA,
                        DEVICE=DEVICE, 
                        logger=logger)

            naive_acc, naive_par, naive_if = evaluate(naive_model, test_df, feature_cols, protected_col, test_pairs, test_pair_dists, GAMMA, DEVICE)
            logger.info(f"[{name} seed={seed} Naive] acc={naive_acc:.4f} parity={naive_par:.6f} if={naive_if:.6f}")

            dro_model = DROModel(len(feature_cols), n_samples=N, hidden=HIDDEN, init_lambda=0.1, DEVICE = DEVICE).to(DEVICE)


            with torch.no_grad():
                dro_model.p_tilde_dp.copy_(torch.tensor(p_hat, device=DEVICE, dtype=torch.float32))
                dro_model.p_tilde_if.copy_(torch.tensor(p_hat, device=DEVICE, dtype=torch.float32))


            _ = train_dro(dro_model, train_df, feature_cols, label_col, protected_col,mb=MB_SIZE, epochs=NUM_EPOCHS,
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
                                pair_distances=train_pair_dists,
                                gamma=GAMMA,)

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
        ("dro_acc", "Accuracy", "acc_plot"),
        ("dro_if", "IFViolation", "if_plot"),
        ("dro_parity", "ParityGap", "parity_plot"),
    ]
    for m, nice, fname in metrics_to_plot:
        plt.figure(figsize=(6, 4))
        x = summary_df["alpha"].values
        y = summary_df[f"{m}_mean"].values
        e = summary_df[f"{m}_sem"].values
        plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=f"DRO ({m})")
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