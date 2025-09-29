import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from corruption import apply_adversarial_corruption

meta_path = Path("/home/kuldeep/Fairness_Poject/Data/bank_meta.json")

with meta_path.open("r", encoding="utf-8") as f:
    meta = json.load(f)

protected_col = meta["protected_col"]
label_col = meta["label_col"]
feature_cols = meta.get("feature_cols", [])
numeric_cols = meta.get("numeric_cols", [])
categorical_cols = meta.get("categorical_cols", [])

bank_df  = pd.read_csv("/home/kuldeep/Fairness_Poject/Data/bank_processed.csv") 

df = bank_df.copy().reset_index(drop=True)

ALPHAS = [0, 0.1, 0.2, 0.3, 0.4]
SEEDS = [1, 12, 123, 1234, 12345, 54321, 5432, 543, 54, 5]

output_base = Path("/home/kuldeep/Fairness_Poject/Dataset/Bank_Datasets")

for seed in SEEDS:
    seed_dir = output_base / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = seed_dir / "train"
    val_dir = seed_dir / "val"
    test_dir = seed_dir / "test"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    train_val_df[feature_cols] = scaler.fit_transform(train_val_df[feature_cols].values).astype(np.float32)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].values).astype(np.float32)
    
    test_df.to_csv(test_dir / "test.csv", index=False)
    train_val_scaled = train_val_df.copy()
    
    for alpha in ALPHAS:

        corrupted_df = apply_adversarial_corruption(train_val_scaled.copy(), alpha, numeric_cols, categorical_cols, protected_col, label_col, seed=seed)
        train_df, val_df = train_test_split(corrupted_df, test_size=0.15, random_state=seed)
        
        train_df.to_csv(train_dir / f"train_alpha_{alpha}.csv", index=False)
        val_df.to_csv(val_dir / f"val_alpha_{alpha}.csv", index=False)