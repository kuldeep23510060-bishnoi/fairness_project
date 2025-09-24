import os
import time
import logging
import pandas as pd
from pathlib import Path
from data.bank import load_ucirepo_350
from data.adult import load_adult
from run_experiments import run_experiments

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("dro")

SEEDS_10 = [123]
ALPHAS = [0.2]
HIDDEN = 128
MB_SIZE = 1024
NUM_EPOCHS = 20
LR_THETA = 1e-3
LR_LAMBDA = 5e-4
LR_P = 1e-3
INNER_P_STEPS = 3
GAMMA = 0.02
BETA_DRO = 0.1

print("Starting multi-dataset DRO-only experiments. Outputs will be saved to:", OUTPUT_DIR)

t0_all = time.time()
datasets = []

# 1) ucirepo id=350
df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols = load_ucirepo_350()
datasets.append(("ucirepo_350", df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols))

# 2) Adult
df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols = load_adult()
datasets.append(("adult", df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols))

all_summaries = {}
for name, df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols in datasets:
    logger.info("=== Running dataset: %s ===", name)
    summary_df = run_experiments(name, df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols, ALPHAS, SEEDS_10, OUTPUT_DIR, logger,
                                 HIDDEN, MB_SIZE, NUM_EPOCHS, LR_THETA,LR_LAMBDA, LR_P, INNER_P_STEPS, GAMMA, BETA_DRO)
    all_summaries[name] = summary_df

if len(all_summaries) > 0:
    combined = pd.concat({k: v.set_index("alpha") for k, v in all_summaries.items()}, axis=0)
    combined.to_csv(Path(OUTPUT_DIR) / "combined_all_datasets_summary.csv")

print("Finished in %.1f seconds. Check outputs/<> directories for CSVs & plots." % (time.time() - t0_all))