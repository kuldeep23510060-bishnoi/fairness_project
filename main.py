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
    summary_df = run_experiments(name, df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols, alphas=ALPHAS, seeds=SEEDS_10, outroot=OUTPUT_DIR, logger=logger)
    all_summaries[name] = summary_df

if len(all_summaries) > 0:
    combined = pd.concat({k: v.set_index("alpha") for k, v in all_summaries.items()}, axis=0)
    combined.to_csv(Path(OUTPUT_DIR) / "combined_all_datasets_summary.csv")

print("Finished in %.1f seconds. Check outputs/<> directories for CSVs & plots." % (time.time() - t0_all))