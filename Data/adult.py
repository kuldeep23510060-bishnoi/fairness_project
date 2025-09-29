import pandas as pd
import json

def load_adult():

    TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    COLS = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    train = pd.read_csv(TRAIN_URL, names=COLS, sep=",", skipinitialspace=True, na_values="?")
    test = pd.read_csv(TEST_URL, names=COLS, sep=",", skipinitialspace=True, skiprows=1, na_values="?")

    test["income"] = test["income"].str.rstrip(".").str.strip()

    df = pd.concat([train, test], ignore_index=True)
    df = df.dropna().reset_index(drop=True)

    df["Y"] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["income"])

    df = df.drop(columns=['education'])

    df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype(int)

    df['workclass'] = df['workclass'].astype('category').cat.codes
    df['marital-status'] = df['marital-status'].astype('category').cat.codes
    df['occupation'] = df['occupation'].astype('category').cat.codes
    df['relationship'] = df['relationship'].astype('category').cat.codes
    df['race'] = df['race'].astype('category').cat.codes
    df['native-country'] = df['native-country'].astype('category').cat.codes

    numeric_cols = ["age", "fnlwgt",  "capital-gain", "capital-loss", "hours-per-week"]
    categorical_cols = ['workclass',  "education-num", 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

    feature_cols = numeric_cols + categorical_cols 
    protected_col = "sex"
    label_col = "Y"

    return df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols



df, protected_col, label_col, feature_cols, numeric_cols, categorical_cols = load_adult()


df.to_csv("data/adult_processed.csv", index=False)

meta = {
    "protected_col": protected_col,
    "label_col": label_col,
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols
}

with open("data/adult_meta.json", "w") as f:
    json.dump(meta, f, indent=2)



