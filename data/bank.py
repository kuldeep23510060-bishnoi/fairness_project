import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_ucirepo_350():

    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    X["X2"] = X["X2"] - 1
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    feature_cols = ['X1', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20' ,'X21', 'X22', 'X23']
    numeric_cols = ['X1', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20' ,'X21', 'X22', 'X23']
    categorical_cols = ['X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

    protected_col = "X2"
    label_col = "Y"
    
    return df, protected_col, label_col,  feature_cols, numeric_cols, categorical_cols

