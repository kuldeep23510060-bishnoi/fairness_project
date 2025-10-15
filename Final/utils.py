import torch
import random
import numpy as np


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_data_tensors(df, features, label_col, prot_col, device):
    
    X = torch.tensor(df[features].values, device=device, dtype=torch.float32)
    y = torch.tensor(df[label_col].values, device=device, dtype=torch.float32).view(-1)
    
    female = torch.tensor((1 - df[prot_col].values), device=device, dtype=torch.float32).view(-1)
    male = torch.tensor(df[prot_col].values, device=device, dtype=torch.float32).view(-1)
    
    return X, y, female, male





