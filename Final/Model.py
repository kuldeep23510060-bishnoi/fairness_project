import torch
import torch.nn as nn
import torch.nn.functional as F

from projection import project_to_simplex_and_l1_around

class SimpleMLP(nn.Module):
    
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):

        super().__init__()
            
        self.hidden1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dropout1 = nn.Dropout(dropout)
        
        self.hidden2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout2 = nn.Dropout(dropout)
        
        self.output = nn.Linear(hidden, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        return self.output(x).squeeze(-1)


class DROModel(nn.Module):
    
    def __init__(self, in_dim: int,  n_samples: int,  hidden: int,  init_lambda: float, device: torch.device, dropout: float = 0.1 ):
        super().__init__()
            
        self.n_samples = n_samples

        self.net = SimpleMLP(in_dim, hidden, dropout)

        self.lambdas = nn.Parameter(torch.full((2,), init_lambda, dtype=torch.float32, device=device))
        
        p0 = torch.full((n_samples,), 1.0 / n_samples, device=device, dtype=torch.float32)
        
        self.p_tilde_dp = nn.Parameter(p0.clone())
        self.p_tilde_if = nn.Parameter(p0.clone())
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def clamp_lambdas(self, max_val: float = 10.0):
        with torch.no_grad():
            self.lambdas.clamp_(min=0.0, max=max_val)
    
    def project_p_tilde_dp(self, p_hat: torch.Tensor, radius: float):
        with torch.no_grad():
            newp = project_to_simplex_and_l1_around(
                self.p_tilde_dp, p_hat, radius, iters=8
            )
            self.p_tilde_dp.copy_(newp)
    
    def project_p_tilde_if(self, p_hat: torch.Tensor, radius: float):
        with torch.no_grad():
            newp = project_to_simplex_and_l1_around(
                self.p_tilde_if, p_hat, radius, iters=8
            )
            self.p_tilde_if.copy_(newp)