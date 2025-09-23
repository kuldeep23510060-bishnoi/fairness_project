import torch
import torch.nn as nn
import torch.nn.functional as F
from projection import project_to_simplex_and_l1_around

class SimpleMLP(nn.Module):

    def __init__(self, in_dim, hidden):
        super().__init__()

        self.hidden1 = nn.Linear(in_dim, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        
        return self.output(x).squeeze(-1)

class DROModel(nn.Module):
    def __init__(self, in_dim, n_samples, hidden, init_lambda, DEVICE):
        super().__init__()
        self.net = SimpleMLP(in_dim, hidden)
        self.lambdas = nn.Parameter(torch.tensor([init_lambda, init_lambda], dtype=torch.float32, device=DEVICE))
        p0 = torch.ones(n_samples, device=DEVICE, dtype=torch.float32) / float(n_samples)
        self.p_tilde_dp = nn.Parameter(p0.clone())
        self.p_tilde_if = nn.Parameter(p0.clone())
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def clamp_lambdas(self):
        with torch.no_grad():
            self.lambdas.clamp_(min=0.0)

    def project_p_tilde_dp(self, p_hat: torch.Tensor, radius: float):
        with torch.no_grad():
            newp = project_to_simplex_and_l1_around(self.p_tilde_dp, p_hat, radius, iters=8)
            self.p_tilde_dp.copy_(newp)

    def project_p_tilde_if(self, p_hat: torch.Tensor, radius: float):
        with torch.no_grad():
            newp = project_to_simplex_and_l1_around(self.p_tilde_if, p_hat, radius, iters=8)
            self.p_tilde_if.copy_(newp)