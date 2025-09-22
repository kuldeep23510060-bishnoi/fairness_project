import torch
import torch.nn as nn
import torch.nn.functional as F
from projection import project_to_simplex_and_l1_around

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class DROJointModel(nn.Module):
    
    def __init__(self, in_dim, n_samples, hidden, init_lambda):

        super().__init__()

        self.net = SimpleMLP(in_dim, hidden)

        self.lambdas = nn.Parameter(torch.tensor([init_lambda, init_lambda, init_lambda], dtype=torch.float32, device=DEVICE))

        p0 = torch.ones(n_samples, device=DEVICE, dtype=torch.float32) / float(n_samples)
        self.p_tilde_dp_0 = nn.Parameter(p0.clone())
        self.p_tilde_dp_1 = nn.Parameter(p0.clone())
        self.p_tilde_if = nn.Parameter(p0.clone())

        self.to(DEVICE)

    def forward(self, x):
        return self.net(x)

    def clamp_lambdas(self):

        with torch.no_grad():
            self.lambdas.clamp_(min=0.0)

    def project_p_tilde_dp(self, p_hat_0, p_hat_1, radius_0, radius_1):
        with torch.no_grad():
            newp = project_to_simplex_and_l1_around(self.p_tilde_dp_0, p_hat_0, radius_0, iters=8)
            self.p_tilde_dp_0.copy_(newp)

            newp = project_to_simplex_and_l1_around(self.p_tilde_dp_1, p_hat_1, radius_1, iters=8)
            self.p_tilde_dp_1.copy_(newp)


    def project_p_tilde_if(self, p_hat_if, radius):

        with torch.no_grad():
            newp = project_to_simplex_and_l1_around(self.p_tilde_if, p_hat_if, radius, iters=8)
            self.p_tilde_if.copy_(newp)