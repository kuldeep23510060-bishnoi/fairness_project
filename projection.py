import torch


def project_onto_l1_ball(v: torch.Tensor, z: float) -> torch.Tensor:

    v = v.view(-1)

    if z <= 0:
        return torch.zeros_like(v)
    
    if torch.norm(v, p=1) <= z + 1e-12:
        return v.clone()
    
    abs_v = v.abs()

    sorted_v, _ = torch.sort(abs_v, descending=True)

    cssv = torch.cumsum(sorted_v, dim=0)

    idxs = torch.arange(1, len(sorted_v) + 1, device=v.device, dtype=sorted_v.dtype)

    cond = sorted_v * idxs > (cssv - z)

    if not cond.any():

        theta = (cssv[-1] - z) / len(v)

        return torch.sign(v) * torch.clamp(abs_v - theta, min=0.0)
    
    rho = torch.nonzero(cond, as_tuple=False)[-1].item()

    theta = (cssv[rho] - z) / (rho + 1)

    return torch.sign(v) * torch.clamp(abs_v - theta, min=0.0)



def project_onto_simplex(p: torch.Tensor) -> torch.Tensor:

    if p.numel() == 0:
        return p
    
    p = p.view(-1)

    u, _ = torch.sort(p, descending=True)

    cssv = torch.cumsum(u, dim=0)

    idxs = torch.arange(1, len(u) + 1, device=p.device, dtype=u.dtype)

    cond = u * idxs > (cssv - 1.0)

    if not cond.any():
        return torch.full_like(p, 1.0 / p.numel())
    
    rho = torch.nonzero(cond, as_tuple=False)[-1].item()
    
    theta = (cssv[rho] - 1.0) / (rho + 1)

    return torch.clamp(p - theta, min=0.0)


def project_to_simplex_and_l1_around(p_vec: torch.Tensor, p_hat: torch.Tensor, l1_radius: float, iters: int = 10) -> torch.Tensor:

    x = p_vec.clone().detach()

    for _ in range(iters):

        x = project_onto_simplex(x)

        delta = x - p_hat

        delta = project_onto_l1_ball(delta, float(l1_radius))

        x = p_hat + delta

    x = project_onto_simplex(x)
    
    return x