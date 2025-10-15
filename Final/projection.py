import torch

def project_to_simplex_and_l1_around(p: torch.Tensor, p_hat: torch.Tensor, radius: float, iters: int = 8,eps: float = 1e-12

                                    ) -> torch.Tensor:
    

    original_shape = p.shape

    p_flat = p.view(-1)
    p_hat_flat = p_hat.view(-1)
    
    assert p_flat.shape == p_hat_flat.shape, \
        f"Shape mismatch: p {p_flat.shape} vs p_hat {p_hat_flat.shape}"
    
    assert radius >= 0, f"Radius must be non-negative, got {radius}"

    assert iters > 0, f"Iterations must be positive, got {iters}"

    x = p_flat.clone()
    q1 = torch.zeros_like(x)
    q2 = torch.zeros_like(x)
    
    for _ in range(iters):
        y = _project_onto_simplex(x + q1, eps)
        q1 = x + q1 - y
        
        x = _project_onto_l1_ball_around_center(y + q2, p_hat_flat, radius, eps)
        q2 = y + q2 - x

    return x.view(original_shape)


def _project_onto_simplex(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    n = p.numel()
    if n == 0:
        return p
    
    sorted_p, _ = torch.sort(p, descending=True)
    cumsum_p = torch.cumsum(sorted_p, dim=0)
    indices = torch.arange(1, n + 1, device=p.device, dtype=sorted_p.dtype)
    
    condition = sorted_p * indices > (cumsum_p - 1.0)
    rho_indices = torch.nonzero(condition, as_tuple=False)
    
    if len(rho_indices) > 0:
        rho = rho_indices[-1].item()
        theta = (cumsum_p[rho] - 1.0) / (rho + 1)
    else:
        theta = (p.sum() - 1.0) / n
    
    result = torch.clamp(p - theta, min=0.0)
    result_sum = result.sum()
    
    if result_sum > eps:
        result = result / result_sum
    else:
        result = torch.ones_like(result) / n
    
    return result


def _project_onto_l1_ball(v: torch.Tensor, z: float, eps: float = 1e-12) -> torch.Tensor:

    if z <= 0:
        return torch.zeros_like(v)
    
    l1_norm = torch.abs(v).sum()
    if l1_norm <= z + eps:
        return v.clone()
    
    abs_v = torch.abs(v)
    sorted_abs, _ = torch.sort(abs_v, descending=True)
    cumsum_abs = torch.cumsum(sorted_abs, dim=0)
    
    n = v.numel()
    indices = torch.arange(1, n + 1, device=v.device, dtype=sorted_abs.dtype)
    
    condition = sorted_abs * indices > (cumsum_abs - z)
    rho_indices = torch.nonzero(condition, as_tuple=False)
    
    if len(rho_indices) > 0:
        rho = rho_indices[-1].item()
        lambda_threshold = (cumsum_abs[rho] - z) / (rho + 1)
    else:
        lambda_threshold = (cumsum_abs[-1] - z) / n
    
    result = torch.sign(v) * torch.clamp(abs_v - lambda_threshold, min=0.0)
    return result


def _project_onto_l1_ball_around_center(x: torch.Tensor, center: torch.Tensor, radius: float, eps: float = 1e-12) -> torch.Tensor:
    
    delta = x - center
    delta_proj = _project_onto_l1_ball(delta, radius, eps)
    return center + delta_proj
