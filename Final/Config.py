import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DROConfig:

    mb: int
    epochs: int
    max_l1_radius_dp: float
    max_l1_radius_if: float
    lr_theta: float
    lr_lambda: float
    lr_p: float
    inner_p_steps: int
    beta: float
    gamma: float
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-5
    dropout: float = 0.01
    temperature: float = 100.0
    eps: float = 1e-12
    lambda_clip_max: float = 10.0
    log_interval: int = 10
    checkpoint_interval: int = 50
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    use_amp: bool = False  
    gradient_accumulation_steps: int = 1
    
    def __post_init__(self):

        validations = [
            (self.mb > 0, "Batch size must be positive"),
            (self.epochs > 0, "Epochs must be positive"),
            (self.max_l1_radius_dp >= 0, "DP radius must be non-negative"),
            (self.max_l1_radius_if >= 0, "IF radius must be non-negative"),
            (self.lr_theta > 0, "lr_theta must be positive"),
            (self.lr_lambda > 0, "lr_lambda must be positive"),
            (self.lr_p > 0, "lr_p must be positive"),
            (self.inner_p_steps > 0, "inner_p_steps must be positive"),
            (self.beta >= 0, "beta must be non-negative"),
            (self.gamma >= 0, "gamma must be non-negative"),
            (self.max_grad_norm > 0, "max_grad_norm must be positive"),
            (self.weight_decay >= 0, "weight_decay must be non-negative"),
            (0 <= self.dropout < 1, "dropout must be in [0, 1)"),
            (self.temperature > 0, "temperature must be positive"),
            (self.eps > 0, "eps must be positive"),
            (self.lambda_clip_max > 0, "lambda_clip_max must be positive"),
            (self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(cls, path: Path) -> 'DROConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)