import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

class MetricsAccumulator:
    
    __slots__ = ['losses', 'cons_dp', 'cons_if', 'num_both', 'grad_norms', 'lambda_dp_vals', 'lambda_if_vals']
    
    def __init__(self):
        self.losses: List[float] = []
        self.cons_dp: List[float] = []
        self.cons_if: List[float] = []
        self.num_both: List[int] = []
        self.grad_norms: List[float] = []
        self.lambda_dp_vals: List[float] = []
        self.lambda_if_vals: List[float] = []
    
    def add(self, loss, cons_dp, cons_if,  num_both, grad_norm, lambda_dp, lambda_if):
        self.losses.append(loss)
        self.cons_dp.append(cons_dp)
        self.cons_if.append(cons_if)
        self.num_both.append(num_both)
        self.grad_norms.append(grad_norm)
        self.lambda_dp_vals.append(lambda_dp)
        self.lambda_if_vals.append(lambda_if)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:

        if not self.losses:
            return {}
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
            }
        
        return {
            'loss': compute_stats(self.losses),
            'cons_dp': compute_stats(self.cons_dp),
            'cons_if': compute_stats(self.cons_if),
            'grad_norm': compute_stats(self.grad_norms),
        }
    
    def get_averages(self) -> Dict[str, float]:

        if not self.losses:
            return {
                'avg_loss': 0.0, 'avg_dp': 0.0, 
                'avg_if': 0.0, 'avg_num_both': 0.0,
                'avg_grad_norm': 0.0
            }
        
        return {
            'avg_loss': float(np.mean(self.losses)),
            'avg_dp': float(np.mean(self.cons_dp)),
            'avg_if': float(np.mean(self.cons_if)),
            'avg_num_both': float(np.mean(self.num_both)),
            'avg_grad_norm': float(np.mean(self.grad_norms))
        }
    
    def reset(self):
        for attr in self.__slots__:
            getattr(self, attr).clear()