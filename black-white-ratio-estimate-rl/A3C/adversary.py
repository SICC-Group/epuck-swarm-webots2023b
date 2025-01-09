import torch
import numpy as np


class Adversary:
    def __init__(self, method) -> None:
        """method: list of str"""
        self.method = method[0]
        if self.method == "ipm":
            self.scale = float(method[1])
        if self.method == "noise":
            self.mu = 2.0
            self.sigma = 1.0
    
    def __call__(self, benign_gradients, *args, **kwargs) -> np.ndarray:
        """benign gradients: shape of (num_benign_agents, length of gradient)
        return an adversary gradient in shape of (1, length of gradient)"""
        method_name = f"_{self.method}"
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Method {self.method} is not defined")
        res: torch.Tensor = method(benign_gradients, *args, **kwargs).cpu()
        
        grad_len = len(benign_gradients[0])
        if res.ndimension() == 2:
            if res.size(0) != 1 or res.size(1) != grad_len:
                raise ValueError(f"excepted the shape of (1, {grad_len}), but got the shape of ({res.size(0)}, {res.size(1)})")
        elif res.ndimension() == 1:
            if res.size(0) != grad_len:
                raise ValueError(f"excepted the shape of ({grad_len},), but got the shape of ({res.size(0)},)")
            res = res.unsqueeze(0)

        return res.numpy()
    

    def _noise(self, benign_gradients, *args, **kwargs):
        return torch.normal(
            self.mu, self.sigma, size=torch.Size([1, len(benign_gradients[0])])
        )

    def _signflip(self, *args, **kwargs):
        pass
    
    def _alie(self, benign_gradients: torch.Tensor, *args, **kwargs):
        num_byzantine = kwargs.get("num_byz")
        num_workers = len(benign_gradients)

        s = torch.floor_divide(num_workers, 2) + 1 - num_byzantine
        cdf_value = (num_workers - num_byzantine - s) / (
            num_workers - num_byzantine
        )
        dist = torch.distributions.normal.Normal(
            torch.tensor(0.0), torch.tensor(1.0)
        )
        z_max = dist.icdf(cdf_value)

        mean = benign_gradients.mean(dim=0)
        std = benign_gradients.std(dim=0)
        
        res = mean + std * z_max
        
        return res
    
    def _ipm(self, benign_gradients: torch.Tensor, *args, **kwargs):
        mean = benign_gradients.mean(dim=0)
        res = -self.scale * mean
        return res