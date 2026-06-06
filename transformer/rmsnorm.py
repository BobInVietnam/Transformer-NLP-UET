import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The only learnable parameter (gain). Initialized to ones.
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute the mean of the squared inputs along the last dimension (d_model)
        # x.pow(2).mean(-1, keepdim=True) calculates: (1/n) * sum(x^2)
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # 2. Divide x by the square root of the variance (RMS value) + eps
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # 3. Apply the learnable element-wise scaling factor
        return self.gamma * x_normed