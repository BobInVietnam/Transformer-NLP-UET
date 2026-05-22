import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
        # Gamma (gain) and Beta (bias) are learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        
        # 1. Compute mean across the last dimension (d_model)
        # keepdim=True preserves the dimension so we can easily subtract it from x
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. Compute variance across the last dimension
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # 3. Normalize the tensor values
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. Scale and shift using our learnable parameters
        return self.gamma * x_norm + self.beta


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        # Dropout is added to regularize and prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # Input 'x' is the raw input to the sublayer (the residual connection)
        # 'sublayer_output' is the tensor produced by Attention or FFN
        
        # 1. Apply dropout to the sublayer's output
        processed_output = self.dropout(sublayer_output)
        
        # 2. Add step: Combine the original input with the processed output
        residual_sum = x + processed_output
        
        # 3. Norm step: Run the sum through Layer Normalization
        return self.layer_norm(residual_sum)