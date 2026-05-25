import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 1. Create a matrix of shape (max_len, d_model) full of zeros
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create a column vector of positions: [[0], [1], [2], ... [max_len - 1]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Compute the division term for the sine and cosine functions in log space
        # Mathematically: exp(2i * (-ln(10000) / d_model)) == 1 / (10000^(2i / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. Apply sine to even indices (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 5. Apply cosine to odd indices (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension -> shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 6. register_buffer saves `pe` to the model's state_dict and moves it to the 
        # correct device (CPU/GPU), but tells PyTorch NOT to update it during backpropagation.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # Slice the pe matrix to match the current sequence length and add it
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x