import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # First linear layer expands the dimension (e.g., 512 -> 2048)
        self.w_1 = nn.Linear(d_model, d_ff)
        # Second linear layer shrinks it back (e.g., 2048 -> 512)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(self.activation(self.w_1(x))))