import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # RoPE operates on pairs of dimensions, so dim must be even (usually equal to head_dim)
        self.dim = dim
        
        # 1. Precompute the inverse frequencies: theta^(-2i/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 2. Generate the sequence position indices: [0, 1, 2, ..., max_seq_len-1]
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # 3. Outer product to calculate angles matrix: shape (max_seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        
        # 4. Duplicate each frequency to match the full head dimension shape (max_seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache the cosine and sine transformations as buffers
        self.register_buffer("cos_cached", emb.cos(), persistent=False) # Shape: (max_seq_len, dim)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the final dimension in half and rotates the coordinates."""
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract the exact cosine and sine slices required for the current batch length
        # Returns shapes: (1, 1, seq_len, dim) to allow broadcasting across heads
        return (
            self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1),
            self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        )