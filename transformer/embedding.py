import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # nn.Embedding is a simple lookup table
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The paper specifies multiplying embeddings by sqrt(d_model)
        # This prevents the embeddings from becoming vanishingly small 
        # compared to the positional encodings we add next.
        return self.embedding(x) * math.sqrt(self.d_model)