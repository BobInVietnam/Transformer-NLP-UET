import torch
import torch.nn as nn
from transformer.rope import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads # e.g., 512 // 8 = 64
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Instantiate RoPE explicitly bound to the head dimension size
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)

    def _apply_rope(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # t shape: (batch_size, num_heads, seq_len, head_dim)
        # Rotary formula: R(t) = t * cos(positions) + rotate_half(t) * sin(positions)
        return (t * cos) + (self.rope._rotate_half(t) * sin)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        k_seq_len = k.size(1)
        
        # 1. Project and reshape into standard 4D attention tensors
        Q = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Fetch the precomputed cosine and sine rotation tensors
        cos_q, sin_q = self.rope(Q, seq_len)
        cos_k, sin_k = self.rope(K, k_seq_len)
        
        # 3. Physically rotate Queries and Keys (Values are left untouched!)
        Q = self._apply_rope(Q, cos_q, sin_q)
        K = self._apply_rope(K, cos_k, sin_k)
        
        # 4. Run standard scaled dot-product attention as normal
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads back together and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(context)