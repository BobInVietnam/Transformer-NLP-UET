import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Define the weight matrices as Linear layers (without bias, per original paper)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Final output projection matrix
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # Input shape: (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        
        # 1. Linear projections to get Q, K, V matrices
        Q = self.w_q(q)  # (batch_size, seq_len, d_model)
        K = self.w_k(k)  # (batch_size, seq_len, d_model)
        V = self.w_v(v)  # (batch_size, seq_len, d_model)
        
        # 2. Split into multiple heads and transpose to shape: (batch_size, num_heads, seq_len, d_k)
        # We view the d_model dimension as (num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Calculate Scaled Dot-Product Attention
        # Transpose K to make it (batch_size, num_heads, d_k, seq_len) for matrix multiplication
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (used in Decoder or padding)
        if mask is not None:
            # Replaces masked out tokens with a massive negative value, so softmax zeroes them out
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Softmax to get probability distribution weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 5. Multiply weights by Values to get context representation
        # context shape: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, V)
        
        # 6. Concatenate heads back together
        # Transpose back to (batch_size, seq_len, num_heads, d_k)
        # contiguous() ensures memory layout is unbroken before calling view()
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)
        
        # 7. Apply the final linear projection layer
        return self.w_o(context)