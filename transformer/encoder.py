import torch
import torch.nn as nn
from addnorm import LayerNorm, AddAndNorm
from embedding import InputEmbedding
from feedforward import PositionWiseFeedForward
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding

class TransformerEncoderLayer(nn.Module):
    """A single modular layer containing Attention, FFN, and Add & Norm wrappers."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # We need two separate Add & Norm structures since there are two sub-layers
        self.add_norm_1 = AddAndNorm(d_model, dropout)
        self.add_norm_2 = AddAndNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Sub-layer 1: Self-Attention. 
        # In an Encoder, Queries, Keys, and Values all stem from the same input tensor 'x'
        attn_out = self.self_attention(q=x, k=x, v=x, mask=mask)
        x = self.add_norm_1(x, attn_out)
        
        # Sub-layer 2: Position-wise Feed-Forward
        ffn_out = self.feed_forward(x)
        x = self.add_norm_2(x, ffn_out)
        
        return x

class TransformerEncoder(nn.Module):
    """The complete wrapper that manages Embeddings, Position, and stacked Layers."""
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Stack N identical encoder layers sequentially using nn.ModuleList
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final normalization block applied to the output of the full stack
        self.final_layer_norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len) -> absolute integer token IDs
        
        # Step 1 & 2: Process Input through Embedding and Positional Encodings
        x = self.embedding(x)                   # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)         # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        # Step 3: Loop sequentially through the stacked Encoder Layers
        for layer in self.layers:
            x = layer(x, mask)                  # (batch_size, seq_len, d_model)
            
        # Step 4: Final structural cleanup normalization
        return self.final_layer_norm(x)

# =====================================================================
# 4. VERIFICATION PIPELINE (Sanity Check Execution)
# =====================================================================
if __name__ == "__main__":
    # Define standard hyperparameters matching the original small-scale model
    VOCAB_SIZE = 10000
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    
    # Initialize our complete Encoder stack
    encoder = TransformerEncoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, 
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, d_ff=D_FF
    )
    
    # Simulate a batch of token inputs: 2 examples, each 10 tokens long
    example_input = torch.randint(low=0, high=VOCAB_SIZE, size=(2, 10))
    print(f"Input Shape (Batch, SeqLen): {example_input.shape}")
    
    # Pass tensor through the model
    with torch.no_grad():
        output_context = encoder(example_input)
        
    print(f"Output Shape (Batch, SeqLen, D_Model): {output_context.shape}")
    assert output_context.shape == (2, 10, D_MODEL), "Shape mismatch error!"
    print("Success! The Encoder infrastructure is executing flawlessly.")