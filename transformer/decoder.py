import torch
import torch.nn as nn
from transformer.addnorm import LayerNorm, AddAndNorm
from transformer.embedding import InputEmbedding
from transformer.feedforward import PositionWiseFeedForward
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import TransformerEncoder

class TransformerDecoderLayer(nn.Module):
    """A single modular layer containing Attention, FFN, and Add & Norm wrappers."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.cross_self_attention = MultiHeadAttention(d_model, num_heads)
        
        self.add_norm_1 = AddAndNorm(d_model, dropout)
        self.add_norm_2 = AddAndNorm(d_model, dropout)
        self.add_norm_3 = AddAndNorm(d_model, dropout)
        
    def forward(self, encoder_output: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        masked_attn_out = self.masked_self_attention(q=x, k=x, v=x, mask=mask)
        x = self.add_norm_1(x, masked_attn_out)
        
        cross_attn_out = self.cross_self_attention(q=x, k=encoder_output, v=encoder_output, mask=src_mask)
        x = self.add_norm_2(x, cross_attn_out)
        
        # Sub-layer 3: Position-wise Feed Forward
        ffn_out = self.feed_forward(x)
        x = self.add_norm_3(x, ffn_out)
        return x
        
class TransformerDecoder(nn.Module):
    """The complete Decoder stack, ending with a Linear mapping to vocabulary logits."""
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Stack N identical decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = LayerNorm(d_model)
        # Final Linear layer projects d_model back to the Vocabulary Size
        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates a lower-triangular causal mask matrix."""
        # Creates a matrix with 1s on and below the diagonal, 0s above it
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        # Shape: (1, 1, seq_len, seq_len) to match MultiHeadAttention expected dimensions
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, tgt_seq_len) -> absolute target token IDs
        batch_size, tgt_seq_len = x.size()
        
        # Step 1: Prep Causal Mask for the sequence length
        mask = self.generate_causal_mask(tgt_seq_len, x.device)
        
        # Step 2: Input Embedding + Positional Encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Step 3: Pass through the stacked Decoder Layers
        for layer in self.layers:
            x = layer(x=x, encoder_output=encoder_output, mask=mask, src_mask=src_mask)
            
        x = self.final_layer_norm(x)
        
        # Step 4: Map back to target vocabulary logits
        return self.fc_out(x) # Output shape: (batch_size, tgt_seq_len, vocab_size)
    
if __name__ == "__main__":
    # Hyperparameters
    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 8000
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    
    # Initialize stacks
    encoder = TransformerEncoder(SRC_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF)
    decoder = TransformerDecoder(TGT_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF)
    
    # 1. Simulating English Source Input: Batch of 2, 12 tokens long
    src_input = torch.randint(0, SRC_VOCAB_SIZE, (2, 12))
    src_mask = (src_input != 0).unsqueeze(1).unsqueeze(2)
    # 2. Simulating Vietnamese Target Input: Batch of 2, 9 tokens long (shifted right)
    tgt_input = torch.randint(0, TGT_VOCAB_SIZE, (2, 9))
    
    # Execution
    with torch.no_grad():
        # Pass through encoder to get semantic context vectors
        enc_context = encoder(x=src_input) # Shape: (2, 12, 512)
        print(enc_context.shape)
        # Pass target tokens + encoder context through decoder
        predictions = decoder(x=tgt_input, encoder_output=enc_context, src_mask=src_mask) # Shape: (2, 9, 8000)
        
    print(f"Final Output Logits Shape: {predictions.shape}")
    assert predictions.shape == (2, 9, TGT_VOCAB_SIZE)
    print("Success! Complete Encoder-Decoder data loop executed perfectly.")