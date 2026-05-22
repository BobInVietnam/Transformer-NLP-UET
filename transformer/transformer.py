import torch
import torch.nn as nn

# Import the existing modular components you created
from embedding import InputEmbedding
from positional_encoding import PositionalEncoding
from multihead_attention import MultiHeadAttention
from addnorm import LayerNorm, AddAndNorm
from feedforward import PositionWiseFeedForward
from encoder import TransformerEncoder
from decoder import TransformerDecoder

class Transformer(nn.Module):
    """
    A complete Encoder-Decoder Transformer architecture.
    Sewing together the standalone encoder and decoder pipelines.
    """
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_max_len: int = 5000,
        tgt_max_len: int = 5000,
        d_model: int = 512, 
        num_layers: int = 6, 
        num_heads: int = 8, 
        d_ff: int = 2048, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Instantiate the comprehensive Encoder Stack
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=src_max_len,
            dropout=dropout
        )
        
        # 2. Instantiate the comprehensive Decoder Stack
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=tgt_max_len,
            dropout=dropout
        )

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: torch.Tensor = None, 
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Execution Pipeline:
        Args:
            src: Source language token sequences (batch_size, src_seq_len)
            tgt: Target language token sequences (batch_size, tgt_seq_len)
            src_mask: Padding mask for the source tokens (batch_size, 1, 1, src_seq_len)
            tgt_mask: Optional extra padding mask or combined causal mask for target
            
        Returns:
            Logits mapped across target vocabulary dimensions (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Step 1: Process the source sentence through the Encoder to harvest memory context
        # Output shape: (batch_size, src_seq_len, d_model)
        encoder_output = self.encoder(x=src, mask=src_mask)
        
        # Step 2: Feed target tokens and encoder context representations into the Decoder
        # Output shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        decoder_logits = self.decoder(x=tgt, encoder_output=encoder_output, src_mask=src_mask)
        
        return decoder_logits

# =====================================================================
# SYSTEM VERIFICATION AND PIPELINE SANITY CHECK
# =====================================================================
if __name__ == "__main__":
    # Define hyperparameter spaces matching the "Attention Is All You Need" base model
    SRC_VOCAB_SIZE = 10000  # Size of source language vocabulary (e.g., English)
    TGT_VOCAB_SIZE = 8000   # Size of target language vocabulary (e.g., Vietnamese)
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    
    # Initialize the top-level unified Transformer
    transformer_model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )
    
    print("--- Model Structural Layout ---")
    print(transformer_model)
    
    # Simulate a batched forward-pass execution
    BATCH_SIZE = 2
    SRC_SEQ_LEN = 12
    TGT_SEQ_LEN = 9
    
    # Mock source tokens (e.g., input sentences)
    mock_src_batch = torch.randint(low=0, high=SRC_VOCAB_SIZE, size=(BATCH_SIZE, SRC_SEQ_LEN))
    # Mock target tokens (e.g., right-shifted target tracking sentences)
    mock_tgt_batch = torch.randint(low=0, high=TGT_VOCAB_SIZE, size=(BATCH_SIZE, TGT_SEQ_LEN))
    
    print("\n--- Processing Tensors ---")
    print(f"Source Batch Input Geometry: {mock_src_batch.shape}")
    print(f"Target Batch Input Geometry: {mock_tgt_batch.shape}")
    
    # Execute full pipeline forward pass under evaluation mode (deactivates dropout)
    transformer_model.eval()
    with torch.no_grad():
        output_logits = transformer_model(src=mock_src_batch, tgt=mock_tgt_batch)
        
    print(f"Resulting Vocabulary Logits Geometry: {output_logits.shape}")
    
    # Verification Assertions
    assert output_logits.shape == (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE), "Output dimensionality mismatch!"
    print("\n[Status] Verification Complete. Complete sequence mapping is structurally operational!")