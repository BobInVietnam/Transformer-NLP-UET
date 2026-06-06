import torch
from torch.utils.data import Dataset

class SummaryDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.df = df
        self.src_vocab = src_vocab  # Instances of SpacyVocabulary
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Tokenize using spaCy's localized Vietnamese engine
        src_tokens = self.src_vocab.tokenize_text(str(row["article"]))
        tgt_tokens = self.tgt_vocab.tokenize_text(str(row["summary"]))
        
        # 2. Numericalize strings into vocabulary index integers
        src_ids = self.src_vocab.numericalize(src_tokens)
        tgt_ids = self.tgt_vocab.numericalize(tgt_tokens)
        
        # 3. Apply sequence truncation & append SOS/EOS boundaries
        sos_id = self.src_vocab.stoi["<SOS>"]
        eos_id = self.src_vocab.stoi["<EOS>"]
        
        tgt_ids = [sos_id] + tgt_ids + [eos_id]
        
        return src_ids, tgt_ids