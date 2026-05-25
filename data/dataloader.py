import torch

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

from data.dataset import SummaryDataset


def collate_fn(batch):
    articles = []
    decoder_inputs = []
    target_labels = []
    
    # Explicit IDs matching your Vocabulary class structure
    sos_id = 2
    eos_id = 3
    
    for article_ids, summary_ids in batch:
        # 1. Process Articles (Encoder side remains unchanged)
        articles.append(torch.tensor(article_ids, dtype=torch.long))
        
        # 2. Construct Decoder Input: Append <SOS> at the front + True summary IDs
        # e.g., [2, word1_id, word2_id, ...]
        dec_in_seq = [sos_id] + summary_ids
        decoder_inputs.append(torch.tensor(dec_in_seq, dtype=torch.long))
        
        # 3. Construct Target Labels: True summary IDs + Append <EOS> at the back
        # e.g., [word1_id, word2_id, ..., 3]
        label_seq = summary_ids + [eos_id]
        target_labels.append(torch.tensor(label_seq, dtype=torch.long))

    # Dynamic padding across the distinct sequences
    articles_padded = pad_sequence(
        articles,
        batch_first=True,
        padding_value=0
    )

    decoder_inputs_padded = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=0
    )
    
    target_labels_padded = pad_sequence(
        target_labels,
        batch_first=True,
        padding_value=0
    )
    
    # Generate Encoder padding mask
    article_mask = (articles_padded != 0).long()

    return {
        "input_ids": articles_padded,
        "attention_mask": article_mask,
        "decoder_input_ids": decoder_inputs_padded,
        "labels": target_labels_padded  # Hand this directly to your cross-entropy loss function!
    }

def get_dataloader(
    dataframe,
    src_vocab,
    tgt_vocab,
    batch_size=32,
    shuffle=True
):
    dataset = SummaryDataset(
        dataframe=dataframe,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return loader

