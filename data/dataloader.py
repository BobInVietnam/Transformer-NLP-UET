import torch

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

from data.spacydataset import SummaryDataset


def collate_fn(batch):
    articles = []
    decoder_inputs = []
    target_labels = []
    
    for src_seq, tgt_seq in batch:
            articles.append(torch.tensor(src_seq))
            
            # Teacher Forcing Shifting:
            # Input drops the final <EOS> token -> Starts with <SOS>
            decoder_inputs.append(torch.tensor(tgt_seq[:-1]))
            # Label drops the starting <SOS> token -> Ends with <EOS>
            target_labels.append(torch.tensor(tgt_seq[1:]))

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
    article_mask = (articles_padded != 0).long().unsqueeze(1).unsqueeze(2)

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
        df=dataframe,
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

