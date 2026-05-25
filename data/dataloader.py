import torch

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

from data.dataset import SummaryDataset


def collate_fn(batch):

    articles = []

    summaries = []

    for article_ids, summary_ids in batch:

        article_tensor = torch.tensor(
            article_ids,
            dtype=torch.long
        )

        summary_tensor = torch.tensor(
            summary_ids,
            dtype=torch.long
        )

        articles.append(article_tensor)

        summaries.append(summary_tensor)

    # dynamic padding
    articles_padded = pad_sequence(

        articles,

        batch_first=True,

        padding_value=0
    )

    summaries_padded = pad_sequence(

        summaries,

        batch_first=True,

        padding_value=0
    )

    # attention mask
    article_mask = (
        articles_padded != 0
    ).long()

    summary_mask = (
        summaries_padded != 0
    ).long()

    return {

        "input_ids":
            articles_padded,

        "attention_mask":
            article_mask,

        "decoder_input_ids":
            summaries_padded,

        "decoder_attention_mask":
            summary_mask
    }


def get_dataloader(

    dataframe,

    vocab,

    batch_size=32,

    shuffle=True
):

    dataset = SummaryDataset(

        dataframe=dataframe,

        vocab=vocab
    )

    loader = DataLoader(

        dataset=dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        collate_fn=collate_fn
    )

    return loader

