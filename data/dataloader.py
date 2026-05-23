from torch.utils.data import DataLoader

from data.dataset import SummaryDataset


def get_dataloader(
    dataframe,
    vocab,
    batch_size=32,
    shuffle=True,
    max_len=128
):

    dataset = SummaryDataset(
        dataframe=dataframe,
        vocab=vocab,
        max_len=max_len
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader