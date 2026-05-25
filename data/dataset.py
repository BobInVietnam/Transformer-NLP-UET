from torch.utils.data import Dataset

from data.preprocessing import text_pipeline


class SummaryDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocab
    ):

        self.article_sequences = [

            text_pipeline(
                text=t,
                vocab=vocab
            )

            for t in dataframe["article"]
        ]

        self.summary_sequences = [

            text_pipeline(
                text=t,
                vocab=vocab
            )

            for t in dataframe["summary"]
        ]

    def __len__(self):

        return len(self.article_sequences)

    def __getitem__(self, idx):

        return (

            self.article_sequences[idx],

            self.summary_sequences[idx]
        )
