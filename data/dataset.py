import torch

from torch.utils.data import Dataset

from data.tokenizer import tokenize

from data.preprocessing import preprocess_text


class SummaryDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocab,
        max_len=128
    ):

        self.dataframe = dataframe

        self.vocab = vocab

        self.max_len = max_len

    def pad(self, ids):

        ids = ids[:self.max_len]

        while len(ids) < self.max_len:

            ids.append(0)

        return ids

    def create_mask(self, ids):

        return [

            0 if token == 0
            else 1

            for token in ids
        ]

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        article = row["article"]

        summary = row["summary"]

        # preprocess
        article = preprocess_text(article)

        summary = preprocess_text(summary)

        # tokenize
        article_tokens = tokenize(article)

        summary_tokens = tokenize(summary)

        # word -> index
        article_ids = self.vocab.numericalize(
            article_tokens
        )

        summary_ids = self.vocab.numericalize(
            summary_tokens
        )

        # padding
        article_ids = self.pad(article_ids)

        summary_ids = self.pad(summary_ids)

        # attention mask
        article_mask = self.create_mask(article_ids)

        summary_mask = self.create_mask(summary_ids)

        return {

            "input_ids": torch.tensor(
                article_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                article_mask,
                dtype=torch.long
            ),

            "decoder_input_ids": torch.tensor(
                summary_ids,
                dtype=torch.long
            ),

            "decoder_attention_mask": torch.tensor(
                summary_mask,
                dtype=torch.long
            )
        }