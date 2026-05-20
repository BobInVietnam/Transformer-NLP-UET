from torch.utils.data import Dataset

class SummaryDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        article = self.dataframe.iloc[idx]["article"]
        summary = self.dataframe.iloc[idx]["summary"]

        return {
            "article": article,
            "summary": summary
        }