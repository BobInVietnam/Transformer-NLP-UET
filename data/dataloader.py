import pandas as pd

from torch.utils.data import DataLoader
from dataset import SummaryDataset

# Load parquet
train_df = pd.read_parquet(
    "dataset/train-00000-of-00001.parquet"
)

valid_df = pd.read_parquet(
    "dataset/valid-00000-of-00001.parquet"
)

# Create dataset
train_dataset = SummaryDataset(train_df)
valid_dataset = SummaryDataset(valid_df)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False
)

# Test
for batch in train_loader:

    print("=" * 50)

    print("ARTICLE:")
    print(batch["article"][0])

    print("\nSUMMARY:")
    print(batch["summary"][0])

    print("\nBATCH SIZE:")
    print(len(batch["article"]))

    print("=" * 50)

    break