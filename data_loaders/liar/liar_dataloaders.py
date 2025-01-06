from torch.utils.data import DataLoader
from constants.liar_constants import PAD_TOKEN
from data_loaders.liar.padding_collate import MyCollate
from dataset_classes.liar.liar_dataset import (
    train_dataset,
    validation_dataset,
    test_dataset,
)
import os

BATCH_SIZE = int(os.getenv("BATCH_SIZE", default=32))
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=MyCollate(pad_idx=train_dataset.statement_vocab[PAD_TOKEN]),
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=MyCollate(pad_idx=train_dataset.statement_vocab[PAD_TOKEN]),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=MyCollate(pad_idx=train_dataset.statement_vocab[PAD_TOKEN]),
)
