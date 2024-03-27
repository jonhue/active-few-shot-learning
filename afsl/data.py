"""
Selection of classes for datasets and data loaders.
"""

from typing import Sized, Tuple
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset


class Dataset(TorchDataset[torch.Tensor], Sized):
    """Dataset over data in "input" space."""


DataLoader = TorchDataLoader[torch.Tensor]
"""Data loader over data in "input" space."""


class InputDataset(Dataset):
    """Given a dataset for supervised training (comprising inputs and labels), constructs a dataset over data in "input" space."""

    pre_data = torch.tensor([])

    def __init__(self, dataset: TorchDataset[Tuple[torch.Tensor, torch.Tensor]]):
        self.dataset = dataset

    def __len__(self):
        return self.k + len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        assert idx >= 0
        if idx < self.k:
            x = self.pre_data[idx]
        else:
            x, _ = self.dataset[idx - self.k]
        return x

    def prepend(self, data: torch.Tensor):
        self.pre_data = data

    @property
    def k(self):
        return self.pre_data.size(0)
