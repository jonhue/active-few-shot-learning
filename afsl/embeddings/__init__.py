from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch
from afsl.model import Model
from afsl.utils import DEFAULT_MINI_BATCH_SIZE

M = TypeVar("M", bound=Model)


class Embedding(ABC, Generic[M]):
    mini_batch_size: int

    def __init__(self, mini_batch_size=DEFAULT_MINI_BATCH_SIZE):
        self.mini_batch_size = mini_batch_size

    @abstractmethod
    def embed(self, model: M, data: torch.Tensor) -> torch.Tensor:
        pass
