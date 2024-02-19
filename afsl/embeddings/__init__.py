from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch
from afsl.model import Model

M = TypeVar("M", bound=Model)


class Embedding(ABC, Generic[M]):
    mini_batch_size: int

    def __init__(self, mini_batch_size: int = 100):
        self.mini_batch_size = mini_batch_size

    @abstractmethod
    def embed(self, model: M, data: torch.Tensor) -> torch.Tensor:
        pass
