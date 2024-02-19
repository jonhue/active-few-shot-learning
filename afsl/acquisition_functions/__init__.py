from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch
from afsl.embeddings import M, Embedding
from afsl.types import Target
from afsl.utils import mini_batch_wrapper, mini_batch_wrapper_non_cat


class AcquisitionFunction(ABC):
    mini_batch_size: int

    def __init__(self, mini_batch_size: int = 100):
        self.mini_batch_size = mini_batch_size

    @abstractmethod
    def select(
        self,
        batch_size: int,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
        force_nonsequential=False,
    ) -> torch.Tensor:
        pass


class BatchAcquisitionFunction(AcquisitionFunction):
    @abstractmethod
    def compute(
        self,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    def select(
        self,
        batch_size: int,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
        force_nonsequential=False,
    ) -> torch.Tensor:
        values = mini_batch_wrapper(
            fn=lambda batch: self.compute(
                embedding=embedding,
                model=model,
                data=batch,
                target=target,
                Sigma=Sigma,
            ),
            data=data,
            batch_size=self.mini_batch_size,
        )
        _, indices = torch.topk(values, batch_size)
        return indices


State = TypeVar("State")


class SequentialAcquisitionFunction(AcquisitionFunction, Generic[State]):
    @abstractmethod
    def initialize(
        self,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> State:
        pass

    @abstractmethod
    def compute(self, state: State) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, state: State, i: int) -> State:
        pass

    def select(
        self,
        batch_size: int,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
        force_nonsequential=False,
    ) -> torch.Tensor:
        states = mini_batch_wrapper_non_cat(
            fn=lambda batch: self.initialize(
                embedding=embedding,
                model=model,
                data=batch,
                target=target,
                Sigma=Sigma,
            ),
            data=data,
            batch_size=self.mini_batch_size,
        )

        if force_nonsequential:
            values = torch.cat([self.compute(state) for state in states], dim=0)
            _, indices = torch.topk(values, batch_size)
            return indices
        else:
            indices = []
            for _ in range(batch_size):
                values = torch.cat([self.compute(state) for state in states], dim=0)
                i = int(torch.argmax(values).item())
                indices.append(i)
                states = [self.step(state, i) for state in states]
            return torch.tensor(indices)
