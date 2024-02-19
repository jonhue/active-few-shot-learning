from abc import ABC, abstractmethod
import math
from typing import Generic, TypeVar
import torch
from afsl.embeddings import M
from afsl.utils import (
    DEFAULT_MINI_BATCH_SIZE,
    mini_batch_wrapper,
    mini_batch_wrapper_non_cat,
)


class AcquisitionFunction(ABC, Generic[M]):
    mini_batch_size: int

    def __init__(self, mini_batch_size=DEFAULT_MINI_BATCH_SIZE):
        self.mini_batch_size = mini_batch_size

    @abstractmethod
    def select(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
        force_nonsequential=False,
    ) -> torch.Tensor:
        pass


class BatchAcquisitionFunction(AcquisitionFunction[M]):
    @abstractmethod
    def compute(
        self,
        model: M,
        data: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def select(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
        force_nonsequential=False,
    ) -> torch.Tensor:
        values = mini_batch_wrapper(
            fn=lambda batch: self.compute(
                model=model,
                data=batch,
            ),
            data=data,
            batch_size=self.mini_batch_size,
        )
        _, indices = torch.topk(values, batch_size)
        return indices


State = TypeVar("State")


class SequentialAcquisitionFunction(AcquisitionFunction[M], Generic[M, State]):
    @abstractmethod
    def initialize(
        self,
        model: M,
        data: torch.Tensor,
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
        model: M,
        data: torch.Tensor,
        force_nonsequential=False,
    ) -> torch.Tensor:
        states = mini_batch_wrapper_non_cat(
            fn=lambda batch: self.initialize(
                model=model,
                data=batch,
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
                i = self.selector(values)
                indices.append(i)
                states = [self.step(state, i) for state in states]
            return torch.tensor(indices)

    @staticmethod
    def selector(values: torch.Tensor) -> int:
        return int(torch.argmax(values).item())


class TargetedAcquisitionFunction(ABC):
    target: torch.Tensor
    r"""Tensor of prediction targets (shape $m \times d$) or `None` if data selection should be "undirected"."""

    def __init__(
        self,
        target: torch.Tensor,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
    ):
        assert target.size(0) > 0, "Target must be non-empty"
        assert (
            subsampled_target_frac > 0 and subsampled_target_frac <= 1
        ), "Fraction of target must be in (0, 1]"
        assert (
            max_target_size is None or max_target_size > 0
        ), "Max target size must be positive"

        m = self.target.size(0)
        max_target_size = max_target_size if max_target_size is not None else m
        self.target = target[
            torch.randperm(m)[
                : min(math.ceil(subsampled_target_frac * m), max_target_size)
            ]
        ]
