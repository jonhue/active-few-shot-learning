"""
`afsl` supports a wide range of acquisition functions which are summarized here.
The default implementation uses [ITL](acquisition_functions/itl).
You can use a custom acquisition function as follows:

```python
from afsl.acquisition_functions.greedy_max_det import GreedyMaxDet

acquisition_function = GreedyMaxDet()
data_loader = afsl.ActiveDataLoader(data, batch_size=64, acquisition_function=acquisition_function)
```

## Overview of Acquisition Functions

The following table provides an overview of the acquisition functions and their properties:

|                                                                    | Relevance? | Informativeness? | Diversity? | Model Requirement   |
|--------------------------------------------------------------------|------------|------------------|------------|---------------------|
| [ITL](acquisition_functions/itl)                                   | ✅          | ✅                | ✅          | embedding / kernel  |
| [VTL](acquisition_functions/vtl)                                   | ✅          | ✅                | ✅          | embedding / kernel  |
| [CTL](acquisition_functions/ctl)                                   | ✅          | ❌                | ✅          | embedding / kernel  |
| [Cosine Similarity](acquisition_functions/cosine_similarity)       | ✅          | ❌                | ❌          | embedding           |
| [GreedyMaxDet](acquisition_functions/greedy_max_det)               | ❌          | ✅                | ✅          | embedding / kernel  |
| [GreedyMaxDist](acquisition_functions/greedy_max_dist)             | ❌          | (✅)                | ✅          | embedding / kernel  |
| [k-means++](acquisition_functions/kmeans_pp)                       | ❌          | (✅)                | ✅          | embedding / kernel  |
| [Uncertainty Sampling](acquisition_functions/uncertainty_sampling) | ❌          | ✅                | ❌          | embedding / kernel  |
| [MaxMargin](acquisition_functions/max_margin)                      | ❌          | (✅)              | ❌          | softmax             |
| [MaxEntropy](acquisition_functions/max_entropy)                    | ❌          | (✅)              | ❌          | softmax             |
| [Information Density](acquisition_functions/information_density)   | (✅)        | (✅)              | ❌          | embedding & softmax |
| [Random](acquisition_functions/random)                             | ❌          | ❌                | (✅)        | -                   |


- **Relevance** and **Informativeness** capture whether obtained data is "useful" as outlined [here](/afsl#why-active-data-selection).
- **Diversity** captures whether the selected batches are diverse, i.e., whether they cover different "useful" parts of the data space. In a non-diverse batch, most data is not useful conditional on the rest of the batch, meaning that most of the batch is "wasted".
- **Model Requirement** describes the type of model required for the acquisition function. For example, some acquisition functions require an *embedding* or a *kernel* (see afsl.model), while others require the model to output a *softmax* distribution (typically in a classification context).

---
"""

from abc import ABC, abstractmethod
import math
from typing import Generic, TypeVar
import torch
from afsl.model import Model
from afsl.utils import (
    DEFAULT_MINI_BATCH_SIZE,
    mini_batch_wrapper,
    mini_batch_wrapper_non_cat,
)

M = TypeVar("M", bound=Model)


class AcquisitionFunction(ABC, Generic[M]):
    """Abstract base class for acquisition functions."""

    mini_batch_size: int
    """Size of mini-batch used for computing the acquisition function."""

    selected: torch.Tensor
    """Indices of the selected data points."""

    def __init__(self, mini_batch_size=DEFAULT_MINI_BATCH_SIZE):
        self.mini_batch_size = mini_batch_size
        self.selected = torch.tensor([])

    @abstractmethod
    def _select(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def select(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Selects the next batch.

        :param batch_size: Size of the batch to be selected.
        :param model: Model used for data selection.
        :param data: Tensor of inputs (shape $n \times d$) to be selected from.
        :return: Indices of the newly selected batch.
        """
        selected = self._select(batch_size, model, data)
        self.selected = torch.cat([self.selected, selected])
        return selected


class BatchAcquisitionFunction(AcquisitionFunction[M]):
    """
    Abstract base class for acquisition functions that select entire batches with a single computation of the acquisition function.
    """

    @abstractmethod
    def compute(
        self,
        model: M,
        data: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes the acquisition function for the given data.

        :param model: Model used for data selection.
        :param data: Tensor of inputs (shape $n \times d$) to be selected from.
        :return: Acquisition function values for the given data.
        """
        pass

    def _select(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
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
    """
    Abstract base class for acquisition functions that select a batch by sequentially adding points.
    """

    force_nonsequential: bool = False
    """Whether to force non-sequential data selection."""

    def __init__(
        self,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        force_nonsequential=False,
    ):
        super().__init__(mini_batch_size=mini_batch_size)
        self.force_nonsequential = force_nonsequential

    @abstractmethod
    def initialize(
        self,
        model: M,
        data: torch.Tensor,
    ) -> State:
        r"""
        Initializes the state for batch selection.

        :param model: Model used for data selection.
        :param data: Tensor of inputs (shape $n \times d$) to be selected from.
        :return: Initial state of batch selection.
        """
        pass

    @abstractmethod
    def compute(self, state: State) -> torch.Tensor:
        """
        Computes the acquisition function for the given state.

        :param state: State of batch selection.
        :return: Acquisition function values for the given state.
        """
        pass

    @abstractmethod
    def step(self, state: State, i: int) -> State:
        """
        Updates the state after adding a data point to the batch.

        :param state: State of batch selection.
        :param i: Index of the data point added to the batch.
        :return: Updated state of batch selection.
        """
        pass

    def _select(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
    ) -> torch.Tensor:
        states = mini_batch_wrapper_non_cat(
            fn=lambda batch: self.initialize(
                model=model,
                data=batch,
            ),
            data=data,
            batch_size=self.mini_batch_size,
        )

        if self.force_nonsequential:
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
        """
        Given acquisition function values, selects the next data point to be added to the batch.

        :param values: Acquisition function values.
        :return: Index of the selected data point.
        """
        return int(torch.argmax(values).item())


class Targeted(ABC):
    r"""
    Abstract base class for acquisition functions that take into account the relevance of data with respect to a specified target (denoted $\spA$).
    """

    target: torch.Tensor
    r"""Tensor of prediction targets (shape $m \times d$)."""

    def __init__(
        self,
        target: torch.Tensor,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
    ):
        r"""
        :param target: Tensor of prediction targets (shape $m \times d$).
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $0.5$. Ignored if `target` is `None`.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large. Ignored if `target` is `None`.
        """

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
