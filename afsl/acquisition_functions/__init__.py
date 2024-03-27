"""
`afsl` supports a wide range of acquisition functions which are summarized here.
The default implementation uses [ITL](acquisition_functions/itl).
You can use a custom acquisition function as follows:

```python
from afsl.acquisition_functions.undirected_itl import UndirectedITL

acquisition_function = UndirectedITL()
data_loader = afsl.ActiveDataLoader(data, batch_size=64, acquisition_function=acquisition_function)
```

## Overview of Acquisition Functions

The following table provides an overview of the acquisition functions and their properties:

|                                                                    | Relevance? | Informativeness? | Diversity? | Model Requirement   |
|--------------------------------------------------------------------|------------|------------------|------------|---------------------|
| [ITL](acquisition_functions/itl)                                   | ✅          | ✅                | ✅          | embedding / kernel  |
| [VTL](acquisition_functions/vtl)                                   | ✅          | ✅                | ✅          | embedding / kernel  |
| [CTL](acquisition_functions/ctl)                                   | ✅          | (✅)                | ✅          | embedding / kernel  |
| [Cosine Similarity](acquisition_functions/cosine_similarity)       | ✅          | ❌                | ❌          | embedding           |
| [Undirected ITL](acquisition_functions/undirected_itl)             | ❌          | ✅                | ✅          | embedding / kernel  |
| [Undirected VTL](acquisition_functions/undirected_vtl)             | ❌          | ✅                | ✅          | embedding / kernel  |
| [MaxDist](acquisition_functions/max_dist)                          | ❌          | (✅)                | ✅          | embedding / kernel  |
| [k-means++](acquisition_functions/kmeans_pp)                       | ❌          | (✅)                | ✅          | embedding / kernel  |
| [Uncertainty Sampling](acquisition_functions/uncertainty_sampling) | ❌          | ✅                | ❌          | embedding / kernel  |
| [MinMargin](acquisition_functions/min_margin)                      | ❌          | (✅)              | ❌          | softmax             |
| [MaxEntropy](acquisition_functions/max_entropy)                    | ❌          | (✅)              | ❌          | softmax             |
| [LeastConfidence](acquisition_functions/least_confidence)                    | ❌          | (✅)              | ❌          | softmax             |
| [Information Density](acquisition_functions/information_density)   | (✅)        | (✅)              | ❌          | embedding & softmax |
| [Random](acquisition_functions/random)                             | ❌          | ❌                | (✅)        | -                   |


- **Relevance** and **Informativeness** capture whether obtained data is "useful" as outlined [here](/afsl/docs/afsl#why-active-data-selection).
- **Diversity** captures whether the selected batches are diverse, i.e., whether they cover different "useful" parts of the data space. In a non-diverse batch, most data is not useful conditional on the rest of the batch, meaning that most of the batch is "wasted".
- **Model Requirement** describes the type of model required for the acquisition function. For example, some acquisition functions require an *embedding* or a *kernel* (see afsl.model), while others require the model to output a *softmax* distribution (typically in a classification context).

---
"""

from abc import ABC, abstractmethod
import math
from typing import Callable, Generic, Tuple, TypeVar
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
from afsl.data import Dataset
from afsl.model import Model, ModelWithEmbedding
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
    get_device,
    mini_batch_wrapper,
)
import warnings

M = TypeVar("M", bound=Model | None)


class _IndexedDataset(TorchDataset[Tuple[torch.Tensor, int]]):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.dataset[idx]
        return data, idx


class AcquisitionFunction(ABC, Generic[M]):
    """Abstract base class for acquisition functions."""

    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE
    """Size of mini batches used for computing the acquisition function."""

    num_workers: int = DEFAULT_NUM_WORKERS
    """Number of workers used for data loading."""

    subsample: bool = DEFAULT_SUBSAMPLE
    """Whether to (uniformly) subsample the data to a single mini batch for faster computation."""

    def __init__(
        self,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
    ):
        self.mini_batch_size = mini_batch_size
        self.num_workers = num_workers
        self.subsample = subsample

    @abstractmethod
    def select(
        self,
        batch_size: int,
        model: M,
        dataset: Dataset,
        selected_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Selects the next batch.

        :param batch_size: Size of the batch to be selected. Needs to be smaller than `mini_batch_size`.
        :param model: Model used for data selection.
        :param dataset: Inputs (shape $n \times d$) to be selected from.
        :param selected_indices: Indices of previously selected data points. Default is `None`.
        :return: Indices of the newly selected batch.
        """
        pass


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

    def select(
        self,
        batch_size: int,
        model: M,
        dataset: Dataset,
        selected_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if selected_indices is not None:
            print("Passing selected indices is not supported.")
        return BatchAcquisitionFunction._select(
            compute_fn=self.compute,
            batch_size=batch_size,
            model=model,
            dataset=dataset,
            mini_batch_size=self.mini_batch_size,
            num_workers=self.num_workers,
            subsample=self.subsample,
        )

    @staticmethod
    def _select(
        compute_fn: Callable[[M, torch.Tensor], torch.Tensor],
        batch_size: int,
        model: M,
        dataset: Dataset,
        mini_batch_size: int,
        num_workers: int,
        subsample: bool,
    ) -> torch.Tensor:
        indexed_dataset = _IndexedDataset(dataset)
        data_loader = DataLoader(
            indexed_dataset,
            batch_size=mini_batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        _values = []
        _original_indices = []
        for data, idx in data_loader:
            _indiv_values = compute_fn(model, data)
            assert _indiv_values.size(0) == data.size(0)
            _values.append(_indiv_values)
            _original_indices.append(idx)
            if subsample:
                break
        values = torch.cat(_values)
        original_indices = torch.cat(_original_indices)

        _, indices = torch.topk(values, batch_size)
        return original_indices[indices.cpu()]


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
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        super().__init__(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
        )
        self.force_nonsequential = force_nonsequential

    @abstractmethod
    def initialize(
        self,
        model: M,
        data: torch.Tensor,
        selected_data: torch.Tensor | None,
        batch_size: int,
    ) -> State:
        r"""
        Initializes the state for batch selection.

        :param model: Model used for data selection.
        :param data: Tensor of inputs (shape $n \times d$) to be selected from.
        :param selected_data: Data points that have already been selected.
        :param batch_size: Size of the batch to be selected.
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
        r"""
        Updates the state after adding a data point to the batch.

        :param state: State of batch selection.
        :param i: Index of selected data point.
        :return: Updated state of batch selection.
        """
        pass

    @staticmethod
    def selector(values: torch.Tensor) -> int:
        """
        Given acquisition function values, selects the next data point to be added to the batch.

        :param values: Acquisition function values.
        :return: Index of the selected data point.
        """
        return int(torch.argmax(values).item())

    def select_from_minibatch(
        self,
        batch_size: int,
        model: M,
        data: torch.Tensor,
        selected_data: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""
        Selects the next batch from the given mini batch `data`.

        :param batch_size: Size of the batch to be selected. Needs to be smaller than `mini_batch_size`.
        :param model: Model used for data selection.
        :param data: Mini batch of inputs (shape $n \times d$) to be selected from.
        :param selected_data: Data points that have already been selected.
        :return: Indices of the newly selected batch (with respect to mini batch).
        """
        state = self.initialize(model, data, selected_data, batch_size)

        indices = []
        for _ in range(batch_size):
            values = self.compute(state)
            assert values.size(0) == data.size(0)
            i = self.selector(values)
            indices.append(i)
            state = self.step(state, i)
        return torch.tensor(indices)

    def select(
        self,
        batch_size: int,
        model: M,
        dataset: Dataset,
        selected_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Selects the next batch. If `force_nonsequential` is `True`, the data is selected analogously to `BatchAcquisitionFunction.select`.
        Otherwise, the data is selected by hierarchical composition of data selected from mini batches.

        :param batch_size: Size of the batch to be selected. Needs to be smaller than `mini_batch_size`.
        :param model: Model used for data selection.
        :param dataset: Inputs (shape $n \times d$) to be selected from.
        :param selected_indices: Indices of previously selected data points. Default is `None`.
        :return: Indices of the newly selected batch.
        """
        selected_data = (
            torch.stack([dataset[i] for i in selected_indices], dim=0)
            if selected_indices is not None and selected_indices.size(0) > 0
            else None
        )

        if self.force_nonsequential:

            def compute_fn(model: M, data: torch.Tensor) -> torch.Tensor:
                return self.compute(
                    self.initialize(model, data, selected_data, batch_size)
                )

            return BatchAcquisitionFunction._select(
                compute_fn=compute_fn,
                batch_size=batch_size,
                model=model,
                dataset=dataset,
                mini_batch_size=self.mini_batch_size,
                num_workers=self.num_workers,
                subsample=self.subsample,
            )

        assert (
            batch_size < self.mini_batch_size
        ), "Batch size must be smaller than `mini_batch_size`."
        if batch_size > self.mini_batch_size / 2:
            warnings.warn(
                "The evaluation of the acquisition function may be slow since `batch_size` is large relative to `mini_batch_size`."
            )

        indexed_dataset = _IndexedDataset(dataset)
        new_selected_indices = range(len(dataset))
        while len(new_selected_indices) > batch_size:
            data_loader = DataLoader(
                indexed_dataset,
                batch_size=self.mini_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )

            new_selected_indices = []
            for data, idx in data_loader:
                new_selected_indices.extend(
                    idx[
                        self.select_from_minibatch(
                            batch_size, model, data, selected_data
                        )
                    ]
                    .cpu()
                    .tolist()
                )
                if self.subsample:
                    break
            indexed_dataset = Subset(indexed_dataset, new_selected_indices)
        return torch.tensor(new_selected_indices)


class EmbeddingBased(ABC):
    r"""
    Abstract base class for acquisition functions that require an embedding of the data.
    """

    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    """Batch size used for computing the embeddings."""

    def __init__(
        self,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
    ):
        """
        :param embedding_batch_size: Batch size used for computing the embeddings.
        """
        self.embedding_batch_size = embedding_batch_size

    @staticmethod
    def compute_embedding(
        model: ModelWithEmbedding | None,
        data: torch.Tensor,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> torch.Tensor:
        r"""
        Returns the embedding of the given data. If `model` is `None`, the data is returned as is (i.e., the data is assumed to be already embedded).

        :param model: Model used for computing the embedding.
        :param data: Tensor of inputs (shape $n \times d$) to be embedded.
        :param batch_size: Batch size used for computing the embeddings.
        :return: Embedding of the given data.
        """
        if model is None:
            return data

        device = get_device(model)
        model.eval()
        with torch.no_grad():
            embeddings = mini_batch_wrapper(
                fn=lambda batch: model.embed(
                    batch.to(device, non_blocking=True)
                ),  # TODO: handle device internally
                data=data,
                batch_size=batch_size,
            )
            return embeddings


class Targeted(ABC):
    r"""
    Abstract base class for acquisition functions that take into account the relevance of data with respect to a specified target (denoted $\spA$).
    """

    max_target_size: int | None
    r"""Maximum size of the target to be subsampled in each iteration."""

    subsampled_target_frac: float
    r"""Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$."""

    def __init__(
        self,
        target: torch.Tensor,
        subsampled_target_frac: float = 1,
        max_target_size: int | None = None,
    ):
        r"""
        :param target: Tensor of prediction targets (shape $m \times d$).
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $1$.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large.
        """

        # assert target.size(0) > 0, "Target must be non-empty"
        assert (
            subsampled_target_frac > 0 and subsampled_target_frac <= 1
        ), "Fraction of target must be in (0, 1]"
        assert (
            max_target_size is None or max_target_size > 0
        ), "Max target size must be positive"

        self._target = target
        self.max_target_size = max_target_size
        self.subsampled_target_frac = subsampled_target_frac

    def add_to_target(self, new_target: torch.Tensor):
        r"""
        Appends new target data to the target.

        :param new_target: Tensor of new prediction targets (shape $m \times d$).
        """
        self._target = torch.cat([self._target, new_target])

    def set_target(self, new_target: torch.Tensor):
        r"""
        Updates the target.

        :param new_target: Tensor of new prediction targets (shape $m \times d$).
        """
        self._target = new_target

    def get_target(self) -> torch.Tensor:
        r"""
        Returns the tensor of (subsampled) prediction target (shape $m \times d$).
        """
        m = self._target.size(0)
        max_target_size = (
            self.max_target_size if self.max_target_size is not None else m
        )
        return self._target[
            torch.randperm(m)[
                : min(math.ceil(self.subsampled_target_frac * m), max_target_size)
            ]
        ]
