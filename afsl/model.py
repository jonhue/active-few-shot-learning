"""
Selection of protocols for PyTorch models.
"""

from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable
import torch


class Model(Protocol):
    """Protocol for PyTorch `nn.Module` instances."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def eval(self) -> Model:
        ...

    def zero_grad(self) -> None:
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...


@runtime_checkable
class ModelWithEmbedding(Model, Protocol):
    """Protocol for PyTorch models with associated embeddings."""

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: Tensor of inputs with shape $n \times d$.
        :return: Tensor of associated latent representation with shape $n \times k$.
        """
        ...


@runtime_checkable
class ModelWithKernel(Model, Protocol):
    """Protocol for PyTorch models with associated kernel."""

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        r"""
        :param x1: Tensor of inputs with shape $n \times d$.
        :param x2: Tensor of inputs with shape $m \times d$.
        :return: Tensor of associated (symmetric and positive semi-definite) kernel matrix with shape $n \times m$.
        """
        ...


ModelWithEmbeddingOrKernel = ModelWithEmbedding | ModelWithKernel
"""Protocol for PyTorch models with associated kernel or associated embeddings."""


class ClassificationModel(Model, Protocol):
    """Protocol for PyTorch classification models."""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: Tensor of inputs with shape $n \times d$.
        :return: Tensor of predicted class labels with shape $n$.
        """
        ...

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: Tensor of inputs with shape $n \times d$.
        :return: Tensor of associated logits with shape $n \times k$.
        """
        ...

    @property
    def final_layer(self) -> torch.nn.Linear:
        """Returns the final linear layer of the model."""
        ...
