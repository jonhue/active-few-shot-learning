"""
Selection of protocols for PyTorch models.
"""

from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable
import torch


class Model(Protocol):
    """"""

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
    """"""

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the latent representation (a tensor with shape $n \times k$) of the input data `x` (of shape $n \times d$)."""
        ...


@runtime_checkable
class ModelWithKernel(Model, Protocol):
    """"""

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor | None) -> torch.Tensor:
        r"""Given inputs `x1` (of shape $n \times d$) and `x2` (of shape $m \times d$), returns their covariance matrix (a tensor with shape $n \times m$). If `x2` is `None`, returns the covariance matrix of `x1` with itself."""
        ...


ModelWithEmbeddingOrKernel = ModelWithEmbedding | ModelWithKernel


class ClassificationModel(Model, Protocol):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the predicted class labels (shape $n$) of the input data `x` (of shape $n \times d$)."""
        ...

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the logits (a tensor with shape $n \times k$) of the input data `x` (of shape $n \times d$)."""
        ...

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    @property
    def final_layer(self) -> torch.nn.Linear:
<<<<<<< HEAD
        """Returns the final linear layer of the model."""
=======
        """Returns the final linear layer of the model. Assumes that this layer does not include an additive bias (TODO: drop assumption)."""
>>>>>>> c134e94 (gradient embeddings)
        ...
