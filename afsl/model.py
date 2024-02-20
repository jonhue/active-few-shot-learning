from __future__ import annotations
from typing import Iterator, Protocol
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


class ModelWithEmbedding(Model, Protocol):
    """"""

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the latent representation (a tensor with shape $n \times k$) of the input data `x` (of shape $n \times d$)."""
        ...


class ClassificationModel(Model, Protocol):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        r"""Returns the logits (a tensor with shape $n \times k$) of the input data `x` (of shape $n \times d$)."""
        ...

    @property
    def final_layer(self) -> torch.nn.Linear:
        """Returns the final linear layer of the model. Assumes that this layer does not include an additive bias (TODO: drop assumption)."""
        ...


class Classifier(ClassificationModel):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self(x)
        _, predicted = torch.max(outputs.data, dim=1)
        return predicted

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)
