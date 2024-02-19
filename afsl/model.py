from typing import Iterator, Protocol
import torch


class Model(Protocol):
    """"""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        ...

    def eval(self) -> None:
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...


class LatentModel(Model):
    """"""

    def latent(self, data: torch.Tensor) -> torch.Tensor:
        r"""Returns the latent representation (a tensor with shape $n \times k$) of the input data (of shape $n \times d$)."""
        ...


class ClassificationModel(LatentModel):
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def final_layer(self) -> torch.nn.Linear:
        """Returns the final linear layer of the model. Assumes that this layer does not include an additive bias."""
        ...
