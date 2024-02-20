from typing import Generic
import torch
from afsl.acquisition_functions import M, AcquisitionFunction
from afsl.acquisition_functions.itl import ITL


class ActiveDataLoader(Generic[M]):
    r"""
    `ActiveDataLoader` can be used as a drop-in replacement for random data selection:

    ```python
    data_loader = ActiveDataLoader.initialize(data, target, batch_size=64)
    batch = data[data_loader.next(model)]
    ```

    where
    - `model` is a PyTorch `nn.Module`,
    - `data` is a tensor of inputs (shape $n \times d$), and
    - `target` is a tensor of prediction targets (shape $m \times d$) or `None`.
    """

    data: torch.Tensor
    r"""Tensor of inputs (shape $n \times d$) to be selected from."""

    batch_size: int
    r"""Size of the batch to be selected."""

    acquisition_function: AcquisitionFunction[M]
    r"""Acquisition function to be used for data selection."""

    subsampled_target_frac: float
    max_target_size: int | None

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        acquisition_function: AcquisitionFunction[M],
    ):
        assert data.size(0) > 0, "Data must be non-empty"
        assert batch_size > 0, "Batch size must be positive"

        self.data = data
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function

    @classmethod
    def initialize(
        cls,
        data: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
        Sigma: torch.Tensor | None = None,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
    ):
        acquisition_function = ITL(
            target=target,
            Sigma=Sigma,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
        )
        return cls(
            data=data,
            batch_size=batch_size,
            acquisition_function=acquisition_function,
        )

    def next(self, model: M) -> torch.Tensor:
        return self.acquisition_function.select(
            batch_size=self.batch_size,
            model=model,
            data=self.data,
        )
