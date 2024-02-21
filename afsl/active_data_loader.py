from typing import Generic
import torch
from afsl.acquisition_functions import M, AcquisitionFunction
from afsl.acquisition_functions.undirected_itl import UndirectedITL
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
    r"""Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $0.5$."""

    max_target_size: int | None
    r"""
    Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large.

    .. warning::

        The computational complexity of `next` scales cubically with the size of the target. If the target is large, it is recommended to set `max_target_size` to value other than `None`.
    """

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        acquisition_function: AcquisitionFunction[M],
    ):
        """
        Explicitly constructs an active data loader with a custom acquisition function.
        `afsl` supports a wide range of acquisition functions which are summarized in `afsl.acquisition_functions`.
        """

        assert data.size(0) > 0, "Data must be non-empty"
        assert batch_size > 0, "Batch size must be positive"

        self.data = data
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function

    @classmethod
    def initialize(
        cls,
        data: torch.Tensor,
        target: torch.Tensor | None,
        batch_size: int,
        Sigma: torch.Tensor | None = None,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
    ):
        r"""
        Initializes an active data loader.

        :param data: Tensor of inputs (shape $n \times d$) to be selected from.
        :param target: Tensor of prediction targets (shape $m \times d$) or `None`.
        :param batch_size: Size of the batch to be selected.
        :param Sigma: Optionally pass a covariance matrix of model parameters. See `afsl.model.ModelWithEmbedding` for more details.
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $0.5$. Ignored if `target` is `None`.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large. Ignored if `target` is `None`.
        """

        if target is not None:
            acquisition_function = ITL(
                target=target,
                Sigma=Sigma,
                subsampled_target_frac=subsampled_target_frac,
                max_target_size=max_target_size,
            )
        else:
            acquisition_function = UndirectedITL(Sigma=Sigma)
        return cls(
            data=data,
            batch_size=batch_size,
            acquisition_function=acquisition_function,
        )

    def next(self, model: M) -> torch.Tensor:
        r"""
        Selects the next batch of data provided a `model` which is a PyTorch `nn.Module`.
        """

        return self.acquisition_function.select(
            batch_size=self.batch_size,
            model=model,
            data=self.data,
        )
