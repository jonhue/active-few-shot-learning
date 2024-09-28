import torch
from activeft.acquisition_functions import BatchAcquisitionFunction
from activeft.model import Model
from activeft.utils import DEFAULT_MINI_BATCH_SIZE, DEFAULT_NUM_WORKERS


class Random(BatchAcquisitionFunction):
    """
    `Random` selects a batch uniformly at random from the data.
    Using `activeft` with the `Random` acquisition function is equivalent to using the classical PyTorch data loader with shuffling:

    ```python
    from torch.utils.data import DataLoader

    data_loader = DataLoader(data, batch_size=64, shuffle=True)
    batch = next(iter(data_loader))
    ```

    Random data selection leads inherently to reasonably diverse batches, yet, it does not consider the "usefulness" of data.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | ❌                | (✅)        | -                  |
    """

    def __init__(
        self,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
    ):
        super().__init__(
            mini_batch_size=mini_batch_size, num_workers=num_workers, subsample=True
        )

    def compute(
        self,
        model: Model,
        data: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.randperm(data.size(0), device=device)
