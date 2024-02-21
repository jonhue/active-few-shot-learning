import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.model import Model


class Random(BatchAcquisitionFunction):
    """
    `Random` selects a batch uniformly at random from the data.
    Using `afsl` with the `Random` acquisition function is equivalent to using the classical PyTorch data loader with shuffling:

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

    def compute(
        self,
        model: Model,
        data: torch.Tensor,
    ) -> torch.Tensor:
        return torch.randperm(data.size(0))
