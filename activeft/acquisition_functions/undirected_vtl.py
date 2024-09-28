import torch
import wandb
from activeft.acquisition_functions.bace import BaCEState, TargetedBaCE
from activeft.acquisition_functions.vtl import VTL
from activeft.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
)


class UndirectedVTL(VTL):
    r"""
    `UndirectedVTL` is the special case of [VTL](vtl) without specified prediction targets.[^1]
    In the literature, this acquisition function is also known as BAIT.[^4][^2]

    | Relevance? | Diversity? | Model Requirement  |
    |------------|------------|--------------------|
    | ❌         | ✅         | embedding / kernel  |

    [^1]: That is, the prediction targets $\spA$ are equal to the data set $\spS$.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^4]: Ash, J., Goel, S., Krishnamurthy, A., and Kakade, S. Gone fishing: Neural active learning with fisher embeddings. NeurIPS, 34, 2021.
    """

    def __init__(
        self,
        noise_std=None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        """
        :param noise_std: Standard deviation of the noise.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        TargetedBaCE.__init__(
            self,
            target=torch.tensor([]),
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            embedding_batch_size=embedding_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )

    def compute(self, state: BaCEState) -> torch.Tensor:
        return self._compute(state=state, _target_indices=torch.arange(state.n))
