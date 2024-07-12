import torch
import wandb
from afsl.acquisition_functions.bace import BaCEState, TargetedBaCE
from afsl.gaussian import get_jitter
from afsl.utils import DEFAULT_MINI_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEFAULT_SUBSAMPLE


class UndirectedVTL(TargetedBaCE):
    r"""
    `UndirectedVTL` is the special case of [VTL](vtl) without specified prediction targets.[^1]
    In the literature, this acquisition function is also known as BAIT.[^4][^2]

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | ✅                | ✅          | embedding / kernel  |

    [^1]: That is, the prediction targets $\spA$ are equal to the data set $\spS$.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^4]: Ash, J., Goel, S., Krishnamurthy, A., and Kakade, S. Gone fishing: Neural active learning with fisher embeddings. NeurIPS, 34, 2021.
    """

    def __init__(
        self,
        noise_std=1.0,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        """
        :param noise_std: Standard deviation of the noise.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        TargetedBaCE.__init__(
            self,
            target=torch.tensor([]),
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )

    def compute(self, state: BaCEState) -> torch.Tensor:
        if self.noise_std is None:
            noise_var = get_jitter(covariance_matrix=state.covariance_matrix, indices=torch.arange(state.n))
        else:
            noise_var = self.noise_std**2

        def compute_posterior_variance(i, j):
            return state.covariance_matrix[i, i] - state.covariance_matrix[
                i, j
            ] ** 2 / (state.covariance_matrix[j, j] + noise_var)

        data_indices = torch.arange(state.n).unsqueeze(
            1
        )  # Expand dims for broadcasting
        target_indices = torch.arange(state.n).unsqueeze(
            0
        )  # Expand dims for broadcasting

        posterior_variances = compute_posterior_variance(target_indices, data_indices)
        total_posterior_variances = torch.sum(posterior_variances, dim=1)
        wandb.log(
            {
                "max_posterior_var": torch.max(posterior_variances),
                "min_posterior_var": torch.min(posterior_variances),
            }
        )
        return -total_posterior_variances
