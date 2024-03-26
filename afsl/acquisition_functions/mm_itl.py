import torch
import wandb
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState
from afsl.acquisition_functions.ctl import _compute_correlations


class MMITL(TargetedBaCE):
    def compute(self, state: BaCEState) -> torch.Tensor:
        correlations = _compute_correlations(
            covariance_matrix=state.covariance_matrix, n=state.n
        )
        sqd_correlations = torch.square(correlations)

        marginal_mi = -0.5 * torch.log(1 - sqd_correlations)
        wandb.log(
            {
                "max_mi": torch.max(marginal_mi),
                "min_mi": torch.min(marginal_mi),
            }
        )
        mean_marginal_mi = torch.mean(marginal_mi, dim=1)
        return mean_marginal_mi
