import torch
import wandb
from afsl.acquisition_functions.bace import BaCE, BaCEState


class GreedyMaxDet(BaCE):
    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[:, :])
        wandb.log(
            {
                "max_var": torch.max(variances),
                "min_var": torch.min(variances),
            }
        )
        return variances
