import torch
import wandb
from afsl.acquisition_functions.bace import BaCE, BaCEState
from afsl.embeddings import M


class GreedyMaxDet(BaCE[M]):
    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[:, :])
        wandb.log(
            {
                "max_var": torch.max(variances),
                "min_var": torch.min(variances),
            }
        )
        return variances
