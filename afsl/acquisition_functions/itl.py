import torch
import wandb
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState


class ITL(TargetedBaCE):
    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[: state.n, : state.n])
        conditional_covariance_matrix = state.covariance_matrix.condition_on(
            torch.arange(start=state.n, end=state.covariance_matrix.dim)
        )[: state.n, : state.n]
        conditional_variances = torch.diag(conditional_covariance_matrix)

        mi = 0.5 * torch.log(variances / conditional_variances)
        wandb.log(
            {
                "max_mi": torch.max(mi),
                "min_mi": torch.min(mi),
            }
        )
        return mi
