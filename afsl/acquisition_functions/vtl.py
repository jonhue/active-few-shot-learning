import torch
import wandb
from afsl.acquisition_functions.bace import BaCE, BaCEState


class VTL(BaCE):
    def compute(self, state: BaCEState) -> torch.Tensor:
        noise_var = self.noise_std**2

        def compute_posterior_variance(i, j):
            return state.covariance_matrix[i, i] - state.covariance_matrix[
                i, j
            ] ** 2 / (state.covariance_matrix[j, j] + noise_var)

        data_indices = torch.arange(state.n).unsqueeze(
            1
        )  # Expand dims for broadcasting
        target_indices = torch.arange(state.n, state.covariance_matrix.dim).unsqueeze(
            0
        )  # Expand dims for broadcasting

        posterior_variances = compute_posterior_variance(target_indices, data_indices)
        total_posterior_variances = torch.sum(posterior_variances, dim=1)
        wandb.log(
            {
                "max_posterior_var": torch.max(total_posterior_variances),
                "min_posterior_var": torch.min(total_posterior_variances),
            }
        )
        return -total_posterior_variances
