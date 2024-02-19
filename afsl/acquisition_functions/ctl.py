import torch
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState


class CTL(TargetedBaCE):
    def compute(self, state: BaCEState) -> torch.Tensor:
        ind_a = torch.arange(state.n)
        ind_b = torch.arange(state.n, state.covariance_matrix.dim)
        covariance_aa = state.covariance_matrix[ind_a, :][:, ind_a]
        covariance_bb = state.covariance_matrix[ind_b, :][:, ind_b]
        covariance_ab = state.covariance_matrix[ind_a, :][:, ind_b]

        std_a = torch.sqrt(torch.diag(covariance_aa))
        std_b = torch.sqrt(torch.diag(covariance_bb))
        std_ab = torch.ger(std_a, std_b)  # outer product of standard deviations

        correlations = covariance_ab / std_ab
        average_correlations = torch.mean(correlations, dim=1)
        return average_correlations
