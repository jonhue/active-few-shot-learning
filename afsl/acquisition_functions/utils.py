import torch
from afsl.gaussian import GaussianCovarianceMatrix

JITTER_ADJUSTMENT = 0.01


def get_jitter(
    covariance_matrix: GaussianCovarianceMatrix, indices: torch.Tensor
) -> float:
    if indices.dim() == 0:
        return JITTER_ADJUSTMENT

    # condition_number = torch.linalg.cond(covariance_matrix[indices, indices])
    # return JITTER_ADJUSTMENT * condition_number

    eigvals = torch.linalg.eigvalsh(covariance_matrix[indices, indices])
    return JITTER_ADJUSTMENT * eigvals[-1]
