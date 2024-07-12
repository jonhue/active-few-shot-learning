from typing import List
import torch

from afsl.acquisition_functions.utils import get_jitter


class GaussianCovarianceMatrix:
    _matrix: torch.Tensor

    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix

    @staticmethod
    def from_embeddings(Embeddings: torch.Tensor, Sigma: torch.Tensor | None = None):
        if Sigma is None:
            Sigma = torch.eye(Embeddings.size(1)).to(Embeddings.device)
        return GaussianCovarianceMatrix(Embeddings @ Sigma @ Embeddings.T)

    def __getitem__(self, indices):
        i, j = indices
        return self._matrix[i, j]

    @property
    def dim(self) -> int:
        return self._matrix.size(0)

    def condition_on(
        self,
        indices: torch.Tensor | List[int] | int,
        target_indices: torch.Tensor | None = None,
        noise_std: float | None = None,
    ):
        """
        Computes the conditional covariance matrix.

        :param indices: Indices on which to condition.
        :param target_indices: Indices on which to compute conditional covariance. All indices if `None`.
        :param noise_std: Standard deviation of observation noise. Determined automatically if `None`.

        :return: Conditional covariance of target_indices upon observing indices
        """
        _indices: torch.Tensor = torch.tensor(indices) if not torch.is_tensor(indices) else indices  # type: ignore
        if _indices.dim() == 0:
            _indices = _indices.unsqueeze(0)
        if target_indices is None:
            target_indices = torch.arange(self.dim)

        if noise_std is None:
            noise_var = get_jitter(covariance_matrix=self, indices=_indices)
        else:
            noise_var = noise_std**2

        Sigma_AA = self._matrix[target_indices][:, target_indices]
        Sigma_ii = self._matrix[_indices][:, _indices]
        Sigma_Ai = self._matrix[target_indices][:, _indices]
        posterior_Sigma_AA = (
            Sigma_AA
            - Sigma_Ai
            @ torch.inverse(
                Sigma_ii + noise_var * torch.eye(Sigma_ii.size(0)).to(Sigma_AA.device)
            )
            @ Sigma_Ai.T
        )
        return GaussianCovarianceMatrix(posterior_Sigma_AA)
