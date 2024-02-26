from typing import List
import torch


class GaussianCovarianceMatrix:
    _matrix: torch.Tensor
    noise_std: float

    def __init__(self, matrix: torch.Tensor, noise_std: float):
        self._matrix = matrix
        self.noise_std = noise_std

    @staticmethod
    def from_embeddings(
        noise_std: float, Embeddings: torch.Tensor, Sigma: torch.Tensor | None = None
    ):
        if Sigma is None:
            Sigma = torch.eye(Embeddings.size(1)).to(Embeddings.device)
        return GaussianCovarianceMatrix(
            Embeddings @ Sigma @ Embeddings.T, noise_std=noise_std
        )

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
    ):
        _indices: torch.Tensor = torch.tensor(indices) if not torch.is_tensor(indices) else indices  # type: ignore
        if _indices.dim() == 0:
            _indices = _indices.unsqueeze(0)
        if target_indices is None:
            target_indices = torch.arange(self.dim)
        noise_var = self.noise_std**2

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
        return GaussianCovarianceMatrix(posterior_Sigma_AA, noise_std=self.noise_std)
