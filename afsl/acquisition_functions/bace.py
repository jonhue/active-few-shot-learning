from typing import NamedTuple
import torch
from afsl.acquisition_functions import SequentialAcquisitionFunction
from afsl.embeddings import M, Embedding
from afsl.gaussian import GaussianCovarianceMatrix
from afsl.types import Target


class BaCEState(NamedTuple):
    covariance_matrix: GaussianCovarianceMatrix
    n: int


class BaCE(SequentialAcquisitionFunction):
    noise_std: float

    def __init__(self, noise_std=1.0):
        super().__init__()
        self.noise_std = noise_std

    def initialize(
        self,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> BaCEState:
        assert target is not None, "Target must be non-empty"

        n = data.size(0)
        data_embeddings = embedding.embed(model, data)
        target_embeddings = embedding.embed(model, target)
        joint_embeddings = torch.cat((data_embeddings, target_embeddings))
        covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
            noise_std=self.noise_std, Embeddings=joint_embeddings, Sigma=Sigma
        )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)

    def step(self, state: BaCEState, i: int) -> BaCEState:
        posterior_covariance_matrix = state.covariance_matrix.condition_on(i)
        return BaCEState(covariance_matrix=posterior_covariance_matrix, n=state.n)
