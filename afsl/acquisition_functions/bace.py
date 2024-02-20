from typing import NamedTuple
import torch
from afsl.acquisition_functions import (
    SequentialAcquisitionFunction,
    Targeted,
)
from afsl.gaussian import GaussianCovarianceMatrix
from afsl.model import ModelWithEmbeddingOrKernel, ModelWithKernel
from afsl.utils import DEFAULT_MINI_BATCH_SIZE, compute_embedding


class BaCEState(NamedTuple):
    covariance_matrix: GaussianCovarianceMatrix
    n: int


class BaCE(SequentialAcquisitionFunction[ModelWithEmbeddingOrKernel, BaCEState]):
    Sigma: torch.Tensor | None
    noise_std: float

    def __init__(
        self,
        Sigma: torch.Tensor | None = None,
        noise_std=1.0,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
    ):
        super().__init__(mini_batch_size=mini_batch_size)
        self.Sigma = Sigma
        self.noise_std = noise_std

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel,
        data: torch.Tensor,
    ) -> BaCEState:
        n = data.size(0)
        if isinstance(model, ModelWithKernel):
            covariance_matrix = GaussianCovarianceMatrix(
                model.kernel(data, None), noise_std=self.noise_std
            )
        else:
            data_embeddings = compute_embedding(
                model, data, mini_batch_size=self.mini_batch_size
            )
            covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
                noise_std=self.noise_std, Embeddings=data_embeddings, Sigma=self.Sigma
            )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)

    def step(self, state: BaCEState, i: int) -> BaCEState:
        posterior_covariance_matrix = state.covariance_matrix.condition_on(i)
        return BaCEState(covariance_matrix=posterior_covariance_matrix, n=state.n)


class TargetedBaCE(Targeted, BaCE):
    def __init__(
        self,
        target: torch.Tensor,
        Sigma: torch.Tensor | None = None,
        noise_std=1.0,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
    ):
        BaCE.__init__(
            self,
            Sigma=Sigma,
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
        )
        Targeted.__init__(
            self,
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
        )

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel,
        data: torch.Tensor,
    ) -> BaCEState:
        n = data.size(0)
        if isinstance(model, ModelWithKernel):
            covariance_matrix = GaussianCovarianceMatrix(
                model.kernel(torch.cat((data, self.target)), None),
                noise_std=self.noise_std,
            )
        else:
            data_embeddings = compute_embedding(
                model, data=data, mini_batch_size=self.mini_batch_size
            )
            target_embeddings = compute_embedding(
                model, data=self.target, mini_batch_size=self.mini_batch_size
            )
            joint_embeddings = torch.cat((data_embeddings, target_embeddings))
            covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
                noise_std=self.noise_std, Embeddings=joint_embeddings, Sigma=self.Sigma
            )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)

    def step(self, state: BaCEState, i: int) -> BaCEState:
        posterior_covariance_matrix = state.covariance_matrix.condition_on(i)
        return BaCEState(covariance_matrix=posterior_covariance_matrix, n=state.n)
