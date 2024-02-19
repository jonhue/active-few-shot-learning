from typing import NamedTuple
import torch
from afsl.acquisition_functions import (
    SequentialAcquisitionFunction,
    Targeted,
)
from afsl.embeddings import M, Embedding
from afsl.embeddings.provided import ProvidedEmbedding
from afsl.gaussian import GaussianCovarianceMatrix
from afsl.utils import DEFAULT_MINI_BATCH_SIZE


class BaCEState(NamedTuple):
    covariance_matrix: GaussianCovarianceMatrix
    n: int


class BaCE(SequentialAcquisitionFunction[M, BaCEState]):
    embedding: Embedding[M]
    Sigma: torch.Tensor | None
    noise_std: float

    def __init__(
        self,
        embedding: Embedding[M] = ProvidedEmbedding(),
        Sigma: torch.Tensor | None = None,
        noise_std=1.0,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
    ):
        super().__init__(mini_batch_size=mini_batch_size)
        self.embedding = embedding
        self.Sigma = Sigma
        self.noise_std = noise_std

    def initialize(
        self,
        model: M,
        data: torch.Tensor,
    ) -> BaCEState:
        n = data.size(0)
        data_embeddings = self.embedding.embed(model, data)
        covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
            noise_std=self.noise_std, Embeddings=data_embeddings, Sigma=self.Sigma
        )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)

    def step(self, state: BaCEState, i: int) -> BaCEState:
        posterior_covariance_matrix = state.covariance_matrix.condition_on(i)
        return BaCEState(covariance_matrix=posterior_covariance_matrix, n=state.n)


class TargetedBaCE(Targeted, BaCE[M]):
    def __init__(
        self,
        target: torch.Tensor,
        embedding: Embedding[M] = ProvidedEmbedding(),
        Sigma: torch.Tensor | None = None,
        noise_std=1.0,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
    ):
        BaCE.__init__(
            self,
            embedding=embedding,
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
        model: M,
        data: torch.Tensor,
    ) -> BaCEState:
        n = data.size(0)
        data_embeddings = self.embedding.embed(model, data)
        target_embeddings = self.embedding.embed(model, self.target)
        joint_embeddings = torch.cat((data_embeddings, target_embeddings))
        covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
            noise_std=self.noise_std, Embeddings=joint_embeddings, Sigma=self.Sigma
        )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)

    def step(self, state: BaCEState, i: int) -> BaCEState:
        posterior_covariance_matrix = state.covariance_matrix.condition_on(i)
        return BaCEState(covariance_matrix=posterior_covariance_matrix, n=state.n)
