import torch
import wandb
from afsl.acquisition_functions.bace import BaCE, BaCEState
from afsl.embeddings import M, Embedding
from afsl.gaussian import GaussianCovarianceMatrix
from afsl.types import Target


class GreedyMaxDet(BaCE):
    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[:, :])
        wandb.log(
            {
                "max_var": torch.max(variances),
                "min_var": torch.min(variances),
            }
        )
        return variances

    def initialize(
        self,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> BaCEState:
        n = data.size(0)
        data_embeddings = embedding.embed(model, data)
        covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
            noise_std=self.noise_std, Embeddings=data_embeddings, Sigma=Sigma
        )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)
