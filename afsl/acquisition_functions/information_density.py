import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.acquisition_functions.cosine_similarity import CosineSimilarity
from afsl.acquisition_functions.max_entropy import MaxEntropy
from afsl.embeddings import Embedding
from afsl.model import LatentModel
from afsl.types import Target


class InformationDensity(BatchAcquisitionFunction):
    def compute(
        self,
        embedding: Embedding,
        model: LatentModel,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        entropy = MaxEntropy(mini_batch_size=self.mini_batch_size).compute(
            embedding, model, data, target, Sigma
        )
        cosine_similarity = CosineSimilarity(
            mini_batch_size=self.mini_batch_size
        ).compute(embedding, model, data, target, Sigma)
        return entropy * cosine_similarity
