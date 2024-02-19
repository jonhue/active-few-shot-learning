import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.acquisition_functions.cosine_similarity import CosineSimilarity
from afsl.acquisition_functions.max_entropy import MaxEntropy
from afsl.model import LatentModel
from afsl.utils import DEFAULT_MINI_BATCH_SIZE


class InformationDensity(BatchAcquisitionFunction):
    cosine_similarity: CosineSimilarity
    max_entropy: MaxEntropy

    def __init__(self, target: torch.Tensor, mini_batch_size=DEFAULT_MINI_BATCH_SIZE):
        super().__init__(mini_batch_size=mini_batch_size)
        self.cosine_similarity = CosineSimilarity(
            target=target, mini_batch_size=mini_batch_size
        )
        self.max_entropy = MaxEntropy(mini_batch_size=mini_batch_size)

    def compute(
        self,
        model: LatentModel,
        data: torch.Tensor,
    ) -> torch.Tensor:
        entropy = self.max_entropy.compute(model, data)
        cosine_similarity = self.cosine_similarity.compute(model, data)
        return entropy * cosine_similarity
