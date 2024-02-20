import torch
import torch.nn.functional as F
from afsl.acquisition_functions import (
    BatchAcquisitionFunction,
    Targeted,
)
from afsl.model import ModelWithEmbedding
from afsl.utils import DEFAULT_MINI_BATCH_SIZE, compute_embedding, get_device


class CosineSimilarity(Targeted, BatchAcquisitionFunction):
    def __init__(
        self,
        target: torch.Tensor,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
    ):
        BatchAcquisitionFunction.__init__(self, mini_batch_size=mini_batch_size)
        Targeted.__init__(
            self,
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
        )

    def compute(
        self,
        model: ModelWithEmbedding,
        data: torch.Tensor,
    ) -> torch.Tensor:
        model.eval()
        device = get_device(model)
        with torch.no_grad():
            data_latent = compute_embedding(
                model, data=data, mini_batch_size=self.mini_batch_size
            )
            target_latent = compute_embedding(
                model, data=self.target, mini_batch_size=self.mini_batch_size
            )

            data_latent_normalized = F.normalize(data_latent, p=2, dim=1)
            target_latent_normalized = F.normalize(target_latent, p=2, dim=1)

            cosine_similarities = torch.matmul(
                data_latent_normalized, target_latent_normalized.T
            )

            average_cosine_similarities = torch.mean(cosine_similarities, dim=1)
            return average_cosine_similarities
