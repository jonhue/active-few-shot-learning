import torch
import torch.nn.functional as F
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.embeddings import Embedding
from afsl.types import LatentModel, Target
from afsl.utils import get_device


class CosineSimilarity(BatchAcquisitionFunction):
    def compute(
        self,
        embedding: Embedding,
        model: LatentModel,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert target is not None, "Target must be non-empty"

        model.eval()
        device = get_device(model)
        with torch.no_grad():
            data_latent = model.latent(data.to(device))
            target_latent = model.latent(target.to(device))

            data_latent_normalized = F.normalize(data_latent, p=2, dim=1)
            target_latent_normalized = F.normalize(target_latent, p=2, dim=1)

            cosine_similarities = torch.matmul(
                data_latent_normalized, target_latent_normalized.T
            )

            average_cosine_similarities = torch.mean(cosine_similarities, dim=1)
            return average_cosine_similarities
