import torch
from afsl.embeddings import Embedding
from afsl.model import LatentModel
from afsl.utils import get_device, mini_batch_wrapper


class LatentEmbedding(Embedding[LatentModel]):
    """is empirical NTK wrt last layer if latent is the last layer"""

    def embed(self, model: LatentModel, data: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            embeddings = mini_batch_wrapper(
                fn=lambda batch: model.latent(batch.to(get_device(model))),
                data=data,
                batch_size=self.mini_batch_size,
            )
            return embeddings
