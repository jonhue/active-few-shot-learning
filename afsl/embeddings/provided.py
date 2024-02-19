import torch
from afsl.embeddings import Embedding
from afsl.model import ModelWithEmbedding
from afsl.utils import get_device, mini_batch_wrapper


class ProvidedEmbedding(Embedding[ModelWithEmbedding]):
    """is empirical NTK wrt last layer if latent is the last layer"""

    def embed(self, model: ModelWithEmbedding, data: torch.Tensor) -> torch.Tensor:
        device = get_device(model)
        model.eval()
        with torch.no_grad():
            embeddings = mini_batch_wrapper(
                fn=lambda batch: model.embed(batch.to(device)),
                data=data,
                batch_size=self.mini_batch_size,
            )
            return embeddings
