import torch
from afsl.embeddings import Embedding
from afsl.model import Model
from afsl.utils import get_device, mini_batch_wrapper


class GradientEmbedding(Embedding[Model]):
    """compute embedding corresponding to empirical NTK. Assumes MLP model!"""

    def embed(self, model: Model, data: torch.Tensor) -> torch.Tensor:
        device = get_device(model)
        model.eval()

        def fn(batch):
            batch = batch.to(device)
            batch.requires_grad = True
            for param in model.parameters():
                param.requires_grad = True

            # forward pass
            z_L = model(batch)

            embeddings = []
            for i in range(z_L.size(0)):
                output = z_L[i].unsqueeze(0)

                # backward pass
                model.zero_grad()
                output.backward(retain_graph=i < z_L.size(0) - 1)

                # compute embedding
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                embedding = torch.cat([g.flatten() for g in gradients])
                embeddings.append(embedding.detach())
            embeddings = torch.stack(embeddings)
            return embeddings

        # compute outputs
        return mini_batch_wrapper(
            fn=fn,
            data=data,
            batch_size=self.mini_batch_size,
        )
