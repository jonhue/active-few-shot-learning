import torch
from afsl.model import ModelWithEmbedding


class EmpiricalNTKEmbedding(ModelWithEmbedding):
    """
    Implements the embedding corresponding to the empirical NTK kernel.
    Requires the model to be a multi-layer perceptron, i.e., a full-connected neural network with linear layers composed by nonlinearities.

    For more details regarding the empirical NTK, see afsl.model.ModelWithEmbedding.
    """

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        data.requires_grad = True
        for param in self.parameters():
            param.requires_grad = True

        # forward pass
        z_L = self(data)

        embeddings = []
        for i in range(z_L.size(0)):
            output = z_L[i].unsqueeze(0)

            # backward pass
            self.zero_grad()
            output.backward(retain_graph=i < z_L.size(0) - 1)

            # compute embedding
            gradients = [p.grad for p in self.parameters() if p.grad is not None]
            embedding = torch.cat([g.flatten() for g in gradients])
            embeddings.append(embedding.detach())
        embeddings = torch.stack(embeddings)
        return embeddings
