import torch
from activeft.acquisition_functions import EmbeddingBased
from activeft.models.simple_mlp import SimpleMLP


class SimpleMLPWithEmbedding(SimpleMLP):
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


model = SimpleMLPWithEmbedding(input_size=10, hidden_sizes=[64, 32])

data = torch.randn(100, 10)


def test_last_layer():
    embeddings = EmbeddingBased.compute_embedding(model, data, batch_size=99)
    assert embeddings.shape == (100, 32)
