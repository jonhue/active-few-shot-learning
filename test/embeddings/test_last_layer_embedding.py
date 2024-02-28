import torch
from afsl.models.simple_mlp import SimpleMLP
from afsl.utils import compute_embedding


class SimpleMLPWithEmbedding(SimpleMLP):
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


model = SimpleMLPWithEmbedding(input_size=10, hidden_sizes=[64, 32])

data = torch.randn(100, 10)


def test_last_layer():
    embeddings = compute_embedding(model, data, batch_size=99)
    assert embeddings.shape == (100, 32)
