import torch
from afsl.embeddings.provided import ProvidedEmbedding
from afsl.models.simple_mlp import SimpleMLP

model = SimpleMLP(input_size=10, hidden_sizes=[64, 32])

data = torch.randn(100, 10)

embedding = ProvidedEmbedding(mini_batch_size=99)


def test_embed():
    embeddings = embedding.embed(model, data)
    assert embeddings.shape == (100, 32)
