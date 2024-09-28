import torch
from activeft.embeddings.empirical_ntk import EmpiricalNTKEmbedding
from activeft.models.simple_mlp import SimpleMLP


class SimpleMLPWithEmbedding(SimpleMLP, EmpiricalNTKEmbedding):
    pass


model = SimpleMLPWithEmbedding(input_size=10, hidden_sizes=[64, 32])
K = sum(p.numel() for p in model.parameters())

data = torch.randn(100, 10)


def test_embed():
    embeddings = model.embed(data)
    assert embeddings.shape == (100, K)
