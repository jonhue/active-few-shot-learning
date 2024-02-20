import torch
import pytest
from afsl.embeddings.classification import ClassificationEmbedding
from afsl.models.simple_convnet import SimpleCNN

model = SimpleCNN(input_channels=3, output_channels=5, k=256)
K = sum(p.numel() for p in model.fc2.parameters())

data = torch.randn(100, 3, 28, 28)


def test_embed_cross_entropy_loss():
    embedding = ClassificationEmbedding(kind="cross_entropy_loss", mini_batch_size=99)
    embeddings = embedding.embed(model, data)
    assert embeddings.shape == (100, K)


def test_embed_summed_cross_entropy_loss():
    embedding = ClassificationEmbedding(
        kind="summed_cross_entropy_loss", mini_batch_size=99
    )
    embeddings = embedding.embed(model, data)
    assert embeddings.shape == (100, K)


def test_embed_norm():
    embedding = ClassificationEmbedding(kind="norm", mini_batch_size=99)
    embeddings = embedding.embed(model, data)
    assert embeddings.shape == (100, K)


def test_embed_invalid_kind():
    embedding = ClassificationEmbedding(kind="invalid_kind", mini_batch_size=99)  # type: ignore
    with pytest.raises(NotImplementedError):
        embedding.embed(model, data)
