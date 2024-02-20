import torch
from afsl.embeddings.classification import (
    CrossEntropyEmbedding,
    SummedCrossEntropyEmbedding,
    OutputNormEmbedding,
)
from afsl.models.simple_convnet import SimpleCNN

data = torch.randn(100, 3, 28, 28)


def test_embed_cross_entropy_loss():
    class SimpleCNNWithEmbedding(SimpleCNN, CrossEntropyEmbedding):
        pass

    model = SimpleCNNWithEmbedding(input_channels=3, output_channels=5, k=256)
    K = sum(p.numel() for p in model.fc2.parameters())
    embeddings = model.embed(data)
    assert embeddings.shape == (100, K)


def test_embed_summed_cross_entropy_loss():
    class SimpleCNNWithEmbedding(SimpleCNN, SummedCrossEntropyEmbedding):
        pass

    model = SimpleCNNWithEmbedding(input_channels=3, output_channels=5, k=256)
    K = sum(p.numel() for p in model.fc2.parameters())
    embeddings = model.embed(data)
    assert embeddings.shape == (100, K)


def test_embed_norm():
    class SimpleCNNWithEmbedding(SimpleCNN, OutputNormEmbedding):
        pass

    model = SimpleCNNWithEmbedding(input_channels=3, output_channels=5, k=256)
    K = sum(p.numel() for p in model.fc2.parameters())
    embeddings = model.embed(data)
    assert embeddings.shape == (100, K)
