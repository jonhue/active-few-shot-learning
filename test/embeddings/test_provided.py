import torch
from torch import nn
from afsl.embeddings.provided import ProvidedEmbedding
from afsl.model import ModelWithEmbedding


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(SimpleMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.output = nn.Linear(input_size, output_size)

    def embed(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, x):
        x = self.latent(x)
        x = self.output(x)
        return x


# Create the model
model = SimpleMLP(input_size=10, hidden_sizes=[64, 32])

data = torch.randn(100, 10)

embedding = ProvidedEmbedding(mini_batch_size=99)


def test_embed():
    embeddings = embedding.embed(model, data)
    assert embeddings.shape == (100, 32)
