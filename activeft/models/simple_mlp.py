import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(SimpleMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.output = nn.Linear(input_size, output_size)

    def logits(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return x

    def forward(self, x):
        x = self.logits(x)
        x = self.output(x)
        return x
