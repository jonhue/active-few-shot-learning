import torch
import torch.nn as nn
import torch.nn.functional as F
from afsl.embeddings.classification import HallucinatedCrossEntropyEmbedding


class SimpleCNN(nn.Module):
    def __init__(self, output_dim: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # MNIST images are 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(
            128 * 3 * 3, 256
        )  # 3*3 comes from the image dimension after pooling layers
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, output_dim, bias=False)

    @property
    def final_layer(self):
        return self.fc2

    def logits(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1st conv layer + activation + pooling
        x = self.pool(F.relu(self.conv2(x)))  # 2nd conv layer + activation + pooling
        x = self.pool(F.relu(self.conv3(x)))  # 3rd conv layer + activation + pooling
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))  # 1st fully connected layer + activation
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self.logits(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        outputs = self(x)
        _, predicted = torch.max(outputs.data, dim=1)
        return predicted

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class SimpleCNNWithHallucinatedCrossEntropyEmbedding(
    SimpleCNN, HallucinatedCrossEntropyEmbedding
):
    pass


class SimpleCNNWithLastLayerEmbedding(SimpleCNN):
    def embed(self, x):
        return self.logits(x)
