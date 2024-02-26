import torch
import torch.nn as nn
from afsl.embeddings.classification import HallucinatedCrossEntropyEmbedding


class EfficientNet(nn.Module):
    def __init__(self, output_dim: int):
        super(EfficientNet, self).__init__()

        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            pretrained=True,
        )

        # Freeze all layers except the penultimate layer
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[-1].parameters():
            param.requires_grad = True

        # Replace the last linear layer
        k = self.model.classifier.fc.in_features
        self.model.classifier.fc = torch.nn.Identity()
        self.fc = nn.Linear(k, output_dim, bias=False)

    @property
    def final_layer(self):
        return self.fc

    def logits(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        x = self.logits(x)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

    def predict(self, x):
        outputs = self(x)
        _, predicted = torch.max(outputs.data, dim=1)
        return predicted

    # def reset(self):
    #     self.fc.reset_parameters()


class EfficientNetWithHallucinatedCrossEntropyEmbedding(
    EfficientNet, HallucinatedCrossEntropyEmbedding
):
    pass


class EfficientNetWithLastLayerEmbedding(EfficientNet):
    def embed(self, x):
        return self.logits(x)