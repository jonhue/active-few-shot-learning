"""works only for classification models; assumes that the final layer is a linear layer without bias"""

import torch
from torch import nn
from afsl.model import ClassificationModel


class CrossEntropyEmbedding(ClassificationModel):
    def embed(self, data: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.final_layer, nn.Linear), "Final layer must be linear."
        assert self.final_layer.bias is None, "Final layer must not have bias."

        logits = self.logits(data)  # (N, K)
        outputs = self(data)  # (N, C)
        pred = self.predict(data)  # (N,)
        # losses = nn.CrossEntropyLoss(reduction="none")(outputs, pred)  # (N,)

        C = outputs.size(1)

        # compute gradient explicitly: eq. (1) of https://arxiv.org/pdf/1906.03671.pdf
        pred_ = torch.nn.functional.one_hot(pred, C)  # (N, C)
        A = (outputs - pred_)[:, :, None]  # (N, C, 1)
        B = logits[:, None, :]  # (N, 1, K)
        J = torch.matmul(A, B).view(-1, A.shape[1] * B.shape[2])  # (N, C * K)
        return J


class SummedCrossEntropyEmbedding(ClassificationModel):
    def embed(self, data: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.final_layer, nn.Linear), "Final layer must be linear."
        assert self.final_layer.bias is None, "Final layer must not have bias."

        logits = self.logits(data)  # (N, K)
        outputs = self(data)  # (N, C)

        C = outputs.size(1)

        A = (C * outputs - 1)[:, :, None]  # (N, C, 1)
        B = logits[:, None, :]  # (N, 1, K)
        J = torch.matmul(A, B).view(-1, A.shape[1] * B.shape[2])  # (N, C * K)
        return J


class OutputNormEmbedding(ClassificationModel):
    def embed(self, data: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.final_layer, nn.Linear), "Final layer must be linear."
        assert self.final_layer.bias is None, "Final layer must not have bias."

        logits = self.logits(data)  # (N, K)
        outputs = self(data)  # (N, C)

        K = logits.size(1)
        C = outputs.size(1)

        J = torch.zeros((data.size(0), C * K), dtype=torch.float32)
        for i in range(data.size(0)):
            W = self.final_layer.weight  # (C, K)
            Z = logits[i, :][:, None] @ logits[i, :][None, :]  # (K, K)
            J[i, :] = (W @ Z).view(-1, J.size(1))
        return J
