import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.embeddings import Embedding
from afsl.types import Model, Target
from afsl.utils import get_device


class MaxEntropy(BatchAcquisitionFunction):
    def compute(
        self,
        embedding: Embedding,
        model: Model,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            output = torch.softmax(model(data.to(get_device(model))), dim=1)
            entropy = -torch.sum(output * torch.log(output), dim=1)
            return entropy
