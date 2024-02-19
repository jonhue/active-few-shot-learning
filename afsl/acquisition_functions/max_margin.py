import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.embeddings import Embedding
from afsl.model import Model
from afsl.types import Target
from afsl.utils import get_device


class MaxMargin(BatchAcquisitionFunction):
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
            top_preds, _ = torch.topk(output, 2, dim=1)
            margins = top_preds[:, 0] - top_preds[:, 1]
            return margins
