import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.model import Model
from afsl.utils import get_device


class MaxEntropy(BatchAcquisitionFunction):
    def compute(
        self,
        model: Model,
        data: torch.Tensor,
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            output = torch.softmax(model(data.to(get_device(model))), dim=1)
            entropy = -torch.sum(output * torch.log(output), dim=1)
            return entropy
