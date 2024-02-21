import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.model import Model
from afsl.utils import get_device


class MaxEntropy(BatchAcquisitionFunction):
    r"""
    Given a model which for an input $\vx$ outputs a (softmax) distribution over classes $p_{\vx}$, `MaxEntropy`[^1] selects the inputs $\vx$ where $p_{\vx}$ has the largest entropy: $\H{p_\vx}$.

    Intuitively, the entropy of $p_{\vx}$ measures the "uncertainty" of $p_{\vx}$, and therefore,
    `MaxEntropy` can be seen as a heuristic for determining informative data points.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | (✅)              | ❌          | softmax            |

    [^1]: Settles, B. and Craven, M. An analysis of active learning strategies for sequence labeling tasks. In EMNLP, 2008.
    """

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
