import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.model import Model
from afsl.utils import get_device


class MaxMargin(BatchAcquisitionFunction):
    r"""
    Given a model which for an input $\vx$ outputs a (softmax) distribution over classes $p_{\vx}$, the margin of $\vx$ is the difference between the largest and second largest class probabilities: \\[\mathrm{margin}(\vx) \defeq \max_i p_\vx(i) - \max_{j, j \neq i} p_\vx(j).\\]
    `MaxMargin`[^1] selects the inputs with the largest margin.
    Intuitively, this leads to the selection of inputs for which the model is uncertain about the correct class.
    This is a commonly used heuristic for determining informative data points.

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
            output = torch.softmax(
                model(data.to(get_device(model), non_blocking=True)), dim=1
            )
            top_preds, _ = torch.topk(output, 2, dim=1)
            margins = top_preds[:, 0] - top_preds[:, 1]
            return margins
