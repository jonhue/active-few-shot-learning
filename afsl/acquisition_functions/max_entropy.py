import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.model import Model
from afsl.utils import get_device, mini_batch_wrapper


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

            def engine(batch: torch.Tensor) -> torch.Tensor:
                output = torch.softmax(
                    model(batch.to(get_device(model), non_blocking=True)), dim=1
                )
                entropy = -torch.sum(output * torch.log(output), dim=1)
                return entropy

            return mini_batch_wrapper(
                fn=engine,
                data=data,
                batch_size=100,
            )
