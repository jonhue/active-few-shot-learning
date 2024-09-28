import torch
from activeft.acquisition_functions import BatchAcquisitionFunction
from activeft.model import Model
from activeft.utils import get_device, mini_batch_wrapper


class MaxEntropy(BatchAcquisitionFunction):
    r"""
    Given a model which for an input $\vx$ outputs a (softmax) distribution over classes $p_{\vx}$, `MaxEntropy`[^1] selects the inputs $\vx$ where $p_{\vx}$ has the largest entropy: $\H{p_\vx}$.

    Intuitively, the entropy of $p_{\vx}$ measures the "uncertainty" of $p_{\vx}$, and therefore,
    `MaxEntropy` can be seen as a heuristic for determining informative data points.

    | Relevance? | Diversity? | Model Requirement  |
    |------------|------------|--------------------|
    | ❌         | ❌          | softmax            |

    [^1]: Settles, B. and Craven, M. An analysis of active learning strategies for sequence labeling tasks. In EMNLP, 2008.
    """

    def compute(
        self,
        model: Model,
        data: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():

            def engine(batch: torch.Tensor) -> torch.Tensor:
                output = torch.softmax(
                    model(batch.to(get_device(model), non_blocking=True)), dim=1
                ).to(device)
                entropy = -torch.sum(output * torch.log(output), dim=1)
                return entropy

            return mini_batch_wrapper(
                fn=engine,
                data=data,
                batch_size=100,
            )