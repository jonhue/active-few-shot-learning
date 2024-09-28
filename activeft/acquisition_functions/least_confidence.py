import torch
from activeft.acquisition_functions import BatchAcquisitionFunction
from activeft.model import Model
from activeft.utils import get_device, mini_batch_wrapper


class LeastConfidence(BatchAcquisitionFunction):
    r"""
    Given a model which for an input $\vx$ outputs a (softmax) distribution over classes $p_{\vx}$, `LeastConfidence`[^1] selects the inputs with the smallest confidence, i.e., minimizing $\max_i p_{\vx}(i)$.
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
        device: torch.device | None = None,
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():

            def engine(batch: torch.Tensor) -> torch.Tensor:
                output = torch.softmax(
                    model(batch.to(get_device(model), non_blocking=True)), dim=1
                ).to(device)
                return -torch.max(output, dim=1).values

            return mini_batch_wrapper(
                fn=engine,
                data=data,
                batch_size=100,
            )
