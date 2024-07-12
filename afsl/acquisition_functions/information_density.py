import torch
from afsl.acquisition_functions import BatchAcquisitionFunction
from afsl.acquisition_functions.cosine_similarity import CosineSimilarity
from afsl.acquisition_functions.max_entropy import MaxEntropy
from afsl.model import ModelWithEmbedding
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
)


class InformationDensity(BatchAcquisitionFunction):
    r"""
    `InformationDensity`[^1] is a heuristic combination of the [MaxEntropy](max_entropy) and [Cosine Similarity](cosine_similarity) acquisition functions: \\[\argmax_{\vx}\quad \H{p_\vx} \cdot \left(\frac{1}{|\spA|} \sum_{\vxp \in \spA} \angle(\vphi(\vx), \vphi(\vxp))\right)^\beta\\] where the parameter $\beta$ trades off informativeness and relevance.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | (✅)        | (✅)              | ❌          | embedding & softmax |

    [^1]: Settles, B. and Craven, M. An analysis of active learning strategies for sequence labeling tasks. In EMNLP, 2008.
    """

    beta: float
    r"""Parameter $\beta$ trading off informativeness and relevance. Default is $1.0$."""

    cosine_similarity: CosineSimilarity
    max_entropy: MaxEntropy

    def __init__(
        self,
        target: torch.Tensor,
        beta=1.0,
        subsampled_target_frac: float = 1,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
    ):
        super().__init__(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
        )
        self.cosine_similarity = CosineSimilarity(
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            embedding_batch_size=embedding_batch_size,
            num_workers=num_workers,
            subsample=subsample,
        )
        self.max_entropy = MaxEntropy(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
        )
        self.beta = beta

    def compute(
        self,
        model: ModelWithEmbedding,
        data: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        entropy = self.max_entropy.compute(model, data, device)
        cosine_similarity = self.cosine_similarity.compute(model, data, device)
        return entropy * cosine_similarity**self.beta
