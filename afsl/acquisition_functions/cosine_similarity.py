import torch
import torch.nn.functional as F
from afsl.acquisition_functions import (
    BatchAcquisitionFunction,
    EmbeddingBased,
    Targeted,
)
from afsl.model import ModelWithEmbedding
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
)


class CosineSimilarity(
    EmbeddingBased, Targeted, BatchAcquisitionFunction[ModelWithEmbedding | None]
):
    r"""
    The cosine similarity between two vectors $\vphi$ and $\vphip$ is \\[\angle(\vphi, \vphip) \defeq \frac{\vphi^\top \vphip}{\|\vphi\|_2 \|\vphip\|_2}.\\]

    Given a set of targets $\spA$ and a model which for an input $\vx$ computes an embedding $\vphi(\vx)$, `CosineSimilarity`[^1] selects the inputs $\vx$ which maximize \\[ \frac{1}{|\spA|} \sum_{\vxp \in \spA} \angle(\vphi(\vx), \vphi(\vxp)). \\]
    Intuitively, this selects the points that are most similar to the targets $\spA$.

    .. note::

        `CosineSimilarity` coincides with [CTL](ctl) with `force_nonsequential=True`.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ✅          | ❌                | ❌          | embedding           |

    [^1]: Settles, B. and Craven, M. An analysis of active learning strategies for sequence labeling tasks. In EMNLP, 2008.
    """

    def __init__(
        self,
        target: torch.Tensor,
        subsampled_target_frac: float = 1,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
    ):
        r"""
        :param target: Tensor of prediction targets (shape $m \times d$).
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $1$. Ignored if `target` is `None`.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large. Ignored if `target` is `None`.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        """

        BatchAcquisitionFunction.__init__(
            self,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
        )
        EmbeddingBased.__init__(self, embedding_batch_size=embedding_batch_size)
        Targeted.__init__(
            self,
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
        )

    def compute(
        self,
        model: ModelWithEmbedding | None,
        data: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        data_latent = self.compute_embedding(
            model=model, data=data, batch_size=self.embedding_batch_size
        ).to(device)
        target_latent = self.compute_embedding(
            model=model, data=self.get_target(), batch_size=self.embedding_batch_size
        ).to(device)

        data_latent_normalized = F.normalize(data_latent, p=2, dim=1)
        target_latent_normalized = F.normalize(target_latent, p=2, dim=1)

        cosine_similarities = torch.matmul(
            data_latent_normalized, target_latent_normalized.T
        )

        average_cosine_similarities = torch.mean(cosine_similarities, dim=1)
        return average_cosine_similarities
