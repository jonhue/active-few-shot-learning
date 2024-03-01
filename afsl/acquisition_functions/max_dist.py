from typing import NamedTuple
import torch
from afsl.acquisition_functions import SequentialAcquisitionFunction
from afsl.model import ModelWithEmbedding, ModelWithEmbeddingOrKernel, ModelWithKernel
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
    compute_embedding,
    get_device,
)

__all__ = ["MaxDist", "DistanceState", "sqd_kernel_distance"]


class DistanceState(NamedTuple):
    """State of sequential batch selection."""

    centroid_indices: torch.Tensor
    """Indices of previously selected centroids."""
    min_sqd_distances: torch.Tensor
    """Minimum squared distances to previously selected centroids. Tensor of shape $n$."""
    kernel_matrix: torch.Tensor
    r"""Kernel matrix of the data. Tensor of shape $n \times n$."""


class MaxDist(SequentialAcquisitionFunction[ModelWithEmbeddingOrKernel, DistanceState]):
    r"""
    Given a model which for two inputs $\vx$ and $\vxp$ induces a distance $d(\vx,\vxp)$, `MaxDist`[^2] constructs the batch by choosing the point with the maximum distance to the nearest previously selected point: \\[ \vx_i = \argmax_{\vx} \min_{j < i} d(\vx, \vx_j). \\]
    The first point $\vx_1$ is chosen randomly.

    .. note::

        This acquisition function is similar to [k-means++](kmeans_pp) but selects the batch deterministically rather than randomly.

    `MaxDist` explicitly enforces *diversity* in the selected batch.
    If the selected centroids from previous batches are used to initialize the centroids for the current batch,[^3] then `MaxDist` heuristically also leads to *informative* samples since samples are chosen to be different from previously seen data.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | (✅)                | ✅          | embedding / kernel  |

    #### Where does the distance come from?

    This acquisition function rests on the assumption that the model induces a distance $d(\vx,\vxp)$ between points $\vx$ and $\vxp$, either via an embedding or a kernel.

    - **Embeddings** $\vphi(\cdot)$ induce the (euclidean) *embedding distance* \\[ d_\vphi(\vx,\vxp) \defeq \norm{\vphi(\vx) - \vphi(\vxp)}_2. \\]
    - A **kernel** $k$ induces the *kernel distance* \\[ d_k(\vx,\vxp) \defeq = \sqrt{k(\vx,\vx) + k(\vxp,\vxp) - 2 k(\vx,\vxp)}. \\]

    It is straightforward to see that if $k(\vx,\vxp) = \vphi(\vx)^\top \vphi(\vxp)$ then embedding and kernel distances coincide, i.e., $d_\vphi(\vx,\vxp) = d_k(\vx,\vxp)$.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^3]: see `initialize_with_previous_samples` (deprecated)
    """

    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    """Batch size used for computing the embeddings."""

    def __init__(
        self,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        """
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param num_workers: Number of workers used for parallelizing the computation of the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        super().__init__(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        self.embedding_batch_size = embedding_batch_size

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel,
        data: torch.Tensor,
    ) -> DistanceState:
        if isinstance(model, ModelWithEmbedding):
            embeddings = compute_embedding(
                model, data, batch_size=self.embedding_batch_size
            )

        centroid_indices = torch.tensor([])
        min_sqd_distances = torch.full(
            size=(data.size(0),), fill_value=torch.inf, device=get_device(model)
        )

        if isinstance(model, ModelWithEmbedding):
            kernel_matrix = embeddings @ embeddings.T
        else:
            kernel_matrix = model.kernel(data, data)

        return DistanceState(
            centroid_indices=centroid_indices,
            min_sqd_distances=min_sqd_distances,
            kernel_matrix=kernel_matrix,
        )

    def compute(self, state: DistanceState) -> torch.Tensor:
        return state.min_sqd_distances

    def step(self, state: DistanceState, i: int) -> DistanceState:
        centroid_indices = torch.cat(
            [
                state.centroid_indices,
                torch.tensor([i]).to(state.centroid_indices.device),
            ]
        )
        new_sqd_distances = (
            state.kernel_matrix[i, i]
            + torch.diag(state.kernel_matrix)
            - 2 * state.kernel_matrix[i, :]
        ).to(state.min_sqd_distances.device)
        min_sqd_distances = torch.min(state.min_sqd_distances, new_sqd_distances)
        return DistanceState(
            centroid_indices=centroid_indices,
            min_sqd_distances=min_sqd_distances,
            kernel_matrix=state.kernel_matrix,
        )


def sqd_kernel_distance(
    x1: torch.Tensor, x2: torch.Tensor, model: ModelWithKernel
) -> torch.Tensor:
    r"""
    Returns the squared *kernel distance* \\[ d_k(\vx,\vxp)^2 \defeq \norm{\vphi(\vx) - \vphi(\vxp)}_2^2 = k(\vx,\vx) + k(\vxp,\vxp) - 2 k(\vx,\vxp) \\] induced by the kernel $k(\vx,\vxp) = \vphi(\vx)^\top \vphi(\vxp)$.

    :param x1: Tensor of shape $n \times d$.
    :param x2: Tensor of shape $m \times d$.
    :param model: Model with a kernel method.
    :return: Tensor of shape $n \times m$ of pairwise squared distances.
    """
    return torch.sqrt(
        model.kernel(x1, x1) + model.kernel(x2, x2) - 2 * model.kernel(x1, x2)
    )
