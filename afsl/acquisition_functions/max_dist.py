from typing import NamedTuple
import torch
from afsl.acquisition_functions import EmbeddingBased, SequentialAcquisitionFunction
from afsl.model import ModelWithEmbedding, ModelWithEmbeddingOrKernel, ModelWithKernel
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
    get_device,
)

__all__ = ["MaxDist", "DistanceState", "sqd_kernel_distance"]


class DistanceState(NamedTuple):
    """State of sequential batch selection."""

    n: int
    centroid_indices: torch.Tensor
    """Indices of previously selected centroids."""
    min_sqd_distances: torch.Tensor
    """Minimum squared distances to previously selected centroids. Tensor of shape $n$."""
    kernel_matrix: torch.Tensor
    r"""Kernel matrix of the data. Tensor of shape $n \times n$."""


class MaxDist(
    EmbeddingBased,
    SequentialAcquisitionFunction[ModelWithEmbeddingOrKernel | None, DistanceState],
):
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

    - **Embeddings** $\vphi(\cdot)$ induce the (Euclidean) *embedding distance* \\[ d_\vphi(\vx,\vxp) \defeq \norm{\vphi(\vx) - \vphi(\vxp)}_2. \\]
    - A **kernel** $k$ induces the *kernel distance* \\[ d_k(\vx,\vxp) \defeq = \sqrt{k(\vx,\vx) + k(\vxp,\vxp) - 2 k(\vx,\vxp)}. \\]

    It is straightforward to see that if $k(\vx,\vxp) = \vphi(\vx)^\top \vphi(\vxp)$ then embedding and kernel distances coincide, i.e., $d_\vphi(\vx,\vxp) = d_k(\vx,\vxp)$.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^3]: see `initialize_with_previous_samples` (deprecated)
    """

    initialize_with_previous_samples: bool = True
    """Whether to initialize the centroids with the samples from previous batches."""

    def __init__(
        self,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
        initialize_with_previous_samples=True,
    ):
        """
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param num_workers: Number of workers used for parallelizing the computation of the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        :param force_nonsequential: Whether to force non-sequential data selection.
        :param initialize_with_previous_samples: Whether to initialize the centroids with the samples from previous batches.
        """
        SequentialAcquisitionFunction.__init__(
            self,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        EmbeddingBased.__init__(self, embedding_batch_size=embedding_batch_size)
        self.initialize_with_previous_samples = initialize_with_previous_samples

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel | None,
        data: torch.Tensor,
        selected_data: torch.Tensor | None,
        batch_size: int,
    ) -> DistanceState:
        n = data.size(0)

        if model is None or isinstance(model, ModelWithEmbedding):
            embeddings = self.compute_embedding(
                model=model, data=data, batch_size=self.embedding_batch_size
            )
            device = embeddings.device
        else:
            device = get_device(model)

        if self.initialize_with_previous_samples and selected_data is not None:
            centroid_indices = torch.arange(
                data.size(0), data.size(0) + selected_data.size(0)
            )
            data = torch.cat([data, selected_data], dim=0)
            if model is None or isinstance(model, ModelWithEmbedding):
                selected_embeddings = self.compute_embedding(
                    model=model,
                    data=selected_data,
                    batch_size=self.embedding_batch_size,
                )
                embeddings = torch.cat([embeddings, selected_embeddings], dim=0)
                centroids = embeddings[centroid_indices.to(embeddings.device)]
                distances = torch.square(
                    torch.cdist(embeddings.unsqueeze(0), centroids.unsqueeze(0), p=2)[0]
                )
            else:
                centroids = data[centroid_indices.to(data.device)]
                distances = sqd_kernel_distance(data, centroids, model)
            min_sqd_distances = torch.min(distances, dim=1).values
        else:
            centroid_indices = torch.tensor([], dtype=torch.long)
            min_sqd_distances = torch.full(
                size=(data.size(0),), fill_value=torch.inf, device=device
            )

        if model is None or isinstance(model, ModelWithEmbedding):
            kernel_matrix = embeddings @ embeddings.T
        else:
            kernel_matrix = model.kernel(data, data)

        return DistanceState(
            n=n,
            centroid_indices=centroid_indices,
            min_sqd_distances=min_sqd_distances,
            kernel_matrix=kernel_matrix,
        )

    def compute(self, state: DistanceState) -> torch.Tensor:
        min_sqd_distances = torch.clone(state.min_sqd_distances)
        if state.centroid_indices.size(0) > 0:
            min_sqd_distances[state.centroid_indices] = 0
        return min_sqd_distances[:state.n]

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
            n=state.n,
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
