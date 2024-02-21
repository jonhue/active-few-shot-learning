from typing import NamedTuple
import torch
from afsl.acquisition_functions import SequentialAcquisitionFunction
from afsl.model import ModelWithEmbedding, ModelWithEmbeddingOrKernel, ModelWithKernel
from afsl.utils import DEFAULT_MINI_BATCH_SIZE, compute_embedding


def kernel_distance(
    x1: torch.Tensor, x2: torch.Tensor, model: ModelWithKernel
) -> torch.Tensor:
    r"""
    Returns the *kernel distance* \\[ d_k(\vx,\vxp) \defeq \norm{\vphi(\vx) - \vphi(\vxp)}_2 = \sqrt{k(\vx,\vx) + k(\vxp,\vxp) - 2 k(\vx,\vxp)} \\] induced by the kernel $k(\vx,\vxp) = \vphi(\vx)^\top \vphi(\vxp)$.
    """
    return torch.sqrt(
        model.kernel(x1, x1) + model.kernel(x2, x2) - 2 * model.kernel(x1, x2)
    )


class DistanceState(NamedTuple):
    model: ModelWithEmbeddingOrKernel
    data: torch.Tensor
    embeddings: torch.Tensor | None
    centroid_indices: torch.Tensor


class DistanceBasedAcquisitionFunction(
    SequentialAcquisitionFunction[ModelWithEmbeddingOrKernel, DistanceState]
):
    r"""
    Abstract base class for acquisition functions which are based on distances between points.
    This rests on the assumption that the model induces a distance $d(\vx,\vxp)$ between points $\vx$ and $\vxp$, either via an embedding or a kernel.

    - **Embeddings** $\vphi(\cdot)$ induce the (euclidean) *embedding distance* \\[ d_\vphi(\vx,\vxp) \defeq \norm{\vphi(\vx) - \vphi(\vxp)}_2. \\]
    - A **kernel** $k$ induces the *kernel distance* \\[ d_k(\vx,\vxp) \defeq = \sqrt{k(\vx,\vx) + k(\vxp,\vxp) - 2 k(\vx,\vxp)}. \\]

    It is straightforward to see that if $k(\vx,\vxp) = \vphi(\vx)^\top \vphi(\vxp)$ then embedding and kernel distances coincide, i.e., $d_\vphi(\vx,\vxp) = d_k(\vx,\vxp)$.
    """

    initialize_with_previous_samples: bool = True
    """Whether to initialize the centroids with the samples from previous batches."""

    def __init__(
        self,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        force_nonsequential=False,
        initialize_with_previous_samples=True,
    ):
        super().__init__(
            mini_batch_size=mini_batch_size, force_nonsequential=force_nonsequential
        )
        self.initialize_with_previous_samples = initialize_with_previous_samples

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel,
        data: torch.Tensor,
    ) -> DistanceState:
        if isinstance(model, ModelWithEmbedding):
            embeddings = compute_embedding(
                model, data, mini_batch_size=self.mini_batch_size
            )
        else:
            embeddings = None
        centroid_indices = (
            self.selected if self.initialize_with_previous_samples else torch.tensor([])
        )
        return DistanceState(
            model=model,
            data=data,
            embeddings=embeddings,
            centroid_indices=centroid_indices,
        )

    def step(self, state: DistanceState, i: int) -> DistanceState:
        centroid_indices = torch.cat(
            [
                state.centroid_indices,
                torch.tensor([i]).to(state.centroid_indices.device),
            ]
        )
        return DistanceState(
            model=state.model,
            data=state.data,
            embeddings=state.embeddings,
            centroid_indices=centroid_indices,
        )

    @staticmethod
    def compute_min_distances(state: DistanceState):
        # Compute the distance of all points from each centroid
        if state.embeddings is not None:
            centroids = state.embeddings[
                state.centroid_indices.to(state.embeddings.device)
            ]
            distances = torch.cdist(state.embeddings, centroids, p=2)
        else:
            assert isinstance(state.model, ModelWithKernel)
            centroids = state.data[state.centroid_indices]
            distances = kernel_distance(state.data, centroids, state.model)
        # Return the minimum distance for each point
        min_distances = torch.min(distances, dim=1).values
        return min_distances
