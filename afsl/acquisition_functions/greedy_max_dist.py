import torch
from afsl.acquisition_functions.distance import (
    DistanceBasedAcquisitionFunction,
    DistanceState,
)


class GreedyMaxDist(DistanceBasedAcquisitionFunction):
    r"""
    Given a model which for two inputs $\vx$ and $\vxp$ induces a distance $d(\vx,\vxp)$,[^1] `GreedyMaxDist`[^2] constructs the batch by choosing the point with the maximum distance to the nearest previously selected point: \\[ \vx_i = \argmax_{\vx} \min_{j < i} d(\vx, \vx_j). \\]
    The first point $\vx_1$ is chosen randomly.

    .. note::

        This acquisition function is similar to [k-means++](kmeans_pp) but selects the batch deterministically rather than randomly.

    `GreedyMaxDist` explicitly enforces *diversity* in the selected batch.
    If the selected centroids from previous batches are used to initialize the centroids for the current batch,[^3] then `GreedyMaxDist` heuristically also leads to *informative* samples since samples are chosen to be different from previously seen data.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | (✅)                | ✅          | embedding / kernel  |

    [^1]: See afsl.acquisition_functions.distance.DistanceBasedAcquisitionFunction for a discussion of how a distance is induced by embeddings or a kernel.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^3]: see `initialize_with_previous_samples`
    """

    def compute(self, state: DistanceState) -> torch.Tensor:
        if len(state.centroid_indices) == 0:
            # Choose the first centroid randomly
            return torch.ones(state.data.size(0))

        # Compute the distance of each point to the nearest centroid
        distances = self.compute_min_distances(state)
        return distances
