import torch
from afsl.acquisition_functions.distance import (
    DistanceBasedAcquisitionFunction,
    DistanceState,
)


class KMeansPP(DistanceBasedAcquisitionFunction):
    r"""
    Given a model which for two inputs $\vx$ and $\vxp$ induces a distance $d(\vx,\vxp)$,[^1] `KMeansPP`[^2] selects the batch via [k-means++ seeding](https://en.wikipedia.org/wiki/K-means%2B%2B).
    That is, the first centroid $\vx_1$ is chosen randomly and the subsequent centroids are chosen with a probability proportional to the square of the distance to the nearest previously selected centroid: \\[ \Pr{\vx_i = \vx} \propto \min_{j < i} d(\vx; \vx_j)^2. \\]

    .. note::

        This acquisition function is similar to [GreedyMaxDist](greedy_max_dist) but selects the batch randomly rather than deterministically.

    `KMeansPP` explicitly enforces *diversity* in the selected batch.
    If the selected centroids from previous batches are used to initialize the centroids for the current batch,[^3] then `KMeansPP` heuristically also leads to *informative* samples since samples are chosen to be different from previously seen data.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | (✅)                | ✅          | embedding / kernel  |

    Using the afsl.embeddings.classification.CrossEntropyEmbedding embeddings, this acquisition function is known as BADGE (*Batch Active learning by Diverse Gradient Embeddings*).[^4]

    [^1]: See afsl.acquisition_functions.distance.DistanceBasedAcquisitionFunction for a discussion of how a distance is induced by embeddings or a kernel.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^3]: see `initialize_with_previous_samples`

    [^4]: Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., and Agarwal, A. Deep batch active learning by diverse, uncertain gradient lower bounds. ICLR, 2020.
    """

    def compute(self, state: DistanceState) -> torch.Tensor:
        if len(state.centroid_indices) == 0:
            # Choose the first centroid randomly
            return torch.ones(state.data.size(0))

        # Compute the distance of each point to the nearest centroid
        sqd_distances = torch.square(self.compute_min_distances(state))
        # Choose the next centroid with a probability proportional to the square of the distance
        probabilities = sqd_distances / sqd_distances.sum()
        return probabilities

    @staticmethod
    def selector(probabilities: torch.Tensor) -> int:
        return int(torch.multinomial(probabilities, num_samples=1).item())
