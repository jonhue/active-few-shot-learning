from typing import List, NamedTuple
import torch
from afsl.acquisition_functions import SequentialAcquisitionFunction
from afsl.model import ModelWithEmbedding
from afsl.utils import compute_embedding


class BADGEState(NamedTuple):
    embeddings: torch.Tensor
    centroid_indices: List[torch.Tensor]


def compute_distances(embeddings, centroids):
    # Compute the distance of all points in embeddings from each centroid
    distances = torch.cdist(embeddings, centroids, p=2)
    # Return the minimum distance for each point
    min_distances = torch.min(distances, dim=1).values
    return min_distances


class BADGE(SequentialAcquisitionFunction[ModelWithEmbedding, BADGEState]):
    def initialize(
        self,
        model: ModelWithEmbedding,
        data: torch.Tensor,
    ) -> BADGEState:
        embeddings = compute_embedding(
            model, data, mini_batch_size=self.mini_batch_size
        )
        # Choose the first centroid randomly
        centroid_indices = [
            torch.randint(0, embeddings.size(0), (1,)).to(embeddings.device)
        ]
        return BADGEState(embeddings=embeddings, centroid_indices=centroid_indices)

    def step(self, state: BADGEState, i: int) -> BADGEState:
        state.centroid_indices.append(torch.tensor(i).to(state.embeddings.device))
        return state

    def compute(self, state: BADGEState) -> torch.Tensor:
        # Compute the distance of each point to the nearest centroid
        centroids = state.embeddings[
            torch.cat(state.centroid_indices).to(state.embeddings.device)
        ]
        sqd_distances = torch.square(compute_distances(state.embeddings, centroids))
        # Choose the next centroid with a probability proportional to the square of the distance
        probabilities = sqd_distances / sqd_distances.sum()
        return probabilities

    @staticmethod
    def selector(probabilities: torch.Tensor) -> int:
        return int(torch.multinomial(probabilities, num_samples=1).item())
