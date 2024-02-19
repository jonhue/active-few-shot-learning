from typing import List, NamedTuple
import torch
from afsl.acquisition_functions import SequentialAcquisitionFunction
from afsl.acquisition_functions.badge import compute_distances
from afsl.embeddings import M, Embedding
from afsl.types import Target


class GreedyMaxDistState(NamedTuple):
    embeddings: torch.Tensor
    centroid_indices: List[torch.Tensor]


class GreedyMaxDist(SequentialAcquisitionFunction[GreedyMaxDistState]):
    def initialize(
        self,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> GreedyMaxDistState:
        embeddings = embedding.embed(model, data)
        # Choose the first centroid randomly
        centroid_indices = [
            torch.randint(0, embeddings.size(0), (1,)).to(embeddings.device)
        ]
        return GreedyMaxDistState(
            embeddings=embeddings, centroid_indices=centroid_indices
        )

    def step(self, state: GreedyMaxDistState, i: int) -> GreedyMaxDistState:
        state.centroid_indices.append(torch.tensor(i).to(state.embeddings.device))
        return state

    def compute(self, state: GreedyMaxDistState) -> torch.Tensor:
        # Compute the distance of each point to the nearest centroid
        centroids = state.embeddings[
            torch.cat(state.centroid_indices).to(state.embeddings.device)
        ]
        distances = compute_distances(state.embeddings, centroids)
        return distances
