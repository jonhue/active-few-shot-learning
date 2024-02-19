from typing import List, NamedTuple
import torch
from afsl.acquisition_functions import SequentialAcquisitionFunction
from afsl.embeddings import M, Embedding
from afsl.types import Target
from afsl.utils import mini_batch_wrapper_non_cat


class BADGEState(NamedTuple):
    embeddings: torch.Tensor
    centroid_indices: List[torch.Tensor]


def compute_distances(embeddings, centroids):
    # Compute the distance of all points in embeddings from each centroid
    distances = torch.cdist(embeddings, centroids, p=2)
    # Return the minimum distance for each point
    min_distances = torch.min(distances, dim=1).values
    return min_distances


class BADGE(SequentialAcquisitionFunction[BADGEState]):
    def initialize(
        self,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
    ) -> BADGEState:
        embeddings = embedding.embed(model, data)
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

    def select(
        self,
        batch_size: int,
        embedding: Embedding[M],
        model: M,
        data: torch.Tensor,
        target: Target,
        Sigma: torch.Tensor | None = None,
        force_nonsequential=False,
    ) -> torch.Tensor:
        assert not force_nonsequential, "Non-sequential selection is not supported"

        states = mini_batch_wrapper_non_cat(
            fn=lambda batch: self.initialize(
                embedding=embedding,
                model=model,
                data=batch,
                target=target,
                Sigma=Sigma,
            ),
            data=data,
            batch_size=self.mini_batch_size,
        )

        indices = []
        for _ in range(batch_size):
            probabilities = torch.cat([self.compute(state) for state in states], dim=0)
            i = int(torch.multinomial(probabilities, num_samples=1).item())
            indices.append(i)
            states = [self.step(state, i) for state in states]
        return torch.tensor(indices)
