from typing import NamedTuple
import torch
import numpy as np
import pandas as pd
from afsl.acquisition_functions import EmbeddingBased, SequentialAcquisitionFunction
from afsl.model import ModelWithEmbedding
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
)

__all__ = ["ProbCover", "ProbCoverState"]


class ProbCoverState(NamedTuple):
    """State of sequential batch selection."""

    i: int
    n: int
    num_selected: int
    cur_df: pd.DataFrame
    covered_samples: np.ndarray


class ProbCover(
    EmbeddingBased,
    SequentialAcquisitionFunction[ModelWithEmbedding, ProbCoverState],
):
    r"""
    Code adapted from: https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/prob_cover.py
    """

    def __init__(
        self,
        delta: float,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        """
        :param delta: Distance threshold for constructing the graph.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param num_workers: Number of workers used for parallelizing the computation of the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        SequentialAcquisitionFunction.__init__(
            self,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        EmbeddingBased.__init__(self, embedding_batch_size=embedding_batch_size)
        self.delta = delta

    def initialize(
        self,
        model: ModelWithEmbedding | None,
        data: torch.Tensor,
        selected_data: torch.Tensor | None,
        batch_size: int,
    ) -> ProbCoverState:
        num_selected = selected_data.size(0) if selected_data is not None else 0
        data = (
            torch.cat([selected_data, data], dim=0)
            if selected_data is not None
            else data
        )
        features = (
            self.compute_embedding(
                model=model, data=data, batch_size=self.embedding_batch_size
            )
            .cpu()
            .numpy()
        )
        graph_df = construct_graph(features=features, delta=self.delta)

        edge_from_seen = np.isin(graph_df.x, np.arange(num_selected))
        covered_samples = graph_df.y[edge_from_seen].unique()
        cur_df = graph_df[(~np.isin(graph_df.y, covered_samples))]

        return ProbCoverState(
            i=0,
            num_selected=num_selected,
            n=data.size(0),
            cur_df=cur_df,  # type: ignore
            covered_samples=covered_samples,
        )

    def compute(self, state: ProbCoverState) -> torch.Tensor:
        coverage = len(state.covered_samples) / state.n
        # selecting the sample with the highest degree
        degrees = torch.tensor(np.bincount(state.cur_df.x, minlength=state.n))
        # remove already selected samples
        degrees = degrees[state.num_selected:]
        print(
            f"Iteration is {state.i}.\tGraph has {len(state.cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}"
        )
        assert degrees.size(0) == state.n - state.num_selected
        return degrees

    def step(self, state: ProbCoverState, i: int) -> ProbCoverState:
        # shift i
        i = i + state.num_selected
        # removing incoming edges to newly covered samples
        new_covered_samples = state.cur_df.y[(state.cur_df.x == i)].values
        assert len(np.intersect1d(state.covered_samples, new_covered_samples)) == 0, "all samples should be new"  # type: ignore
        cur_df = state.cur_df[(~np.isin(state.cur_df.y, new_covered_samples))]  # type: ignore
        covered_samples = np.concatenate([state.covered_samples, new_covered_samples])  # type: ignore
        return ProbCoverState(
            i=state.i + 1,
            num_selected=state.num_selected,
            n=state.n,
            cur_df=cur_df,  # type: ignore
            covered_samples=covered_samples,
        )


def construct_graph(features, delta, batch_size=500):
    """
    creates a directed graph where:
    x->y iff l2(x,y) < delta.

    represented by a list of edges (a sparse matrix).
    stored in a dataframe
    """
    xs, ys, ds = [], [], []
    print(f"Start constructing graph using delta={delta}")
    # distance computations are done in GPU
    cuda_feats = torch.tensor(features).cuda()
    for i in range(len(features) // batch_size + 1): # FIXED BUG: added +1
        # distance comparisons are done in batches to reduce memory consumption
        cur_feats = cuda_feats[i * batch_size : (i + 1) * batch_size]
        dist = torch.cdist(cur_feats, cuda_feats)
        mask = dist < delta
        # saving edges using indices list - saves memory.
        x, y = mask.nonzero().T
        xs.append(x.cpu() + batch_size * i)
        ys.append(y.cpu())
        ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({"x": xs, "y": ys, "d": ds})
    print(f"Finished constructing graph using delta={delta}")
    print(f"Graph contains {len(df)} edges.")
    return df
