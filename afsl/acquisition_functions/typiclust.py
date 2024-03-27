from typing import NamedTuple
import torch
import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
from afsl.acquisition_functions import EmbeddingBased, SequentialAcquisitionFunction
from afsl.model import ModelWithEmbedding
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
)

__all__ = ["TypiClust", "TypiClustState"]

MIN_CLUSTER_SIZE = 5
MAX_NUM_CLUSTERS = 500
K_NN = 20


class TypiClustState(NamedTuple):
    """State of sequential batch selection."""

    i: int
    num_selected: int
    n: int
    clusters_df: pd.DataFrame
    labels: np.ndarray
    features: np.ndarray


class TypiClust(
    EmbeddingBased,
    SequentialAcquisitionFunction[ModelWithEmbedding, TypiClustState],
):
    r"""
    Code adapted from: https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py
    """

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
        SequentialAcquisitionFunction.__init__(
            self,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        EmbeddingBased.__init__(self, embedding_batch_size=embedding_batch_size)

    def initialize(
        self,
        model: ModelWithEmbedding | None,
        data: torch.Tensor,
        selected_data: torch.Tensor | None,
        batch_size: int,
    ) -> TypiClustState:
        num_selected = selected_data.size(0) if selected_data is not None else 0
        num_clusters = min(num_selected + batch_size, MAX_NUM_CLUSTERS)
        print(f"Clustering into {num_clusters} clusters.")
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
        clusters = kmeans(features=features, num_clusters=num_clusters)

        # using only labeled+unlabeled indices, without validation set.
        labels = np.copy(clusters)  # type: ignore

        # FIXED BUG: cluster_labeled_counts not matching cluster_ids when there are missing cluster numbers
        id_mapping = {
            original_id: new_id for new_id, original_id in enumerate(np.unique(labels))
        }
        labels = np.vectorize(id_mapping.get)(labels)

        existing_indices = np.arange(num_selected)
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)

        cluster_labeled_counts = np.bincount(
            labels[existing_indices], minlength=len(cluster_ids)
        )
        assert len(cluster_ids) == len(cluster_sizes)
        assert len(cluster_ids) == len(cluster_labeled_counts)
        clusters_df = pd.DataFrame(
            {
                "cluster_id": cluster_ids,
                "cluster_size": cluster_sizes,
                "existing_count": cluster_labeled_counts,
                "neg_cluster_size": -1 * cluster_sizes,
            }
        )
        # drop too small clusters, FIXED BUG: prevented zero remaining clusters
        if len(clusters_df[clusters_df.cluster_size > MIN_CLUSTER_SIZE]) > 0:
            clusters_df = clusters_df[clusters_df.cluster_size > MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(["existing_count", "neg_cluster_size"])  # type: ignore
        labels[existing_indices] = -1

        return TypiClustState(
            i=0,
            num_selected=num_selected,
            n=data.size(0),
            clusters_df=clusters_df,
            labels=labels,
            features=features,
        )

    def compute(self, state: TypiClustState) -> torch.Tensor:
        cluster = state.clusters_df.iloc[state.i % len(state.clusters_df)].cluster_id
        indices = (state.labels == cluster).nonzero()[0]
        rel_feats = state.features[indices]
        typicality = torch.zeros(state.n)
        # in case we have too small cluster, calculate density among half of the cluster
        typicality[indices] = torch.tensor(
            calculate_typicality(rel_feats, min(K_NN, len(indices) // 2))
        )
        # remove already selected samples
        typicality = typicality[state.num_selected:]
        assert typicality.size(0) == state.n - state.num_selected
        return typicality

    def step(self, state: TypiClustState, i: int) -> TypiClustState:
        state.labels[i] = -1
        return TypiClustState(
            i=state.i + 1,
            num_selected=state.num_selected,
            n=state.n,
            clusters_df=state.clusters_df,
            labels=state.labels,
            features=state.features,
        )


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    cpu_index.add(features)  # type: ignore
    distances, indices = cpu_index.search(features, num_neighbors + 1)  # type: ignore
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)  # type: ignore
    return typicality
