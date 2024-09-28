from typing import NamedTuple, Tuple
from warnings import warn
from activeft.acquisition_functions import AcquisitionFunction, Targeted
from activeft.acquisition_functions.lazy_vtl import LazyVTL
from activeft.acquisition_functions.vtl import VTL
import faiss
import torch
import time
import concurrent.futures
import numpy as np
from activeft import ActiveDataLoader
from activeft.data import Dataset as AbstractDataset


class Dataset(AbstractDataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


class RetrievalTime(NamedTuple):
    faiss: float
    """Time spent with Faiss retrieval."""
    sift: float
    """Additional time spent with SIFT."""


class Retriever:
    """
    Adapter for the [Faiss](https://github.com/facebookresearch/faiss) library.
    First preselects a large number of candidates using Faiss, and then uses VTL to select the final results.

    `Retriever` can be used as a wrapper around a Faiss index object:

    ```python
    d = 768  # Dimensionality of the embeddings
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    retriever = Retriever(index, acquisition_function)
    indices = retriever.search(query_embeddings, N=10)
    ```
    """

    index: faiss.Index
    only_faiss: bool = False

    def __init__(
        self,
        index: faiss.Index,
        acquisition_function: AcquisitionFunction | None = None,
        llambda: float = 0.01,
        fast: bool = False,
        also_query_opposite: bool = False,
        only_faiss: bool = False,
        device: torch.device | None = None,
    ):
        """
        :param index: Faiss index object.
        :param acquisition_function: Acquisition function object.
        :param llambda: Value of the lambda parameter of SIFT. Ignored if `acquisition_function` is set.
        :param fast: Whether to use the SIFT-Fast. Ignored if `acquisition_function` is set.
        :param also_query_opposite: If using an inner product index, setting this to `True` will also query the opposite of the query embeddings, pre-selecting points with high *absolute* inner product.
        :param only_faiss: Whether to only use Faiss for search.
        :param device: Device to use for computation.
        """
        self.index = index
        self.also_query_opposite = also_query_opposite
        self.only_faiss = only_faiss
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if acquisition_function is not None:
            self.acquisition_function = acquisition_function
            if fast:
                warn(
                    "Ignoring `fast` parameter, as `acquisition_function` is set.",
                    UserWarning,
                )
        elif fast:
            assert llambda is not None
            self.acquisition_function = LazyVTL(
                target=torch.Tensor(),
                num_workers=1,
                subsample=False,
                noise_std=np.sqrt(llambda),
            )
        else:
            assert llambda is not None
            self.acquisition_function = VTL(
                target=torch.Tensor(),
                num_workers=1,
                subsample=False,
                force_nonsequential=False,
                noise_std=np.sqrt(llambda),
            )

    def search(
        self,
        query: np.ndarray,
        N: int,
        K: int | None,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RetrievalTime]:
        r"""
        :param query: Query embedding (of shape $m \times d$), comprised of $m$ individual embeddings.
        :param N: Number of results to return.
        :param K: Number of results to pre-sample with Faiss. Does not pre-sample if set to `None`.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of acquisition values (of length $N$), array of selected indices (of length $N$), array of corresponding embeddings (of shape $N \times d$), retrieval time.
        """
        D, I, V, retrieval_time = self.batch_search(
            queries=np.array([query]),
            N=N,
            K=K,
            mean_pooling=mean_pooling,
            threads=threads,
        )
        return D[0], I[0], V[0], retrieval_time

    def batch_search(
        self,
        queries: np.ndarray,
        N: int,
        K: int | None = None,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RetrievalTime]:
        r"""
        :param queries: $n$ query embeddings (of combined shape $n \times m \times d$), each comprised of $m$ individual embeddings.
        :param N: Number of results to return.
        :param K: Number of results to pre-sample with Faiss. Does not pre-sample if set to `None`.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of acquisition values (of shape $n \times N$), array of selected indices (of shape $n \times N$), array of corresponding embeddings (of shape $n \times N \times d$), retrieval time.
        """
        queries = queries.astype("float32")
        n, m, d = queries.shape
        assert d == self.index.d
        mean_queries = np.mean(queries, axis=1)

        k = K or self.index.ntotal
        t_start = time.time()
        faiss.omp_set_num_threads(threads)
        D, I, V = self.index.search_and_reconstruct(mean_queries, k)  # type: ignore
        if self.also_query_opposite:
            assert (
                self.index.metric_type == faiss.METRIC_INNER_PRODUCT
            ), "`also_query_opposite` should only be used with inner product indexes."
            D_, I_, V_ = self.index.search_and_reconstruct(-mean_queries, k)  # type: ignore
            D__, I__, V__ = (
                np.concatenate([D, D_]),
                np.concatenate([I, I_]),
                np.concatenate([V, V_]),
            )
            sorted_indices = np.argsort(-D__)[:k]
            D, I, V = D__[sorted_indices], I__[sorted_indices], V__[sorted_indices]
        t_faiss = time.time() - t_start

        if self.only_faiss:
            retrieval_time = RetrievalTime(faiss=t_faiss, sift=0)
            return D[:, :N], I[:, :N], V[:, :N], retrieval_time

        def engine(i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            dataset = Dataset(torch.tensor(V[i]))
            target = torch.tensor(
                queries[i] if not mean_pooling else mean_queries[i].reshape(1, -1)
            )

            if isinstance(self.acquisition_function, Targeted):
                self.acquisition_function.set_target(target)

            sub_indexes, values = ActiveDataLoader(
                dataset=dataset,
                batch_size=N,
                acquisition_function=self.acquisition_function,
                device=self.device,
            ).next()
            return (
                np.array(values),
                np.array(I[i][sub_indexes]),
                np.array(V[i][sub_indexes]),
            )

        t_start = time.time()
        resulting_values = []
        resulting_indices = []
        resulting_embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(engine, i))
            for future in concurrent.futures.as_completed(futures):
                values, indices, embeddings = future.result()
                resulting_values.append(values)
                resulting_indices.append(indices)
                resulting_embeddings.append(embeddings)
        t_sift = time.time() - t_start
        retrieval_time = RetrievalTime(faiss=t_faiss, sift=t_sift)
        return (
            np.array(resulting_values),
            np.array(resulting_indices),
            np.array(resulting_embeddings),
            retrieval_time,
        )
