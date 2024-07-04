import faiss
import torch
import concurrent.futures
import numpy as np
from afsl import ActiveDataLoader
from afsl.data import Dataset as AbstractDataset
from torch.utils.data import Dataset as TorchDataset

from examples.acquisition_functions import get_acquisition_function


class Dataset(AbstractDataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


class ITLSearcher:
    """
    Adapter for the [Faiss](https://github.com/facebookresearch/faiss) library.
    First preselects a large number of candidates using Faiss, and then uses ITL to select the final results.

    `ITLSearcher` can be used as a wrapper around a Faiss index object:

    ```python
    d = 768  # Dimensionality of the embeddings
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    itl = ITLSearcher(index)
    indices = itl.search(query_embeddings, k=10)
    ```
    """

    index: faiss.Index  # type: ignore
    force_nonsequential: bool = False
    skip_itl: bool = False

    def __init__(
        self,
        index: faiss.Index,  # type: ignore
        alg: str,
        noise_std: float,
        force_nonsequential: bool = False,
        skip_itl: bool = False,
    ):
        self.index = index
        self.alg = alg
        self.noise_std = noise_std
        self.force_nonsequential = force_nonsequential
        self.skip_itl = skip_itl

    def search(
        self,
        query: np.ndarray,
        k: int,
        k_mult: float = 100.0,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> np.ndarray:
        r"""
        :param query: Query embedding (of shape $m \times d$), comprised of $m$ individual embeddings.
        :param k: Number of results to return.
        :param k_mult: `k * k_mult` is the number of results to pre-sample before executing ITL.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of selected indices (of length $k$).
        """
        return self.batch_search(
            queries=np.array([query]),
            k=k,
            k_mult=k_mult,
            mean_pooling=mean_pooling,
            threads=threads,
        )[0]

    def batch_search(
        self,
        queries: np.ndarray,
        k: int,
        k_mult: float = 100.0,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> np.ndarray:
        r"""
        :param queries: $n$ query embeddings (of combined shape $n \times m \times d$), each comprised of $m$ individual embeddings.
        :param k: Number of results to return.
        :param k_mult: `k * k_mult` is the number of results to pre-sample before executing ITL.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of selected indices (of shape $n \times k$).
        """
        assert k_mult > 1

        queries = queries.astype("float32")
        n, m, d = queries.shape
        assert d == self.index.d
        mean_queries = np.mean(queries, axis=1)

        if self.alg == "Random":
            indices = np.arange(self.index.ntotal)
            subsets = np.empty((n, k), dtype=int)
            for i in range(n):
                subsets[i] = np.random.choice(indices, k, replace=False)
            return subsets

        faiss.omp_set_num_threads(threads)  # type: ignore
        D, I, V = self.index.search_and_reconstruct(mean_queries, int(k * k_mult))

        if self.skip_itl:
            return I[:, :k]

        def engine(i: int) -> np.ndarray:
            dataset = Dataset(torch.tensor(V[i]))
            target = torch.tensor(
                queries[i] if not mean_pooling else mean_queries[i].reshape(1, -1)
            )
            acquisition_function = get_acquisition_function(
                alg=self.alg,
                target=target,
                noise_std=self.noise_std,
                num_workers=threads,
                subsample_acquisition=False,
                subsampled_target_frac=1.0,
                max_target_size=None,
                # force_nonsequential=self.force_nonsequential,
                mini_batch_size=10_000,  # TODO: this parameter is unused
            )
            sub_indexes = ActiveDataLoader(
                dataset=dataset,
                batch_size=k,
                acquisition_function=acquisition_function,
            ).next()
            return np.array(I[i][sub_indexes])

        result = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(engine, i))
            for future in concurrent.futures.as_completed(futures):
                result.append(future.result())
        return np.array(result)
