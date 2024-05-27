import faiss
import torch
import concurrent.futures
import numpy as np
from afsl import ActiveDataLoader
from afsl.acquisition_functions.itl import ITL
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset[torch.Tensor]):
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

    index: faiss.Index
    force_nonsequential: bool = False
    skip_itl: bool = False

    def __init__(
        self,
        index: faiss.Index,
        force_nonsequential: bool = False,
        skip_itl: bool = False,
    ):
        self.index = index
        self.force_nonsequential = force_nonsequential
        self.skip_itl = skip_itl

    def search(
        self,
        query: np.ndarray,
        k: int,
        k_mult: float = 100.0,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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

        faiss.omp_set_num_threads(threads)
        D, I, V = self.index.search_and_reconstruct(mean_queries, int(k * k_mult))

        if self.skip_itl:
            result = []
            for i, v in zip(I, V):
                result.append((np.array(i[:k]), np.array(v[:k])))
            return result

        def engine(i: int) -> tuple[np.ndarray, np.ndarray]:
            dataset = Dataset(torch.tensor(V[i]))
            target = torch.tensor(
                queries[i] if not mean_pooling else mean_queries[i].reshape(1, -1)
            )
            acquisition_function = ITL(
                target=target,
                num_workers=threads,
                subsample=False,
                force_nonsequential=self.force_nonsequential,
            )
            sub_indexes = ActiveDataLoader(
                dataset=dataset,
                batch_size=k,
                acquisition_function=acquisition_function,
            ).next()
            return np.array(I[i][sub_indexes]), np.array(V[i][sub_indexes])

        result = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(engine, i))
            for future in concurrent.futures.as_completed(futures):
                result.append(future.result())
        return result
    