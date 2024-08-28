from typing import Tuple
from afsl.acquisition_functions import AcquisitionFunction, Targeted
import faiss  # type: ignore
import torch
import concurrent.futures
import numpy as np
from afsl import ActiveDataLoader
from afsl.data import Dataset as AbstractDataset


class Dataset(AbstractDataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


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

    index: faiss.Index  # type: ignore
    only_faiss: bool = False

    def __init__(
        self,
        index: faiss.Index,  # type: ignore
        acquisition_function: AcquisitionFunction,
        only_faiss: bool = False,
        device: torch.device | None = None,
    ):
        self.index = index
        self.acquisition_function = acquisition_function
        self.only_faiss = only_faiss
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def search(
        self,
        query: np.ndarray,
        N: int,
        k: int | None,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        :param query: Query embedding (of shape $m \times d$), comprised of $m$ individual embeddings.
        :param N: Number of results to return.
        :param k: Number of results to pre-sample with Faiss. Does not pre-sample if set to `None`.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of acquisition values (of length $N$), array of selected indices (of length $N$), and array of corresponding embeddings (of shape $N \times d$).
        """
        D, I, V = self.batch_search(
            queries=np.array([query]),
            N=N,
            k=k,
            mean_pooling=mean_pooling,
            threads=threads,
        )
        return D[0], I[0], V[0]

    def batch_search(
        self,
        queries: np.ndarray,
        N: int,
        k: int | None = None,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        :param queries: $n$ query embeddings (of combined shape $n \times m \times d$), each comprised of $m$ individual embeddings.
        :param N: Number of results to return.
        :param k: Number of results to pre-sample with Faiss. Does not pre-sample if set to `None`.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of acquisition values (of shape $n \times N$), array of selected indices (of shape $n \times N$), and array of corresponding embeddings (of shape $n \times N \times d$).
        """
        assert k is None or k >= N

        queries = queries.astype("float32")
        n, m, d = queries.shape
        assert d == self.index.d
        mean_queries = np.mean(queries, axis=1)

        faiss.omp_set_num_threads(threads)  # type: ignore
        D, I, V = self.index.search_and_reconstruct(mean_queries, k or self.index.ntotal)  # type: ignore

        if self.only_faiss:
            return D[:, :N], I[:, :N], V[:, :N]

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
            return np.array(values), np.array(I[i][sub_indexes]), np.array(V[i][sub_indexes])

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
        return np.array(resulting_values), np.array(resulting_indices), np.array(resulting_embeddings)
