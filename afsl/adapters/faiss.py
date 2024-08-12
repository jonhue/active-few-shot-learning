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
    indices = retriever.search(query_embeddings, k=10)
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
        k: int,
        k_mult: float = 100.0,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        :param query: Query embedding (of shape $m \times d$), comprised of $m$ individual embeddings.
        :param k: Number of results to return.
        :param k_mult: `k * k_mult` is the number of results to pre-sample before executing VTL.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of selected indices (of length $k$) and array of corresponding embeddings (of shape $k \times d$).
        """
        I, V = self.batch_search(
            queries=np.array([query]),
            k=k,
            k_mult=k_mult,
            mean_pooling=mean_pooling,
            threads=threads,
        )
        return I[0], V[0]

    def batch_search(
        self,
        queries: np.ndarray,
        k: int,
        k_mult: float = 100.0,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        :param queries: $n$ query embeddings (of combined shape $n \times m \times d$), each comprised of $m$ individual embeddings.
        :param k: Number of results to return.
        :param k_mult: `k * k_mult` is the number of results to pre-sample before executing VTL.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of selected indices (of shape $n \times k$) and array of corresponding embeddings (of shape $n \times k \times d$).
        """
        assert k_mult > 1

        queries = queries.astype("float32")
        n, m, d = queries.shape
        assert d == self.index.d
        mean_queries = np.mean(queries, axis=1)

        faiss.omp_set_num_threads(threads)  # type: ignore
        D, I, V = self.index.search_and_reconstruct(mean_queries, int(k * k_mult))  # type: ignore

        if self.only_faiss:
            return I[:, :k], V[:, :k]

        def engine(i: int) -> Tuple[np.ndarray, np.ndarray]:
            dataset = Dataset(torch.tensor(V[i]))
            target = torch.tensor(
                queries[i] if not mean_pooling else mean_queries[i].reshape(1, -1)
            )

            if isinstance(self.acquisition_function, Targeted):
                self.acquisition_function.set_target(target)

            sub_indexes = ActiveDataLoader(
                dataset=dataset,
                batch_size=k,
                acquisition_function=self.acquisition_function,
                device=self.device,
            ).next()
            return np.array(I[i][sub_indexes]), np.array(V[i][sub_indexes])

        resulting_indices = []
        resulting_values = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(engine, i))
            for future in concurrent.futures.as_completed(futures):
                indices, values = future.result()
                resulting_indices.append(indices)
                resulting_values.append(values)
        return np.array(resulting_indices), np.array(resulting_values)
