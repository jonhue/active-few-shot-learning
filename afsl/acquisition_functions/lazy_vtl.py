from typing import List, NamedTuple, Tuple
import numpy as np
import torch
from afsl.acquisition_functions import (
    EmbeddingBased,
    SequentialAcquisitionFunction,
    Targeted,
)
from afsl.gaussian import GaussianCovarianceMatrix, get_jitter
from afsl.model import (
    ModelWithEmbeddingOrKernel,
    ModelWithKernel,
    ModelWithLatentCovariance,
)
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
    PriorityQueue,
    get_device,
)

Cache = Tuple[GaussianCovarianceMatrix, torch.Tensor]

__all__ = ["LazyVTL", "LazyVTLState"]


class LazyVTLState(NamedTuple):
    """State of lazy VTL."""

    covariance_matrix: GaussianCovarianceMatrix
    r"""Kernel matrix of the data. Tensor of shape $n \times n$."""
    m: int
    """Size of the target space."""
    selected_indices: List[int]
    """Indices of points that were already observed."""
    covariance_matrix_indices: List[int]
    """Indices of points that were added to the covariance matrix (excluding the initially added target space)."""
    target: torch.Tensor
    r"""Tensor of shape $m \times d$ which includes target space."""
    data: torch.Tensor
    r"""Tensor of shape $n \times d$ which includes sample space."""
    current_inv: torch.Tensor
    """Current inverse of the covariance matrix of selected data."""


class LazyVTL(
    Targeted,
    EmbeddingBased,
    SequentialAcquisitionFunction[ModelWithEmbeddingOrKernel | None, LazyVTLState],
):
    noise_std: float
    """Standard deviation of the noise. Determined automatically if set to `None`."""

    priority_queue: PriorityQueue | None
    """Priority queue over the data set."""

    def __init__(
        self,
        target: torch.Tensor,
        noise_std: float,
        subsampled_target_frac: float = 1,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
    ):
        """
        :param target: Tensor of prediction targets (shape $m \times d$).
        :param noise_std: Standard deviation of the noise.
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $1$. Ignored if `target` is `None`.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large. Ignored if `target` is `None`.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        """
        SequentialAcquisitionFunction.__init__(
            self,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=False,
        )
        Targeted.__init__(
            self,
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
        )
        EmbeddingBased.__init__(self, embedding_batch_size=embedding_batch_size)
        self.noise_std = noise_std

    def set_initial_priority_queue(
        self,
        indices: np.ndarray,
        embeddings: np.ndarray,
        target_embedding: np.ndarray,
        inner_products: np.ndarray | None = None,
    ):
        r"""
        Constructs the initial priority queue (of length $k$) over the data set.

        :param indices: Array of length $k$ containing indices of the data points.
        :param embeddings: Array of shape $k \times d$ containing the data point embeddings.
        :param target_embedding: Array of shape $d$ containing the (mean) target embedding.
        :param inner_products: Array of length $k$ containing precomputed (absolute) inner products of the data point embeddings with the query embedding.
        """
        self_inner_products = np.sum(embeddings * embeddings, axis=1)
        target_inner_product = target_embedding @ target_embedding
        if inner_products is None:
            inner_products = embeddings @ target_embedding
        values = target_inner_product - inner_products**2 / (
            self_inner_products + self.noise_var
        )

        self.priority_queue = PriorityQueue(
            indices=indices.tolist(), values=values.tolist()
        )

    def select_from_minibatch(
        self,
        batch_size: int,
        model: ModelWithEmbeddingOrKernel | None,
        data: torch.Tensor,
        device: torch.device | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Selects the next batch from the given mini batch `data`.

        :param batch_size: Size of the batch to be selected. Needs to be smaller than `mini_batch_size`.
        :param model: Model used for data selection.
        :param data: Mini batch of inputs (shape $n \times d$) to be selected from.
        :param device: Device used for computation of the acquisition function.
        :return: Indices of the newly selected batch (with respect to mini batch) and corresponding values of the acquisition function.
        """
        state = self.initialize(model, data, device)

        assert (
            self.priority_queue is not None
        ), "Initial priority queue must be set."  # TODO: make optional, compute from scratch if not provided
        assert self.priority_queue.size() == data.size(
            0
        ), "Size of the priority queue must match the size of the data set."

        selected_values = []
        for _ in range(batch_size):
            while True:
                stopping_value = np.inf
                i, value = self.priority_queue.pop()
                if (
                    value >= stopping_value
                ):  # done if the value is larger than a previously updated value
                    break  # (i, value) contains the next selected index and its value

                new_value, cache = self.recompute(state, i)

                stopping_value = np.minimum(
                    stopping_value, new_value
                )  # stores the smallest updated value
                self.priority_queue.push(i, new_value)
            selected_values.append(value)
            state = self.step(state, i, cache)
        return torch.tensor(state.selected_indices), torch.tensor(selected_values)

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel | None,
        data: torch.Tensor,
        device: torch.device | None,
    ) -> LazyVTLState:
        target = self.get_target()
        m = target.size(0)

        # Compute covariance matrix of targets
        assert model is None, "embedding computation via model not supported"
        covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
            Embeddings=target.to(device)
        )

        return LazyVTLState(
            covariance_matrix=covariance_matrix,
            m=m,
            selected_indices=[],
            covariance_matrix_indices=[],
            target=target,
            data=data,
            current_inv=torch.empty(0, 0),
        )

    def recompute(
        self, state: LazyVTLState, data_idx: int
    ) -> Tuple[float, Cache | None]:
        """
        Update value of data point `data_idx`.
        """
        try:  # index of data point within covariance matrix
            idx = state.m + state.covariance_matrix_indices.index(data_idx)
            new_covariance_matrix = state.covariance_matrix
            cache = None
        except (
            ValueError
        ):  # expand the stored covariance matrix if selected data has not been selected before, O(n^2)
            cache = expand_covariance_matrix(
                covariance_matrix=state.covariance_matrix,
                current_inv=state.current_inv,
                data=state.data,
                target=state.target,
                data_idx=data_idx,
                covariance_matrix_indices=state.covariance_matrix_indices,
                selected_indices=state.selected_indices,
            )
            new_covariance_matrix, _ = cache
            idx = new_covariance_matrix.dim - 1
        value = compute(
            covariance_matrix=new_covariance_matrix,
            idx=idx,
            noise_var=self.noise_var,
            m=state.m,
        )

        return value, cache

    def step(
        self, state: LazyVTLState, data_idx: int, cache: Cache | None
    ) -> LazyVTLState:
        """
        Advances the state.
        Updates the stored covariance matrix and the inverse of the covariance matrix (restricted to selected data).
        """
        # update cached inverse covariance matrix of selected data, O(n^2)
        if data_idx not in state.covariance_matrix_indices:
            assert cache is not None
            new_covariance_matrix, covariance_vector = cache
            state.covariance_matrix_indices.append(
                data_idx
            )  # Note: not treating as immutable!
            u = covariance_vector[:-1]
            alpha = covariance_vector[-1].item()
        else:
            new_covariance_matrix = state.covariance_matrix
            idx = state.m + state.covariance_matrix_indices.index(data_idx)
            covariance_vector = state.covariance_matrix._matrix[idx]
            u = torch.cat((covariance_vector[:idx], covariance_vector[idx + 1 :]))
            alpha = covariance_vector[idx].item()
        new_inv = update_inverse(
            A_inv=state.current_inv,
            u=u,
            alpha=alpha + self.noise_var,
        )

        # update the stored covariance matrix by conditioning on the new data point, O(n^2)
        idx = state.m + state.covariance_matrix_indices.index(data_idx)
        posterior_covariance_matrix = new_covariance_matrix.condition_on(
            idx, noise_std=self.noise_std
        )

        state.selected_indices.append(data_idx)  # Note: not treating as immutable!
        return LazyVTLState(
            covariance_matrix=posterior_covariance_matrix,
            m=state.m,
            selected_indices=state.selected_indices,
            covariance_matrix_indices=state.covariance_matrix_indices,
            target=state.target,
            data=state.data,
            current_inv=new_inv,
        )

    @property
    def noise_var(self):
        return self.noise_std**2


def compute(
    covariance_matrix: GaussianCovarianceMatrix, idx: int, noise_var: float, m: int
) -> float:
    """
    Computes the acquisition value of the data point at index `idx` within the covariance matrix.

    Time complexity: O(1)
    """

    def compute_posterior_variance(i, j):
        return covariance_matrix[i, i] - covariance_matrix[i, j] ** 2 / (
            covariance_matrix[j, j] + noise_var
        )

    target_indices = torch.arange(m).unsqueeze(0)  # Expand dims for broadcasting
    posterior_variances = compute_posterior_variance(target_indices, idx)
    total_posterior_variance = torch.sum(posterior_variances, dim=1).cpu().item()
    return total_posterior_variance


def expand_covariance_matrix(
    covariance_matrix: GaussianCovarianceMatrix,
    current_inv: torch.Tensor,
    data: torch.Tensor,
    target: torch.Tensor,
    data_idx: int,
    covariance_matrix_indices: List[int],
    selected_indices: List[int],
) -> Tuple[GaussianCovarianceMatrix, torch.Tensor]:
    """
    Expands the given covariance matrix with `data_idx`.

    :return: Expanded covariance matrix and covariance vector (i.e., the final row/column of the expanded covariance matrix).

    Time complexity: O(n^2)
    """
    unique_selected_data = data[
        torch.tensor(covariance_matrix_indices).to(data.device)
    ]  # (n', d)
    selected_data = data[torch.tensor(selected_indices).to(data.device)]  # (n, d)
    new_data = data[data_idx]  # (d,)
    joint_data = torch.cat(
        (target, unique_selected_data, new_data.unsqueeze(0)), dim=0
    )  # (m+n'+1, d)
    I_d = torch.eye(selected_data.size(1)).to(selected_data.device)
    covariance_vector = (
        joint_data @ (I_d - selected_data.T @ current_inv @ selected_data) @ new_data
    )  # (m+n'+1,)
    assert covariance_vector.size(0) == covariance_matrix.dim + 1
    return covariance_matrix.expand(covariance_vector), covariance_vector


def update_inverse(A_inv: torch.Tensor, u: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""
    Updates the inverse of $n \times n$ a matrix $A$ after adding a new column $u$ with a given diagonal element $\alpha$.
    Uses the Sherman-Morrison-Woodbury formula.

    Time complexity: O(n^2)
    """
    if A_inv.numel() == 0:  # Check if the previous matrix is empty
        return torch.tensor([[1.0 / alpha]])

    u = u.view(-1, 1)  # Ensure u is a column vector
    S = alpha - torch.matmul(torch.matmul(u.t(), A_inv), u)
    S_inv = 1.0 / S

    A_inv_u = torch.matmul(A_inv, u)
    upper_left = A_inv + torch.matmul(A_inv_u, A_inv_u.t()) * S_inv
    upper_right = -A_inv_u * S_inv
    lower_left = -A_inv_u.t() * S_inv
    lower_right = S_inv

    # Combine blocks into the full inverse matrix
    upper = torch.cat((upper_left, upper_right), dim=1)
    lower = torch.cat((lower_left, lower_right), dim=1)
    A_inv_extended = torch.cat((upper, lower), dim=0)

    return A_inv_extended
