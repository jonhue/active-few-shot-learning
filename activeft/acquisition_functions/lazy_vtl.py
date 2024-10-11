from typing import List, NamedTuple, Tuple
import numpy as np
import torch
from activeft.acquisition_functions import (
    EmbeddingBased,
    SequentialAcquisitionFunction,
    Targeted,
)
from activeft.gaussian import GaussianCovarianceMatrix
from activeft.model import ModelWithEmbeddingOrKernel
from activeft.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
    PriorityQueue,
)

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
    """
    Lazy Implementation of [VTL](vtl).[^1]

    See Appendix F.2 of [Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08020).

    [^1]: Hübotter, J., Bongni, S., Hakimi, I., and Krause, A. Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs. Preprint, 2024.
    """

    noise_std: float
    """Standard deviation of the noise. Determined automatically if set to `None`."""

    priority_queue: PriorityQueue | None = None
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
        embeddings: torch.Tensor,
        target_embedding: torch.Tensor,
        inner_products: torch.Tensor | None = None,
    ):
        r"""
        Constructs the initial priority queue (of length $k$) over the data set.

        :param embeddings: Array of shape $k \times d$ containing the data point embeddings.
        :param target_embedding: Array of shape $d$ containing the (mean) target embedding.
        :param inner_products: Array of length $k$ containing precomputed (absolute) inner products of the data point embeddings with the query embedding.
        """
        self_inner_products = torch.sum(embeddings * embeddings, dim=1)
        if inner_products is None:
            inner_products = embeddings @ target_embedding
        values = inner_products**2 / (  # target_inner_product -
            self_inner_products + self.noise_var
        )

        self.priority_queue = PriorityQueue(values=values.cpu().tolist())

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
                i, _ = self.priority_queue.pop()
                new_value, new_covariance_matrix, state = self.recompute(state, i)

                prev_top_value = self.priority_queue.top_value
                self.priority_queue.push(i, new_value)
                if (
                    new_value >= prev_top_value
                ):  # done if the value is larger than the largest upper bound of other points
                    break  # (i, new_value) contains the next selected index and its value
            selected_values.append(new_value)
            state = self.step(state, i, new_covariance_matrix)
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

        if self.priority_queue is None:
            self.set_initial_priority_queue(
                embeddings=data,
                target_embedding=torch.mean(target, dim=0),
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
    ) -> Tuple[float, GaussianCovarianceMatrix, LazyVTLState]:
        """
        Update value of data point `data_idx`.
        """
        try:  # index of data point within covariance matrix
            idx = state.m + state.covariance_matrix_indices.index(data_idx)
            new_covariance_matrix = state.covariance_matrix
            new_state = state
        except ValueError:
            # update cached inverse covariance matrix of previously selected data, O(n^2)
            i = state.current_inv.size(0)
            if len(state.selected_indices) > i:
                d = state.data.size(1)
                prev_data = (
                    state.data[
                        torch.tensor(state.selected_indices[:i]).to(state.data.device)
                    ]
                    if i > 0
                    else torch.empty(0, d)
                )  # (i, d)
                new_data = state.data[
                    torch.tensor(state.selected_indices[i:]).to(state.data.device)
                ]  # (n-1-i, d)
                B = prev_data @ new_data.T  # (n-1,)
                I = torch.eye(new_data.size(0)).to(new_data.device)
                C = new_data @ new_data.T + self.noise_var * I  # (n-1-i, n-1-i)
                new_inv = update_inverse(A_inv=state.current_inv, B=B, C=C)
                new_state = LazyVTLState(
                    covariance_matrix=state.covariance_matrix,
                    m=state.m,
                    selected_indices=state.selected_indices,
                    covariance_matrix_indices=state.covariance_matrix_indices,
                    target=state.target,
                    data=state.data,
                    current_inv=new_inv,
                )
            else:
                new_state = state

            # expand the stored covariance matrix if selected data has not been selected before, O(n^2)
            new_covariance_matrix = expand_covariance_matrix(
                covariance_matrix=state.covariance_matrix,
                current_inv=new_state.current_inv,
                data=state.data,
                target=state.target,
                data_idx=data_idx,
                covariance_matrix_indices=state.covariance_matrix_indices,
                selected_indices=state.selected_indices,
            )
            idx = new_covariance_matrix.dim - 1

        value = compute(
            covariance_matrix=new_covariance_matrix,
            idx=idx,
            noise_var=self.noise_var,
            m=state.m,
        )
        return value, new_covariance_matrix, new_state

    def step(
        self,
        state: LazyVTLState,
        data_idx: int,
        new_covariance_matrix: GaussianCovarianceMatrix,
    ) -> LazyVTLState:
        """
        Advances the state.
        Updates the stored covariance matrix and the inverse of the covariance matrix (restricted to selected data).
        """
        if data_idx not in state.covariance_matrix_indices:
            state.covariance_matrix_indices.append(
                data_idx
            )  # Note: not treating as immutable!

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
            current_inv=state.current_inv,
        )

    @property
    def noise_var(self):
        return self.noise_std**2

    def compute(self, state: LazyVTLState) -> torch.Tensor:
        raise NotImplementedError


def compute(
    covariance_matrix: GaussianCovarianceMatrix, idx: int, noise_var: float, m: int
) -> float:
    """
    Computes the acquisition value of the data point at index `idx` within the covariance matrix.

    Time complexity: O(1)
    """

    def engine(i, j):
        return covariance_matrix[i, j] ** 2 / (covariance_matrix[j, j] + noise_var)

    target_indices = torch.arange(m)
    values = engine(target_indices, idx)
    value = torch.sum(values, dim=0).cpu().item()
    return value


def expand_covariance_matrix(
    covariance_matrix: GaussianCovarianceMatrix,
    current_inv: torch.Tensor,
    data: torch.Tensor,
    target: torch.Tensor,
    data_idx: int,
    covariance_matrix_indices: List[int],
    selected_indices: List[int],
) -> GaussianCovarianceMatrix:
    """
    Expands the given covariance matrix with `data_idx`.

    :return: Expanded covariance matrix.

    Time complexity: O(n^2)
    """
    d = data.size(1)
    unique_selected_data = (
        data[torch.tensor(covariance_matrix_indices).to(data.device)]
        if len(covariance_matrix_indices) > 0
        else torch.empty(0, d)
    )  # (n', d)
    selected_data = (
        data[torch.tensor(selected_indices).to(data.device)]
        if len(selected_indices) > 0
        else torch.empty(0, d)
    )  # (n, d)
    new_data = data[data_idx]  # (d,)
    joint_data = torch.cat(
        (target, unique_selected_data, new_data.unsqueeze(0)), dim=0
    )  # (m+n'+1, d)
    I_d = torch.eye(d).to(selected_data.device)
    covariance_vector = (
        joint_data @ (I_d - selected_data.T @ current_inv @ selected_data) @ new_data
    )  # (m+n'+1,)
    assert covariance_vector.size(0) == covariance_matrix.dim + 1
    return covariance_matrix.expand(covariance_vector)


def update_inverse(
    A_inv: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    r"""
    Updates the inverse of $n \times n$ a matrix $A$ after adding a new off-diagonal $n \times m$ block $B$ with diagonal $m \times m$ block $C$.
    Uses the Sherman-Morrison-Woodbury formula.

    Time complexity: O(n^2 + m^3)
    """
    if A_inv.numel() == 0:  # Check if the previous matrix is empty
        return torch.inverse(C)

    D = torch.inverse(C - B.T @ A_inv @ B)
    A_inv_B = A_inv @ B

    upper_left = A_inv + A_inv_B @ D @ A_inv_B.T
    upper_right = -A_inv_B @ D
    lower_left = -D @ A_inv_B.T
    lower_right = D

    # Combine blocks into the full inverse matrix
    upper = torch.cat((upper_left, upper_right), dim=1)
    lower = torch.cat((lower_left, lower_right), dim=1)
    A_inv_extended = torch.cat((upper, lower), dim=0)

    return A_inv_extended
