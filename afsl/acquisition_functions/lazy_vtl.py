from typing import List, NamedTuple, Tuple
import numpy as np
import torch
from afsl.acquisition_functions import EmbeddingBased, SequentialAcquisitionFunction, Targeted
from afsl.gaussian import GaussianCovarianceMatrix, get_jitter
from afsl.model import ModelWithEmbeddingOrKernel, ModelWithKernel, ModelWithLatentCovariance
from afsl.utils import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_MINI_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEFAULT_SUBSAMPLE, PriorityQueue, get_device

__all__ = ["LazyVTL", "LazyVTLState"]


class LazyVTLState(NamedTuple):
    """State of lazy VTL."""

    covariance_matrix: GaussianCovarianceMatrix
    r"""Kernel matrix of the data. Tensor of shape $n \times n$."""
    m: int
    """Size of the target space."""
    selected_indices: List[int]
    """Indices of points that were already observed."""
    joint_data: torch.Tensor
    r"""Tensor of shape $(n + m) \times d$ which includes both sample space and target space."""


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

    def set_initial_priority_queue(self, initial_priority_queue: PriorityQueue):
        """
        :param initial_priority_queue: Initial priority queue over the data set.
        """
        self.priority_queue = initial_priority_queue

    def select_from_minibatch(
        self, batch_size: int, model: ModelWithEmbeddingOrKernel | None, data: torch.Tensor, device: torch.device | None
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

        assert self.priority_queue is not None, "Initial priority queue must be set."  # TODO: make optional, compute from scratch if not provided
        assert self.priority_queue.size() == data.size(0), "Size of the priority queue must match the size of the data set."

        selected_values = []
        for _ in range(batch_size):
            cached_matrix = None
            while True:
                stopping_value = -np.inf
                i, value = self.priority_queue.pop()
                if value <= stopping_value:  # done if the value is smaller than a previously updated value
                    break  # (i, value) contains the next selected index and its value

                # new_value = self.recompute(state, i)
                if i in state.selected_indices:
                    new_value = self.recompute_previous(state, i)
                else:
                    new_value, cached_matrix = self.recompute_new(state, i, cached_matrix)
                    # Sigma_AA = self._matrix[target_indices][:, target_indices]
                    # Sigma_ii = self._matrix[_indices][:, _indices]
                    # Sigma_Ai = self._matrix[target_indices][:, _indices]
                    # posterior_Sigma_AA = (
                    #     Sigma_AA
                    #     - Sigma_Ai
                    #     @ torch.inverse(
                    #         Sigma_ii + noise_var * torch.eye(Sigma_ii.size(0)).to(Sigma_AA.device)
                    #     )
                    #     @ Sigma_Ai.T
                    # )

                stopping_value = np.maximum(stopping_value, new_value)  # stores the largest updated value
                self.priority_queue.push(i, new_value)
            selected_values.append(value)
            state = self.step(state, i, cached_matrix)
        return torch.tensor(state.selected_indices), torch.tensor(selected_values)

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel | None,
        data: torch.Tensor,
        device: torch.device | None,
    ) -> LazyVTLState:
        target = self.get_target()
        m = target.size(0)
        joint_data = torch.cat((target, data))

        # Compute covariance matrix of targets
        assert model is None, "embedding computation via model not supported"
        covariance_matrix = GaussianCovarianceMatrix.from_embeddings(Embeddings=target.to(device))

        return LazyVTLState(
            covariance_matrix=covariance_matrix,
            m=m,
            selected_indices=[],
            joint_data=joint_data,
        )

    def recompute_previous(self, state: LazyVTLState, data_idx: int) -> float:
        """
        Update value using stored covariance matrix.
        """
        noise_var = self.noise_std**2

        def compute_posterior_variance(i, j):
            return state.covariance_matrix[i, i] - state.covariance_matrix[
                i, j
            ] ** 2 / (state.covariance_matrix[j, j] + noise_var)

        target_indices = torch.arange(state.m).unsqueeze(0)  # Expand dims for broadcasting
        idx = state.m + state.selected_indices.index(data_idx)  # Index of data point within covariance matrix

        posterior_variances = compute_posterior_variance(target_indices, idx)
        total_posterior_variance = torch.sum(posterior_variances, dim=1).cpu().item()
        return -total_posterior_variance

    def recompute_new(self, state: LazyVTLState, data_idx: int, cached_matrix: torch.Tensor | None) -> Tuple[float, torch.Tensor]:
        """
        TODO
        """
        # compute inverse of current covar matrix
        # update value using the inverse
        # store the intermediate computation to expand the stored covar. matrix
        ...

    def step(self, state: LazyVTLState, data_idx: int, cached_matrix: torch.Tensor | None) -> LazyVTLState:
        """
        TODO
        """
        # TODO: update and expand covariance matrix

        state.selected_indices.append(data_idx)  # Note: not treating as immutable!
        idx = state.m + state.selected_indices.index(data_idx)
        posterior_covariance_matrix = state.covariance_matrix.condition_on(
            idx, noise_std=self.noise_std
        )
        return LazyVTLState(
            covariance_matrix=posterior_covariance_matrix,
            m=state.m,
            selected_indices=state.selected_indices,
            joint_data=state.joint_data,
        )
