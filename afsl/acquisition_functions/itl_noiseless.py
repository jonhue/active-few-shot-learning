import torch
import wandb
import numpy as np
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState
from afsl.utils import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
)


ABS_TOL = 1e-5
JITTER_ADJUSTMENT = 1e-10


class ITLNoiseless(TargetedBaCE):
    def __init__(
        self,
        target: torch.Tensor,
        target_is_nonobersavble: bool = True,
        subsampled_target_frac: float = 1,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        r"""
        :param target: Tensor of prediction targets (shape $m \times d$).
        :param target_is_nonobersavble: Indicated whether (some of) the target points can be observed.
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $1$. Ignored if `target` is `None`.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large. Ignored if `target` is `None`.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        TargetedBaCE.__init__(
            self,
            target=target,
            noise_std=torch.nan,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=mini_batch_size,
            embedding_batch_size=embedding_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        self.target_is_nonobersavble = target_is_nonobersavble

    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[: state.n, : state.n])
        conditional_variances = torch.empty_like(variances)

        unobserved_indices = torch.arange(state.n)[~state.observed_indices]
        target_indices = torch.arange(start=state.n, end=state.covariance_matrix.dim)
        if not self.target_is_nonobersavble and state.observed_indices.size(0) > 0:
            observed_target_mask = get_observed_target_mask(state)
            target_indices = target_indices[~observed_target_mask]

        # compute jitter depending on how ill-conditioned the matrix is
        state.covariance_matrix.noise_std = get_jitter(state, target_indices)

        if unobserved_indices.size(0) > 0:
            conditional_variances[unobserved_indices] = torch.diag(
                state.covariance_matrix.condition_on(
                    indices=target_indices,
                    target_indices=unobserved_indices,
                )[:, :]
            )

        mi = 0.5 * torch.clamp(torch.log(variances / conditional_variances), min=0)
        if state.observed_indices.size(0) > 0:
            mi.index_fill_(dim=0, index=state.observed_indices, value=-torch.inf)

        wandb.log(
            {
                "max_mi": torch.max(mi),
                "min_mi": torch.min(mi),
            }
        )
        return mi


def get_observed_target_mask(state: BaCEState) -> torch.Tensor:
    """
    :return: Indices of unobserved points in target space
    """
    target_indices = torch.arange(start=state.n, end=state.covariance_matrix.dim)
    if state.observed_indices.size(0) == 0:
        return target_indices

    samples = state.joint_data[state.observed_indices]
    targets = state.joint_data[state.n :]

    cdist = torch.cdist(targets, samples, p=2)
    observed_targets = torch.any(cdist < ABS_TOL, dim=1)
    return observed_targets


def get_jitter(state: BaCEState, target_indices: torch.Tensor) -> float:
    if target_indices.dim() == 0:
        return JITTER_ADJUSTMENT

    condition_number = torch.linalg.cond(
        state.covariance_matrix[target_indices, target_indices]
    )
    return JITTER_ADJUSTMENT * condition_number
