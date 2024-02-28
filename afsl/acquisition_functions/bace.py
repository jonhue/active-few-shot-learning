from typing import NamedTuple
import torch
from afsl.acquisition_functions import (
    SequentialAcquisitionFunction,
    Targeted,
)
from afsl.gaussian import GaussianCovarianceMatrix
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
    compute_embedding,
)

__all__ = ["BaCE", "BaCEState", "TargetedBaCE"]


class BaCEState(NamedTuple):
    """State of sequential batch selection."""

    covariance_matrix: GaussianCovarianceMatrix
    r"""Kernel matrix of the data. Tensor of shape $n \times n$."""
    n: int
    """Length of the data set."""


class BaCE(SequentialAcquisitionFunction[ModelWithEmbeddingOrKernel, BaCEState]):
    r"""
    `BaCE` [^1] (*Batch selection via Conditional Embeddings*)

    Abstract base class for acquisition functions that select batches sequentially using "conditional embeddings".

    Provided an initial kernel $k_0$ induced by the model,[^2] `BaCE` sequentially updates this kernel by conditioning on the selected data points: \\[\begin{align}
        k_{i}(\vx,\vxp) &= k_{i-1}(\vx,\vxp) - \frac{k_{i-1}(\vx,\vx_i) \cdot k_{i-1}(\vx_i, \vxp)}{k_{i-1}(\vx_i,\vx_i) + \sigma^2} \\\\
    \end{align}\\] where $\sigma^2$ is the noise variance and $\vx_i$ is the $i$-th data point of the batch.

    Using the conditional kernel $k_i$ (or equivalently the "conditional embedding") rather than the initial kernel $k_0$ leads to *diverse* batch selection since $k_i$ reflects the information gained from the previously selected data points $\vx_{1:i}$.

    [^1]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Information-based Transductive Active Learning. arXiv preprint, 2024.

    [^2]: A kernel is also induced by embeddings. See afsl.model.ModelWithEmbedding.
    """

    noise_std: float
    """Standard deviation of the noise."""

    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    """Batch size used for computing the embeddings."""

    def __init__(
        self,
        noise_std=1.0,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        """
        :param noise_std: Standard deviation of the noise.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        super().__init__(
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        self.noise_std = noise_std
        self.embedding_batch_size = embedding_batch_size

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel,
        data: torch.Tensor,
    ) -> BaCEState:
        n = data.size(0)
        if isinstance(model, ModelWithKernel):
            covariance_matrix = GaussianCovarianceMatrix(
                model.kernel(data, data), noise_std=self.noise_std
            )
        else:
            data_embeddings = compute_embedding(
                model, data, batch_size=self.embedding_batch_size
            )
            covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
                noise_std=self.noise_std,
                Embeddings=data_embeddings,
                Sigma=(
                    model.latent_covariance()
                    if isinstance(model, ModelWithLatentCovariance)
                    else None
                ),
            )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)

    def step(self, state: BaCEState, i: int) -> BaCEState:
        posterior_covariance_matrix = state.covariance_matrix.condition_on(i)
        return BaCEState(covariance_matrix=posterior_covariance_matrix, n=state.n)


class TargetedBaCE(Targeted, BaCE):
    r"""
    Abstract base class for acquisition functions that select batches sequentially using "conditional embeddings" while targeting a specific set of prediction targets $\spA$.
    """

    def __init__(
        self,
        target: torch.Tensor,
        noise_std=1.0,
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
        force_nonsequential=False,
    ):
        r"""
        :param target: Tensor of prediction targets (shape $m \times d$).
        :param noise_std: Standard deviation of the noise.
        :param subsampled_target_frac: Fraction of the target to be subsampled in each iteration. Must be in $(0,1]$. Default is $0.5$. Ignored if `target` is `None`.
        :param max_target_size: Maximum size of the target to be subsampled in each iteration. Default is `None` in which case the target may be arbitrarily large. Ignored if `target` is `None`.
        :param mini_batch_size: Size of mini-batch used for computing the acquisition function.
        :param embedding_batch_size: Batch size used for computing the embeddings.
        :param num_workers: Number of workers used for parallel computation.
        :param subsample: Whether to subsample the data set.
        :param force_nonsequential: Whether to force non-sequential data selection.
        """
        BaCE.__init__(
            self,
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            embedding_batch_size=embedding_batch_size,
            num_workers=num_workers,
            subsample=subsample,
            force_nonsequential=force_nonsequential,
        )
        Targeted.__init__(
            self,
            target=target,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
        )

    def initialize(
        self,
        model: ModelWithEmbeddingOrKernel,
        data: torch.Tensor,
    ) -> BaCEState:
        n = data.size(0)
        target = self.get_target()
        if isinstance(model, ModelWithKernel):
            joint_data = torch.cat((data, target))
            covariance_matrix = GaussianCovarianceMatrix(
                model.kernel(joint_data, joint_data),
                noise_std=self.noise_std,
            )
        else:
            data_embeddings = compute_embedding(
                model, data=data, batch_size=self.embedding_batch_size
            )
            target_embeddings = (
                compute_embedding(
                    model, data=target, batch_size=self.embedding_batch_size
                )
                if target.size(0) > 0
                else torch.tensor([])
            )
            joint_embeddings = torch.cat((data_embeddings, target_embeddings))
            covariance_matrix = GaussianCovarianceMatrix.from_embeddings(
                noise_std=self.noise_std,
                Embeddings=joint_embeddings,
                Sigma=(
                    model.latent_covariance()
                    if isinstance(model, ModelWithLatentCovariance)
                    else None
                ),
            )
        return BaCEState(covariance_matrix=covariance_matrix, n=n)
