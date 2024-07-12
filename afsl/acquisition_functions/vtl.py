import torch
import wandb
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState
from afsl.gaussian import get_jitter


class VTL(TargetedBaCE):
    r"""
    `VTL` [^2][^3] (*variance-based transductive learning*) composes the batch by sequentially selecting the samples which minimize the posterior marginal variances of the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \tr{\Var{\vf(\spA) \mid \spD_i, y(\vx)}}.
    \end{align}\\]
    Here, $\spS$ denotes the data set, $f$ is the stochastic process induced by the kernel $k$.[^1]
    We denote (noisy) observations of $\vx_{1:i}$ by $y_{1:i}$ and the first $i$ selected samples by $\spD_i = \\{(\vx_j, y_j)\\}_{j=1}^i$.

    .. note::

        The special case where the prediction targets $\spA$ are equal to $\spS$ (i.e., the prediction targets include "everything") is [Undirected VTL](undirected_vtl).

    `VTL` selects batches via *conditional embeddings*,[^4] leading to diverse batches.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ✅          | ✅                | ✅          | embedding / kernel  |

    #### Comparison to ITL

    [ITL](itl) can be expressed as \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \det{\Var{\vf(\spA) \mid \spD_{i}, y(\vx)}}.
    \end{align}\\]
    That is, [ITL](itl) minimizes the determinant of the posterior covariance matrix of $\vf(\spA)$ whereas `VTL` minimizes the trace of the posterior covariance matrix of $\vf(\spA)$.
    In practice, this difference amounts to a different "weighting" of the prediction targets in $\spA$.
    While `VTL` attributes equal importance to all prediction targets, [ITL](itl) attributes more importance to the "most uncertain" prediction targets.

    [^1]: A kernel $k$ on domain $\spX$ induces a stochastic process $\\{f(\vx)\\}_{\vx \in \spX}$. See afsl.model.ModelWithKernel.

    [^2]: Seo, S., Wallat, M., Graepel, T., and Obermayer, K. Gaussian process regression: Active data selection and test point rejection. In Mustererkennung 2000. Springer, 2000.

    [^3]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Information-based Transductive Active Learning. arXiv preprint, 2024.

    [^4]: see afsl.acquisition_functions.bace.BaCE
    """

    def compute(self, state: BaCEState) -> torch.Tensor:
        if self.noise_std is None:
            noise_var = get_jitter(covariance_matrix=state.covariance_matrix, indices=torch.arange(state.n))
        else:
            noise_var = self.noise_std**2

        def compute_posterior_variance(i, j):
            return state.covariance_matrix[i, i] - state.covariance_matrix[
                i, j
            ] ** 2 / (state.covariance_matrix[j, j] + noise_var)

        data_indices = torch.arange(state.n).unsqueeze(
            1
        )  # Expand dims for broadcasting
        target_indices = torch.arange(state.n, state.covariance_matrix.dim).unsqueeze(
            0
        )  # Expand dims for broadcasting

        posterior_variances = compute_posterior_variance(target_indices, data_indices)
        total_posterior_variances = torch.sum(posterior_variances, dim=1)
        wandb.log(
            {
                "max_posterior_var": torch.max(posterior_variances),
                "min_posterior_var": torch.min(posterior_variances),
            }
        )
        return -total_posterior_variances
