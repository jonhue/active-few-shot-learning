import torch
import wandb
from activeft.acquisition_functions.bace import TargetedBaCE, BaCEState
from activeft.acquisition_functions.ctl import _compute_correlations


class MMITL(TargetedBaCE):
    r"""
    `MMITL` [^3] (*mean-marginal information-based transductive learning*)[^2] composes the batch by sequentially selecting the samples with the largest marginal information gain to the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmax_{\vx}\ \sum_{\vxp \in \spA} \I{f(\vxp)}{y(\vx) \mid \spD_i}.
    \end{align}\\]
    Here, $\spS$ denotes the data set, $f$ is the stochastic process induced by the kernel $k$.[^1]
    We denote (noisy) observations of $\vx_{1:i}$ by $y_{1:i}$ and the first $i$ selected samples by $\spD_i = \\{(\vx_j, y_j)\\}_{j=1}^i$.

    | Relevance? | Diversity? | Model Requirement  |
    |------------|------------|--------------------|
    | ✅         | ✅          | embedding / kernel  |

    [^1]: A kernel $k$ on domain $\spX$ induces a stochastic process $\\{f(\vx)\\}_{\vx \in \spX}$. See activeft.model.ModelWithKernel.

    [^2]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Transductive Active Learning: Theory and Applications. NeurIPS, 2024.
    """

    def compute(self, state: BaCEState) -> torch.Tensor:
        correlations = _compute_correlations(
            covariance_matrix=state.covariance_matrix, n=state.n
        )
        sqd_correlations = torch.square(correlations)

        marginal_mi = -0.5 * torch.log(1 - sqd_correlations)
        wandb.log(
            {
                "max_mi": torch.max(marginal_mi),
                "min_mi": torch.min(marginal_mi),
            }
        )
        mean_marginal_mi = torch.mean(marginal_mi, dim=1)
        return mean_marginal_mi
