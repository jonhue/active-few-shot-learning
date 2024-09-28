import torch
from activeft.acquisition_functions.bace import BaCE, BaCEState
from activeft.utils import wandb_log


class UndirectedITL(BaCE):
    r"""
    `UndirectedITL` composes the batch by sequentially selecting the samples with the largest prior variance: \\[\begin{align}
        \vx_{i+1} &= \argmax_\vx\ \sigma_{i}^2(\vx)
    \end{align}\\] where $\sigma_i^2(\vx) = k_i(\vx, \vx)$ denotes the variance of $\vx$ arising from the "conditional" kernel $k_i$ conditioned on (noisy) observations of $\vx_{1:i}$.[^4]
    $f$ is the stochastic process induced by the kernel $k$.[^1]
    We denote (noisy) observations of $\vx_{1:i}$ by $y_{1:i}$ and the first $i$ selected samples by $\spD_i = \\{(\vx_j, y_j)\\}_{j=1}^i$.

    .. note::

        This acquisition function is analogous to [Uncertainty Sampling](uncertainty_sampling) but selects the batch sequentially by conditioning on previously selected points.
        In other words, (undirected) ITL is a generalization to batch sizes larger than one.

    | Relevance? | Diversity? | Model Requirement  |
    |------------|------------|--------------------|
    | ❌         | ✅          | embedding / kernel  |

    #### Interpretation: Greedy Determinant Maximization

    `UndirectedITL` can be shown to be equivalent to \\[\begin{align}
        \vx_{i+1} &= \argmax_\vx\ \det{\\!(\mK_i(X, X) + \sigma^2 \mI)} \qquad\text{with $X = \vx_{1:i} \cup \\{\vx\\}$}
    \end{align}\\] where $\sigma^2$ is the noise variance and $\mK_i(X, X) = (k_i(\vx, \vxp))_{\vx \in X, \vxp \in X}$.
    This is also known as *greedy determinant maximization*.[^2]

    #### Interpretation: Special "Undirected" Case of ITL

    Similarly, denoting the data set to be selected from by $\spS$, `UndirectedITL` is equivalent to maximizing the information gain of the next observation $y(\vx_{i+1})$ with $\vf(\spS)$ (i.e., minimizing the "posterior" entropy of $\vf(\spS)$): \\[\begin{align}
        \vx_{i+1} &= \argmax_{\vx \in \spS}\ \I{f(\vx)}{y(\vx) \mid \spD_{i}} \\\\
        &= \argmax_{\vx \in \spS}\ \I{\vf(\spS)}{y(\vx) \mid \spD_{i}} \\\\
        &= \argmin_{\vx \in \spS}\ \H{\vf(\spS) \mid \spD_{i}, y(\vx)}.
    \end{align}\\]

    If the set of prediction targets $\spA$ of [ITL](itl) includes all of $\spS$ (i.e., $\spS \subseteq \spA$) then `UndirectedITL` is equivalent to [ITL](itl).[^3]
    Intuitively, such prediction targets include "everything", and hence, ITL is "undirected".

    .. note::

        **Prior vs Posterior Uncertainty:** Note that while `UndirectedITL` selects $\vx_{i+1}$ so as to maximize the *prior* uncertainty $\sigma_i^2(\vx)$, [ITL](itl) selects $\vx_{i+1}$ so as to minimize the *posterior* uncertainty at the prediction targets.
        Remarkably, if $\spS \subseteq \spA$ then the two approaches are equivalent.

    #### Comparison to VTL

    `UndirectedITL` can be equivalently expressed as \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \det{\Var{\vf(\spS) \mid \spD_{i}, y(\vx)}}.
    \end{align}\\]
    That is, `UndirectedITL` minimizes the determinant of the posterior covariance matrix of $\vf(\spS)$ whereas [Undirected VTL](undirected_vtl) minimizes the trace of the posterior covariance matrix of $\vf(\spS)$.

    [^1]: A kernel $k$ on domain $\spX$ induces a stochastic process $\\{f(\vx)\\}_{\vx \in \spX}$. See activeft.model.ModelWithKernel.

    [^2]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^3]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Transductive Active Learning: Theory and Applications. NeurIPS, 2024.

    [^4]: see activeft.acquisition_functions.bace.BaCE
    """

    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[:, :])
        wandb_log(
            {
                "max_var": torch.max(variances),
                "min_var": torch.min(variances),
            }
        )
        return variances
