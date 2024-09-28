import torch
from activeft.acquisition_functions.bace import TargetedBaCE, BaCEState


class CTL(TargetedBaCE):
    r"""
    `CTL` [^3] (*correlation-based transductive learning*) composes the batch by sequentially selecting the samples with the largest correlation to the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmax_{\vx}\ \sum_{\vxp \in \spA} \Cor{f(\vx), f(\vxp) \mid \spD_i}.
    \end{align}\\]
    Here, $\spS$ denotes the data set, $f$ is the stochastic process induced by the kernel $k$.[^1]
    We denote (noisy) observations of $\vx_{1:i}$ by $y_{1:i}$ and the first $i$ selected samples by $\spD_i = \\{(\vx_j, y_j)\\}_{j=1}^i$.

    `CTL` explicitly obtains the most relevant samples with respect to the prediction targets.
    Moreover, `CTL` can be shown to be a tight approximation of [VTL](vtl), and hence, implicitly leads also to informative samples.[^3]

    .. note::

        `CTL` is a generalization of [Cosine Similarity](cosine_similarity) to batch sizes larger than one.

    `CTL` selects batches via *conditional embeddings*,[^4] leading to diverse batches.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ✅          | (✅)                | ✅          | embedding / kernel  |

    [^1]: A kernel $k$ on domain $\spX$ induces a stochastic process $\\{f(\vx)\\}_{\vx \in \spX}$. See activeft.model.ModelWithKernel.

    [^3]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Information-based Transductive Active Learning. arXiv preprint, 2024.

    [^4]: see activeft.acquisition_functions.bace.BaCE
    """

    def compute(self, state: BaCEState) -> torch.Tensor:
        ind_a = torch.arange(state.n)
        ind_b = torch.arange(state.n, state.covariance_matrix.dim)
        covariance_aa = state.covariance_matrix[ind_a, :][:, ind_a]
        covariance_bb = state.covariance_matrix[ind_b, :][:, ind_b]
        covariance_ab = state.covariance_matrix[ind_a, :][:, ind_b]

        std_a = torch.sqrt(torch.diag(covariance_aa))
        std_b = torch.sqrt(torch.diag(covariance_bb))
        std_ab = torch.ger(std_a, std_b)  # outer product of standard deviations

        correlations = covariance_ab / std_ab
        average_correlations = torch.mean(correlations, dim=1)
        return average_correlations
