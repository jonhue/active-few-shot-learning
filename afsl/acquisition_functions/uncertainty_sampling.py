from afsl.acquisition_functions.undirected_itl import UndirectedITL
from afsl.utils import DEFAULT_MINI_BATCH_SIZE


class UncertaintySampling(UndirectedITL):
    r"""
    `UncertaintySampling`[^1] selects the most uncertain data point: \\[ \argmax_\vx\ \sigma^2(\vx) \\] where $\sigma^2(\vx) = k(\vx, \vx)$ denotes the variance of $\vx$ induced by the kernel $k$.[^3]

    If the kernel $k(\vx,\vxp) = \vphi(\vx)^\top \vphi(\vxp)$ is induced by embeddings $\vphi$, then \\[\sigma^2(\vx) = \norm{\vphi(\vx)}_2^2.\\]

    If coupled with gradient embeddings $\vphi(\vx) = \grad[\vtheta] \ell(\vx, \widehat{\vy}; \vtheta)$ (see afsl.embeddings), this is similar to *gradient length* acquisition functions.[^2]

    .. note::

        `UncertaintySampling` coincides with [Undirected ITL](undirected_itl) with `force_nonsequential=True`.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ❌          | ✅                | ❌          | embedding / kernel  |

    [^1]: Lewis, D. D. and Catlett, J. Heterogeneous uncertainty sampling for supervised learning. In Machine learning proceedings 1994. Elsevier, 1994.

    [^2]: Settles, B. and Craven, M. An analysis of active learning strategies for sequence labeling tasks. In EMNLP, 2008.

    [^3]: see afsl.model.ModelWithKernel
    """

    def __init__(self, noise_std=1.0, mini_batch_size=DEFAULT_MINI_BATCH_SIZE):
        super().__init__(
            noise_std=noise_std,
            mini_batch_size=mini_batch_size,
            force_nonsequential=True,
        )
