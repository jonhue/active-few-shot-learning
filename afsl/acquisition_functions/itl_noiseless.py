import torch
import wandb
import numpy as np
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState



class ITLNoiseless(TargetedBaCE):
    r"""
    `ITL` [^3] (*information-based transductive learning*) composes the batch by sequentially selecting the samples with the largest information gain about the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmax_{\vx \in \spS}\ \I{\vf(\spA)}{y(\vx) \mid \spD_i}.
    \end{align}\\]
    Here, $\spS$ denotes the data set, $f$ is the stochastic process induced by the kernel $k$.[^1]
    We denote (noisy) observations of $\vx_{1:i}$ by $y_{1:i}$ and the first $i$ selected samples by $\spD_i = \\{(\vx_j, y_j)\\}_{j=1}^i$.

    `ITL` can equivalently be interpreted as minimizing the posterior entropy of the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \H{\vf(\spA) \mid \spD_i, y(\vx)}.
    \end{align}\\]

    .. note::

        The special case where the prediction targets $\spA$ include $\spS$ ($\spS \subseteq \spA$, i.e., the prediction targets include "everything") is [Undirected ITL](undirected_itl).

    `ITL` selects batches via *conditional embeddings*,[^4] leading to diverse batches.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ✅          | ✅                | ✅          | embedding / kernel  |

    #### Comparison to VTL

    `ITL` can be expressed as \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \det{\Var{\vf(\spA) \mid \spD_{i}, y(\vx)}}.
    \end{align}\\]
    That is, `ITL` minimizes the determinant of the posterior covariance matrix of $\vf(\spA)$ whereas [VTL](vtl) minimizes the trace of the posterior covariance matrix of $\vf(\spA)$.
    In practice, this difference amounts to a different "weighting" of the prediction targets in $\spA$.
    While `VTL` attributes equal importance to all prediction targets, [ITL](itl) attributes more importance to the "most uncertain" prediction targets.

    #### Computation

    `ITL` is computed using $\I{\vf(\spA)}{y(\vx) \mid \spD_i} \approx \I{\vy(\spA)}{y(\vx) \mid \spD_i}$ with \\[\begin{align}
        \I{\vy(\spA)}{y(\vx) \mid \spD_i} &= \frac{1}{2} \log\left( \frac{k_i(\vx,\vx) + \sigma^2}{\tilde{k}_i(\vx,\vx) + \sigma^2} \right) \qquad\text{where} \\\\
        \tilde{k}_i(\vx,\vx) &= k_i(\vx,\vx) - \vk_i(\vx,\spA) (\mK_i(\spA,\spA) + \sigma^2 \mI)^{-1} \vk_i(\spA,\vx)
    \end{align}\\] where $\sigma^2$ is the noise variance and $k_i$ denotes the conditional kernel (see afsl.acquisition_functions.bace.BaCE).

    [^1]: A kernel $k$ on domain $\spX$ induces a stochastic process $\\{f(\vx)\\}_{\vx \in \spX}$. See afsl.model.ModelWithKernel.

    [^3]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Information-based Transductive Active Learning. arXiv preprint, 2024.

    [^4]: see afsl.acquisition_functions.bace.BaCE
    """

    def compute(self, state: BaCEState) -> torch.Tensor:
        variances = torch.diag(state.covariance_matrix[: state.n, : state.n])
        conditional_variances = torch.empty_like(variances)

        unobserved_points = torch.tensor([i for i in torch.arange(state.n) if not ITLNoiseless.observed(i, state)], device=ITLNoiseless.get_device())
        observed_points = torch.tensor([i for i in torch.arange(state.n) if ITLNoiseless.observed(i, state)], device=ITLNoiseless.get_device())

        adapted_target_space = torch.tensor([i for i in torch.arange(start=state.n, end=state.covariance_matrix.dim) if not ITLNoiseless.observed(i, state)], device=ITLNoiseless.get_device())

        only_sample_indices, target_and_sample_indices_sample_indexing, target_and_sample_indices_target_indexing = ITLNoiseless.split(state, unobserved_points)
        
        #
        #   Compute conditional_variances
        #

        #   Unobserved indices contained in sample and target space

        if target_and_sample_indices_sample_indexing.size(dim=0) > 0:
            conditional_variances[target_and_sample_indices_sample_indexing] = ITLNoiseless.compute_conditional_variance(state, target_and_sample_indices_target_indexing, adapted_target_space)

        #   Unobserved indices contained only in sample space

        if only_sample_indices.size(dim=0) > 0:
            conditional_variances[only_sample_indices] = torch.diag(state.covariance_matrix.condition_on(
                indices=adapted_target_space,
                target_indices=only_sample_indices,
            )[:, :])

        #
        #   Compute mutual information
        #

        mi = 0.5 * torch.clamp(torch.log(variances / conditional_variances), min=0)
        if observed_points.size(dim = 0) > 0:
            mi.index_fill_(0, observed_points, -float('inf'))

        wandb.log(
            {
                "max_mi": torch.max(mi),
                "min_mi": torch.min(mi),
            }
        )

        return mi          
    
    @staticmethod
    def observed(idx, state: BaCEState) -> bool:
        """Checks if point has been observed yet

        Parameters
        ----------
        idx : int, index of the point
        state : BaCEState

        Returns
        -------
        True if points has already been observed
        """
        return ITLNoiseless.containsFloat(state.joint_data[idx], state.observed_points)
    
    @staticmethod
    def containsFloat(x, set):
        """Checks if x is in set

        Parameters
        ----------
        x : float, value to search in set
        set : list, set to search in

        Returns
        -------
        Returns True if value was found
        """
        return any(ITLNoiseless.isClose(x, y) for y in set)
    
    @staticmethod
    def containsInt(x, set):
        """Checks if x is in set

        Parameters
        ----------
        x : int, value to search in set
        set : list, set to search in

        Returns
        -------
        Returns True if value was found
        """
        return any(x == y for y in set)
    
    @staticmethod
    def isClose(x, y, rel_tol=1e-09, abs_tol=0.0) -> bool:
        """Checks if two float vectors are almost equal

        Parameters
        ----------
        x : vector, value 1 to check
        y : vector, value 2 to check
        rel_tol : float, optional
            standard value is 0.000001
        abs_tol : float, optional
            standard value is 0.000001

        Returns
        -------
        If the vector x is close to the vector y
        """
        
        return  np.bool_(np.linalg.norm(x - y) <= max(rel_tol * max(np.linalg.norm(x), np.linalg.norm(y)), abs_tol)).item()
    
    @staticmethod
    def split(state: BaCEState, unobserved_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        only_sample_indices = []
        target_and_sample_indices_sample_indexing = []
        target_and_sample_indices_target_indexing = []

        for i in unobserved_points:
            if ITLNoiseless.containsFloat(state.sample_points[i], state.target_points):
                target_and_sample_indices_sample_indexing.append(i)
                target_and_sample_indices_target_indexing.append(ITLNoiseless.to_target_index(state, i))
            else:
                only_sample_indices.append(i)

        return torch.tensor(only_sample_indices), torch.tensor(target_and_sample_indices_sample_indexing), torch.tensor(target_and_sample_indices_target_indexing)
            
    @staticmethod
    def to_target_index(state: BaCEState, i):
        return next((j for j in torch.arange(start=state.n, end=state.covariance_matrix.dim) if ITLNoiseless.isClose(state.sample_points[i], state.joint_data[j])), -1)
    
    @staticmethod
    def compute_conditional_variance(state: BaCEState, unobserved_target_indices: torch.Tensor, adapted_target_space: torch.Tensor) -> torch.Tensor:

        #
        #   Vectorize conditional_covariance computation
        #

        def conditional_variance(i, adapted_target_space) -> torch.Tensor:
            conditional_covariance_matrix = state.covariance_matrix.condition_on(
                indices=adapted_target_space.int(),
                target_indices=torch.reshape(i, [1]),
            )[:, :]
            return torch.diag(conditional_covariance_matrix)

        batch_conditional_variance = torch.vmap(conditional_variance)

        #   Prepare adapted target spaces

        adapted_target_spaces = torch.empty(unobserved_target_indices.size(dim=0), adapted_target_space.size(dim=0) - 1)

        for i, idx in enumerate(unobserved_target_indices):
            adapted_target_spaces[i] = adapted_target_space[adapted_target_space != idx]

        #   Compute conditional variances

        if adapted_target_space.size(dim=0) > 1:
            return batch_conditional_variance(unobserved_target_indices, adapted_target_spaces)
        else:
            return torch.diag(state.covariance_matrix[:, :])
    
    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
