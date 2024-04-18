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

        adapted_target_space = ITLNoiseless.adapted_target_space(state)

        #
        #   Compute conditional_variances
        #

        #   Unobserved indices contained in sample and target space

        unobserved_target_indices, unobserved_target_indices_target_index = ITLNoiseless.get_unobserved_target_indices(state, adapted_target_space) #TODO try to improve

        if unobserved_target_indices.size(dim=0) > 0:
            conditional_variances[unobserved_target_indices] = ITLNoiseless.compute_conditional_variance(state, unobserved_target_indices_target_index, adapted_target_space)

        #   Unobserved indices contained only in sample space

        unobserved_sample_indices = ITLNoiseless.get_unobserved_sample_indices(state, unobserved_points) #TODO try to improve

        if unobserved_sample_indices.size(dim=0) > 0:
            conditional_variances[unobserved_sample_indices] = torch.diag(state.covariance_matrix.condition_on(
                indices=adapted_target_space,
                target_indices=unobserved_sample_indices,
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
        return any(ITLNoiseless.isClose(state.joint_data[idx], y) for y in state.observed_points)
    
    @staticmethod
    def contains(x, set) -> int:
        return next((y for y in set if ITLNoiseless.isClose(x, y)), -1)
    
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
        ------
        If the vector x is close to the vector y
        """
        
        return  np.bool_(np.linalg.norm(x - y) <= max(rel_tol * max(np.linalg.norm(x), np.linalg.norm(y)), abs_tol)).item()
    
    @staticmethod
    def adapted_target_space(state: BaCEState) -> torch.Tensor:
        return torch.tensor([i for i in torch.arange(start=state.n, end=state.covariance_matrix.dim) if not ITLNoiseless.observed(i, state)], device=ITLNoiseless.get_device())
    
    @staticmethod
    def get_unobserved_target_indices(state: BaCEState, adapted_target_space: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [
            [sample_index, target_index] for target_index in adapted_target_space if 
            (sample_index := ITLNoiseless.contains(state.joint_data[target_index], state.sample_points)) > -1
        ]

        if len(indices) == 0:
            return torch.tensor([], device=ITLNoiseless.get_device()), torch.tensor([], device=ITLNoiseless.get_device())
        
        return torch.tensor(indices[:][0], device=ITLNoiseless.get_device()), torch.tensor(indices[:][1], device=ITLNoiseless.get_device())
    
    @staticmethod
    def get_unobserved_sample_indices(state: BaCEState, unobserved_points: torch.Tensor) -> torch.Tensor:
        #print("unobserved_points " + str(unobserved_points))
        return torch.tensor(
            [i for i in unobserved_points if 
                not ITLNoiseless.contains(state.joint_data[i], state.target_points)
            ], 
            device=ITLNoiseless.get_device()
        )
    
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
