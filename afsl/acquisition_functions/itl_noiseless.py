import torch
import wandb
import numpy as np
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState



REL_TOL = 1e-5

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

        observed_points, unobserved_points = ITLNoiseless.observed_points(state)
        adapted_target_space = ITLNoiseless.get_adapted_target_space(state)

        #only_sample_indices, target_and_sample_indices_sample_indexing, target_and_sample_indices_target_indexing = ITLNoiseless.split(state, unobserved_points)
        
        #
        #   Compute conditional_variances
        #

        #   Compute Jitter depending on how ill conditioned the matrix is

        state.covariance_matrix.noise_std = ITLNoiseless.get_jitter(state, adapted_target_space)

        #   Unobserved indices contained in sample and target space

        #if target_and_sample_indices_sample_indexing.size(dim=0) > 0:
        #    print("There is overlap ...")
        #    conditional_variances[target_and_sample_indices_sample_indexing] = ITLNoiseless.compute_conditional_variance(state, target_and_sample_indices_target_indexing, adapted_target_space)
            
        #   Unobserved indices contained only in sample space

        if unobserved_points.size(dim=0) > 0:
            conditional_variances[unobserved_points] = torch.diag(state.covariance_matrix.condition_on(
                indices=adapted_target_space,
                target_indices=unobserved_points,
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
    def observed_points(state: BaCEState) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the sample space into observed and not observed points

        Parameters
        ----------
        state : BaCEState

        Returns
        -------
        Tuple (observed_indices, unobserved_indices) where indices are respective to sample space
        """
        sample_indices = torch.arange(state.n, device=ITLNoiseless.get_device())

        if(state.observed_points.size(dim=0) == 0):
            return torch.tensor([], device=ITLNoiseless.get_device()), sample_indices
        
        sample_points = state.sample_points.view(state.n, -1)
        observed_points = state.observed_points.view(state.observed_points.size(dim=0), -1)
        abs_tol = REL_TOL * sample_points.shape[1]

        cdist = torch.cdist(sample_points, observed_points, p=2.0)
        observed_map = torch.any(cdist < abs_tol, dim=1)

        return sample_indices[observed_map], sample_indices[~observed_map]      
    
    @staticmethod
    def get_adapted_target_space(state: BaCEState) -> torch.Tensor:
        """Get unobserved points from target space

        Parameters
        ----------
        state : BaCEState

        Returns
        -------
        Returns unobserved points in target space
        """
        target_indices = torch.arange(start=state.n, end=state.covariance_matrix.dim, device=ITLNoiseless.get_device())

        if(state.observed_points.size(dim=0) == 0):
            return target_indices
        
        target_points = state.target_points.view(state.target_points.size(dim=0), -1)
        observed_points = state.observed_points.view(state.observed_points.size(dim=0), -1)
        abs_tol = REL_TOL * target_points.shape[1]

        cdist = torch.cdist(target_points, observed_points, p=2.0)
        observed_map = torch.any(cdist < abs_tol, dim=1)

        return target_indices[~observed_map]    

    @staticmethod
    def get_jitter(state: BaCEState, adapted_target_space: torch.Tensor) -> float:
        _indices: torch.Tensor = adapted_target_space
        if _indices.dim() == 0:
            _indices = _indices.unsqueeze(0)

        condition_number = torch.linalg.cond(state.covariance_matrix[_indices, :][:, _indices])
        return 1e-10 * condition_number

    @staticmethod
    def split(state: BaCEState, unobserved_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split domain into points only contained in sample space and points contained in sample and target space 
        indexed by sample indexing and target indexing

        Parameters
        ----------
        state : BaCEState
        unobserved_points : torch.Tensor, unobserved sample points with sample indexing

        Returns
        -------
        Returns (points only contained in sample space, points contained in sample and target space indexed by sample indexing, points contained in sample and target space indexed by target indexing)
        """
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
    def isClose(x, y) -> bool:
        """Checks if two float vectors are almost equal

        Parameters
        ----------
        x : vector, value 1 to check
        y : vector, value 2 to check

        Returns
        -------
        If the vector x is close to the vector y
        """
        return  np.bool_(np.linalg.norm(x - y) <= REL_TOL * x.size(dim=0)).item()
            
    @staticmethod
    def to_target_index(state: BaCEState, i):
        """Convert index from sample space to index in target space

        Parameters
        ----------
        state : BaCEState
        i : int, index in sample space

        Returns
        -------
        Returns index in target space or -1
        """
        return next((j for j in torch.arange(start=state.n, end=state.covariance_matrix.dim) if ITLNoiseless.isClose(state.sample_points[i], state.joint_data[j])), -1)
    
    @staticmethod
    def compute_conditional_variance(state: BaCEState, target_and_sample_indices_target_indexing: torch.Tensor, adapted_target_space: torch.Tensor) -> torch.Tensor:
        """Compute conditional variance for points that are in target and sample space

        Parameters
        ----------
        state : BaCEState
        target_and_sample_indices_target_indexing : torch.Tensor, indices of points that are contained in target and sample space with target space indexing
        adapted_target_space : torch.Tensor, unobserved points in target space
        
        Returns
        -------
        Returns conditional variance of points that are in target and sample space
        """

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

        adapted_target_spaces = torch.empty(target_and_sample_indices_target_indexing.size(dim=0), adapted_target_space.size(dim=0) - 1)

        for i, idx in enumerate(target_and_sample_indices_target_indexing):
            adapted_target_spaces[i] = adapted_target_space[adapted_target_space != idx]

        #   Compute conditional variances

        if adapted_target_space.size(dim=0) > 1:
            return batch_conditional_variance(target_and_sample_indices_target_indexing, adapted_target_spaces)
        else:
            return torch.diag(state.covariance_matrix[:, :])
    
    @staticmethod
    def get_device():
        """Define device on which to create tensor ("cuda:0", "cpu")

        Parameters
        ----------
        None

        Returns
        -------
        Returns "cuda:0" if available and "cpu" otherwise
        """
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
