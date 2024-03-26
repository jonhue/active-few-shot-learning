"""
Active data selection requires that the model has an associated kernel $k$ or (equivalently) associated embeddings.
This page summarizes these requirements and afsl.embeddings provides common implementation of kernels and embeddings.

This module contains a selection of protocols for PyTorch models implementing these requirements.
"""

from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable
import torch


class Model(Protocol):
    """Protocol for vanilla PyTorch `nn.Module` instances."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def eval(self) -> Model:
        ...

    def train(self) -> Model:
        ...

    def zero_grad(self) -> None:
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...


@runtime_checkable
class ModelWithEmbedding(Model, Protocol):
    r"""
    Protocol for PyTorch models with associated embeddings.

    An embedding $\vphi(\vx)$ is a latent representation of an input $\vx$.
    Collecting the embeddings as rows in the design matrix $\mPhi$ of a set of inputs $X$, one can approximate the network by the linear function ${\vf(X) = \mPhi \vtheta}$ with weights $\vtheta$.
    Approximating the weights by ${\vtheta \sim \N{\vmu}{\mSigma}}$ implies ${\vf(X) \sim \N{\mPhi\vmu}{\mPhi \mSigma \mPhi^\top}}$.

    The covariance matrix $\mPhi \mSigma \mPhi^\top$ can be succinctly represented in terms of its associated kernel $k(\vx,\vxp) = \vphi(\vx)^\top \mSigma \vphi(\vxp)$.
    Here,
    - $\vphi(\vx)$ is the latent representation of $\vx$, and
    - $\mSigma$ captures the dependecies in the latent space.

    ##### Common Embeddings

    While any choice of $\vphi$ is possible, the following are common choices:

    - **Last-Layer:** A common choice for $\vphi(\vx)$ is the representation of $\vx$ from the penultimate layer of the neural network.[^1]
    - **Output Gradients (empirical NTK):** Another common choice is $\vphi(\vx) = \grad[\vtheta] \vf(\vx; \vtheta)$ where $\vtheta$ are the network parameters.
    Its associated kernel is known as the *(empirical) Neural Tangent Kernel* (NTK).[^4][^3][^5]
    If $\vtheta$ is restricted to the weights of the final linear layer, then this embedding is simply the last-layer embedding.
    - **Loss Gradients:** Another possible choice is $\vphi(\vx) = \grad[\vtheta] \ell(\vf(\vx; \vtheta); \widehat{\vy}(\vx))$ where $\ell$ is a loss function and $\widehat{\vy}(\vx)$ is some hallucinated label (see afsl.embeddings.classification.HallucinatedCrossEntropyEmbedding).[^6]
    - **Outputs (empirical NNGP):** Another possible choice is $\vphi(\vx) = \vf(\vx)$ (i.e., the output of the network).
    Its associated kernel is known as the *(empirical) Neural Network Gaussian Process* (NNGP) kernel.[^2]

    ##### Uncertainty Quantification

    The latent covariance matrix $\mSigma$ is commonly neglected (i.e., $\mSigma = \mI$), but can be used to quantify the uncertainty in the latent space.
    For example, with gradient embeddings, the latent space is the networks parameter space and approximating $\mSigma$ has been the subject of recent research.[^7][^8][^9]

    If `ModelWithLatentCovariance` is implemented, $\mSigma$ is used for computation of the associated kernel.
    Otherwise, it is assumed that $\mSigma = \mI$.

    [^1]: Holzmüller, D., Zaverkin, V., Kästner, J., and Steinwart, I. A framework and benchmark for deep batch active learning for regression. JMLR, 24(164), 2023.

    [^2]: Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., and Sohl-Dickstein, J. Deep neural networks as gaussian processes. ICLR, 2018.

    [^3]: Khan, M. E. E., Immer, A., Abedi, E., and Korzepa, M. Approximate inference turns deep networks into gaussian processes. NeurIPS, 32, 2019.

    [^4]: Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R. R., and Wang, R. On exact computation with an infinitely wide neural net. NeurIPS, 32, 2019.

    [^5]: He, B., Lakshminarayanan, B., and Teh, Y. W. Bayesian deep ensembles via the neural tangent kernel. NeurIPS, 33, 2020.

    [^6]: Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., and Agarwal, A. Deep batch active learning by diverse, uncertain gradient lower bounds. ICLR, 2020.

    [^7]: Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., and Wilson, A. G. A simple baseline for bayesian uncertainty in deep learning. NeurIPS, 32, 2019.

    [^8]: Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. Weight uncertainty in neural network. In ICML, 2015.

    [^9]: Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., and Hennig, P. Laplace redux-effortless bayesian deep learning. NeurIPS, 34, 2021.
    """

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: Tensor of inputs $\vx$ with shape $n \times d$.
        :return: Tensor of associated latent representation $\vphi(\vx)$ with shape $n \times k$.
        """
        ...


@runtime_checkable
class ModelWithLatentCovariance(ModelWithEmbedding, Protocol):
    r"""
    Protocol for PyTorch models with associated latent covariance matrix $\mSigma$.

    If this protocol is implemented, $\mSigma$ is used in computation of the associated kernel $k(\vx,\vxp) = \vphi(\vx)^\top \mSigma \vphi(\vxp)$.
    """

    def latent_covariance(self) -> torch.Tensor:
        r"""
        :return: Tensor of associated latent covariance matrix $\mSigma$ with shape $k \times k$.
        """
        ...


@runtime_checkable
class ModelWithKernel(Model, Protocol):
    r"""
    Protocol for PyTorch models with associated kernel.

    A kernel $k(\vx,\vxp)$ is a symmetric and positive semi-definite function that measures the similarity between two inputs $\vx$ and $\vxp$.
    This protocol can be implemented alternatively to `ModelWithEmbedding` for direct computation of the kernel $k$.

    https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html describes how to compute the empirical NTK with PyTorch.
    """

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        r"""
        :param x1: Tensor of inputs with shape $n \times d$.
        :param x2: Tensor of inputs with shape $m \times d$.
        :return: Tensor of associated (symmetric and positive semi-definite) kernel matrix with shape $n \times m$.
        """
        ...


ModelWithEmbeddingOrKernel = ModelWithEmbedding | ModelWithKernel
"""Protocol for PyTorch models with either an associated kernel or with associated embeddings."""


class ClassificationModel(Model, Protocol):
    """Protocol for PyTorch classification models. The `forward` function is expected to return a distribution over classes."""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: Tensor of inputs with shape $n \times d$.
        :return: Tensor of predicted class labels with shape $n$.
        """
        ...

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: Tensor of inputs with shape $n \times d$.
        :return: Tensor of associated logits with shape $n \times k$.
        """
        ...

    @property
    def final_layer(self) -> torch.nn.Linear:
        """Returns the final linear layer of the model."""
        ...
