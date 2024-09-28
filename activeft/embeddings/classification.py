"""
Implementation of embeddings for classification models whose final layer is a linear layer. The `forward` function is expected to return a distribution over classes.

For more details regarding embeddings, see activeft.model.ModelWithEmbedding.
"""

import torch
from torch import nn
from activeft.model import ClassificationModel, ModelWithEmbedding


class HallucinatedCrossEntropyEmbedding(ClassificationModel, ModelWithEmbedding):
    r"""
    Computes the embedding \\[\vphi(\vx) = \grad[\vthetap] \left. \ell_{\mathrm{CE}}(\vf(\vx; \vtheta); \widehat{y}(\vx)) \right\rvert_{\vtheta = \widehat{\vtheta}}\\] where $\vthetap$ refers to the parameters of the final output layer, $\widehat{\vtheta}$ are the current parameter estimates, $\widehat{y}(\vx) = \argmax_i f_i(\vx; \widehat{\vtheta})$ are the associated "hallucinated" labels, and $\ell_{\mathrm{CE}}$ is the [cross-entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

    This embedding was originally proposed by [^1].

    [^1]: Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., and Agarwal, A. Deep batch active learning by diverse, uncertain gradient lower bounds. ICLR, 2020.
    """

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.final_layer, nn.Linear), "Final layer must be linear."

        logits = self.logits(data)  # (N, K)
        probabilities = self(data)  # (N, C)
        pred = self.predict(data)  # (N,)

        C = probabilities.size(1)

        # compute gradient explicitly: eq. (1) of https://arxiv.org/pdf/1906.03671.pdf
        pred_ = torch.nn.functional.one_hot(pred, C)  # (N, C)
        A = (probabilities - pred_)[:, :, None]  # (N, C, 1)
        if self.final_layer.bias is not None:
            logits = torch.cat(
                (logits, torch.ones(logits.size(0), 1, device=logits.device)), dim=1
            )  # (N, K+1)
        B = logits[:, None, :]  # (N, 1, K+1)
        J = torch.matmul(A, B).view(-1, A.shape[1] * B.shape[2])  # (N, C * (K+1))
        return J


# class ExpectedCrossEntropyEmbedding(ClassificationModel, ModelWithEmbedding):
#     r"""
#     Computes the embedding \\[\vphi(\vx) = \E{y(\vx)}{\grad[\vthetap] \left. \ell_{\mathrm{CE}}(\vf(\vx; \vtheta); y(\vx)) \right\rvert_{\vtheta = \widehat{\vtheta}}}\\] where $\vthetap$ refers to the parameters of the final output layer, $\widehat{\vtheta}$ are the current parameter estimates, $\ell_{\mathrm{CE}}$ is the [cross-entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), and the expectation is over the predictions of the model $\widehat{\vtheta}$.
#     """

#     def embed(self, data: torch.Tensor) -> torch.Tensor:
#         assert isinstance(self.final_layer, nn.Linear), "Final layer must be linear."
#         assert self.final_layer.bias is None, "Final layer must not have bias."

#         logits = self.logits(data)  # (N, K)
#         probabilities = self(data)  # (N, C)

#         C = probabilities.size(1)

#         A = (C * probabilities - 1)[:, :, None]  # (N, C, 1)
#         B = logits[:, None, :]  # (N, 1, K)
#         J = torch.matmul(A, B).view(-1, A.shape[1] * B.shape[2])  # (N, C * K)
#         return J


# class OutputNormEmbedding(ClassificationModel, ModelWithEmbedding):
#     def embed(self, data: torch.Tensor) -> torch.Tensor:
#         assert isinstance(self.final_layer, nn.Linear), "Final layer must be linear."
#         assert self.final_layer.bias is None, "Final layer must not have bias."

#         logits = self.logits(data)  # (N, K)
#         probabilities = self(data)  # (N, C)

#         K = logits.size(1)
#         C = probabilities.size(1)

#         J = torch.zeros((data.size(0), C * K), dtype=torch.float32)
#         for i in range(data.size(0)):
#             W = self.final_layer.weight  # (C, K)
#             Z = logits[i, :][:, None] @ logits[i, :][None, :]  # (K, K)
#             J[i, :] = (W @ Z).view(-1, J.size(1))
#         return J
