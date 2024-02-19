import math
from typing import Generic
import torch
from afsl.acquisition_functions import AcquisitionFunction
from afsl.acquisition_functions.itl import ITL
from afsl.embeddings import M, Embedding
from afsl.embeddings.latent import LatentEmbedding
from afsl.types import Target


class ActiveDataLoader(Generic[M]):
    data: torch.Tensor
    target: Target
    batch_size: int
    acquisition_function: AcquisitionFunction
    embedding: Embedding[M]
    subsampled_target_frac: float
    max_target_size: int | None

    def __init__(
        self,
        data: torch.Tensor,
        target: Target,
        batch_size: int,
        acquisition_function: AcquisitionFunction = ITL(),
        embedding: Embedding[M] = LatentEmbedding(),
        subsampled_target_frac: float = 0.5,
        max_target_size: int | None = None,
    ):
        assert data.size(0) > 0, "Data must be non-empty"
        assert batch_size > 0, "Batch size must be positive"
        assert (
            subsampled_target_frac > 0 and subsampled_target_frac <= 1
        ), "Fraction of target must be in (0, 1]"
        assert (
            max_target_size is None or max_target_size > 0
        ), "Max target size must be positive"

        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.embedding = embedding
        self.subsampled_target_frac = subsampled_target_frac
        self.max_target_size = max_target_size

    def next(self, model: M, Sigma: torch.Tensor | None = None) -> torch.Tensor:
        target = self._subsample_target()
        return self.acquisition_function.select(
            batch_size=self.batch_size,
            embedding=self.embedding,
            model=model,
            data=self.data,
            target=target,
            Sigma=Sigma,
        )

    def _subsample_target(self) -> Target:
        if self.target is None:
            return None

        m = self.target.size(0)
        max_target_size = (
            self.max_target_size if self.max_target_size is not None else m
        )
        return self.target[
            torch.randperm(m)[
                : min(math.ceil(self.subsampled_target_frac * m), max_target_size)
            ]
        ]
