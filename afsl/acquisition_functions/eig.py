import torch
import torch.nn.functional as F
from afsl.acquisition_functions import (
    BatchAcquisitionFunction,
    EmbeddingBased,
    Targeted,
)
from afsl.model import ClassificationModel
from afsl.utils import (
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SUBSAMPLE,
    get_device,
    mini_batch_wrapper,
)


class EIG(EmbeddingBased, Targeted, BatchAcquisitionFunction[ClassificationModel]):
    def __init__(
        self,
        target: torch.Tensor,
        mini_batch_size=DEFAULT_MINI_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        subsample=DEFAULT_SUBSAMPLE,
    ):
        BatchAcquisitionFunction.__init__(
            self,
            mini_batch_size=mini_batch_size,
            num_workers=num_workers,
            subsample=subsample,
        )
        Targeted.__init__(self, target=target)

    def compute(
        self,
        model: ClassificationModel,
        data: torch.Tensor,
    ) -> torch.Tensor:
        prior_entropy = self.compute_model_entropy(model)
        probs = mini_batch_wrapper(
            fn=lambda batch: self.compute_model_probs(model=model, data=batch),
            data=data,
            batch_size=100,
        ).cpu()

        device = get_device(model)
        n = data.size(0)
        C = model.final_layer.out_features
        posterior_entropy = torch.zeros(size=(n, C))
        for i in range(n):
            x = data[i].unsqueeze(0)
            original_params = model.final_layer.weight.data.clone()
            for c in range(C):
                new_params = model.final_layer.weight.clone()
                with torch.no_grad():
                    model.final_layer.weight.data.copy_(new_params)

                model.train()
                optimizer = torch.optim.SGD([model.final_layer.weight], lr=0.01)
                optimizer.zero_grad()
                loss = F.cross_entropy(
                    model(x.to(device)), torch.tensor([c], device=device)
                )
                loss.backward()
                optimizer.step()

                posterior_entropy[i, c] = self.compute_model_entropy(model)
            with torch.no_grad():
                model.final_layer.weight.data.copy_(original_params)

        conditional_entropy = torch.sum(probs * posterior_entropy.cpu(), dim=1)
        mi = prior_entropy - conditional_entropy
        return mi

    def compute_model_probs(
        self, model: ClassificationModel, data: torch.Tensor
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(
                model(data.to(get_device(model), non_blocking=True)), dim=1
            )
            return probs

    def compute_model_entropy(self, model: ClassificationModel) -> float:
        probs = self.compute_model_probs(model=model, data=self._target)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        return (
            entropies.sum().item()
        )  # sum of individual entropies for all target points