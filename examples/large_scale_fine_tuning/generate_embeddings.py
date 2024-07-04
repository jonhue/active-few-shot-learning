from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate_embeddings(model: torch.nn.Module, device: torch.device, dataloader: DataLoader, restrict_to_labels: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        embeddings = []
        labels = []
        for x, y in tqdm(dataloader):
            mask = torch.any(y == restrict_to_labels.reshape(-1, 1), dim=0) if restrict_to_labels is not None else torch.ones_like(y, dtype=torch.bool)
            if torch.sum(mask) == 0:
                continue
            z = model.embed(x[mask].to(device))
            embeddings.append(z)
            labels.append(y[mask])
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels
