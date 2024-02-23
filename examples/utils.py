import torch
from torch.utils.data import DataLoader, TensorDataset

from afsl.utils import get_device


def accuracy(
    model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor
) -> float:
    model.eval()

    correct = 0
    total = 0

    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=1_000)
    device = get_device(model)
    with torch.no_grad():
        for inputs, labels in dataloader:
            predicted = model.predict(inputs.to(device))
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return 100 * correct / total
