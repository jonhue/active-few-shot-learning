from typing import NamedTuple
import torch
from torchvision.datasets.vision import VisionDataset


class CollectedData(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return self.inputs.size(0)


class Dataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(Dataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.data = []
        self.targets = []

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def add_data(self, images, targets):
        self.data.extend(images)
        self.targets.extend(targets)

    def valid_perc(self, labels):
        return (torch.tensor(self.targets)[:, None] == labels).any(dim=1).sum() / len(
            self.targets
        )
