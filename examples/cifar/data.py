import math
from typing import NamedTuple
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from PIL import Image


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


class ImbalancedDataset(Dataset):
    def __init__(
        self,
        dataset: torchvision.datasets.CIFAR100,
        transform,
        drop_labels=range(10),
        drop_prob=0.8,
    ):
        """
        :param dataset: The original dataset.
        :param transform: The transformation to apply to the images.
        :param drop_labels: The labels to drop data from.
        :param drop_prob: The percentage of dropped data from the specified labels.
        """

        self.transform = transform

        drop_mask = torch.isin(
            torch.tensor(dataset.targets), torch.tensor(drop_labels)
        ) & (torch.rand(len(dataset)) < drop_prob)
        self.data = dataset.data[~drop_mask]
        self.targets = torch.tensor(dataset.targets)[~drop_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        return self.transform(img=Image.fromarray(img)), target


def get_datasets(imbalanced_train_perc=None):
    # Transform images to correct inputs for EfficientNet
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load the CIFAR100 dataset
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    if imbalanced_train_perc is not None:
        trainset = ImbalancedDataset(
            dataset=trainset, transform=transform, drop_prob=imbalanced_train_perc
        )

    return trainset, testset


def collect_data(dataloader: DataLoader):
    inputs = []
    labels = []
    for data in dataloader:
        inputs.append(data[0])
        labels.append(data[1])
    return torch.cat(inputs), torch.cat(labels)


def collect_dataset(
    dataset: torchvision.datasets.CIFAR100,
    restrict_to_labels: torch.Tensor | None = None,
):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    inputs = []
    labels = []
    for data in dataloader:
        if restrict_to_labels is not None:
            mask = torch.any(data[1] == restrict_to_labels.reshape(-1, 1), dim=0)
        else:
            mask = torch.ones(data[0].shape[0], dtype=torch.bool)
        inputs.append(data[0][mask])
        labels.append(data[1][mask])
    return torch.cat(inputs), torch.cat(labels)


class ImbalancedTestConfig(NamedTuple):
    drop_perc: float
    drop_labels: torch.Tensor


def collect_test_data(
    _testset: torchvision.datasets.CIFAR100,
    n_test: int,
    restrict_to_labels: torch.Tensor | None = None,
    imbalanced_test_config: ImbalancedTestConfig | None = None,
):
    test_inputs, test_labels = collect_dataset(
        _testset, restrict_to_labels=restrict_to_labels
    )
    shuffle_mask = torch.randperm(test_labels.size(0))
    n = math.floor(test_inputs.size(0) / 2)
    val_inputs = test_inputs[shuffle_mask][:n]
    val_labels = test_labels[shuffle_mask][:n]
    valset = CollectedData(val_inputs, val_labels)
    test_inputs = test_inputs[shuffle_mask][n:]
    test_labels = test_labels[shuffle_mask][n:]
    if imbalanced_test_config is not None:
        drop_mask = torch.any(
            test_labels == imbalanced_test_config.drop_labels.reshape(-1, 1), dim=0
        )
        true_indices = torch.where(drop_mask)[0]
        indices_to_flip = true_indices[
            torch.randperm(len(true_indices))[
                : int(imbalanced_test_config.drop_perc * drop_mask.sum())
            ]
        ]
        drop_mask[indices_to_flip] = False

        test_inputs = test_inputs[~drop_mask]
        test_labels = test_labels[~drop_mask]
    test_inputs = test_inputs[:n_test]
    test_labels = test_labels[:n_test]
    testset = CollectedData(test_inputs, test_labels)
    return testset, valset
