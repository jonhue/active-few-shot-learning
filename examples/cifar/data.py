import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from PIL import Image


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


def get_data_loaders(
    batch_size=64,
    train_batch_size=None,
    imbalanced_train_perc=None,
):
    trainset, testset = get_datasets(imbalanced_train_perc=imbalanced_train_perc)

    # Define data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size if train_batch_size is not None else batch_size,
        shuffle=True,
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def collect_data(dataloader: DataLoader):
    inputs = []
    labels = []
    for data in dataloader:
        inputs.append(data[0])
        labels.append(data[1])
    return torch.cat(inputs), torch.cat(labels)


def collect_test_data(testloader, labels=None):
    test_inputs, test_labels = collect_data(testloader)
    if labels is None:
        return test_inputs, test_labels
    else:
        mask = torch.any(test_labels == torch.tensor(labels).reshape(-1, 1), dim=0)
        return test_inputs[mask], test_labels[mask]
