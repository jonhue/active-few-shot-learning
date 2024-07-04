import argparse
import copy
import time
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from afsl.utils import get_device
from examples.large_scale_fine_tuning.cifar_100.data import get_datasets

from examples.large_scale_fine_tuning.cifar_100.model import (
    EfficientNetWithLastLayerEmbedding,
)
from examples.utils import accuracy_from_dataloader

LR = 0.001
EPOCHS = 1_000
TRAIN_BATCH_SIZE = 8
MODEL = EfficientNetWithLastLayerEmbedding


# sbatch --gpus=1 --time=8:00:00 --mem-per-cpu=12000 --wrap="python examples/large_scale_fine_tuning/cifar_100/pretrain_model.py"


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    device = get_device(model)
    running_loss = 0.0
    num_batches = 0

    model.train()
    for _, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
):
    device = get_device(model)
    running_loss = 0.0
    num_batches = 0

    model.train()
    for _, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def train(
    model: torch.nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    num_epochs=1_000,
    use_best_model=True,
):
    best_val_loss = torch.inf
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopper = EarlyStopper(patience=3, min_delta=10)

    wandb.log({"epoch": 0, "train_loss": torch.inf, "val_loss": torch.inf, "train_acc": 0, "val_acc": 0})

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, trainloader, criterion, optimizer)
        val_loss = validate_epoch(model, valloader, criterion)
        if early_stopper.early_stop(val_loss):
            break

        scheduler.step()

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc": accuracy_from_dataloader(model, trainloader), "val_acc": accuracy_from_dataloader(model, valloader)})

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'interm_pretrained_model_weights.pth')

    if use_best_model:
        model.load_state_dict(best_model_wts)
    return best_epoch, best_val_loss


def runner(debug: bool):
    wandb.init(
        name="Unfrozen model & Augmentations & SGD",
        dir="/cluster/scratch/jhuebotter/wandb/cifar-pretraining",
        project="Pretrain EfficientNet-B0 on CIFAR-100",
        config={
            "learning_rate": LR,
            "architecture": "EfficientNet (unfrozen) with-bias",
            "dataset": "CIFAR-100",
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "model": MODEL,
        },
        mode="offline" if debug else "online",
    )
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = MODEL(output_dim=100)
    model.to(device)

    # Define the loss criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Load dataset
    # _trainset, _testset = get_datasets()
    # train_idx, valid_idx = train_test_split(np.arange(len(_trainset)), test_size=0.2, shuffle=True, stratify=np.array(_trainset.targets), random_state=1)
    # trainset = Subset(_trainset, train_idx)
    # valset = Subset(_trainset, valid_idx)
    trainset, valset = get_datasets()

    trainloader = DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    train(model, trainloader, valloader, criterion, optimizer, scheduler, num_epochs=EPOCHS)

    torch.save(model.state_dict(), 'pretrained_model_weights.pth')


def main(args):
    t_start = time.process_time()
    runner(debug=args.debug)
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
