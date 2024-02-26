import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import afsl
from afsl.acquisition_functions import AcquisitionFunction
from afsl.active_data_loader import ActiveDataLoader
from afsl.utils import get_device
from examples.cifar.data import CollectedData, Dataset
from examples.utils import accuracy


def train(
    model: torch.nn.Module,
    trainloader: DataLoader,
    valset: CollectedData,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs=5,
):
    device = get_device(model)
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        num_batches = 0
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        acc = accuracy(model, valset.inputs, valset.labels)
        wandb.log(
            {"epoch": epoch + 1, "loss": running_loss / num_batches, "accuracy": acc}
        )


def train_loop(
    model: torch.nn.Module,
    labels: torch.Tensor,
    train_inputs: afsl.data.Dataset,
    train_labels: torch.Tensor,
    valset: CollectedData,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    acquisition_function: AcquisitionFunction,
    num_rounds=1_000,
    num_epochs=5,
    query_batch_size=10,
    train_batch_size=64,
    # randomize=None,  # every k steps, select data u.a.r.
    # test_subset_size=None,  # use a random subset of the test data for the acquisition function
    reweighting=True,  # dynamically reweight loss to address imbalanced dataset
    reset_parameters=False,  # reset parameters after each round
):
    data = Dataset(root="./data")
    wandb.log({"round": 0, "round_accuracy": 0.0})

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:

    # unique_test_labels = torch.unique(testset.labels)

    data_loader = ActiveDataLoader(
        dataset=train_inputs,
        batch_size=query_batch_size,
        acquisition_function=acquisition_function,
    )

    for i in range(num_rounds):
        # sub_test_inputs = (
        #     test_inputs[torch.randperm(len(test_inputs))[:test_subset_size]]
        #     if test_subset_size is not None
        #     else test_inputs
        # )
        indices = data_loader.next(model).cpu()
        batch_labels = train_labels[indices]
        mask = (batch_labels[:, None] == labels).any(dim=1)
        batch = CollectedData(
            inputs=torch.stack([train_inputs[i] for i in indices])[mask],
            labels=batch_labels[mask],
        )
        data.add_data(batch.inputs, batch.labels)

        if len(data) > 0:
            trainloader = DataLoader(data, batch_size=train_batch_size, shuffle=True)

            print("data labels:", torch.unique(torch.tensor(data.targets)))
            if reweighting:
                criterion.weight = (
                    len(data)
                    / torch.bincount(
                        torch.tensor(data.targets), minlength=labels.size(0)
                    )
                ).to(get_device(model))

            if reset_parameters:
                model.reset()
            train(
                model=model,
                trainloader=trainloader,
                valset=valset,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
            )

        acc = accuracy(model, valset.inputs, valset.labels)
        wandb.log(
            {
                "round": i,
                "round_accuracy": acc,
                # "valid_perc": prior_data.valid_perc(unique_test_labels),
                "data_len": len(data),
                "missing_perc": data.valid_perc(torch.tensor(range(5))),
                # "label": labels[train_mask][indices],
            }
        )
        # wandb.log({"round": j + 1, "round_accuracy": acc, "label": labels[indices]})
    wandb.log(
        {
            "label_histogram": wandb.Histogram(
                np_histogram=np.histogram(
                    torch.tensor(data.targets).cpu().numpy(), bins=np.arange(11)
                )
            )
        }
    )

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))