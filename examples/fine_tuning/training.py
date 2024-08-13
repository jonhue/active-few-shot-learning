import copy
import wandb
import numpy as np
from tqdm import tqdm
import faiss  # type: ignore
import torch
from torch.utils.data import DataLoader
import afsl
from afsl.acquisition_functions import AcquisitionFunction, EmbeddingBased, Targeted
from afsl.active_data_loader import ActiveDataLoader
from afsl.adapters.faiss import Retriever
from afsl.utils import get_device
from examples.fine_tuning.data import CollectedData, Dataset
from examples.utils import accuracy


def train(
    model: torch.nn.Module,
    trainloader: DataLoader,
    valset: CollectedData,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,  # type: ignore
    num_epochs=5,
    use_best_model=False,
):
    best_acc = 0.0
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())

    device = get_device(model)
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        num_batches = 0

        if use_best_model:  # TODO: should refactor this
            model.train()
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        new_acc = accuracy(model, valset.inputs, valset.labels)
        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": running_loss / num_batches,
                "accuracy": new_acc,
            }
        )

        if new_acc >= best_acc:
            best_acc = new_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

    if use_best_model:
        model.load_state_dict(best_model_wts)
    return best_epoch, best_acc


def train_loop(
    model: torch.nn.Module,
    labels: torch.Tensor,
    train_inputs: afsl.data.Dataset,
    train_labels: torch.Tensor,
    valset: CollectedData,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,  # type: ignore
    acquisition_function: AcquisitionFunction,
    num_rounds=1_000,
    num_epochs=5,
    query_batch_size=10,
    train_batch_size=64,
    update_target=False,
    reweighting=True,
    reset_parameters=False,
    use_best_model=False,
    faiss_index_path=None,
    target_embeddings=None,
):
    data = Dataset(root="./data")
    wandb.log({"round": 0, "round_accuracy": 0.0})

    if faiss_index_path is not None:
        acquisition_function.subsample = False

        res = faiss.StandardGpuResources()
        index = faiss.read_index(faiss_index_path)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        retriever = Retriever(index, acquisition_function)  # type: ignore
    else:
        data_loader = ActiveDataLoader(
            dataset=train_inputs,
            batch_size=query_batch_size,
            acquisition_function=acquisition_function,
        )

    last_best_epoch, last_best_acc = -1, 0.0
    for i in range(num_rounds):
        if faiss_index_path is not None:
            assert isinstance(acquisition_function, Targeted)
            assert target_embeddings is not None

            acquisition_function.set_target(
                target_embeddings
            )  # ensure target set is reset to correct length
            query = acquisition_function.get_target().cpu().numpy()
            _batch_indices, _ = retriever.search(
                query=query, k=query_batch_size, k_mult=100
            )
            batch_indices = torch.tensor(_batch_indices)
        else:
            batch_indices = data_loader.next(model)
        batch_labels = train_labels[batch_indices]
        batch_mask = (batch_labels[:, None] == labels).any(dim=1)
        batch_inputs = [train_inputs[i] for i in batch_indices[batch_mask]]

        best_epoch, best_acc = -1, 0.0
        if len(batch_inputs) > 0:
            batch = CollectedData(
                inputs=torch.stack(batch_inputs),
                labels=batch_labels[batch_mask],
            )
            data.add_data(batch.inputs, batch.labels)
            trainloader = DataLoader(data, batch_size=train_batch_size, shuffle=True)

            if update_target and isinstance(
                acquisition_function, afsl.acquisition_functions.Targeted
            ):
                acquisition_function.add_to_target(batch.inputs)

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
            best_epoch, best_acc = train(
                model=model,
                trainloader=trainloader,
                valset=valset,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                use_best_model=use_best_model,
            )
            last_best_epoch, last_best_acc = best_epoch, best_acc
        else:
            best_epoch, best_acc = last_best_epoch, last_best_acc

        acc = accuracy(model, valset.inputs, valset.labels)
        wandb.log(
            {
                "round": i,
                "round_accuracy": acc,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
                "data_len": len(data),
                "missing_perc": data.valid_perc(torch.arange(5)),
            }
        )
    wandb.log(
        {
            "label_histogram": wandb.Histogram(
                np_histogram=np.histogram(
                    torch.tensor(data.targets).cpu().numpy(), bins=np.arange(11)
                )
            )
        }
    )
