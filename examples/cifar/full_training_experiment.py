import argparse
import math
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from afsl.acquisition_functions.itl import ITL
from examples.cifar.data import collect_data, collect_test_data, get_data_loaders

from examples.cifar.model import EfficientNet
from examples.cifar.training import CollectedData, train_loop

LR = 0.001
EPOCHS = 5
TRAIN_BATCH_SIZE = 64
REWEIGHTING = True
EMBEDDING = "hallucinated cross-entropy"
RESET_PARAMS = False
LABELS = torch.arange(10)  # torch.tensor([3, 6, 9])
IMBALANCED_TEST_PERC = None  # 0.5
IMBALANCED_TEST_DROP_LABELS = torch.arange(5)
IMBALANCED_TRAIN_PERC = None  # 0.8

MINI_BATCH_SIZE = 1_000
NUM_ROUNDS = 300

DEFAULT_NOISE_STD = 1.0
DEFAULT_QUERY_BATCH_SIZE = 10
DEFAULT_N_INIT = 100


def experiment(
    seed: int,
    alg: str,
    noise_std: float,
    n_init: int,
    query_batch_size: int,
    subsampled_target_frac: float,
    max_target_size: int | None,
):
    wandb.init(
        name="First experiment",
        dir="/cluster/scratch/jhuebotter/wandb/cifar-fine-tuning",
        project="Fine-tuning CIFAR",
        config={
            "learning_rate": LR,
            "architecture": "EfficientNet (partially frozen)",
            "dataset": "CIFAR-100",
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "model": "Pretrained EfficientNet",
            "reweighting": REWEIGHTING,
            "noise_std": noise_std,
            "seed": seed,
            "alg": alg,
            "validation": "hold-out",
            "embedding": EMBEDDING,
            "reset_params": RESET_PARAMS,
            "imbalanced_test_perc": IMBALANCED_TEST_PERC,
            "imbalanced_train_perc": IMBALANCED_TRAIN_PERC,
            "query_batch_size": query_batch_size,
            "n_init": n_init,
            "labels": LABELS.tolist(),
            "subsampled_target_frac": subsampled_target_frac,
            "max_target_size": max_target_size,
        },
        # mode="offline"
    )

    print("SEED:", seed, "LABELS:", LABELS, "ALG:", alg)
    torch.manual_seed(seed)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = EfficientNet()
    model.to(device)

    # Define the loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Define trainset
    trainloader, testloader = get_data_loaders(
        imbalanced_train_perc=IMBALANCED_TRAIN_PERC
    )
    trainset = CollectedData(*collect_data(trainloader))

    # Define testset and valset
    test_inputs, test_labels = collect_test_data(testloader, labels=LABELS)
    shuffle_mask = torch.randperm(test_labels.size(0))
    n = math.floor(test_inputs.size(0) / 2)
    val_inputs = test_inputs[shuffle_mask][:n]
    val_labels = test_labels[shuffle_mask][:n]
    valset = CollectedData(val_inputs, val_labels)
    test_inputs = test_inputs[shuffle_mask][n:]
    test_labels = test_labels[shuffle_mask][n:]
    if IMBALANCED_TEST_PERC is not None:
        drop_mask = torch.any(
            test_labels == IMBALANCED_TEST_DROP_LABELS.reshape(-1, 1), dim=0
        )
        true_indices = torch.where(drop_mask)[0]
        indices_to_flip = true_indices[
            torch.randperm(len(true_indices))[
                : int(IMBALANCED_TEST_PERC * drop_mask.sum())
            ]
        ]
        drop_mask[indices_to_flip] = False

        test_inputs = test_inputs[~drop_mask]
        test_labels = test_labels[~drop_mask]
    test_inputs = test_inputs[:n_init]
    test_labels = test_labels[:n_init]
    testset = CollectedData(test_inputs, test_labels)
    target = testset.inputs

    print("validation labels:", torch.unique(valset.labels))

    use_oracle_train_labels = False
    if alg == "ITL":
        acquisition_function = ITL(
            target=target,
            noise_std=noise_std,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=MINI_BATCH_SIZE,
        )
    else:
        raise NotImplementedError

    train_loop(
        model=model,
        labels=LABELS,
        trainset=trainset,
        valset=valset,
        criterion=criterion,
        optimizer=optimizer,
        acquisition_function=acquisition_function,
        num_rounds=NUM_ROUNDS,
        num_epochs=EPOCHS,
        query_batch_size=query_batch_size,
        train_batch_size=TRAIN_BATCH_SIZE,
        use_oracle_train_labels=use_oracle_train_labels,
        reweighting=REWEIGHTING,
        reset_parameters=RESET_PARAMS,
    )
    wandb.finish()


def main(args):
    t_start = time.process_time()
    experiment(
        seed=args.seed,
        alg=args.alg,
        noise_std=args.noise_std,
        n_init=args.n_init,
        query_batch_size=args.query_batch_size,
        subsampled_target_frac=args.subsampled_target_frac,
        max_target_size=args.max_target_size,
    )
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alg", type=str, default="ITL")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    parser.add_argument(
        "--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE
    )
    parser.add_argument("--subsampled-target-frac", type=float, default=0.5)
    parser.add_argument("--max-target-size", type=int, default=None)
    args = parser.parse_args()
    main(args)
