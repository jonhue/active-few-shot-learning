import argparse
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from afsl.acquisition_functions.itl import ITL
from afsl.acquisition_functions.random import Random
from afsl.acquisition_functions.undirected_itl import UndirectedITL
from afsl.data import InputDataset
from examples.cifar.data import collect_test_data, get_datasets

from examples.cifar.model import (
    EfficientNetWithHallucinatedCrossEntropyEmbedding,
    EfficientNetWithLastLayerEmbedding,
)
from examples.cifar.training import train_loop
from examples.launch_utils import int_or_none

LR = 0.001
EPOCHS = 5
TRAIN_BATCH_SIZE = 64
REWEIGHTING = True
MODEL = EfficientNetWithLastLayerEmbedding  #  EfficientNetWithHallucinatedCrossEntropyEmbedding
RESET_PARAMS = False
LABELS = torch.arange(10)
IMBALANCED_TEST = (
    None  # ImbalancedTestConfig(drop_perc=0.5, drop_labels=torch.arange(5))
)
IMBALANCED_TRAIN_PERC = None  # 0.8

MINI_BATCH_SIZE = 1_000
NUM_WORKERS = 4
NUM_ROUNDS = 100

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
    subsample_acquisition: bool,
    update_target: bool,
    debug: bool,
):
    wandb.init(
        name="First experiment",
        dir="/cluster/scratch/jhuebotter/wandb/cifar-fine-tuning",
        project="Fine-tuning CIFAR",
        config={
            "learning_rate": LR,
            "architecture": "EfficientNet (partially frozen) with-bias",
            "dataset": "CIFAR-100",
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "model": MODEL,
            "reweighting": REWEIGHTING,
            "subsample_acquisition": subsample_acquisition,
            "noise_std": noise_std,
            "seed": seed,
            "alg": alg,
            "validation": "hold-out",
            "reset_params": RESET_PARAMS,
            "imbalanced_test": IMBALANCED_TEST,
            "query_batch_size": query_batch_size,
            "n_init": n_init,
            "labels": LABELS.tolist(),
            "subsampled_target_frac": subsampled_target_frac,
            "max_target_size": max_target_size,
            "update_target": update_target,
            "variation": "old-data",
        },
        mode="offline" if debug else "online",
    )

    print("SEED:", seed, "LABELS:", LABELS, "ALG:", alg)
    torch.manual_seed(seed)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = MODEL(output_dim=LABELS.size(0))
    model.to(device)

    # Define the loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Define trainset
    trainset, _testset = get_datasets(imbalanced_train_perc=IMBALANCED_TRAIN_PERC)
    train_labels = torch.tensor(trainset.targets)
    if alg == "OracleRandom":
        mask = (train_labels[:, None] == LABELS).any(dim=1)
        trainset.data = trainset.data[mask]
        train_labels = train_labels[mask]
    if debug:
        trainset.data = trainset.data[:10]
        train_labels = train_labels[:10]
    train_inputs = InputDataset(trainset)

    # Define testset and valset
    testset, valset = collect_test_data(
        _testset,
        n_test=n_init,
        restrict_to_labels=LABELS,
        imbalanced_test_config=IMBALANCED_TEST,
    )
    target = testset.inputs

    print("validation labels:", torch.unique(valset.labels))

    num_workers = NUM_WORKERS if not debug else 0
    if alg == "Random" or alg == "OracleRandom":
        acquisition_function = Random(
            mini_batch_size=MINI_BATCH_SIZE,
            num_workers=num_workers,
        )
    elif alg == "ITL" or alg == "ITL-nonsequential":
        acquisition_function = ITL(
            target=target,
            noise_std=noise_std,
            subsampled_target_frac=subsampled_target_frac,
            max_target_size=max_target_size,
            mini_batch_size=MINI_BATCH_SIZE,
            num_workers=num_workers,
            subsample=subsample_acquisition,
            force_nonsequential=(alg == "ITL-nonsequential"),
        )
    elif alg == "UndirectedITL":
        acquisition_function = UndirectedITL(
            noise_std=noise_std,
            mini_batch_size=MINI_BATCH_SIZE,
            num_workers=num_workers,
            subsample=subsample_acquisition,
        )
    else:
        raise NotImplementedError

    train_loop(
        model=model,
        labels=LABELS,
        train_inputs=train_inputs,
        train_labels=train_labels,
        valset=valset,
        criterion=criterion,
        optimizer=optimizer,
        acquisition_function=acquisition_function,
        num_rounds=NUM_ROUNDS,
        num_epochs=EPOCHS,
        query_batch_size=query_batch_size,
        train_batch_size=TRAIN_BATCH_SIZE,
        update_target=update_target,
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
        subsample_acquisition=bool(args.subsample_acquisition),
        update_target=bool(args.update_target),
        debug=args.debug,
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
    parser.add_argument("--max-target-size", type=int_or_none, default=None)
    parser.add_argument("--subsample-acquisition", type=int, default=1)
    parser.add_argument("--update-target", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
