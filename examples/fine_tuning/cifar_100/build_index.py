import argparse
import time
import wandb
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from afsl.utils import DEFAULT_EMBEDDING_BATCH_SIZE, mini_batch_wrapper
from examples.acquisition_functions import get_acquisition_function
from examples.fine_tuning.cifar_100.data import collect_test_data, get_datasets

from examples.fine_tuning.cifar_100.model import (
    EfficientNetWithHallucinatedCrossEntropyEmbedding,
    EfficientNetWithLastLayerEmbedding,
)
from examples.fine_tuning.training import train_loop
from examples.utils import int_or_none

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
NUM_ROUNDS = 101

DEFAULT_NOISE_STD = None
DEFAULT_QUERY_BATCH_SIZE = 10
DEFAULT_N_INIT = 100


def experiment(
    debug: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = MODEL(output_dim=LABELS.size(0))
    model.to(device)

    # Define trainset
    trainset, _testset = get_datasets(imbalanced_train_perc=IMBALANCED_TRAIN_PERC)
    train_labels = torch.tensor(trainset.targets)
    if debug:
        trainset.data = trainset.data[:10]
        train_labels = train_labels[:10]
    data_loader = DataLoader(
        trainset,
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        num_workers=1,
        shuffle=False,
    )

    model.eval()
    with torch.no_grad():
        embeddings = []
        for data, _ in data_loader:
            embeddings.append(model.embed(data.to(device, non_blocking=True)))
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    d = embeddings.shape[1]

    l2_index = faiss.IndexFlatL2(d)
    l2_index.add(embeddings)  # type: ignore
    faiss.write_index(l2_index, "examples/fine_tuning/cifar_100/index/l2_index.faiss")

    ip_index = faiss.IndexFlatIP(d)
    ip_index.add(embeddings)  # type: ignore
    faiss.write_index(ip_index, "examples/fine_tuning/cifar_100/index/ip_index.faiss")

    # absip_index = faiss.IndexFlat(d, faiss.METRIC_ABS_INNER_PRODUCT)
    # absip_index.add(embeddings)  # type: ignore
    # faiss.write_index(absip_index, "examples/fine_tuning/cifar_100/index/absip_index.faiss")


def main(args):
    t_start = time.process_time()
    experiment(
        debug=args.debug,
    )
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
