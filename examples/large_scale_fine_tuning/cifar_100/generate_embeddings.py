import argparse
import time
from typing import Tuple
from tqdm import tqdm
import wandb
import faiss
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from afsl.adapters.faiss import ITLSearcher
from afsl.data import InputDataset
from afsl.utils import get_device
from examples.acquisition_functions import get_acquisition_function
from examples.large_scale_fine_tuning.data import Dataset
from examples.large_scale_fine_tuning.generate_embeddings import generate_embeddings
from examples.large_scale_fine_tuning.cifar_100.data import collect_test_data, get_datasets

from examples.large_scale_fine_tuning.cifar_100.model import (
    EfficientNetWithHallucinatedCrossEntropyEmbedding,
    EfficientNetWithLastLayerEmbedding,
)
from examples.utils import int_or_none

LR = 0.001
EPOCHS = 1
TRAIN_BATCH_SIZE = 64
REWEIGHTING = True
MODEL = EfficientNetWithLastLayerEmbedding  #  EfficientNetWithHallucinatedCrossEntropyEmbedding
RESET_PARAMS = False
LABELS = torch.arange(10)
LEN_TESTSET = 1_000

MINI_BATCH_SIZE = 1_000
NUM_WORKERS = 4
NUM_ROUNDS = 101

DEFAULT_NOISE_STD = 1.0
DEFAULT_QUERY_BATCH_SIZE = 10
DEFAULT_N_INIT = 100


# sbatch --gpus=1 --time=8:00:00 --mem-per-cpu=12000 --wrap="python examples/large_scale_fine_tuning/cifar_100/generate_embeddings.py"


def runner():
    torch.manual_seed(0)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = MODEL(output_dim=100)
    model.to(device)
    pretrained_model_wts = torch.load('./examples/large_scale_fine_tuning/cifar_100/pretrained_model_weights.pth', map_location=device)
    model.load_state_dict(pretrained_model_wts)

    trainset, _testset = get_datasets()
    # train_labels = torch.tensor(trainset.targets)
    # train_inputs = InputDataset(trainset)

    # __testset, _ = collect_test_data(
    #     _testset,
    #     n_test=LEN_TESTSET,
    #     restrict_to_labels=LABELS,
    # )
    # testset = Dataset(root="./data")
    # testset.add_data(images=__testset.inputs, targets=__testset.labels)
    # # target = testset.inputs
    testset = _testset

    print("Loaded datasets")

    train_dataloader = DataLoader(trainset, batch_size=100, shuffle=False)
    device = get_device(model)
    train_embeddings, train_labels = generate_embeddings(model, device, train_dataloader)
    torch.save(train_embeddings, 'train_embeddings.pt')
    torch.save(train_labels, 'train_labels.pt')

    test_dataloader = DataLoader(testset, batch_size=100, shuffle=False)
    device = get_device(model)
    test_embeddings, test_labels = generate_embeddings(model, device, test_dataloader, restrict_to_labels=LABELS)
    torch.save(test_embeddings, 'test_embeddings.pt')
    torch.save(test_labels, 'test_labels.pt')


def main(args):
    t_start = time.process_time()
    runner()
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
