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
from examples.acquisition_functions import get_acquisition_function
from examples.fine_tuning.imagenet.data import ImageNetKaggle, collect_test_data, get_datasets

from examples.fine_tuning.imagenet.model import (
    EfficientNetWithHallucinatedCrossEntropyEmbedding,
    EfficientNetWithLastLayerEmbedding,
)
from examples.fine_tuning.imagenet.training import train_loop
from examples.utils import int_or_none

LR = 0.001
EPOCHS = 1
TRAIN_BATCH_SIZE = 64
REWEIGHTING = True
MODEL = EfficientNetWithLastLayerEmbedding  #  EfficientNetWithHallucinatedCrossEntropyEmbedding
RESET_PARAMS = False
LABELS = torch.tensor([436, 705, 751, 817, 864])  # classes containing "car,"

MINI_BATCH_SIZE = 1_000
NUM_WORKERS = 4
NUM_ROUNDS = 101

DEFAULT_NOISE_STD = 1.0
DEFAULT_QUERY_BATCH_SIZE = 10
DEFAULT_N_INIT = 100


def generate_embeddings(model, device, dataloader, restrict_to_labels: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        embeddings = []
        labels = []
        for x, y in tqdm(dataloader):
            mask = torch.any(y == restrict_to_labels.reshape(-1, 1), dim=0) if restrict_to_labels is not None else torch.ones_like(y)
            if torch.sum(mask) == 0:
                continue
            z = torch.nn.DataParallel(model).embed(x[mask].to(device))
            embeddings.append(z)
            labels.append(y[mask])
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels


def experiment(
    seed: int,
):
    print("SEED:", seed, "LABELS:", LABELS)
    torch.manual_seed(seed)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = MODEL(output_dim=LABELS.size(0))
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    # torch.save(torch.tensor([0, 1, 2]), "test.pt")
    # print("TEST COMPLETE")

    # train_dataset = ImageNetKaggle("/cluster/scratch/jhuebotter/imagenet", "train", transform)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=64, # may need to reduce this depending on your GPU
    #     num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
    #     shuffle=False,
    #     drop_last=False,
    #     pin_memory=True
    # )

    test_dataset = ImageNetKaggle("/cluster/scratch/jhuebotter/imagenet", "val", transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64, # may need to reduce this depending on your GPU
        num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # train_embeddings, _ = generate_embeddings(model, device, train_dataloader)
    # torch.save(train_embeddings, 'train_embeddings.pt')

    test_embeddings, test_labels = generate_embeddings(model, device, test_dataloader, restrict_to_labels=LABELS)
    torch.save(test_embeddings, 'test_embeddings.pt')
    torch.save(test_labels, 'test_labels.pt')


def main(args):
    t_start = time.process_time()
    experiment(
        seed=args.seed,
    )
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
