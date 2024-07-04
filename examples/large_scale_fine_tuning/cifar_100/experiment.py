import argparse
import copy
import time
from tqdm import tqdm
import wandb
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from afsl.adapters.faiss import ITLSearcher
from afsl.data import InputDataset
from examples.acquisition_functions import get_acquisition_function
# from examples.fine_tuning.cifar_100.data import collect_test_data, get_datasets

from examples.large_scale_fine_tuning.cifar_100.data import collect_dataset, collect_test_data, get_datasets
from examples.large_scale_fine_tuning.cifar_100.model import (
    EfficientNetWithHallucinatedCrossEntropyEmbedding,
    EfficientNetWithLastLayerEmbedding,
)
from examples.large_scale_fine_tuning.data import CollectedData, Dataset
# from examples.fine_tuning.training import train_loop
from examples.large_scale_fine_tuning.training import train
from examples.utils import accuracy, int_or_none

LR = 0.0001
EPOCHS = 100
USE_BEST_MODEL = True
TRAIN_BATCH_SIZE = 1
REWEIGHTING = True
MODEL = EfficientNetWithLastLayerEmbedding  #  EfficientNetWithHallucinatedCrossEntropyEmbedding
RESET_PARAMS = True
LABELS = torch.arange(10)

NUM_WORKERS = 4
# NUM_ROUNDS = 101

DEFAULT_NOISE_STD = 1.0
# DEFAULT_QUERY_BATCH_SIZE = 10
DEFAULT_N_INIT = 10  # maximum: 500

K = 11


def experiment(
    seed: int,
    idx: int,
    alg: str,
    noise_std: float,
    n_init: int,
    epochs: int,
    # query_batch_size: int,
    # subsampled_target_frac: float,
    # max_target_size: int | None,
    # subsample_acquisition: bool,
    # update_target: bool,
    indiv_fine_tuning: bool,
    lr: float,
    debug: bool,
):
    wandb.init(
        name="Fifth experiment with weaker model",
        dir="/cluster/scratch/jhuebotter/wandb/cifar-large-scale-fine-tuning",
        project="Large-scale Fine-tuning CIFAR",
        config={
            "learning_rate": lr,
            "architecture": "Pretrained EfficientNet (unfrozen) with-bias",
            "dataset": "CIFAR-100",
            "epochs": epochs,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "model": MODEL,
            "reweighting": REWEIGHTING,
            # "subsample_acquisition": subsample_acquisition,
            "noise_std": noise_std,
            "seed": seed,
            # "idx": idx,
            "alg": alg,
            "reset_params": RESET_PARAMS,
            # "imbalanced_test": IMBALANCED_TEST,
            # "query_batch_size": query_batch_size,
            "n_init": n_init,
            "labels": LABELS.tolist(),
            "indiv_fine_tuning": indiv_fine_tuning,
            # "subsampled_target_frac": subsampled_target_frac,
            # "max_target_size": max_target_size,
            # "update_target": update_target,
        },
        mode="offline" if debug else "online",
    )

    print("SEED:", seed, "LABELS:", LABELS, "ALG:", alg)
    torch.manual_seed(0)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    trainset, testset = get_datasets()
    test_inputs, test_labels = collect_dataset(testset)
    shuffle_mask = torch.randperm(test_labels.size(0))
    test_inputs = test_inputs[shuffle_mask][:n_init]
    test_labels = test_labels[shuffle_mask][:n_init]
    # testset, _ = collect_test_data(
    #     _testset,
    #     n_test=n_init,
    #     restrict_to_labels=LABELS,
    # )

    # Load embeddings
    train_embeddings = torch.load("./examples/large_scale_fine_tuning/cifar_100/embeddings/train_embeddings.pt", map_location=device)
    train_labels = torch.load("./examples/large_scale_fine_tuning/cifar_100/embeddings/train_labels.pt", map_location=device)
    test_embeddings = torch.load("./examples/large_scale_fine_tuning/cifar_100/embeddings/test_embeddings.pt", map_location=device)[shuffle_mask.cuda()][:n_init]
    # test_labels = torch.load("./examples/large_scale_fine_tuning/cifar_100/embeddings/test_labels.pt", map_location=device)[:n_init]
    # assert (trainset[0][1] == train_labels[0]) and (testset.labels[0] == test_labels[0]), "Embeddings are out of order!"
    assert trainset[0][1] == train_labels[0], "Embeddings are out of order!"

    torch.manual_seed(seed)

    # Define model
    model = MODEL(output_dim=100)
    model.to(device)
    pretrained_model_wts = torch.load('./examples/large_scale_fine_tuning/cifar_100/weak_pretrained_model_weights.pth', map_location=device)
    model.load_state_dict(pretrained_model_wts)

    # Define the loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    d = train_embeddings[0].size(0)
    index = faiss.IndexFlatIP(d)
    index.add(train_embeddings.cpu().numpy())
    searcher = ITLSearcher(index, alg=alg, noise_std=noise_std)

    if indiv_fine_tuning:
        raw_indices = searcher.batch_search(queries=test_embeddings.cpu().numpy()[:, np.newaxis, :], k=K, k_mult=100.0)
        indices = torch.tensor(raw_indices).reshape(n_init, K)

        # num = 0
        for k in torch.arange(start=0, step=1, end=K):
            if k == 0:
                best_epoch = torch.nan
                best_acc = accuracy(model, test_inputs, test_labels)
                wandb.log(
                    {
                        "k": k,
                        # "i": i,
                        "accuracy": best_acc,
                        "epoch": best_epoch,
                        "data_len": 0,
                    }
                )
            else:
                correct = 0
                total = 0
                for i in tqdm(range(n_init)):
                    model.load_state_dict(pretrained_model_wts)
                    # mask = (train_labels[indices[i][:int(k)], None].cpu() == LABELS).any(dim=1)  # TODO: also test architecture which includes outputs for all labels
                    subtrainset = Subset(trainset, tuple(indices[i][:int(k)]))
                    # subtrainset = Subset(trainset, tuple(indices[:int(k)]))
                    wandb.log(
                        {
                            "k": k,
                            "data_len": len(subtrainset),
                        }
                    )
                    assert len(subtrainset) > 0
                    # if len(subtrainset) == 0:
                    #     continue
                    subtestset = CollectedData(test_inputs[i][None, :], test_labels[i].reshape(-1))
                    subtrainloader = DataLoader(subtrainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
                    best_epoch, best_acc = train(
                        model=model,
                        trainloader=subtrainloader,
                        testset=subtestset,
                        criterion=criterion,
                        optimizer=optimizer,
                        # scheduler=scheduler,
                        num_epochs=epochs,
                        use_best_model=USE_BEST_MODEL,
                    )
                    # if RESET_PARAMS:
                    #     model.load_state_dict(pretrained_model_wts)
                    total += 1
                    correct += int(best_acc == 100.0)
                wandb.log(
                    {
                        "k": k,
                        # "i": i,
                        "accuracy": 100 * correct / total,
                        "data_len": len(subtrainset),
                    }
                )
        #     num += 1
        # wandb.log(
        #     {
        #         "num": num,
        #     }
        # )
    else:
        indices = torch.tensor(searcher.search(query=test_embeddings.cpu().numpy(), k=K)).reshape(-1)
        for k in torch.arange(start=0, step=10, end=K):
            if k == 0:
                best_epoch = torch.nan
                best_acc = accuracy(model, test_inputs, test_labels)
                wandb.log(
                    {
                        "k": k,
                        "accuracy": best_acc,
                        "epoch": best_epoch,
                        "data_len": 0,
                    }
                )
                continue
            # mask = (train_labels[indices[:int(k)], None].cpu() == LABELS).any(dim=1)  # TODO: also test architecture which includes outputs for all labels
            # subtrainset = Subset(trainset, tuple(indices[:int(k)][mask]))
            subtrainset = Subset(trainset, tuple(indices[:int(k)]))
            if len(subtrainset) == 0:
                continue
            subtrainloader = DataLoader(subtrainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            best_epoch, best_acc = train(
                model=model,
                trainloader=subtrainloader,
                testset=CollectedData(test_inputs, test_labels),
                criterion=criterion,
                optimizer=optimizer,
                # scheduler=scheduler,
                num_epochs=epochs,
                use_best_model=USE_BEST_MODEL,
            )
            if RESET_PARAMS:
                model.load_state_dict(pretrained_model_wts)
            wandb.log(
                {
                    "k": k,
                    "accuracy": best_acc,
                    "epoch": best_epoch,
                    "data_len": len(subtrainset),
                }
            )
    wandb.finish()


def main(args):
    t_start = time.process_time()
    experiment(
        seed=args.seed,
        idx=args.idx,
        alg=args.alg,
        noise_std=args.noise_std,
        n_init=args.n_init,
        epochs=args.epochs,
        # query_batch_size=args.query_batch_size,
        # subsampled_target_frac=args.subsampled_target_frac,
        # max_target_size=args.max_target_size,
        # subsample_acquisition=bool(args.subsample_acquisition),
        # update_target=bool(args.update_target),
        indiv_fine_tuning=bool(args.indiv_fine_tuning),
        lr=args.lr,
        debug=args.debug,
    )
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--alg", type=str, default="ITL")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    # parser.add_argument(
    #     "--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE
    # )
    # parser.add_argument("--subsampled-target-frac", type=float, default=0.5)
    # parser.add_argument("--max-target-size", type=int_or_none, default=None)
    # parser.add_argument("--subsample-acquisition", type=int, default=1)
    # parser.add_argument("--update-target", type=int, default=0)
    parser.add_argument("--indiv-fine-tuning", type=int, default=1)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
