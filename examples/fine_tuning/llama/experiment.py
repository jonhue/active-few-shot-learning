import argparse
import time
import wandb
import torch
import afsl
from examples.acquisition_functions import get_acquisition_function
from examples.fine_tuning.llama.data import get_datasets, tokenize

from trl import SFTTrainer

from examples.fine_tuning.llama.model import get_model

from examples.utils import int_or_none

LR = 0.001
EPOCHS = 100
USE_BEST_MODEL = True
TRAIN_BATCH_SIZE = 64
REWEIGHTING = True
RESET_PARAMS = True
LABELS = torch.arange(3)
IMBALANCED_TEST = (
    None  # ImbalancedTestConfig(drop_perc=0.5, drop_labels=torch.arange(5))
)
IMBALANCED_TRAIN_PERC = None  # 0.8

MINI_BATCH_SIZE = 1_000
NUM_WORKERS = 4
NUM_ROUNDS = 101

DEFAULT_NOISE_STD = 0.01
DEFAULT_QUERY_BATCH_SIZE = 1
DEFAULT_N_INIT = 30


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
    #wandb.init(
    #    name="MNIST First Test",
    #    dir="/cluster/scratch/sbongni/wandb/mnist",
    #    project="Fine-tuning MNIST",
    #    config={
    #        "learning_rate": LR,
    #        "architecture": "CNN",
    #        "dataset": "MNIST",
    #        "epochs": EPOCHS,
    #        "use_best_model": USE_BEST_MODEL,
    #        "train_batch_size": TRAIN_BATCH_SIZE,
    #        "model": MODEL,
    #        "reweighting": REWEIGHTING,
    #        "subsample_acquisition": subsample_acquisition,
    #        "noise_std": noise_std,
    #        "seed": seed,
    #        "alg": alg,
    #        "validation": "hold-out",
    #        "reset_params": RESET_PARAMS,
    #        "imbalanced_test": IMBALANCED_TEST,
    #        "query_batch_size": query_batch_size,
    #        "n_init": n_init,
    #        "labels": LABELS.tolist(),
    #        "subsampled_target_frac": subsampled_target_frac,
    #        "max_target_size": max_target_size,
    #        "update_target": update_target,
    #    },
    #    mode="offline" if debug else "online",
    #)

    print("SEED:", seed, "LABELS:", LABELS, "ALG:", alg)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    #   Model
    #

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_id = "OpenAssistant/oasst1"

    #
    #   Train / Test set
    #

    trainset, testset = get_datasets(dataset_id)

    sample = tokenize(model_id, trainset)   # type: ignore       
    target = tokenize(model_id, testset)    # type: ignore

    #
    #   Acquisition Function
    #

    model = get_model(model_id)

    acquisition_function = get_acquisition_function(
        alg=alg,
        target=target,
        noise_std=noise_std,
        mini_batch_size=MINI_BATCH_SIZE,
        num_workers=NUM_WORKERS if not debug else 0,
        subsample_acquisition=subsample_acquisition,
        subsampled_target_frac=subsampled_target_frac,
        max_target_size=max_target_size,
    )

    data_loader = afsl.ActiveDataLoader(
        dataset=trainset,       # TODO
        batch_size=query_batch_size,
        acquisition_function=acquisition_function
    )
    
    #
    #   Trainer
    #

    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,     # TODO
        max_seq_length=2,
        #tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    #
    #   Training Loop
    #

    num_batches = int(len(train_inputs) / query_batch_size)

    #for batch_idx in range(num_batches):
    batch_indices = data_loader.next(model)

    input = ... # TODO

    trainer.training_step(model, input)

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
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--subsampled-target-frac", type=float, default=0.5)
    parser.add_argument("--max-target-size", type=int_or_none, default=None)
    parser.add_argument("--subsample-acquisition", type=int, default=1)
    parser.add_argument("--update-target", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
