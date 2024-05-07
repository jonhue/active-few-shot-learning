import argparse
import time
import torch
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForTokenClassification
from peft import LoraConfig     # type: ignore



import afsl
from examples.fine_tuning.training import LlamaTrainer
from examples.acquisition_functions import get_acquisition_function
from examples.fine_tuning.llama.data import get_oasst1
from examples.fine_tuning.llama.model import get_model, get_tokenizer
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
    print("SEED:", seed, "LABELS:", LABELS, "ALG:", alg)
    torch.manual_seed(seed)

    #
    #   Model
    #

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    model = get_model(model_id)
    tokenizer = get_tokenizer(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = DataCollatorForTokenClassification(tokenizer)

    #def collate_tokenize(data):
    #    print(data)
    #    text_batch = [element["input_ids"] for element in data]
    #    tokenized = tokenizer(text_batch, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    #    return tokenized

    #
    #   Train / Test set
    #

    train_set, test_set = get_oasst1()

    #
    #   Active DataLoader
    #

    acquisition_function = get_acquisition_function(
        alg=alg,
        target=torch.Tensor(),      #TODO give test_set
        noise_std=noise_std,
        mini_batch_size=MINI_BATCH_SIZE,
        num_workers=NUM_WORKERS if not debug else 0,
        subsample_acquisition=subsample_acquisition,
        subsampled_target_frac=subsampled_target_frac,
        max_target_size=max_target_size,
    )

    #
    #   Trainer
    #

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=64,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    
    default_args = {
        "output_dir": "tmp",
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "log_level": "error",
        "report_to": "none",
    }

    training_args = TrainingArguments(
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        gradient_checkpointing=True, 
        fp16=True, 
        max_steps=1000,
        **default_args
    )

    trainer = LlamaTrainer(
        model=model,
        train_dataset=train_set,
        data_collator=data_collator,
        acquisition_function=acquisition_function,
        query_batch_size=query_batch_size,
        args=training_args,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=128,
    )

    trainer.train()     # type: ignore

    # wandb.finish()


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
