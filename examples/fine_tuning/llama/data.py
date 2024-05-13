import seqio
import math
import datasets
from datasets import Dataset
from torch.utils.data import random_split
from transformers import AutoTokenizer

# Fix bug with sll-certificates
import os, certifi

os.environ["CURL_CA_BUNDLE"] = certifi.where()


class NLDataset(Dataset):
    def __init__(self, tf_dataset):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        def tokenize(set):
            return tokenizer(
                set, max_length=256, padding="max_length", return_tensors="pt"
            )

        self.inputs = []
        self.labels = []

        for ex in tf_dataset:
            self.inputs.append(
                tokenize(ex["inputs_pretokenized"].numpy().decode())["input_ids"]
            )
            self.labels.append(
                tokenize(ex["targets_pretokenized"].numpy().decode())["input_ids"]
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.inputs[i] for i in index], [self.labels[i] for i in index]
        else:
            return self.inputs[index], self.labels[index]


def get_flanv2(imbalanced_train_perc=None) -> tuple[Dataset, Dataset]:
    selected_mixture = seqio.get_mixture_or_task("")

    INPUT_SEQ_LEN = 2056
    TARGET_SEQ_LEN = 512
    tf_dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,  # type: ignore
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"],  # type: ignore
    )

    dataset = NLDataset(tf_dataset)

    train, test = random_split(
        dataset, [math.ceil(0.1 * len(dataset)), math.floor(0.9 * len(dataset))]
    )

    return train, test


def get_oasst1(imbalanced_train_perc=None) -> tuple[Dataset, Dataset]:
    datasets.config.DOWNLOADED_DATASETS_PATH = (
        "~/../../scratch/sbongni/.cache/huggingface/downloads"
    )
    dataset = datasets.load_dataset(
        "OpenAssistant/oasst1", verification_mode="no_checks"
    )

    return dataset["train"], dataset["validation"]
