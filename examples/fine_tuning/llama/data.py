import seqio
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import flan.v2.mixtures

# Fix bug with sll-certificates
import os, certifi
os.environ["CURL_CA_BUNDLE"] = certifi.where()



def get_datasets(dataset_id: str, imbalanced_train_perc=None):
    selected_mixture = seqio.get_mixture_or_task(dataset_id)

    INPUT_SEQ_LEN = 2056
    TARGET_SEQ_LEN = 512
    dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,
        # The passthrough features let you track the source/task/template metadata for the example
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    )

    inputs = []
    targets = []

    for i, ex in enumerate(dataset):
        print(ex)
        inputs.append(ex["inputs_pretokenized"].numpy().decode())
        targets.append(ex["targets_pretokenized"].numpy().decode())

    return inputs, targets

def tokenize(model_id: str, dataset: Dataset) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize(set):
        return tokenizer(set["text"], padding=True, truncation=True, max_length=128, return_tensors="pt")

    return dataset.map(tokenize, batched=True)["input_ids"]       # type: ignore
