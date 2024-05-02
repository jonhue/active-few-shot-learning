import seqio
import math
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer

import flan.v2.mixtures     # type: ignore

# Fix bug with sll-certificates
import os, certifi
os.environ["CURL_CA_BUNDLE"] = certifi.where()



class NLDataset(Dataset):
    def __init__(self, tf_dataset, model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize(set):
            return tokenizer(set, max_length=256, padding="max_length", return_tensors="pt")
        

        self.inputs = []
        self.labels = []

        for ex in tf_dataset:
            self.inputs.append(tokenize(ex["inputs_pretokenized"].numpy().decode())['input_ids'])
            self.labels.append(tokenize(ex["targets_pretokenized"].numpy().decode())['input_ids'])

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def get_datasets(dataset_id: str, model_id: str, split_ratio: float, imbalanced_train_perc=None) -> tuple[Dataset, Dataset]:
    selected_mixture = seqio.get_mixture_or_task(dataset_id)

    INPUT_SEQ_LEN = 2056
    TARGET_SEQ_LEN = 512
    dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,     # type: ignore
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]     # type: ignore
    )

    dataset = NLDataset(dataset, model_id)

    train, test = random_split(
        dataset, 
        [
            math.ceil(split_ratio * len(dataset)), 
            math.floor((1 - split_ratio) * len(dataset))
        ]
    )

    return train, test
