import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset



def get_datasets(dataset_id: str, imbalanced_train_perc=None):
    dataset = load_dataset(dataset_id)

    return dataset["train"], dataset["validation"]      # type: ignore

def tokenize(model_id: str, dataset: Dataset) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize(set):
        return tokenizer(set["text"], padding=True, truncation=True, max_length=128, return_tensors="pt")

    return dataset.map(tokenize, batched=True)["input_ids"]       # type: ignore

train, test = get_datasets("SirNeural/flan_v2")

print(train[0])
print(train[1])
print(train[2])
print(train[3])