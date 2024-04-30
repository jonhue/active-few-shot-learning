import torch
from transformers import AutoTokenizer
from datasets import load_dataset



def get_datasets(model_id: str, imbalanced_train_perc=None):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = load_dataset("OpenAssistant/oasst1")

    def tokenize(set):
        return tokenizer(set["text"], padding=True, truncation=True, max_length=128, return_tensors="pt")

    trainset = dataset["train"].map(tokenize, batched=True)       # type: ignore
    testset = dataset["validation"].map(tokenize, batched=True)   # type: ignore

    return (
        torch.tensor(trainset["input_ids"]),
        torch.tensor(testset["input_ids"]),
    )