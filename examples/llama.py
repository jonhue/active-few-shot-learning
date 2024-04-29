import transformers
import torch
import argparse

from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer

def experiment():
    access_token = "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"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    dataset = load_dataset(
        "json", 
        data_files="./data/test.json",
        field="data",
        split="train"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=access_token
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
    )

    #trainer.train()

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

def main(args):
    experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
