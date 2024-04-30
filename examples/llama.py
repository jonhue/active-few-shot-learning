import transformers
import torch
import argparse

from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

def experiment():
    access_token = "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"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    #
    #   Dataset
    #

    dataset = load_dataset(
        "json", 
        data_files="./data/train_dataset.json",
        split="train"
    )

    #
    #   Model
    #

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=access_token,
        quantization_config=quant_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1 

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

    args = TrainingArguments(
        output_dir="code-llama-7b-text-to-sql", # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=3,          # batch size per device during training
        gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        fp16=True,                              # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2,
        #tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    #
    #   Training loop
    #

    #batch_size = 1
    #num_batches = len(dataset) / batch_size
#
    #for batch_idx in range(num_batches):
    #    batch = dataset[batch_idx]
#
    #    input = ...
#
    #    trainer.training_step(model, input)

    trainer.train()

def main(args):
    experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
