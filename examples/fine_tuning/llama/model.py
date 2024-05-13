import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

ACCESS_TOKEN = "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"


def _get_quantization_config():
    compute_dtype = getattr(torch, "float16")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def get_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=ACCESS_TOKEN, 
        quantization_config=_get_quantization_config()
    )
    
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=ACCESS_TOKEN, 
        quantization_config=_get_quantization_config()
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model
