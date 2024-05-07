import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

ACCESS_TOKEN = "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"

def _get_quantization_config():
    compute_dtype = getattr(torch, "float16")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

def get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(
        model_id, token=ACCESS_TOKEN, quantization_config=_get_quantization_config()
    )

def get_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=ACCESS_TOKEN, quantization_config=_get_quantization_config()
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model
