import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig



def get_model(model_id: str):
    access_token = "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"

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        token=access_token,
        quantization_config=quant_config
    )
