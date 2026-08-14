import torch
from transformers import BitsAndBytesConfig

BF16 = "bf16"
INT8 = "int8"
NF4 = "nf4"


def quantization_config(precision: str):
    if precision == BF16:
        return None
    if precision == INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"unknown precision {precision!r}, expected one of "
                     f"{BF16}, {INT8}, {NF4}")


def is_quantized(precision: str) -> bool:
    return precision != BF16
