import copy
import gc
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_decoder_layers(model) -> nn.ModuleList:
    if hasattr(model, "transformer"):
        return model.transformer.h
    return model.model.layers


def set_decoder_layers(model, layers: nn.ModuleList) -> None:
    if hasattr(model, "transformer"):
        model.transformer.h = layers
    else:
        model.model.layers = layers


def restore_removed_layers(base_model_path: str, pruned_dir: str,
                           adapter_dir: str, output_dir: str,
                           block: Tuple[int, int]) -> Path:
    start, end = block

    pruned = AutoModelForCausalLM.from_pretrained(
        pruned_dir, dtype=torch.bfloat16, device_map="cpu")
    healed = PeftModel.from_pretrained(pruned, adapter_dir).merge_and_unload()

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map="cpu")
    base_layers = get_decoder_layers(base)

    layers = list(get_decoder_layers(healed))
    for index in range(start, end):
        layers.insert(index, copy.deepcopy(base_layers[index]))

    set_decoder_layers(healed, nn.ModuleList(layers))
    healed.config.num_hidden_layers = len(layers)
    base_layer_types = getattr(base.config, "layer_types", None)
    if base_layer_types is not None:
        healed.config.layer_types = list(base_layer_types)

    if len(layers) != base.config.num_hidden_layers:
        raise RuntimeError(
            f"restored {len(layers)} layers, base has {base.config.num_hidden_layers}")

    output_dir = Path(output_dir)
    healed.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(str(output_dir))

    del healed, base, pruned
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir
