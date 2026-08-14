import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import build_prompts, extract_final_answer, load_gsm8k
from .quantization import BF16, quantization_config


def load_for_generation(model_path: str, adapter_path: Optional[str] = None,
                        precision: str = BF16):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=quantization_config(precision))
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    for attribute in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(model.generation_config, attribute):
            setattr(model.generation_config, attribute, None)
    return model, tokenizer


def evaluate_gsm8k(model, tokenizer, batch_size: int, max_new_tokens: int,
                   limit: Optional[int] = None) -> dict:
    dataset = load_gsm8k("test", limit)
    prompts = build_prompts(dataset)
    gold = [extract_final_answer(answer) for answer in dataset["answer"]]

    prompt_lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: prompt_lengths[i])

    predictions = [None] * len(prompts)
    generations = [None] * len(prompts)
    started = time.perf_counter()

    for start in tqdm(range(0, len(order), batch_size), desc="Evaluating GSM8K"):
        indices = order[start:start + batch_size]
        encoded = tokenizer([prompts[i] for i in indices],
                            return_tensors="pt", padding=True)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id)

        completions = tokenizer.batch_decode(
            outputs[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for index, completion in zip(indices, completions):
            generations[index] = completion
            predictions[index] = extract_final_answer(completion)

    correct = sum(p is not None and g is not None and p == g
                  for p, g in zip(predictions, gold))
    elapsed = time.perf_counter() - started

    return {
        "total": len(prompts),
        "correct": correct,
        "exact_match": correct / len(prompts),
        "seconds": round(elapsed, 1),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "predictions": [
            {"id": i, "predicted": predictions[i], "gold": gold[i],
             "correct": predictions[i] is not None and predictions[i] == gold[i],
             "generation": generations[i]}
            for i in range(len(prompts))
        ],
    }


def run_evaluation(config, stage: str, model_path: str,
                   adapter_path: Optional[str] = None) -> dict:
    config.prepare_dirs()
    model, tokenizer = load_for_generation(model_path, adapter_path,
                                           config.precision)

    result = evaluate_gsm8k(model, tokenizer, config.eval_batch_size,
                            config.eval_max_new_tokens, config.eval_limit)
    result["stage"] = stage
    result["model_path"] = str(model_path)
    result["adapter_path"] = str(adapter_path) if adapter_path else None
    result["layers"] = model.config.num_hidden_layers
    result["precision"] = config.precision

    suffix = "" if config.precision == BF16 else f"_{config.precision}"
    output_path = Path(config.results_dir) / f"{stage}{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{stage}: EM {result['exact_match'] * 100:.2f}% "
          f"({result['correct']}/{result['total']}) "
          f"on {result['layers']} layers in {result['seconds'] / 60:.1f} min "
          f"-> {output_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
