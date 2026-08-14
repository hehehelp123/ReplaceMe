import gc
import json
from pathlib import Path
from typing import Optional

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from .evaluate import load_for_generation
from .quantization import BF16


def run_harness(config, stage: str, model_path: str,
                adapter_path: Optional[str] = None) -> dict:
    config.prepare_dirs()
    model, tokenizer = load_for_generation(model_path, adapter_path,
                                           config.precision)

    wrapped = HFLM(pretrained=model, tokenizer=tokenizer,
                   batch_size=config.harness_batch_size)
    output = simple_evaluate(model=wrapped, tasks=config.harness_tasks,
                             num_fewshot=config.harness_num_fewshot,
                             limit=config.harness_limit, verbosity="ERROR")

    result = {
        "stage": stage,
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "layers": model.config.num_hidden_layers,
        "precision": config.precision,
        "num_fewshot": config.harness_num_fewshot,
        "limit": config.harness_limit,
        "results": output["results"],
    }

    suffix = "" if config.precision == BF16 else f"_{config.precision}"
    output_path = Path(config.results_dir) / f"{stage}_harness{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    for task, metrics in output["results"].items():
        reported = {k: round(v, 4) for k, v in metrics.items()
                    if isinstance(v, float) and not k.endswith("_stderr,none")}
        print(f"{stage} {task}: {reported} -> {output_path}")

    del wrapped, model
    gc.collect()
    torch.cuda.empty_cache()
    return result
