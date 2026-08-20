import gc
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoConfig

from . import pipeline


def candidate_blocks(num_layers: int, layers_to_skip: int):
    return [(start, start + layers_to_skip)
            for start in range(1, num_layers - layers_to_skip + 1)]


def _norm_ratio(pruned_dir: str):
    payload = torch.load(f"{pruned_dir}_transform", map_location="cpu",
                         weights_only=False)
    ratios = payload.get("norm_ratios")
    return ratios[0] if ratios else None


def _completed(record_path: Path):
    if not record_path.exists():
        return set()
    done = set()
    for line in record_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(tuple(json.loads(line)["block"]))
    return done


def _discard(*paths):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def sweep_blocks(config) -> Path:
    num_layers = AutoConfig.from_pretrained(config.model_path).num_hidden_layers
    blocks = candidate_blocks(num_layers, config.layers_to_skip)

    record_path = Path(config.runs_dir) / f"{config.model_tag}_sweep.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(record_path)

    print(f"sweep {config.model_tag}: {len(blocks)} candidates, "
          f"{len(done)} already done, layers={num_layers}", flush=True)

    for start, end in blocks:
        if (start, end) in done:
            continue

        candidate = replace(config,
                            model_tag=f"{config.model_tag}/b{start:02d}",
                            selected_block=[start, end])
        started = time.time()
        record = {"block": [start, end], "host_layer": start - 1}

        try:
            pruned_dir = str(pipeline.prune(candidate))
            record["norm_ratio"] = _norm_ratio(pruned_dir)

            pipeline.heal(candidate)
            record["pruned_healed"] = pipeline.eval_pruned(candidate)["exact_match"]

            pipeline.restore(candidate)
            record["restored"] = pipeline.eval_replaceme(candidate)["exact_match"]
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            _discard(candidate.run_dir / "pruned_ReplaceMe_joint_lstsq_1",
                     str(candidate.run_dir / "pruned_ReplaceMe_joint_lstsq_1_transform"),
                     candidate.restored_dir,
                     candidate.heal_adapter_dir / "checkpoints")
            gc.collect()
            torch.cuda.empty_cache()

        record["minutes"] = round((time.time() - started) / 60, 1)
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"block ({start},{end}): {record}", flush=True)

    return record_path
