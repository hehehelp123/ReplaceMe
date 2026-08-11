import json
from pathlib import Path
from typing import Tuple

from ReplaceMe import distance_scored, lstsq_joint

from .config import ExperimentConfig
from .evaluate import run_evaluation
from .restore import restore_removed_layers
from .train import train_lora

DATASET = "openai/gsm8k"
DATASET_COLUMN = "text"
DATASET_SUBSET = "train"


def _state_path(config: ExperimentConfig) -> Path:
    return config.run_dir / "state.json"


def read_state(config: ExperimentConfig) -> dict:
    path = _state_path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_state(config: ExperimentConfig, **updates) -> dict:
    state = read_state(config)
    state.update(updates)
    config.prepare_dirs()
    _state_path(config).write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def selected_block(config: ExperimentConfig) -> Tuple[int, int]:
    if config.selected_block is not None:
        return tuple(config.selected_block)
    block = read_state(config).get("selected_block")
    if block is None:
        raise RuntimeError("no selected block: run find-candidates first "
                           "or set selected_block in the config")
    return tuple(block)


def find_candidates(config: ExperimentConfig) -> Tuple[int, int]:
    config.prepare_dirs()
    signals = distance_scored.profile_distances(
        model_path=config.model_path,
        dataset=DATASET,
        dataset_column=DATASET_COLUMN,
        batch_size=config.calib_batch_size,
        max_length=config.calib_max_length,
        layers_to_skip=config.layers_to_skip,
        dataset_size=config.calib_size,
        dataset_subset=DATASET_SUBSET,
        use_4bit=False,
        compute_taylor=config.compute_taylor,
        taylor_veto_quantile=config.taylor_veto_quantile,
        answer_only_loss=config.answer_only_loss,
        csv_save_path=str(config.candidates_path),
        distances_save_path=str(config.distances_path),
    )
    block = tuple(signals["selected_block"])
    write_state(config, selected_block=list(block))
    return block


def prune(config: ExperimentConfig) -> Path:
    block = selected_block(config)
    output_dir = lstsq_joint.lstsq(
        model_path=config.model_path,
        dataset=DATASET,
        dataset_column=DATASET_COLUMN,
        batch_size=config.calib_batch_size,
        max_length=config.calib_max_length,
        layers_to_skip=config.layers_to_skip,
        dataset_size=config.calib_size,
        dataset_subset=DATASET_SUBSET,
        use_4bit=False,
        save_path=str(config.run_dir / "pruned"),
        alpha=config.alpha_reg,
        alpha_act=config.alpha_act,
        alpha_grad=config.alpha_grad,
        identity_blend_threshold=config.identity_blend_threshold,
        num_A=1,
        selected_blocks=[block],
        save_transform_only=True,
    )
    write_state(config, pruned_dir=str(output_dir))
    return Path(output_dir)


def heal(config: ExperimentConfig) -> Path:
    state = read_state(config)
    pruned_dir = state.get("pruned_dir")
    if pruned_dir is None:
        raise RuntimeError("no pruned model: run prune first")
    return train_lora(config, pruned_dir, config.heal_adapter_dir)


def restore(config: ExperimentConfig) -> Path:
    state = read_state(config)
    return restore_removed_layers(
        base_model_path=config.model_path,
        pruned_dir=state["pruned_dir"],
        adapter_dir=str(config.heal_adapter_dir),
        output_dir=str(config.restored_dir),
        block=selected_block(config),
    )


def eval_baseline(config: ExperimentConfig) -> dict:
    return run_evaluation(config, "baseline", config.model_path)


def train_sft(config: ExperimentConfig) -> Path:
    return train_lora(config, config.model_path, config.sft_adapter_dir)


def eval_sft(config: ExperimentConfig) -> dict:
    return run_evaluation(config, "sft", config.model_path,
                          str(config.sft_adapter_dir))


def eval_pruned(config: ExperimentConfig) -> dict:
    state = read_state(config)
    return run_evaluation(config, "pruned_healed", state["pruned_dir"],
                          str(config.heal_adapter_dir))


def eval_replaceme(config: ExperimentConfig) -> dict:
    return run_evaluation(config, "replaceme", str(config.restored_dir))


def scenario_1(config: ExperimentConfig) -> dict:
    return eval_baseline(config)


def scenario_2(config: ExperimentConfig) -> dict:
    train_sft(config)
    return eval_sft(config)


def scenario_3(config: ExperimentConfig) -> dict:
    find_candidates(config)
    prune(config)
    heal(config)
    restore(config)
    return eval_replaceme(config)
