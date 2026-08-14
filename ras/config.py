from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ExperimentConfig:
    model_path: str
    model_tag: str
    layers_to_skip: int
    runs_dir: str = "runs"
    seed: int = 42

    calib_size: int = 128
    calib_max_length: int = 256
    calib_batch_size: int = 2
    answer_only_loss: bool = True
    compute_taylor: bool = False
    taylor_veto_quantile: Optional[float] = None

    alpha_act: float = 1.0
    alpha_grad: float = 0.5
    alpha_reg: float = 0.0
    identity_blend_threshold: float = 0.15

    train_batch_size: int = 8
    train_grad_accum: int = 1
    train_max_length: int = 384
    train_learning_rate: float = 2e-4
    train_epochs: float = 1.0
    train_warmup_steps: int = 20
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])

    eval_batch_size: int = 64
    eval_max_new_tokens: int = 256
    eval_limit: Optional[int] = None
    precision: str = "bf16"

    selected_block: Optional[List[int]] = None
    pruned_dir: Optional[str] = None

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.model_tag

    @property
    def candidates_path(self) -> Path:
        return self.run_dir / "candidates.csv"

    @property
    def distances_path(self) -> Path:
        return self.run_dir / "distances.pth"

    @property
    def sft_adapter_dir(self) -> Path:
        return self.run_dir / "sft_adapter"

    @property
    def heal_adapter_dir(self) -> Path:
        return self.run_dir / "heal_adapter"

    @property
    def restored_dir(self) -> Path:
        return self.run_dir / "restored"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def block_path(self) -> Path:
        return self.run_dir / "selected_block.json"

    def prepare_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        return ExperimentConfig(**yaml.safe_load(f))
