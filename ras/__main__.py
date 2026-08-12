import argparse

from . import pipeline
from .config import load_config
from .rescore import rescore_all

STAGES = {
    "find-candidates": pipeline.find_candidates,
    "prune": pipeline.prune,
    "heal": pipeline.heal,
    "restore": pipeline.restore,
    "eval-baseline": pipeline.eval_baseline,
    "train-sft": pipeline.train_sft,
    "eval-sft": pipeline.eval_sft,
    "eval-pruned": pipeline.eval_pruned,
    "eval-replaceme": pipeline.eval_replaceme,
    "scenario-1": pipeline.scenario_1,
    "scenario-2": pipeline.scenario_2,
    "scenario-3": pipeline.scenario_3,
    "rescore": lambda config: rescore_all(config.runs_dir),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="ras")
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.eval_limit is not None:
        config.eval_limit = args.eval_limit

    STAGES[args.stage](config)


if __name__ == "__main__":
    main()
