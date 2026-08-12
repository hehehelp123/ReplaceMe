import json
from pathlib import Path

from .data import extract_final_answer


def rescore_file(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    predictions = result["predictions"]

    correct = 0
    for record in predictions:
        predicted = extract_final_answer(record["generation"])
        record["predicted"] = predicted
        record["correct"] = predicted is not None and predicted == record["gold"]
        correct += record["correct"]

    previous = result["exact_match"]
    result["correct"] = correct
    result["exact_match"] = correct / len(predictions)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return {"path": path, "before": previous, "after": result["exact_match"]}


def rescore_all(runs_dir: str = "runs") -> None:
    for path in sorted(Path(runs_dir).glob("*/results/*.json")):
        changed = rescore_file(path)
        print(f"{path.relative_to(runs_dir)}: "
              f"{changed['before'] * 100:.2f}% -> {changed['after'] * 100:.2f}%")
