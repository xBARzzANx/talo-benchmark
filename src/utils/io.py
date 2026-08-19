"""Task loading and result persistence."""
import json, csv
from pathlib import Path
from typing import Any


def load_tasks(task_class: str) -> list[dict]:
    path = Path("benchmark") / task_class / "tasks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No tasks found at {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_result(result: dict, phase: str, run_id: str) -> Path:
    out_dir = Path("results") / phase / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"
    with open(out_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    return out_file


def load_results(phase: str, run_id: str) -> list[dict]:
    path = Path("results") / phase / run_id / "results.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def export_to_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
