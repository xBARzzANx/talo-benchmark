"""
Phase 1 runner: All benchmark models × all tasks × zero-shot.
Usage: python run_phase1.py [--dry-run] [--task-class sql_generation]
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import track

from src.models.model_client import call_model, MODEL_MAP
from src.utils.io import load_tasks, save_result
from src.prompts.templates import get_prompt

logging.basicConfig(level=logging.INFO)
console = Console()

PHASE1_MODELS = [m for m in MODEL_MAP if m != "claude-sonnet-4-6"]
GOLD_STANDARD = "claude-sonnet-4-6"
TASK_CLASSES = ["sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation"]


def run(task_class: str, dry_run: bool, run_id: str) -> None:
    tasks = load_tasks(task_class)
    models = PHASE1_MODELS + [GOLD_STANDARD]
    console.print(f"\n[bold]Phase 1 — {task_class}[/bold] | {len(tasks)} tasks × {len(models)} models")

    for task in track(tasks, description=f"{task_class}"):
        for model_id in models:
            if dry_run:
                console.print(f"  [DRY RUN] {model_id} → task {task['id']}")
                continue

            prompt = get_prompt(task_class, "zero_shot", **task["prompt_vars"])
            response = call_model(model_id, prompt)

            result = {
                "run_id": run_id,
                "phase": "phase1",
                "task_class": task_class,
                "task_id": task["id"],
                "difficulty": task.get("difficulty"),
                "model_id": model_id,
                "strategy": "zero_shot",
                **response,
            }
            save_result(result, "phase1", run_id)

    console.print(f"  [green]Done: {task_class}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-class", default="all")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    classes = TASK_CLASSES if args.task_class == "all" else [args.task_class]

    for tc in classes:
        run(tc, args.dry_run, run_id)

    console.print(f"\n[bold green]Phase 1 complete. Run ID: {run_id}[/bold green]")
