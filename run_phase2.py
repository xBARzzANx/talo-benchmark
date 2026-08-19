"""
Phase 2 runner: Top-Modell pro Task-Klasse x 3 Prompt-Strategien.
Alignment Matrix (aus Phase 1):
  sql_generation      -> gemini-2.5-flash
  anomaly_detection   -> gpt-4.1-mini
  kpi_interpretation  -> claude-haiku-4-5
  report_generation   -> gpt-4.1-mini

Usage:
  python run_phase2.py
  python run_phase2.py --dry-run
  python run_phase2.py --task-class sql_generation
"""
import argparse
import logging
from datetime import datetime
from rich.console import Console
from rich.progress import track

from src.models.model_client import call_model
from src.utils.io import load_tasks, save_result
from src.prompts.templates import get_prompt
from src.prompts.examples import FEW_SHOT_EXAMPLES

logging.basicConfig(level=logging.WARNING)
console = Console()

ALIGNMENT_MATRIX = {
    "sql_generation":     "gemini-2.5-flash",
    "anomaly_detection":  "gpt-4.1-mini",
    "kpi_interpretation": "claude-haiku-4-5",
    "report_generation":  "gpt-4.1-mini",
}

PHASE2_STRATEGIES = ["few_shot", "chain_of_thought", "structured_output"]

TASK_CLASSES = list(ALIGNMENT_MATRIX.keys())


def run(task_class: str, dry_run: bool, run_id: str) -> None:
    model_id = ALIGNMENT_MATRIX[task_class]
    tasks = load_tasks(task_class)
    examples = FEW_SHOT_EXAMPLES.get(task_class, "")

    console.print(
        f"\n[bold]Phase 2 — {task_class}[/bold] | "
        f"Modell: [cyan]{model_id}[/cyan] | "
        f"{len(tasks)} tasks × {len(PHASE2_STRATEGIES)} strategies"
    )

    for strategy in PHASE2_STRATEGIES:
        console.print(f"  Strategie: [yellow]{strategy}[/yellow]")

        for task in track(tasks, description=f"  {strategy}"):
            if dry_run:
                console.print(
                    f"    [DRY RUN] {model_id} | {strategy} → {task['id']}"
                )
                continue

            # Prompt zusammenbauen
            prompt_vars = dict(task["prompt_vars"])
            if strategy == "few_shot":
                prompt_vars["examples"] = examples

            try:
                prompt = get_prompt(task_class, strategy, **prompt_vars)
            except KeyError as e:
                console.print(f"    [red]Prompt-Fehler {task['id']}: {e}[/red]")
                continue

            response = call_model(model_id, prompt)

            result = {
                "run_id":      run_id,
                "phase":       "phase2",
                "task_class":  task_class,
                "task_id":     task["id"],
                "difficulty":  task.get("difficulty"),
                "model_id":    model_id,
                "strategy":    strategy,
                **response,
            }
            save_result(result, "phase2", run_id)

    console.print(f"  [green]Done: {task_class}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-class", default="all")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    classes = TASK_CLASSES if args.task_class == "all" else [args.task_class]

    console.print(f"\n[bold]TALO Phase 2[/bold] | Run ID: {run_id}")
    console.print(f"Task-Klassen: {classes}")
    console.print(f"Strategien: {PHASE2_STRATEGIES}")
    console.print(f"Dry Run: {args.dry_run}\n")

    for tc in classes:
        run(tc, args.dry_run, run_id)

    console.print(f"\n[bold green]Phase 2 complete. Run ID: {run_id}[/bold green]")