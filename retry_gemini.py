import json
from datetime import datetime
from src.models.model_client import call_model
from src.utils.io import load_tasks, save_result
from src.prompts.templates import get_prompt

FAILED = {
    "sql_generation":    ["sql_006"],
    "anomaly_detection": ["anm_002", "anm_006", "anm_009", "anm_012", "anm_014"],
    "kpi_interpretation":["kpi_007"],
}

run_id = datetime.now().strftime("%Y%m%d_%H%M") + "_gemini_retry"

for task_class, task_ids in FAILED.items():
    tasks = load_tasks(task_class)
    tasks = [t for t in tasks if t["id"] in task_ids]
    print(f"\n{task_class}: retrying {len(tasks)} tasks...")
    for task in tasks:
        prompt = get_prompt(task_class, "zero_shot", **task["prompt_vars"])
        response = call_model("gemini-2.5-flash", prompt)
        result = {
            "run_id": run_id,
            "phase": "phase1",
            "task_class": task_class,
            "task_id": task["id"],
            "difficulty": task.get("difficulty"),
            "model_id": "gemini-2.5-flash",
            "strategy": "zero_shot",
            **response,
        }
        save_result(result, "phase1", run_id)
        status = "OK" if not response["error"] else f"ERROR: {response['error'][:50]}"
        print(f"  {task['id']}: {status}")

print(f"\nDone. Run ID: {run_id}")