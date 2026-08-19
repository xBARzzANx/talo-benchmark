"""
End-to-end pipeline smoke test.
Tests one task from each class against two models.
Run from the repo root: python scripts/check_connectivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model_client import call_model
from src.evaluators.anomaly_eval import evaluate_anomaly
from rich.console import Console
from rich.table import Table

console = Console()

TEST_MODELS = ["gpt-4.1-mini", "claude-haiku-4-5"]


def check_sql():
    prompt = ("Given schema: orders(id, customer_id, amount, date)\n"
              "Question: What is the total revenue in January 2024?\n"
              "Return ONLY the SQL query.")
    results = []
    for model in TEST_MODELS:
        r = call_model(model, prompt)
        if r["error"]:
            results.append({"model": model, "status": "ERROR", "content": r["error"]})
        else:
            results.append({"model": model, "status": "OK", "content": r["content"][:80]})
    return "SQL Generation", results


def check_anomaly():
    prompt = ("Identify anomalies in this dataset (CSV):\n"
              "row,value\n0,100\n1,102\n2,99\n3,5000\n4,101\n"
              "Return ONLY a JSON list of anomalous row indices.")
    results = []
    for model in TEST_MODELS:
        r = call_model(model, prompt)
        if r["error"]:
            results.append({"model": model, "status": "ERROR", "content": r["error"]})
        else:
            eval_r = evaluate_anomaly(r["content"], ground_truth_indices=[3], n_rows=5)
            results.append({"model": model, "status": f"F1={eval_r['f1']}", "content": r["content"][:60]})
    return "Anomaly Detection", results


def check_connectivity():
    """Smoke test: can we reach each model?"""
    table = Table(title="Model Connectivity Check")
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Latency")

    for model_id in ["gpt-4.1-mini", "claude-haiku-4-5", "gemini-2.5-flash",
                      "llama-3.1-8b", "mistral-7b", "claude-sonnet-4-6"]:
        r = call_model(model_id, "Reply with only the word: OK", max_tokens=10)
        if r["error"]:
            table.add_row(model_id, "[red]FAIL[/red]", "--")
        else:
            table.add_row(model_id, "[green]OK[/green]", f"{r['latency_s']}s")

    console.print(table)


if __name__ == "__main__":
    console.print("\n[bold]TALO Pipeline Smoke Test[/bold]\n")
    check_connectivity()

    for check_fn in [check_sql, check_anomaly]:
        name, results = check_fn()
        console.print(f"\n[bold]{name}[/bold]")
        for r in results:
            status_color = "green" if r["status"] != "ERROR" else "red"
            console.print(f"  {r['model']}: [{status_color}]{r['status']}[/{status_color}] -- {r['content']}")

    console.print("\n[bold green]Smoke test complete.[/bold green]")
