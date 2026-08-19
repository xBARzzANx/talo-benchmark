"""
Implements the classification-accuracy evaluation for the task
classification module described in Section 5.2: accuracy over the 60
benchmark tasks (ground truth = task_class) plus the ambiguous test queries
in benchmark/ambiguous_queries.jsonl (ground truth = expected_class).

The 60 benchmark tasks are structured task dicts (schema+question, CSV
data, KPI fields, audience+sections), not natural-language queries. To
classify them, each task's prompt_vars are serialized into one
representative query string per class (see `serialize_task`). This
serialization is a fixed, documented convention for THIS evaluation only --
it is not used anywhere in the TALO pipeline itself, where run_talo.py and
app.py always classify a raw user-provided query string directly.

Usage:
  python analysis/classifier_eval.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from talo.classifier import TaskClassifier
from src.utils.io import load_tasks

OUTPUT_DIR = REPO_ROOT / "analysis" / "output"
TASK_CLASSES = ["sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation"]


def serialize_task(task_class: str, task: dict) -> str:
    """Fixed serialization of a benchmark task's prompt_vars into one query string, for evaluation only."""
    pv = task["prompt_vars"]
    if task_class == "sql_generation":
        return f"Schema: {pv['schema']}\nQuestion: {pv['question']}"
    if task_class == "anomaly_detection":
        return f"Dataset:\n{pv['data']}\nAny anomalies?"
    if task_class == "kpi_interpretation":
        return (f"KPI: {pv['kpi_name']} Value: {pv['value']} Benchmark: {pv['benchmark']} "
                f"Prior period: {pv['prior_period']} Context: {pv['context']}")
    if task_class == "report_generation":
        return f"Data:\n{pv['data']}\nAudience: {pv['audience']}\nSections: {pv['sections']}"
    raise ValueError(f"Unknown task class: {task_class}")


def evaluate_benchmark_tasks(classifier: TaskClassifier) -> list[dict]:
    rows = []
    for tc in TASK_CLASSES:
        for task in load_tasks(tc):
            query = serialize_task(tc, task)
            result = classifier.classify(query)
            rows.append({
                "source": "benchmark_tasks", "task_id": task["id"], "difficulty": task.get("difficulty"),
                "ground_truth": tc, "predicted": result.task_class,
                "correct": result.task_class == tc, "confidence": result.confidence,
                "used_fallback": result.used_fallback,
            })
    return rows


def evaluate_ambiguous_queries(classifier: TaskClassifier) -> list[dict]:
    rows = []
    path = REPO_ROOT / "benchmark" / "ambiguous_queries.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            result = classifier.classify(row["query"])
            rows.append({
                "source": "ambiguous_queries", "task_id": row["id"], "difficulty": None,
                "ground_truth": row["expected_class"], "predicted": result.task_class,
                "correct": result.task_class == row["expected_class"], "confidence": result.confidence,
                "used_fallback": result.used_fallback, "ambiguity_type": row.get("ambiguity_type"),
            })
    return rows


def confusion_matrix(rows: list[dict]) -> dict[tuple, int]:
    matrix = defaultdict(int)
    for r in rows:
        matrix[(r["ground_truth"], r["predicted"])] += 1
    return matrix


def accuracy(rows: list[dict]) -> float:
    return sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0.0


def per_class_accuracy(rows: list[dict]) -> dict[str, float]:
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["ground_truth"]].append(r["correct"])
    return {tc: sum(v) / len(v) for tc, v in by_class.items()}


# --------------------------------------------------------------------------
# File output
# --------------------------------------------------------------------------

def write_report_md(benchmark_rows: list[dict], ambiguous_rows: list[dict]) -> str:
    lines = ["# Classifier Evaluation\n"]

    lines.append("## Benchmark Tasks (N=60, ground truth = task_class)\n")
    lines.append(
        f"Overall accuracy: **{accuracy(benchmark_rows):.3f}** "
        f"({sum(r['correct'] for r in benchmark_rows)}/{len(benchmark_rows)})\n"
    )
    lines.append("| Task Class | Accuracy | n |")
    lines.append("|---|---|---|")
    per_class = per_class_accuracy(benchmark_rows)
    by_class_n = defaultdict(int)
    for r in benchmark_rows:
        by_class_n[r["ground_truth"]] += 1
    for tc in TASK_CLASSES:
        lines.append(f"| {tc} | {per_class.get(tc, 0):.3f} | {by_class_n[tc]} |")

    lines.append("\n### Confusion Matrix\n")
    cm = confusion_matrix(benchmark_rows)
    lines.append("| Ground Truth \\ Predicted | " + " | ".join(TASK_CLASSES) + " |")
    lines.append("|---|" + "---|" * len(TASK_CLASSES))
    for gt in TASK_CLASSES:
        row = [str(cm.get((gt, pred), 0)) for pred in TASK_CLASSES]
        lines.append(f"| {gt} | " + " | ".join(row) + " |")

    misclassified = [r for r in benchmark_rows if not r["correct"]]
    if misclassified:
        lines.append("\n### Misclassified\n")
        lines.append("| Task ID | Ground Truth | Predicted | Confidence | Fallback |")
        lines.append("|---|---|---|---|---|")
        for r in misclassified:
            lines.append(
                f"| {r['task_id']} | {r['ground_truth']} | {r['predicted']} | "
                f"{r['confidence']:.2f} | {r['used_fallback']} |"
            )
    else:
        lines.append("\nNo misclassifications.\n")

    lines.append(f"\n## Ambiguous Queries (N={len(ambiguous_rows)}, ground truth = expected_class)\n")
    lines.append(
        f"Overall accuracy: **{accuracy(ambiguous_rows):.3f}** "
        f"({sum(r['correct'] for r in ambiguous_rows)}/{len(ambiguous_rows)})\n"
    )
    lines.append("| Query ID | Ambiguity Type | Ground Truth | Predicted | Confidence | Fallback | Correct |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in ambiguous_rows:
        lines.append(
            f"| {r['task_id']} | {r.get('ambiguity_type', '')} | {r['ground_truth']} | {r['predicted']} | "
            f"{r['confidence']:.2f} | {r['used_fallback']} | {'yes' if r['correct'] else 'no'} |"
        )

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Console reporting
# --------------------------------------------------------------------------

def print_summary(benchmark_rows: list[dict], ambiguous_rows: list[dict]) -> None:
    print(f"\nBenchmark tasks (N={len(benchmark_rows)}): accuracy = {accuracy(benchmark_rows):.3f}")
    per_class = per_class_accuracy(benchmark_rows)
    for tc in TASK_CLASSES:
        print(f"  {tc}: {per_class.get(tc, 0):.3f}")

    print(f"\nAmbiguous queries (N={len(ambiguous_rows)}): accuracy = {accuracy(ambiguous_rows):.3f}")
    for r in ambiguous_rows:
        status = "OK" if r["correct"] else "FAIL"
        print(f"  [{status}] {r['task_id']}: predicted={r['predicted']} expected={r['ground_truth']}")


def main() -> None:
    classifier = TaskClassifier()
    benchmark_rows = evaluate_benchmark_tasks(classifier)
    ambiguous_rows = evaluate_ambiguous_queries(classifier)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "classifier_eval.md").write_text(write_report_md(benchmark_rows, ambiguous_rows), encoding="utf-8")
    with open(OUTPUT_DIR / "classifier_eval.jsonl", "w", encoding="utf-8") as f:
        for r in benchmark_rows + ambiguous_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Analyse-Outputs geschrieben nach: {OUTPUT_DIR}")
    print("  classifier_eval.md")
    print("  classifier_eval.jsonl")

    print_summary(benchmark_rows, ambiguous_rows)


if __name__ == "__main__":
    main()
