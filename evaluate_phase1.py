"""
Phase 1 Evaluation Script
Berechnet Scores für alle 4 Task-Klassen:
- SQL Generation: Execution Accuracy (0 oder 1)
- Anomaly Detection: Precision, Recall, F1
- KPI Interpretation: LLM-as-Judge (1-5)
- Report Generation: LLM-as-Judge (1-5)
"""
import json
from pathlib import Path
from datetime import datetime
from src.evaluators.sql_eval import evaluate_sql
from src.evaluators.anomaly_eval import evaluate_anomaly
from src.evaluators.llm_judge import judge_output
from src.evaluators.extractors import extract_sql_fence_only
from src.utils.io import load_tasks

RUN_IDS = ["20260522_2030", "20260522_2219_gemini_retry"]
OUTPUT_FILE = f"results/phase1_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

# Alle Tasks laden als lookup dict
tasks_lookup = {}
for tc in ["sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation"]:
    for t in load_tasks(tc):
        tasks_lookup[(tc, t["id"])] = t

# Alle Runs laden, deduplizieren
results = []
for run_id in RUN_IDS:
    path = Path(f"results/phase1/{run_id}/results.jsonl")
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))

# Deduplizieren: neueste Version pro task+model behalten
seen = {}
for r in results:
    key = (r["task_class"], r["task_id"], r["model_id"])
    seen[key] = r
results = list(seen.values())
results = [r for r in results if not r["error"]]
print(f"Zu bewertende Runs: {len(results)}")

# Scores berechnen
scored = []
sql_count = anm_count = kpi_count = rep_count = 0

for i, r in enumerate(results):
    tc = r["task_class"]
    task = tasks_lookup.get((tc, r["task_id"]))
    if not task:
        continue

    score_data = {}

    if tc == "sql_generation":
        predicted = extract_sql_fence_only(r["content"] or "")
        score_data = evaluate_sql(predicted, task["ground_truth_sql"], task["db_path"])
        sql_count += 1
        if sql_count % 15 == 0:
            print(f"  SQL: {sql_count}/90 bewertet")

    elif tc == "anomaly_detection":
        score_data = evaluate_anomaly(
            r["content"] or "",
            task["ground_truth_indices"],
            task["n_rows"]
        )
        anm_count += 1
        if anm_count % 15 == 0:
            print(f"  Anomaly: {anm_count}/90 bewertet")

    elif tc == "kpi_interpretation":
        task_input = json.dumps(task["prompt_vars"], ensure_ascii=False)
        score_data = judge_output(
            tc, task_input, r["content"] or "",
            ground_truth=task.get("reference_interpretation", "")
        )
        kpi_count += 1
        if kpi_count % 5 == 0:
            print(f"  KPI: {kpi_count}/90 bewertet")

    elif tc == "report_generation":
        task_input = json.dumps(task["prompt_vars"], ensure_ascii=False)
        score_data = judge_output(tc, task_input, r["content"] or "")
        rep_count += 1
        if rep_count % 5 == 0:
            print(f"  Report: {rep_count}/90 bewertet")

    scored.append({
        "task_class":  tc,
        "task_id":     r["task_id"],
        "difficulty":  r.get("difficulty"),
        "model_id":    r["model_id"],
        "strategy":    r.get("strategy", "zero_shot"),
        "latency_s":   r.get("latency_s"),
        "input_tokens":r.get("input_tokens"),
        "output_tokens":r.get("output_tokens"),
        **score_data,
    })

# Speichern
Path("results").mkdir(exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    for s in scored:
        f.write(json.dumps(s) + "\n")

print(f"\nFertig! {len(scored)} Scores gespeichert in: {OUTPUT_FILE}")
print("\nSchnellübersicht:")
from collections import defaultdict
by_class = defaultdict(list)
for s in scored:
    tc = s["task_class"]
    if tc == "sql_generation":
        by_class[tc].append(s.get("score", 0))
    elif tc == "anomaly_detection":
        by_class[tc].append(s.get("f1", 0))
    else:
        by_class[tc].append(s.get("mean_score", 0) or 0)

for tc, vals in sorted(by_class.items()):
    avg = sum(vals) / len(vals)
    print(f"  {tc}: Ø {avg:.3f} ({len(vals)} Runs)")