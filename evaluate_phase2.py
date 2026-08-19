"""Phase 2 Evaluation Script"""
import json
from pathlib import Path
from datetime import datetime
from src.evaluators.sql_eval import evaluate_sql
from src.evaluators.anomaly_eval import evaluate_anomaly
from src.evaluators.llm_judge import judge_output
from src.evaluators.extractors import extract_sql_keyword_line
from src.utils.io import load_tasks

RUN_ID = "20260523_1406"
OUTPUT_FILE = f"results/phase2_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

tasks_lookup = {}
for tc in ["sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation"]:
    for t in load_tasks(tc):
        tasks_lookup[(tc, t["id"])] = t

results = []
with open(f"results/phase2/{RUN_ID}/results.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

results = [r for r in results if not r["error"]]
print(f"Zu bewertende Runs: {len(results)}")

scored = []
counters = {"sql_generation": 0, "anomaly_detection": 0,
            "kpi_interpretation": 0, "report_generation": 0}

for r in results:
    tc = r["task_class"]
    task = tasks_lookup.get((tc, r["task_id"]))
    if not task:
        continue

    if tc == "sql_generation":
        # CoT gibt Text + SQL zurück — Keyword-Line-Extraktion (Section 4.4.5)
        predicted = extract_sql_keyword_line(r["content"] or "")
        score_data = evaluate_sql(predicted, task["ground_truth_sql"], task["db_path"])

    elif tc == "anomaly_detection":
        score_data = evaluate_anomaly(
            r["content"] or "", task["ground_truth_indices"], task["n_rows"]
        )

    elif tc == "kpi_interpretation":
        task_input = json.dumps(task["prompt_vars"], ensure_ascii=False)
        score_data = judge_output(
            tc, task_input, r["content"] or "",
            ground_truth=task.get("reference_interpretation", "")
        )

    elif tc == "report_generation":
        task_input = json.dumps(task["prompt_vars"], ensure_ascii=False)
        score_data = judge_output(tc, task_input, r["content"] or "")

    scored.append({
        "task_class":   tc,
        "task_id":      r["task_id"],
        "difficulty":   r.get("difficulty"),
        "model_id":     r["model_id"],
        "strategy":     r["strategy"],
        "latency_s":    r.get("latency_s"),
        "input_tokens": r.get("input_tokens"),
        "output_tokens":r.get("output_tokens"),
        **score_data,
    })

    counters[tc] += 1
    if counters[tc] % 15 == 0:
        label = tc.replace("_", " ").title()
        print(f"  {label}: {counters[tc]}/45 bewertet")

with open(OUTPUT_FILE, "w") as f:
    for s in scored:
        f.write(json.dumps(s) + "\n")

print(f"\nFertig! {len(scored)} Scores in: {OUTPUT_FILE}")

from collections import defaultdict
by_class_strategy = defaultdict(lambda: defaultdict(list))
for s in scored:
    tc = s["task_class"]
    st = s["strategy"]
    if tc == "sql_generation":
        by_class_strategy[tc][st].append(s.get("score", 0))
    elif tc == "anomaly_detection":
        by_class_strategy[tc][st].append(s.get("f1", 0))
    else:
        by_class_strategy[tc][st].append(s.get("mean_score") or 0)

print("\nSchnellübersicht (Ø Score pro Klasse × Strategie):")
for tc in sorted(by_class_strategy):
    print(f"\n  {tc}:")
    for st in ["few_shot", "chain_of_thought", "structured_output"]:
        vals = by_class_strategy[tc][st]
        if vals:
            print(f"    {st}: {sum(vals)/len(vals):.3f}")