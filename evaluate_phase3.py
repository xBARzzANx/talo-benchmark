"""
Phase 3 evaluation: scores all five conditions (B1-B5) x four task classes,
using the same evaluators and the same primary extractor variants as
Phase 1/2 (src/evaluators/extractors.py, keyword_line for SQL and permissive
for anomaly parsing -- see CLAUDE.md, Section 5).

Data sources per condition/class cell, cache-first (CLAUDE.md, Regel 3):
  1. Phase 1 cache (zero-shot) for B1/B2/B3, and for any B4/B5 cell whose
     resolved strategy happens to be zero_shot (none currently).
  2. Phase 2 cache (top-model-per-class, 3 strategies) for B4 cells that
     match Phase 2's model, and for all of B5 (whose per-class model+
     strategy is exactly Phase 2's top-model-per-class).
  3. results/phase3/<run-id>/results.jsonl (purpose="condition_fill") for
     B4's two new cells (sql_generation/few_shot, kpi_interpretation/
     chain_of_thought on gpt-4.1-mini). Requires --run-id, produced by
     run_phase3.py.

B5's mandatory live-verification rows (purpose="live_verification") are
NEVER used for scoring -- B5 is always scored from the Phase 2 cache, per
Section 6, Design-Entscheidung E4. Their only purpose is the reproducibility
check performed by run_phase3.py itself.

Normalization (Section 6, E1): SQL execution accuracy and anomaly F1 are
already on [0, 1]. KPI/Report LLM-as-Judge mean scores (1-5) are normalized
via (score - 1) / 4. A condition's overall score is the unweighted macro-
mean of its four (normalized) class scores.

LLM-as-Judge scores for KPI/Report are reused from the existing Phase 1/2
score files wherever the exact (task_class, task_id, model_id, strategy)
combination was already judged (cache-first). Any remaining combination is
judged live -- this costs real API calls, so --dry-run and a confirmation
prompt gate it, exactly as in run_phase3.py.

Usage:
  python evaluate_phase3.py --dry-run
  python evaluate_phase3.py --run-id 20260524_1000
  python evaluate_phase3.py --run-id 20260524_1000 --condition B4 --yes
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.evaluators.sql_eval import evaluate_sql
from src.evaluators.anomaly_eval import evaluate_anomaly
from src.evaluators.llm_judge import judge_output
from src.evaluators.extractors import extract_sql_keyword_line
from src.utils.io import load_tasks
from src.utils.phase3 import (
    TASK_CLASSES, CONDITIONS, CONDITION_IDS,
    load_phase1_cache, load_phase2_cache, load_phase3_cache,
    load_existing_judge_scores, resolve_row, normalize,
)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_row(tc: str, row: dict, task: dict, existing_judge_scores: dict,
              model_id: str, strategy: str, dry_run: bool, new_judge_calls: list) -> dict:
    content = row.get("content") or ""

    if tc == "sql_generation":
        predicted = extract_sql_keyword_line(content)
        result = evaluate_sql(predicted, task["ground_truth_sql"], task["db_path"])
        return {"raw_score": result["score"], "detail": result}

    if tc == "anomaly_detection":
        result = evaluate_anomaly(content, task["ground_truth_indices"], task["n_rows"])
        return {"raw_score": result["f1"], "detail": result}

    key = (tc, task["id"], model_id, strategy)
    cached_judge = existing_judge_scores.get(key)
    if cached_judge:
        return {"raw_score": cached_judge["mean_score"], "detail": {"judge_status": "cached"}}

    if dry_run:
        new_judge_calls.append(key)
        return {"raw_score": None, "detail": {"judge_status": "pending"}}

    task_input = json.dumps(task["prompt_vars"], ensure_ascii=False)
    ground_truth = task.get("reference_interpretation", "") if tc == "kpi_interpretation" else ""
    judged = judge_output(tc, task_input, content, ground_truth=ground_truth)
    return {"raw_score": judged.get("mean_score"), "detail": judged}


def run_evaluation(
    condition_ids: list[str], run_id: str | None, dry_run: bool,
) -> tuple[list[dict], list[tuple], list[tuple]]:
    phase1_cache = load_phase1_cache()
    phase2_cache = load_phase2_cache()
    phase3_cache = load_phase3_cache(run_id)
    existing_judge_scores = load_existing_judge_scores()

    tasks_lookup = {}
    for tc in TASK_CLASSES:
        for t in load_tasks(tc):
            tasks_lookup[(tc, t["id"])] = t

    scored = []
    new_judge_calls: list[tuple] = []
    missing = []

    for cid in condition_ids:
        resolve = CONDITIONS[cid]["resolve"]
        for tc in TASK_CLASSES:
            model_id, strategy = resolve(tc)
            for task in load_tasks(tc):
                row, source = resolve_row(tc, task["id"], model_id, strategy, phase1_cache, phase2_cache, phase3_cache)
                if row is None:
                    missing.append((cid, tc, task["id"], model_id, strategy))
                    continue
                score_data = score_row(
                    tc, row, task, existing_judge_scores, model_id, strategy, dry_run, new_judge_calls,
                )
                raw_score = score_data["raw_score"]
                scored.append({
                    "condition": cid, "task_class": tc, "task_id": task["id"],
                    "difficulty": task.get("difficulty"), "model_id": model_id, "strategy": strategy,
                    "source": source, "raw_score": raw_score,
                    "normalized_score": normalize(tc, raw_score) if raw_score is not None else None,
                    **{k: v for k, v in score_data["detail"].items() if k not in ("judge_model",)},
                })

    return scored, missing, new_judge_calls


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary(scored: list[dict], missing: list[tuple]) -> None:
    by_condition_class = defaultdict(list)
    for s in scored:
        if s["normalized_score"] is not None:
            by_condition_class[(s["condition"], s["task_class"])].append(s["normalized_score"])

    print(f"\n{'Condition':<6}{'sql':>8}{'anomaly':>10}{'kpi':>8}{'report':>8}{'overall':>10}")
    print("-" * 50)
    for cid in CONDITION_IDS:
        class_means = []
        cells = {}
        for tc in TASK_CLASSES:
            vals = by_condition_class.get((cid, tc), [])
            mean = sum(vals) / len(vals) if vals else None
            cells[tc] = mean
            if mean is not None:
                class_means.append(mean)
        overall = sum(class_means) / len(class_means) if len(class_means) == 4 else None
        row = f"{cid:<6}"
        for tc in TASK_CLASSES:
            v = cells[tc]
            row += f"{v:>8.3f}" if v is not None else f"{'--':>8}"
        row += f"{overall:>10.3f}" if overall is not None else f"{'incomplete':>11}"
        print(row)

    if missing:
        print(f"\n{len(missing)} Zellen ohne Daten (run_phase3.py zuerst ausfuehren):")
        for cid, tc, task_id, model_id, strategy in missing[:10]:
            print(f"  {cid} | {tc} | {task_id} | {model_id}/{strategy}")
        if len(missing) > 10:
            print(f"  ... und {len(missing) - 10} weitere")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Nur Deckung/geplante Judge-Calls anzeigen, keine Judge-Calls.")
    parser.add_argument("--condition", default="all", choices=CONDITION_IDS + ["all"])
    parser.add_argument("--run-id", default=None, help="results/phase3/<run-id>/ mit neu generierten B4-Zellen.")
    parser.add_argument("--yes", action="store_true", help="Bestaetigungsabfrage vor echten Judge-Calls ueberspringen.")
    args = parser.parse_args()

    condition_ids = CONDITION_IDS if args.condition == "all" else [args.condition]

    scored, missing, new_judge_calls = run_evaluation(condition_ids, args.run_id, dry_run=True)

    if new_judge_calls and not args.dry_run:
        print(f"{len(new_judge_calls)} neue Judge-Calls erforderlich (KPI/Report ohne gecachten Score).")
        if not args.yes:
            answer = input("Fortfahren? [y/N] ").strip().lower()
            if answer != "y":
                print("Abgebrochen.")
                return
        scored, missing, new_judge_calls = run_evaluation(condition_ids, args.run_id, dry_run=False)
    elif new_judge_calls:
        print(f"[Dry Run] {len(new_judge_calls)} neue Judge-Calls waeren erforderlich:")
        for key in new_judge_calls[:10]:
            print(f"  {key}")

    print_summary(scored, missing)

    if not args.dry_run:
        out_path = Path(f"results/phase3_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, default=str) + "\n")
        print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
