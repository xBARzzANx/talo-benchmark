"""
Shared constants and cache-resolution helpers for the Phase 3 evaluation
design described in Section 6 of the thesis: the five conditions B1-B5, and
lookup into the Phase 1 / Phase 2 / Phase 3 raw-result and score caches.

Used by run_phase3.py, evaluate_phase3.py, and analysis/phase3_analysis.py
so all three agree on the same condition definitions and data sources.
"""
import json
from pathlib import Path

import yaml

from talo.router import ALIGNMENT_MATRIX
from src.utils.io import load_tasks

PHASE1_RUN_IDS = ["20260522_2030", "20260522_2219_gemini_retry"]
PHASE2_RUN_ID = "20260523_1406"
PRICING_PATH = Path("configs/pricing.yaml")

TASK_CLASSES = ["sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation"]
JUDGED_CLASSES = {"kpi_interpretation", "report_generation"}
ZERO_SHOT = "zero_shot"

CONDITIONS = {
    "B1": {"label": "Static Single-Model", "resolve": lambda tc: ("gpt-4.1-mini", ZERO_SHOT)},
    "B2": {"label": "Single-Model Top Performer", "resolve": lambda tc: ("gemini-2.5-flash", ZERO_SHOT)},
    "B3": {"label": "Routing-Exclusive", "resolve": lambda tc: (ALIGNMENT_MATRIX[tc]["model_id"], ZERO_SHOT)},
    "B4": {"label": "Prompt-Exclusive", "resolve": lambda tc: ("gpt-4.1-mini", ALIGNMENT_MATRIX[tc]["strategy"])},
    "B5": {"label": "TALO", "resolve": lambda tc: (ALIGNMENT_MATRIX[tc]["model_id"], ALIGNMENT_MATRIX[tc]["strategy"])},
}
CONDITION_IDS = list(CONDITIONS.keys())

CLASS_LABELS = {
    "sql_generation": "SQL", "anomaly_detection": "Anomaly",
    "kpi_interpretation": "KPI", "report_generation": "Report",
}

# Fixed categorical color order (dataviz skill, references/palette.md) --
# assigned by condition identity, never re-cycled or re-ranked by value.
# Shared by analysis/phase3_analysis.py's chart and app.py's cost comparison.
CONDITION_COLORS = {
    "B1": "#2a78d6",  # blue
    "B2": "#eb6834",  # orange
    "B3": "#1baf7a",  # aqua
    "B4": "#eda100",  # yellow
    "B5": "#e87ba4",  # magenta
}

# Judge model is called via its litellm string in src/evaluators/llm_judge.py
# (anthropic/claude-sonnet-4-5). The pricing.yaml lookup key is the legacy
# "claude-sonnet-4-6" -- see configs/pricing.yaml for the full note.
# Do not rename this key.
JUDGE_PRICING_KEY = "claude-sonnet-4-6"

# Rough estimate: judge prompt = task input (JSON) + model output + a fixed
# ~180-word rubric, max_tokens=256. No historical token counts exist for
# judge calls (llm_judge.py does not record them), so this is a documented
# approximation, not a cache-derived average like the generation token counts.
JUDGE_EST_INPUT_TOKENS = 550
JUDGE_EST_OUTPUT_TOKENS = 120


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def load_pricing() -> dict:
    with open(PRICING_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_cost(model_id: str, input_tokens: float, output_tokens: float, pricing: dict) -> float:
    cfg = pricing["models"].get(model_id)
    if not cfg:
        return 0.0
    return (input_tokens * cfg["input"] + output_tokens * cfg["output"]) / 1_000_000


def estimate_tokens(all_rows: list[dict], task_class: str, model_id: str, strategy: str) -> tuple[float, float, str]:
    """
    Estimate (avg_input_tokens, avg_output_tokens, basis) for a not-yet-made
    generation call, from historical Phase 1/2 token counts. Falls back to
    progressively coarser proxies when no exact-match history exists:
    input tokens are driven mostly by (class, strategy) -- same schema/data/
    examples regardless of model; output tokens are driven mostly by
    (class, model) -- how verbose that model tends to be on that class.
    """
    def avg(rows: list[dict]) -> tuple[float, float] | None:
        if not rows:
            return None
        return (
            sum(r.get("input_tokens") or 0 for r in rows) / len(rows),
            sum(r.get("output_tokens") or 0 for r in rows) / len(rows),
        )

    def match(rows, **filters):
        return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]

    exact = match(all_rows, task_class=task_class, model_id=model_id, strategy=strategy)
    if exact:
        i, o = avg(exact)
        return i, o, "exact (class+model+strategy)"

    same_class_strategy = match(all_rows, task_class=task_class, strategy=strategy)
    same_class_model = match(all_rows, task_class=task_class, model_id=model_id)
    in_avg = avg(same_class_strategy)
    out_avg = avg(same_class_model)
    if in_avg and out_avg:
        return in_avg[0], out_avg[1], "approx (in: class+strategy / out: class+model)"

    same_class = match(all_rows, task_class=task_class)
    if same_class:
        i, o = avg(same_class)
        return i, o, "approx (class average)"

    return 0.0, 0.0, "no historical data"


# --------------------------------------------------------------------------
# Cache loading
# --------------------------------------------------------------------------

def load_phase1_cache() -> dict[tuple, dict]:
    """(task_class, task_id, model_id) -> raw result row. Zero-shot only."""
    rows: dict[tuple, dict] = {}
    for run_id in PHASE1_RUN_IDS:
        path = Path(f"results/phase1/{run_id}/results.jsonl")
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("error"):
                    continue
                rows[(r["task_class"], r["task_id"], r["model_id"])] = r
    return rows


def load_phase2_cache() -> dict[tuple, dict]:
    """(task_class, task_id, model_id, strategy) -> raw result row."""
    rows: dict[tuple, dict] = {}
    path = Path(f"results/phase2/{PHASE2_RUN_ID}/results.jsonl")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("error"):
                    continue
                rows[(r["task_class"], r["task_id"], r["model_id"], r["strategy"])] = r
    return rows


def load_phase3_cache(run_id: str | None, purpose: str = "condition_fill") -> dict[tuple, dict]:
    """(task_class, task_id, model_id, strategy) -> raw result row for a given run_id/purpose."""
    rows: dict[tuple, dict] = {}
    if not run_id:
        return rows
    path = Path(f"results/phase3/{run_id}/results.jsonl")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("error") and r.get("purpose") == purpose:
                    rows[(r["task_class"], r["task_id"], r["model_id"], r["strategy"])] = r
    return rows


def load_all_phase3_cache(purpose: str = "condition_fill", root: str = "results/phase3") -> dict[tuple, dict]:
    """Aggregate condition_fill rows across every results/phase3/<run_id>/ directory."""
    rows: dict[tuple, dict] = {}
    base = Path(root)
    if not base.exists():
        return rows
    for run_dir in sorted(base.iterdir()):
        if run_dir.is_dir():
            rows.update(load_phase3_cache(run_dir.name, purpose=purpose))
    return rows


def find_latest(pattern: str, root: str = "results") -> Path | None:
    candidates = sorted(Path(root).glob(pattern))
    return candidates[-1] if candidates else None


def load_existing_judge_scores() -> dict[tuple, dict]:
    """(task_class, task_id, model_id, strategy) -> already-computed judge score row."""
    scores: dict[tuple, dict] = {}
    for pattern in ("phase1_scores_*.jsonl", "phase2_scores_*.jsonl", "phase3_scores_*.jsonl"):
        path = find_latest(pattern)
        if not path:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["task_class"] not in JUDGED_CLASSES or row.get("mean_score") is None:
                    continue
                key = (row["task_class"], row["task_id"], row["model_id"], row.get("strategy", ZERO_SHOT))
                scores[key] = row
    return scores


def load_judged_keys() -> set[tuple]:
    return set(load_existing_judge_scores().keys())


def resolve_row(
    tc: str, task_id: str, model_id: str, strategy: str,
    phase1_cache: dict, phase2_cache: dict, phase3_cache: dict,
) -> tuple[dict | None, str]:
    """Find the raw generation row for (task_class, task_id, model_id, strategy)."""
    if strategy == ZERO_SHOT:
        row = phase1_cache.get((tc, task_id, model_id))
        if row:
            return row, "phase1"
    else:
        row = phase2_cache.get((tc, task_id, model_id, strategy))
        if row:
            return row, "phase2"
    row = phase3_cache.get((tc, task_id, model_id, strategy))
    if row:
        return row, "phase3"
    return None, "missing"


# --------------------------------------------------------------------------
# Cost / latency (full per-condition cost, not just this run's new/marginal
# calls -- "what would it cost to run this condition from scratch")
# --------------------------------------------------------------------------

def condition_cost_latency(pricing: dict) -> dict[str, dict]:
    """
    For each condition: total cost across all 60 tasks (generation, exact
    tokens from the Phase 1/2/3 cache, aggregated over all four task
    classes -- not scoped to a single class) plus an approximate judge cost
    for the judged classes, and mean generation latency.

    Single source of truth for analysis/output/cost_latency.md and for
    app.py's cost comparison chart, which reads only `generation_cost` from
    this and ignores `judge_cost` (Design-Entscheidung E6): the demo's cost
    comparison is about routing/prompting choices, not the fixed per-task
    evaluation overhead, so it shows generation cost only, aggregated
    across all four classes.
    """
    phase1_cache = load_phase1_cache()
    phase2_cache = load_phase2_cache()
    phase3_cache = load_all_phase3_cache()

    out = {}
    for cid in CONDITION_IDS:
        gen_cost = 0.0
        judge_cost = 0.0
        latencies = []
        n_gen = 0
        n_judge = 0
        for tc in TASK_CLASSES:
            model_id, strategy = CONDITIONS[cid]["resolve"](tc)
            for task in load_tasks(tc):
                row, _source = resolve_row(tc, task["id"], model_id, strategy, phase1_cache, phase2_cache, phase3_cache)
                if row is None:
                    continue
                gen_cost += call_cost(model_id, row.get("input_tokens") or 0, row.get("output_tokens") or 0, pricing)
                if row.get("latency_s") is not None:
                    latencies.append(row["latency_s"])
                n_gen += 1
                if tc in JUDGED_CLASSES:
                    judge_cost += call_cost(JUDGE_PRICING_KEY, JUDGE_EST_INPUT_TOKENS, JUDGE_EST_OUTPUT_TOKENS, pricing)
                    n_judge += 1
        out[cid] = {
            "generation_cost": gen_cost, "judge_cost": judge_cost,
            "total_cost": gen_cost + judge_cost,
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
            "n_generation_calls": n_gen, "n_judge_calls": n_judge,
        }
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def normalize(task_class: str, raw_score: float) -> float:
    """Section 6, E1: SQL/Anomaly already on [0,1]; KPI/Report (1-5) -> [0,1]."""
    if task_class in ("sql_generation", "anomaly_detection"):
        return raw_score
    return (raw_score - 1) / 4
