"""
Phase 3 runner: the five evaluation conditions (B1-B5) for Chapter 6.

Cache-first: reuses Phase 1 (zero-shot, all models) and
Phase 2 (three strategies, top model per class) raw model responses wherever
a condition's resolved (model, strategy) per task class is already covered.
New generation calls are only made where a cache gap exists.

Per the Chapter 6 design (Section 6, Design-Entscheidung E4):
  - B1, B2, B3 are always 100% Phase-1-cache (every model was run zero-shot
    on every class in Phase 1).
  - B4 needs new generation for exactly the task classes where GPT-4.1-mini's
    Phase-3-optimal strategy was not part of Phase 2 (sql_generation/few_shot,
    kpi_interpretation/chain_of_thought -- 2 x 15 = 30 calls).
  - B5's reported scores always come from the Phase-2-cache (its per-class
    model+strategy is exactly Phase 2's top-model-per-class). Additionally,
    B5 requires ONE mandatory live end-to-end run across all 60 tasks as a
    functional proof of the orchestrator pipeline (60 calls). Live results
    must match the cached ones at temperature=0.0; deviations are logged,
    not used to override the reported (cache) scores.

Expected total when running B4 + B5 for real: ~90 new API calls.

No API calls are made without an explicit, itemized cost estimate and a
confirmation prompt, unless --yes is passed.

Usage:
  python run_phase3.py --dry-run
  python run_phase3.py --dry-run --condition B4
  python run_phase3.py --condition B4            # shows cost, asks to confirm
  python run_phase3.py --condition all --yes      # scripted, skips the prompt
"""
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import track

from talo.prompt_selector import PromptSelector, PromptSelectionError
from src.models.model_client import call_model
from src.utils.io import load_tasks, save_result
from src.utils.phase3 import (
    TASK_CLASSES, JUDGED_CLASSES, CONDITIONS, CONDITION_IDS,
    JUDGE_PRICING_KEY, JUDGE_EST_INPUT_TOKENS, JUDGE_EST_OUTPUT_TOKENS,
    load_pricing, call_cost, estimate_tokens, resolve_row,
    load_phase1_cache, load_phase2_cache, load_all_phase3_cache, load_judged_keys,
)

console = Console(width=120)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

@dataclass
class TaskPlan:
    task_id: str
    difficulty: str | None
    cached: bool
    source: str  # "phase1" | "phase2" | "new"
    judge_needed: bool


@dataclass
class ClassPlan:
    task_class: str
    model_id: str
    strategy: str
    tasks: list[TaskPlan] = field(default_factory=list)

    @property
    def n_cached(self) -> int:
        return sum(1 for t in self.tasks if t.cached)

    @property
    def n_new(self) -> int:
        return sum(1 for t in self.tasks if not t.cached)

    @property
    def n_judge_new(self) -> int:
        return sum(1 for t in self.tasks if t.judge_needed)


@dataclass
class ConditionPlan:
    condition_id: str
    label: str
    classes: dict[str, ClassPlan] = field(default_factory=dict)

    @property
    def n_new_total(self) -> int:
        return sum(c.n_new for c in self.classes.values())

    @property
    def n_judge_new_total(self) -> int:
        return sum(c.n_judge_new for c in self.classes.values())


def build_plan(
    condition_ids: list[str],
    phase1_cache: dict, phase2_cache: dict, phase3_cache: dict, judged_keys: set,
) -> dict[str, ConditionPlan]:
    plans: dict[str, ConditionPlan] = {}
    for cid in condition_ids:
        cond = CONDITIONS[cid]
        classes: dict[str, ClassPlan] = {}
        for tc in TASK_CLASSES:
            model_id, strategy = cond["resolve"](tc)
            cplan = ClassPlan(task_class=tc, model_id=model_id, strategy=strategy)
            for t in load_tasks(tc):
                row, source = resolve_row(tc, t["id"], model_id, strategy, phase1_cache, phase2_cache, phase3_cache)
                judge_needed = (
                    tc in JUDGED_CLASSES
                    and (tc, t["id"], model_id, strategy) not in judged_keys
                )
                cplan.tasks.append(TaskPlan(
                    task_id=t["id"], difficulty=t.get("difficulty"),
                    cached=row is not None, source=source, judge_needed=judge_needed,
                ))
            classes[tc] = cplan
        plans[cid] = ConditionPlan(condition_id=cid, label=cond["label"], classes=classes)
    return plans


# --------------------------------------------------------------------------
# Token / cost estimation
# --------------------------------------------------------------------------

@dataclass
class CostLine:
    label: str
    n_calls: int
    est_cost: float
    basis: str = ""


def estimate_condition_cost(
    plan: ConditionPlan, all_rows: list[dict], pricing: dict,
) -> list[CostLine]:
    lines = []
    for tc, cplan in plan.classes.items():
        if cplan.n_new == 0:
            continue
        in_tok, out_tok, basis = estimate_tokens(all_rows, tc, cplan.model_id, cplan.strategy)
        cost = call_cost(cplan.model_id, in_tok, out_tok, pricing) * cplan.n_new
        lines.append(CostLine(
            label=f"{tc} generation ({cplan.model_id}/{cplan.strategy})",
            n_calls=cplan.n_new, est_cost=cost, basis=basis,
        ))
        if cplan.n_judge_new:
            judge_cost = call_cost(
                JUDGE_PRICING_KEY, JUDGE_EST_INPUT_TOKENS, JUDGE_EST_OUTPUT_TOKENS, pricing,
            ) * cplan.n_judge_new
            lines.append(CostLine(
                label=f"{tc} judge scoring", n_calls=cplan.n_judge_new,
                est_cost=judge_cost, basis="approx (no historical judge token data)",
            ))
    return lines


def estimate_b5_live_verification_cost(all_rows: list[dict], pricing: dict) -> list[CostLine]:
    """
    B5's mandatory live end-to-end run (Section 6, E4): all 60 tasks, each
    class's model+strategy is exactly Phase 2's top-model-per-class, so an
    exact-match token estimate is available for all four classes.
    """
    lines = []
    for tc in TASK_CLASSES:
        model_id, strategy = CONDITIONS["B5"]["resolve"](tc)
        in_tok, out_tok, basis = estimate_tokens(all_rows, tc, model_id, strategy)
        n = len(load_tasks(tc))
        cost = call_cost(model_id, in_tok, out_tok, pricing) * n
        lines.append(CostLine(
            label=f"{tc} live verification ({model_id}/{strategy})",
            n_calls=n, est_cost=cost, basis=basis,
        ))
    return lines


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_plan(plans: dict[str, ConditionPlan], all_rows: list[dict], pricing: dict) -> float:
    grand_total = 0.0

    for cid, plan in plans.items():
        table = Table(title=f"{cid} -- {plan.label}")
        table.add_column("Task Class")
        table.add_column("Model")
        table.add_column("Strategy")
        table.add_column("Cached", justify="right")
        table.add_column("New", justify="right")
        table.add_column("Source")
        table.add_column("New Judge", justify="right")

        for tc, cplan in plan.classes.items():
            cached_sources = sorted({t.source for t in cplan.tasks if t.cached})
            source = "+".join(cached_sources) if cached_sources else "new"
            if cplan.n_new and cached_sources:
                source += " + new"
            table.add_row(
                tc, cplan.model_id, cplan.strategy,
                str(cplan.n_cached), str(cplan.n_new), source,
                str(cplan.n_judge_new) if tc in JUDGED_CLASSES else "-",
            )
        console.print(table)

        if plan.n_new_total == 0 and plan.n_judge_new_total == 0:
            console.print(f"  [green]{cid} vollstaendig aus dem Cache bedient -- 0 neue Calls.[/green]\n")
            continue

        cost_lines = estimate_condition_cost(plan, all_rows, pricing)
        condition_total = sum(cl.est_cost for cl in cost_lines)
        grand_total += condition_total
        for cl in cost_lines:
            console.print(
                f"    {cl.label}: {cl.n_calls} Calls, ~${cl.est_cost:.4f} ({cl.basis})"
            )
        console.print(f"  [yellow]{cid} Zwischensumme: ~${condition_total:.4f}[/yellow]\n")

    if "B5" in plans:
        console.print("[bold]B5 -- Pflicht-Live-Verifikation (Section 6, E4)[/bold]")
        console.print(
            "  Die B5-Scores in der Ergebnistabelle stammen immer aus dem Phase-2-Cache "
            "(oben: 0 neue Calls). Zusaetzlich schreibt das Framework-Design einen "
            "einmaligen Live-Durchlauf ueber alle 60 Tasks als Funktionsnachweis vor:"
        )
        b5_lines = estimate_b5_live_verification_cost(all_rows, pricing)
        b5_total = sum(cl.est_cost for cl in b5_lines)
        grand_total += b5_total
        for cl in b5_lines:
            console.print(f"    {cl.label}: {cl.n_calls} Calls, ~${cl.est_cost:.4f} ({cl.basis})")
        console.print(f"  [yellow]Live-Verifikation Zwischensumme: ~${b5_total:.4f}[/yellow]\n")

    console.print(f"[bold]Geschaetzte Gesamtkosten: ~${grand_total:.4f}[/bold]")
    return grand_total


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def run_condition(cid: str, plan: ConditionPlan, run_id: str, prompt_selector: PromptSelector) -> None:
    for tc, cplan in plan.classes.items():
        new_tasks = [t for t in cplan.tasks if not t.cached]
        if not new_tasks:
            continue
        console.print(f"  [cyan]{cid}/{tc}[/cyan]: {len(new_tasks)} neue Calls ({cplan.model_id}/{cplan.strategy})")
        tasks_by_id = {t["id"]: t for t in load_tasks(tc)}
        for tp in track(new_tasks, description=f"    {tc}"):
            task = tasks_by_id[tp.task_id]
            try:
                prompt = prompt_selector.build_prompt(tc, cplan.strategy, **task["prompt_vars"])
            except PromptSelectionError as e:
                console.print(f"    [red]Prompt-Fehler {tp.task_id}: {e}[/red]")
                continue
            response = call_model(cplan.model_id, prompt)
            result = {
                "run_id": run_id, "phase": "phase3", "condition": cid, "purpose": "condition_fill",
                "task_class": tc, "task_id": tp.task_id, "difficulty": tp.difficulty,
                "model_id": cplan.model_id, "strategy": cplan.strategy,
                **response,
            }
            save_result(result, "phase3", run_id)


def run_b5_live_verification(run_id: str, phase2_cache: dict, prompt_selector: PromptSelector) -> None:
    console.print("\n[bold]B5 Pflicht-Live-Verifikation ueber alle 60 Tasks[/bold]")
    deviations = []
    checked = 0
    for tc in TASK_CLASSES:
        model_id, strategy = CONDITIONS["B5"]["resolve"](tc)
        for task in track(load_tasks(tc), description=f"  {tc}"):
            try:
                prompt = prompt_selector.build_prompt(tc, strategy, **task["prompt_vars"])
            except PromptSelectionError as e:
                console.print(f"    [red]Prompt-Fehler {task['id']}: {e}[/red]")
                continue
            response = call_model(model_id, prompt)
            result = {
                "run_id": run_id, "phase": "phase3", "condition": "B5", "purpose": "live_verification",
                "task_class": tc, "task_id": task["id"], "difficulty": task.get("difficulty"),
                "model_id": model_id, "strategy": strategy,
                **response,
            }
            save_result(result, "phase3", run_id)
            checked += 1

            cached = phase2_cache.get((tc, task["id"], model_id, strategy))
            if cached and response.get("content") != cached.get("content"):
                deviations.append({
                    "task_class": tc, "task_id": task["id"], "model_id": model_id, "strategy": strategy,
                    "cached_content": cached.get("content"), "live_content": response.get("content"),
                })

    report_path = Path("results/phase3") / run_id / "live_verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"checked": checked, "deviations": deviations}, f, indent=2, ensure_ascii=False)

    if deviations:
        console.print(f"  [red]{len(deviations)}/{checked} Tasks weichen vom Cache ab.[/red] Details: {report_path}")
    else:
        console.print(f"  [green]Alle {checked} Live-Ergebnisse stimmen mit dem Phase-2-Cache ueberein.[/green]")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Nur Plan und Kostenschaetzung anzeigen, keine Calls.")
    parser.add_argument("--condition", default="all", choices=CONDITION_IDS + ["all"])
    parser.add_argument("--yes", action="store_true", help="Bestaetigungsabfrage vor echten Calls ueberspringen.")
    args = parser.parse_args()

    condition_ids = CONDITION_IDS if args.condition == "all" else [args.condition]

    console.print("[bold]TALO Phase 3 Runner[/bold]")
    console.print(f"Bedingungen: {condition_ids}\n")

    phase1_cache = load_phase1_cache()
    phase2_cache = load_phase2_cache()
    phase3_cache = load_all_phase3_cache()
    judged_keys = load_judged_keys()
    pricing = load_pricing()
    all_rows = list(phase1_cache.values()) + list(phase2_cache.values())

    plans = build_plan(condition_ids, phase1_cache, phase2_cache, phase3_cache, judged_keys)
    total_cost = print_plan(plans, all_rows, pricing)

    if args.dry_run:
        console.print("\n[yellow]Dry Run -- keine Calls ausgefuehrt.[/yellow]")
        return

    if total_cost == 0 and "B5" not in condition_ids:
        console.print("\n[green]Nichts zu tun -- alle gewaehlten Bedingungen sind vollstaendig gecacht.[/green]")
        return

    if not args.yes:
        console.print(
            f"\n[bold]Bestaetigung erforderlich:[/bold] geschaetzte Kosten ~${total_cost:.4f} "
            f"fuer echte API-Calls."
        )
        answer = input("Fortfahren? [y/N] ").strip().lower()
        if answer != "y":
            console.print("Abgebrochen.")
            return

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    prompt_selector = PromptSelector()

    for cid in condition_ids:
        run_condition(cid, plans[cid], run_id, prompt_selector)

    if "B5" in condition_ids:
        run_b5_live_verification(run_id, phase2_cache, prompt_selector)

    console.print(f"\n[bold green]Phase 3 Run abgeschlossen. Run ID: {run_id}[/bold green]")


if __name__ == "__main__":
    main()
