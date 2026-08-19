"""
Diagnose-Skript: Prueft zwei potenzielle Confounds in der Evaluationslogik.

A) SQL-Extraktion: Phase 1 und Phase 2 verwenden unterschiedliche Extraktoren.
   Dieses Skript re-scored alle SQL-Tasks beider Phasen unter drei Varianten
   und zeigt, ob und wo sich die Scores unterscheiden.

B) Anomaly-Parsing: Der Regex-Fallback koennte bei Chain-of-Thought-Outputs
   Zahlen aus dem Begleittext als Anomalie-Indizes einsammeln. Dieses Skript
   vergleicht den aktuellen Parser mit einem strikten Parser, der nur die
   letzte JSON-Liste im Output beruecksichtigt.

KOSTEN: 0 API-Calls. Nur lokale SQLite-Ausfuehrung und String-Parsing.

Ausfuehren im Repo-Root:  python scripts/diagnose_scoring.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluators.sql_eval import evaluate_sql
from src.evaluators.anomaly_eval import evaluate_anomaly, parse_anomaly_indices
from src.utils.io import load_tasks

# Identisch zu evaluate_phase1.py / evaluate_phase2.py
PHASE1_RUN_IDS = ["20260522_2030", "20260522_2219_gemini_retry"]
PHASE2_RUN_ID = "20260523_1406"

TASK_CLASSES = ["sql_generation", "anomaly_detection",
                "kpi_interpretation", "report_generation"]


# --------------------------------------------------------------------------
# Daten laden (exakt die Filterlogik der bestehenden Evaluationsskripte)
# --------------------------------------------------------------------------

def load_phase1():
    results = []
    for run_id in PHASE1_RUN_IDS:
        path = Path(f"results/phase1/{run_id}/results.jsonl")
        if not path.exists():
            print(f"  WARNUNG: {path} nicht gefunden")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    seen = {}
    for r in results:
        seen[(r["task_class"], r["task_id"], r["model_id"])] = r
    return [r for r in seen.values() if not r.get("error")]


def load_phase2():
    path = Path(f"results/phase2/{PHASE2_RUN_ID}/results.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return [r for r in out if not r.get("error")]


def load_task_lookup():
    lookup = {}
    for tc in TASK_CLASSES:
        for t in load_tasks(tc):
            lookup[(tc, t["id"])] = t
    return lookup


# --------------------------------------------------------------------------
# A) SQL-Extraktoren
# --------------------------------------------------------------------------

def extract_sql_p1(content: str) -> str:
    """Extraktor aus evaluate_phase1.py."""
    return re.sub(r"```sql|```", "", content or "").strip()


def extract_sql_p2(content: str) -> str:
    """Extraktor aus evaluate_phase2.py."""
    predicted = re.sub(r"```sql|```", "", content or "").strip()
    if "SELECT" in predicted.upper():
        sql_lines, in_sql = [], False
        for line in predicted.split("\n"):
            if any(kw in line.upper() for kw in ["SELECT", "WITH", "INSERT"]):
                in_sql = True
            if in_sql:
                sql_lines.append(line)
        predicted = "\n".join(sql_lines).strip()
    return predicted


def extract_sql_unified(content: str) -> str:
    """
    Einheitlicher Extraktor fuer alle Bedingungen.

    Vorgehen, in dieser Reihenfolge:
    1. Wenn ein ```sql-Fence existiert: dessen Inhalt nehmen (letzter Block).
    2. Sonst: ab dem letzten Vorkommen von SELECT/WITH bis zum Ende bzw.
       bis zur ersten Leerzeile nach einem Semikolon.
    3. Trailing-Text nach dem ersten Semikolon abschneiden.
    """
    if not content:
        return ""
    text = content.strip()

    fences = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fences:
        candidate = fences[-1].strip()
    else:
        m = list(re.finditer(r"\b(SELECT|WITH)\b", text, re.IGNORECASE))
        if not m:
            return text
        candidate = text[m[-1].start():].strip()

    if ";" in candidate:
        candidate = candidate.split(";")[0].strip() + ";"
    return candidate


SQL_EXTRACTORS = {
    "p1_logic": extract_sql_p1,
    "p2_logic": extract_sql_p2,
    "unified": extract_sql_unified,
}


def diagnose_sql(phase1, phase2, tasks):
    print("\n" + "=" * 78)
    print("A) SQL-EXTRAKTION — Score je Extraktor")
    print("=" * 78)

    rows = []
    for r in phase1:
        if r["task_class"] == "sql_generation":
            rows.append(("phase1", r))
    for r in phase2:
        if r["task_class"] == "sql_generation":
            rows.append(("phase2", r))

    # scores[(phase, model, strategy)][extractor] = [0/1, ...]
    scores = defaultdict(lambda: defaultdict(list))
    disagreements = []

    for phase, r in rows:
        task = tasks.get(("sql_generation", r["task_id"]))
        if not task:
            continue
        key = (phase, r["model_id"], r.get("strategy", "zero_shot"))
        per_task = {}
        for name, fn in SQL_EXTRACTORS.items():
            res = evaluate_sql(fn(r["content"]), task["ground_truth_sql"], task["db_path"])
            scores[key][name].append(res["score"])
            per_task[name] = res["score"]
        if len(set(per_task.values())) > 1:
            disagreements.append((phase, r["model_id"], r.get("strategy", "zero_shot"),
                                  r["task_id"], per_task))

    print(f"\n{'Phase':<8}{'Modell':<20}{'Strategie':<20}"
          f"{'p1_logic':>10}{'p2_logic':>10}{'unified':>10}")
    print("-" * 78)
    for key in sorted(scores):
        phase, model, strategy = key
        vals = scores[key]
        n = len(vals["unified"])
        line = f"{phase:<8}{model:<20}{strategy:<20}"
        for name in ["p1_logic", "p2_logic", "unified"]:
            mean = sum(vals[name]) / n if n else 0
            line += f"{mean:>10.3f}"
        print(line + f"   (n={n})")

    print(f"\nTasks mit abweichendem Score je nach Extraktor: {len(disagreements)}")
    for d in disagreements[:15]:
        print(f"  {d[0]} | {d[1]} | {d[2]} | {d[3]} -> {d[4]}")
    if len(disagreements) > 15:
        print(f"  ... und {len(disagreements) - 15} weitere")

    if not disagreements:
        print("  -> KEINE Abweichung. Der Extraktor-Unterschied ist folgenlos.")
        print("     Kapitel 4 bleibt unveraendert.")
    else:
        print("\n  -> ABWEICHUNG VORHANDEN. Der Extraktor beeinflusst die Scores.")
        print("     Fuer Kapitel 6 muss ein einheitlicher Extraktor verwendet werden.")
        print("     Pruefen, ob sich dadurch Kapitel-4-Zahlen aendern.")


# --------------------------------------------------------------------------
# B) Anomaly-Parser
# --------------------------------------------------------------------------

def parse_anomaly_strict(model_output: str, n_rows: int) -> list[int]:
    """
    Strikter Parser: beruecksichtigt nur eine explizite JSON-Liste.

    1. Gesamten Output als JSON parsen.
    2. Sonst: letzte eckige Klammer-Gruppe im Text als JSON parsen.
    3. Sonst: leere Liste (KEIN Fallback auf freie Zahlen im Text).
    """
    if not model_output:
        return []
    try:
        parsed = json.loads(model_output.strip())
        if isinstance(parsed, list):
            return sorted({int(i) for i in parsed if 0 <= int(i) < n_rows})
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    cleaned = re.sub(r"```json|```", "", model_output)
    blocks = re.findall(r"\[[^\[\]]*\]", cleaned)
    for block in reversed(blocks):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                return sorted({int(i) for i in parsed
                               if isinstance(i, (int, float)) and 0 <= int(i) < n_rows})
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return []


def f1_from_indices(pred: list[int], gt: list[int], n_rows: int) -> float:
    pred_s, gt_s = set(pred), set(gt)
    tp = len(pred_s & gt_s)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_s)
    recall = tp / len(gt_s)
    return round(2 * precision * recall / (precision + recall), 3)


def diagnose_anomaly(phase1, phase2, tasks):
    print("\n" + "=" * 78)
    print("B) ANOMALY-PARSING — aktueller Parser vs. strikter Parser")
    print("=" * 78)

    rows = []
    for r in phase1:
        if r["task_class"] == "anomaly_detection":
            rows.append(("phase1", r))
    for r in phase2:
        if r["task_class"] == "anomaly_detection":
            rows.append(("phase2", r))

    scores = defaultdict(lambda: {"current": [], "strict": []})
    fallback_hits = []

    for phase, r in rows:
        task = tasks.get(("anomaly_detection", r["task_id"]))
        if not task:
            continue
        n_rows = task["n_rows"]
        gt = task["ground_truth_indices"]
        content = r["content"] or ""

        cur = evaluate_anomaly(content, gt, n_rows)
        strict_idx = parse_anomaly_strict(content, n_rows)
        strict_f1 = f1_from_indices(strict_idx, gt, n_rows)

        key = (phase, r["model_id"], r.get("strategy", "zero_shot"))
        scores[key]["current"].append(cur["f1"])
        scores[key]["strict"].append(strict_f1)

        cur_idx = sorted(set(parse_anomaly_indices(content, n_rows)))
        if cur_idx != strict_idx:
            fallback_hits.append({
                "phase": phase,
                "model": r["model_id"],
                "strategy": r.get("strategy", "zero_shot"),
                "task_id": r["task_id"],
                "ground_truth": sorted(gt),
                "current_parser": cur_idx,
                "strict_parser": strict_idx,
                "f1_current": cur["f1"],
                "f1_strict": strict_f1,
            })

    print(f"\n{'Phase':<8}{'Modell':<20}{'Strategie':<20}"
          f"{'F1 aktuell':>12}{'F1 strikt':>12}{'Delta':>9}")
    print("-" * 81)
    for key in sorted(scores):
        phase, model, strategy = key
        cur = scores[key]["current"]
        strict = scores[key]["strict"]
        n = len(cur)
        mc = sum(cur) / n if n else 0
        ms = sum(strict) / n if n else 0
        print(f"{phase:<8}{model:<20}{strategy:<20}"
              f"{mc:>12.3f}{ms:>12.3f}{ms - mc:>+9.3f}")

    print(f"\nTasks mit abweichendem Index-Set: {len(fallback_hits)}")
    for h in fallback_hits[:15]:
        print(f"  {h['phase']} | {h['model']} | {h['strategy']} | {h['task_id']}")
        print(f"      GT={h['ground_truth']}  aktuell={h['current_parser']} "
              f"(F1 {h['f1_current']})  strikt={h['strict_parser']} (F1 {h['f1_strict']})")
    if len(fallback_hits) > 15:
        print(f"  ... und {len(fallback_hits) - 15} weitere")

    cot = [h for h in fallback_hits if h["strategy"] == "chain_of_thought"]
    print(f"\n  Davon Chain-of-Thought: {len(cot)}")
    if cot:
        delta = sum(h["f1_strict"] - h["f1_current"] for h in cot) / len(cot)
        print(f"  Mittleres F1-Delta bei CoT durch strikten Parser: {delta:+.3f}")
        if delta > 0.05:
            print("  -> Der Regex-Fallback benachteiligt CoT messbar.")
            print("     Die Aussage in 4.6.2 ('reasoning errors rather than formatting")
            print("     artifacts') ist so nicht haltbar und muss revidiert werden.")
        else:
            print("  -> Kein nennenswerter Effekt. Die Aussage in 4.6.2 ist gedeckt.")

    Path("results").mkdir(exist_ok=True)
    out = Path("results/diagnose_parser_disagreements.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fallback_hits, f, indent=2, ensure_ascii=False)
    print(f"\n  Vollstaendige Abweichungsliste gespeichert: {out}")


# --------------------------------------------------------------------------

def main():
    print("Lade Daten ...")
    tasks = load_task_lookup()
    phase1 = load_phase1()
    phase2 = load_phase2()
    print(f"  Phase 1: {len(phase1)} Results (nach Dedup und Fehlerfilter)")
    print(f"  Phase 2: {len(phase2)} Results")
    print(f"  Tasks:   {len(tasks)}")

    diagnose_sql(phase1, phase2, tasks)
    diagnose_anomaly(phase1, phase2, tasks)

    print("\n" + "=" * 78)
    print("FERTIG.")
    print("=" * 78)


if __name__ == "__main__":
    main()